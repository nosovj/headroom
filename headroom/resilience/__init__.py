"""Upstream resilience: retry logic and circuit breaker.

This module provides:
- Exponential backoff retry with jitter for transient errors
- Circuit breaker pattern for fail-fast behavior
- Per-upstream URL tracking

Retry config:
- Delays: 100ms, 200ms, 400ms, 800ms, 1600ms (2x exponential)
- ±20% jitter to prevent thundering herd
- Max 5 retries
- Only retries on transient errors (502, 503, 504, timeout)
- Does NOT retry on 400 Bad Request or 401 Unauthorized

Circuit breaker config:
- Trip after 5 consecutive failures
- 30s recovery window before allowing probe
- Open→probe→close state transitions
- Per-upstream URL tracking
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar
from functools import wraps

import httpx

logger = logging.getLogger(__name__)

# Feature flags
RETRY_ENABLED = os.environ.get("HEADROOM_FEATURE_RETRY", "true").lower() == "true"
CIRCUIT_BREAKER_ENABLED = os.environ.get("HEADROOM_FEATURE_CIRCUIT_BREAKER", "true").lower() == "true"

# Retry defaults
DEFAULT_MAX_RETRIES = 5
DEFAULT_BASE_DELAY_MS = 100
DEFAULT_JITTER = 0.2  # ±20%

# Circuit breaker defaults
DEFAULT_FAILURE_THRESHOLD = 5
DEFAULT_RECOVERY_TIMEOUT_SECONDS = 30


# =============================================================================
# Retry Logic
# =============================================================================

@dataclass
class RetryStats:
    """Statistics for retry behavior."""
    total_retries: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    total_retry_time_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.successful_retries + self.failed_retries
        return self.successful_retries / total if total > 0 else 0.0


class RetryError(Exception):
    """Raised when all retries are exhausted."""
    def __init__(self, message: str, last_error: Exception | None = None):
        super().__init__(message)
        self.last_error = last_error


def calculate_backoff_delay(attempt: int, base_delay_ms: int = DEFAULT_BASE_DELAY_MS, jitter: float = DEFAULT_JITTER) -> float:
    """Calculate exponential backoff delay with jitter.

    Args:
        attempt: Retry attempt number (0-indexed).
        base_delay_ms: Base delay in milliseconds.
        jitter: Jitter factor (0.2 = ±20%).

    Returns:
        Delay in seconds.
    """
    # Exponential backoff: base * 2^attempt
    delay_ms = base_delay_ms * (2 ** attempt)

    # Add jitter: ±jitter percent
    jitter_range = delay_ms * jitter
    delay_ms += random.uniform(-jitter_range, jitter_range)

    # Cap at 1600ms (5th attempt)
    return min(delay_ms / 1000.0, 1.6)


def is_retryable_error(status_code: int | None, error: Exception | None) -> bool:
    """Check if an error is retryable (transient).

    Args:
        status_code: HTTP status code if available.
        error: Exception if available.

    Returns:
        True if the error is retryable.
    """
    # Transient HTTP errors
    if status_code in {502, 503, 504}:
        return True

    # Timeout errors
    if isinstance(error, (asyncio.TimeoutError, httpx.TimeoutException)):
        return True

    # Connection errors
    if isinstance(error, (httpx.ConnectError, httpx.NetworkError)):
        return True

    return False


def is_non_retryable_error(status_code: int | None) -> bool:
    """Check if an error is non-retryable (should fail fast).

    Args:
        status_code: HTTP status code if available.

    Returns:
        True if the error should NOT be retried.
    """
    # Client errors - don't retry
    if status_code in {400, 401, 403, 404, 422}:
        return True
    return False


async def retry_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay_ms: int = DEFAULT_BASE_DELAY_MS,
    jitter: float = DEFAULT_JITTER,
    retryable: Callable[[int, Exception | None], bool] | None = None,
    stats: RetryStats | None = None,
    **kwargs: Any,
) -> Any:
    """Execute a function with exponential backoff retry.

    Args:
        func: Async function to execute.
        *args: Arguments to pass to func.
        max_retries: Maximum number of retry attempts.
        base_delay_ms: Base delay in milliseconds.
        jitter: Jitter factor (0.2 = ±20%).
        retryable: Custom function to determine if error is retryable.
            Takes (status_code, error) and returns bool.
        stats: Optional RetryStats to accumulate statistics.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        Result of func.

    Raises:
        RetryError: If all retries are exhausted.
    """
    if not RETRY_ENABLED:
        return await func(*args, **kwargs)

    retryable_fn = retryable or (lambda sc, e: is_retryable_error(sc, e))

    last_error: Exception | None = None
    last_status_code: int | None = None

    for attempt in range(max_retries + 1):
        try:
            result = await func(*args, **kwargs)
            if attempt > 0 and stats:
                stats.successful_retries += 1
                logger.info(f"Retry succeeded on attempt {attempt}")
            return result

        except Exception as e:
            last_error = e
            last_status_code = getattr(e, 'response', None)
            if hasattr(last_status_code, 'status_code'):
                last_status_code = last_status_code.status_code
            elif last_status_code is not None and not isinstance(last_status_code, int):
                last_status_code = None

            # Check if should retry
            if attempt >= max_retries:
                logger.warning(f"Max retries ({max_retries}) exhausted")
                break

            if is_non_retryable_error(last_status_code):
                logger.info(f"Non-retryable error (status={last_status_code}), failing fast")
                break

            if not retryable_fn(last_status_code, e):
                logger.info(f"Non-retryable error: {type(e).__name__}, failing fast")
                break

            # Calculate and apply delay
            delay = calculate_backoff_delay(attempt, base_delay_ms, jitter)
            if stats:
                stats.total_retry_time_ms += delay * 1000
                stats.total_retries += 1

            logger.debug(
                f"Retry attempt {attempt + 1}/{max_retries} after {delay*1000:.0f}ms delay "
                f"(status={last_status_code}): {type(e).__name__}"
            )
            await asyncio.sleep(delay)

    if stats:
        stats.failed_retries += 1

    raise RetryError(
        f"All {max_retries} retries exhausted. Last error: {last_error}",
        last_error=last_error
    )


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests pass through
    OPEN = "open"          # Fail-fast, requests rejected immediately
    HALF_OPEN = "half_open"  # Testing if upstream recovered


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls rejected due to open circuit
    state_changes: int = 0


class CircuitBreaker:
    """Circuit breaker for upstream URL.

    Implements the circuit breaker pattern with:
    - CLOSED: Normal operation, failures increment counter
    - OPEN: After threshold failures, reject requests immediately
    - HALF_OPEN: After recovery timeout, allow one probe request

    Transitions:
    - CLOSED → OPEN: After failure_threshold consecutive failures
    - OPEN → HALF_OPEN: After recovery_timeout seconds
    - HALF_OPEN → CLOSED: On successful probe
    - HALF_OPEN → OPEN: On failed probe
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        recovery_timeout: int = DEFAULT_RECOVERY_TIMEOUT_SECONDS,
        stats: CircuitBreakerStats | None = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for this circuit (e.g., upstream URL).
            failure_threshold: Number of failures before opening circuit.
            recovery_timeout: Seconds to wait before probing for recovery.
            stats: Optional CircuitBreakerStats to accumulate statistics.
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._stats = stats or CircuitBreakerStats()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (fail-fast mode)."""
        return self._state == CircuitState.OPEN

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal mode)."""
        return self._state == CircuitState.CLOSED

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (probing mode)."""
        return self._state == CircuitState.HALF_OPEN

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._stats.successful_calls += 1
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                # Successful probe - close the circuit
                logger.info(f"Circuit {self.name}: probe succeeded, closing circuit")
                self._state = CircuitState.CLOSED
                self._stats.state_changes += 1

    async def record_failure(self) -> None:
        """Record a failed call."""
        async with self._lock:
            self._stats.failed_calls += 1
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Failed probe - reopen the circuit
                logger.warning(f"Circuit {self.name}: probe failed, reopening circuit")
                self._state = CircuitState.OPEN
                self._stats.state_changes += 1

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    logger.warning(
                        f"Circuit {self.name}: failure threshold reached "
                        f"({self._failure_count}/{self.failure_threshold}), opening circuit"
                    )
                    self._state = CircuitState.OPEN
                    self._stats.state_changes += 1

    async def can_attempt(self) -> bool:
        """Check if a request can be attempted.

        Returns:
            True if request can proceed, False if should fail-fast.
        """
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.recovery_timeout:
                        # Transition to half-open
                        logger.info(
                            f"Circuit {self.name}: recovery timeout passed, "
                            f"probing with half-open circuit"
                        )
                        self._state = CircuitState.HALF_OPEN
                        self._stats.state_changes += 1
                        self._failure_count = 0
                        return True
                return False

            # HALF_OPEN - allow probe
            return True

    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return CircuitBreakerStats(
            total_calls=self._stats.total_calls,
            successful_calls=self._stats.successful_calls,
            failed_calls=self._stats.failed_calls,
            rejected_calls=self._stats.rejected_calls,
            state_changes=self._stats.state_changes,
        )


class CircuitBreakerRegistry:
    """Registry for circuit breakers per upstream URL."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for the given name/URL."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name=name)
                logger.info(f"Circuit breaker created for: {name}")
            return self._breakers[name]

    async def get_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all circuit breakers."""
        async with self._lock:
            return {
                name: {
                    "state": cb.state.value,
                    **vars(cb.get_stats()),
                }
                for name, cb in self._breakers.items()
            }


# Global registry
_registry: CircuitBreakerRegistry | None = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry


async def call_with_circuit_breaker(
    func: Callable[..., Any],
    upstream_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Execute a function with circuit breaker protection.

    Args:
        func: Async function to execute.
        upstream_name: Name/URL of upstream for circuit tracking.
        *args: Arguments to pass to func.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        Result of func.

    Raises:
        Exception: If circuit is open or call fails.
    """
    if not CIRCUIT_BREAKER_ENABLED:
        return await func(*args, **kwargs)

    registry = get_circuit_breaker_registry()
    cb = await registry.get_or_create(upstream_name)

    # Check if circuit allows the call
    if not await cb.can_attempt():
        cb._stats.rejected_calls += 1
        raise CircuitOpenError(f"Circuit breaker is open for {upstream_name}")

    cb._stats.total_calls += 1

    try:
        result = await func(*args, **kwargs)
        await cb.record_success()
        return result
    except Exception as e:
        await cb.record_failure()
        raise


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and request is rejected."""
    pass
