"""Token count cache with session scope.

This module provides an LRU cache for token counts to avoid
redundant token counting. Cache is scoped to a session and invalidated
when the session ends.

Key features:
- LRU eviction (collections.OrderedDict)
- Session-scoped with invalidation
- SHA256 content hash as key
- Secondary validation via content length check
- Statistics tracking (hits, misses, evictions)
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Default max cache size
DEFAULT_MAX_CACHE_SIZE = 10_000


@dataclass
class CacheStats:
    """Statistics for token count cache."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    invalidations: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


@dataclass
class TokenCacheEntry:
    """Cache entry with validation."""
    token_count: int
    content_hash: str
    content_length: int


class TokenCountCache:
    """LRU cache for token counts with session scope.

    Uses session_id + SHA256(content) as cache key.
    Entries include content_length for secondary validation.
    """

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_CACHE_SIZE,
        stats: CacheStats | None = None,
    ):
        """
        Initialize token count cache.

        Args:
            max_size: Maximum number of entries in cache.
            stats: Optional CacheStats to accumulate statistics.
        """
        self._cache = OrderedDict()
        self._max_size = max_size
        self._stats = stats or CacheStats()
        self._lock = threading.RLock()
        self._sessions: set[str] = set()

    def get(
        self,
        session_id: str,
        content: str | bytes,
        expected_length: int | None = None,
    ) -> int | None:
        """Get token count from cache.

        Args:
            session_id: Session identifier.
            content: Content to look up.
            expected_length: Optional content length for validation.

        Returns:
            Cached token count, or None if not found or validation failed.
        """
        with self._lock:
            key = self._make_key(session_id, content)
            entry = self._cache.get(key)

            if entry is None:
                self._stats.misses += 1
                return None

            # Secondary validation - check content length
            if expected_length is not None and entry.content_length != expected_length:
                # Content length mismatch - hash collision or modified content
                logger.debug(
                    f"Cache invalidation: length mismatch "
                    f"(expected={expected_length}, stored={entry.content_length})"
                )
                del self._cache[key]
                self._stats.misses += 1
                return None

            # Hit!
            self._stats.hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return entry.token_count

    def put(
        self,
        session_id: str,
        content: str | bytes,
        token_count: int,
    ) -> None:
        """Add token count to cache.

        Args:
            session_id: Session identifier.
            content: Content that was tokenized.
            token_count: Token count result.
        """
        with self._lock:
            # Evict oldest if at capacity
            if len(self._cache) >= self._max_size and self._cache:
                evicted_key = next(iter(self._cache))
                del self._cache[evicted_key]
                self._stats.evictions += 1

            key = self._make_key(session_id, content)
            content_bytes = content.encode() if isinstance(content, str) else content

            self._cache[key] = TokenCacheEntry(
                token_count=token_count,
                content_hash=hashlib.sha256(content_bytes).hexdigest(),
                content_length=len(content_bytes),
            )

            # Track session
            self._sessions.add(session_id)

    def invalidate_session(self, session_id: str) -> int:
        """Invalidate all cache entries for a session.

        Args:
            session_id: Session to invalidate.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            if session_id not in self._sessions:
                return 0

            # Find and remove all keys for this session
            prefix = f"{session_id}:"
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(prefix)]

            for key in keys_to_remove:
                del self._cache[key]

            self._sessions.discard(session_id)
            self._stats.invalidations += len(keys_to_remove)

            logger.debug(f"Invalidated {len(keys_to_remove)} cache entries for session {session_id}")
            return len(keys_to_remove)

    def clear(self) -> int:
        """Clear entire cache.

        Returns:
            Number of entries removed.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._sessions.clear()
            return count

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                invalidations=self._stats.invalidations,
            )

    @staticmethod
    def _make_key(session_id: str, content: str | bytes) -> str:
        """Create cache key from session and content."""
        content_bytes = content.encode() if isinstance(content, str) else content
        content_hash = hashlib.sha256(content_bytes).hexdigest()
        return f"{session_id}:{content_hash}"

    @property
    def size(self) -> int:
        """Current number of entries in cache."""
        with self._lock:
            return len(self._cache)

    @property
    def max_size(self) -> int:
        """Maximum cache size."""
        return self._max_size


# Global cache instance
_global_cache: TokenCountCache | None = None
_cache_lock = threading.Lock()


def get_token_cache() -> TokenCountCache:
    """Get or create the global token cache instance."""
    global _global_cache
    with _cache_lock:
        if _global_cache is None:
            _global_cache = TokenCountCache(max_size=DEFAULT_MAX_CACHE_SIZE)
            logger.info(f"Token cache initialized (max_size={DEFAULT_MAX_CACHE_SIZE})")
        return _global_cache


def count_tokens_cached(
    session_id: str,
    content: str | bytes,
    tokenizer: Any,
    expected_length: int | None = None,
) -> int:
    """Count tokens with caching.

    First checks cache, then counts and caches result.

    Args:
        session_id: Session identifier.
        content: Content to tokenize.
        tokenizer: Tokenizer with count_text method.
        expected_length: Optional content length for validation.

    Returns:
        Token count.
    """
    cache = get_token_cache()

    # Try cache first
    cached = cache.get(session_id, content, expected_length)
    if cached is not None:
        return cached

    # Count tokens
    content_str = content.decode() if isinstance(content, bytes) else content
    token_count = tokenizer.count_text(content_str)

    # Cache result
    cache.put(session_id, content, token_count)

    return token_count


class CachingTokenCounter:
    """Wrapper that adds LRU caching to any tokenizer.

    This wraps a tokenizer and caches count_text() and count_messages()
    results using the global token cache.
    """

    def __init__(self, tokenizer: Any, session_id: str = "default"):
        """
        Args:
            tokenizer: Any object implementing TokenCounter protocol.
            session_id: Session ID for cache keying.
        """
        self._tokenizer = tokenizer
        self._session_id = session_id
        self._cache = get_token_cache()

    def count_text(self, text: str) -> int:
        """Count tokens with caching."""
        # Try cache first
        cached = self._cache.get(self._session_id, text)
        if cached is not None:
            return cached

        # Count tokens
        result = self._tokenizer.count_text(text)

        # Cache result
        self._cache.put(self._session_id, text, result)

        return result

    def count_messages(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens in messages with caching."""
        import json
        # Serialize messages for cache key
        cache_key = json.dumps(messages, sort_keys=True)

        # Try cache first
        cached = self._cache.get(self._session_id, cache_key)
        if cached is not None:
            return cached

        # Count tokens
        result = self._tokenizer.count_messages(messages)

        # Cache result
        self._cache.put(self._session_id, cache_key, result)

        return result

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to underlying tokenizer."""
        return getattr(self._tokenizer, name)


def invalidate_session(session_id: str) -> int:
    """Invalidate cache for a session."""
    cache = get_token_cache()
    return cache.invalidate_session(session_id)
