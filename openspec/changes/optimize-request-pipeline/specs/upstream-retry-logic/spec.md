## ADDED Requirements

### Requirement: Upstream errors trigger automatic retry with exponential backoff

The system SHALL retry failed upstream requests (401 Unauthorized, 502 Bad Gateway, 503 Service Unavailable, 504 Gateway Timeout) with exponential backoff. The retry schedule SHALL be: 100ms, 200ms, 400ms, 800ms, 1600ms (max 5 retries). Retries SHALL only occur for transient errors, not for client errors (400 Bad Request).

#### Scenario: Automatic retry on 502 error

- **WHEN** an upstream request returns 502 Bad Gateway
- **THEN** the system SHALL wait 100ms and retry the request
- **AND** if the retry fails, wait 200ms and retry again
- **AND** continue until success or max retries reached

#### Scenario: No retry on 401 with invalid credentials

- **WHEN** an upstream request returns 401 Unauthorized due to invalid API key
- **THEN** the system SHALL NOT retry the request
- **AND** SHALL immediately return the error to the client

### Requirement: Retry state is tracked per-request for debugging

The system SHALL log retry attempts including the error type, retry count, and total time spent in retries. This information SHALL be accessible via the proxy stats endpoint for monitoring retry behavior.

#### Scenario: Retry logging

- **WHEN** a request undergoes 3 retries before succeeding
- **THEN** the proxy log SHALL contain entries showing retry 1, retry 2, retry 3
- **AND** the final log entry SHALL include total_retry_time_ms

### Requirement: Circuit breaker prevents repeated upstream failures

The system SHALL implement a circuit breaker that trips after 5 consecutive failures to an upstream. When tripped, subsequent requests SHALL fail immediately (fail-fast) for 30 seconds before allowing one probe request. This prevents the proxy from being blocked by a degraded upstream.

#### Scenario: Circuit opens after consecutive failures

- **WHEN** 5 consecutive requests to https://api.anthropic.com fail
- **THEN** the circuit breaker SHALL open
- **AND** subsequent requests SHALL immediately return 503 Service Unavailable
- **AND** after 30 seconds, one probe request SHALL be allowed

#### Scenario: Circuit closes after successful probe

- **WHEN** the circuit breaker is open and the probe request succeeds
- **THEN** the circuit SHALL close
- **AND** normal request processing SHALL resume

### Requirement: Request timeout is configurable per upstream

The system SHALL support configurable timeouts per upstream URL. Default timeout SHALL be 30 seconds. If an upstream request exceeds its timeout, it SHALL be treated as a retryable error.

#### Scenario: Configurable timeout per upstream

- **WHEN** a request to MiniMax upstream times out after 15 seconds (configurable)
- **THEN** the timeout SHALL be treated as a retryable error
- **AND** retry logic SHALL be applied