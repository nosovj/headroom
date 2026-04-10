## ADDED Requirements

### Requirement: Token counting happens exactly once per content block

The system SHALL count tokens for each unique content block only once and cache the result. Subsequent requests for the same content SHALL retrieve the cached count without re-tokenizing. The cache SHALL be keyed by content hash to enable O(1) lookups.

#### Scenario: Cache hit on repeated content

- **WHEN** content "Hello world" has already been token counted with result 2 tokens
- **AND** a new request contains the same content "Hello world"
- **THEN** the system SHALL return the cached result of 2 tokens
- **AND** no tokenization SHALL occur

#### Scenario: Cache miss triggers counting

- **WHEN** content "New content" has not been token counted before
- **THEN** the system SHALL compute token count via the Rust worker
- **AND** the result SHALL be stored in the cache keyed by SHA256(content)

### Requirement: Cache is shared across requests within a session

The system SHALL maintain a session-scoped cache that persists across multiple requests in the same conversation. The cache SHALL be cleared when the session ends to prevent unbounded memory growth.

#### Scenario: Cross-request cache persistence

- **WHEN** request 1 contains content X with token count 10
- **AND** request 2 contains the same content X within the same session
- **THEN** the token count SHALL be retrieved from cache for request 2

#### Scenario: Cache cleared on session end

- **WHEN** a session ends (no activity for 30 minutes or explicit close)
- **THEN** all cached token counts for that session SHALL be evicted
- **AND** memory SHALL be released

### Requirement: Cache size is bounded to prevent memory exhaustion

The system SHALL enforce a maximum cache size (configurable, default 10,000 entries). When the limit is reached, the system SHALL evict the least recently used entries using an LRU policy.

#### Scenario: LRU eviction when cache is full

- **WHEN** the cache contains 10,000 entries and a new unique content arrives
- **THEN** the least recently used entry SHALL be evicted
- **AND** the new content token count SHALL be cached

### Requirement: Token count operations are async and non-blocking

The system SHALL perform token counting asynchronously via Rust workers. The Python async event loop SHALL NOT block waiting for token counts. Results SHALL be communicated via futures that resolve when computation completes.

#### Scenario: Async token count resolution

- **WHEN** a request needs token counts for 5 content blocks
- **THEN** the system SHALL spawn async tasks for each count
- **AND** the request SHALL wait for all futures to resolve
- **AND** no Python threads SHALL be blocked during computation