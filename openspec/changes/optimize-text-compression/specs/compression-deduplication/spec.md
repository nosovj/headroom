# compression-deduplication

## ADDED Requirements

### Requirement: Content similarity detection via MinHash

The system SHALL use MinHash (via datasketch library) to detect content similarity before compression. Content with similarity > 0.95 to already-compressed content in the session SHALL skip recompression and reuse the existing compression result.

#### Scenario: Similar content detected and skipped
- **WHEN** new content has MinHash similarity > 0.95 to previously compressed content in the same session
- **THEN** system SHALL reuse the existing compression result instead of recompressing

#### Scenario: Unique content gets compressed
- **WHEN** new content has MinHash similarity < 0.95 to all previously compressed content
- **THEN** system SHALL proceed with compression normally

### Requirement: Session-scoped LSH index

The system SHALL maintain an LSH (Locality-Sensitive Hash) index per session for MinHash lookups. The index SHALL be session-scoped and evicted when the session ends.

#### Scenario: Session cleanup
- **WHEN** session expires or is cleared
- **THEN** LSH index and all MinHash signatures SHALL be evicted from memory

### Requirement: Similarity threshold configuration

The similarity threshold for deduplication SHALL be configurable via `HEADROOM_SIMILARITY_THRESHOLD` env var (default 0.95).

#### Scenario: Custom similarity threshold
- **WHEN** `HEADROOM_SIMILARITY_THRESHOLD=0.90` is set
- **THEN** content with similarity > 0.90 (instead of 0.95) SHALL trigger deduplication

### Requirement: Per-content-type MinHash

The system SHALL maintain separate MinHash signatures per content type (tool_output, log, code, general) to improve deduplication accuracy.

#### Scenario: Tool output deduplication
- **WHEN** new tool output content is presented
- **THEN** system SHALL compare against tool_output MinHash signatures only