# rust-text-compressor

## ADDED Requirements

### Requirement: Fast Rust-based text compression via zstd

The system SHALL provide a Rust-based text compression function accessible from Python via PyO3. This function SHALL use the zstd algorithm and complete compression in under 5ms for typical tool output content (<10KB).

### Requirement: Latency-budget fallback chain

The system SHALL support a fallback chain for text compression:
1. Attempt Rust zstd compression first
2. If compression exceeds latency budget (configurable, default 20ms), fall back to ML compression
3. If ML exceeds budget or ratio is poor, use heuristic TextCompressor

#### Scenario: Fast path compression succeeds
- **WHEN** content is presented for compression with latency budget of 20ms
- **THEN** Rust zstd compression SHALL complete in under 5ms and produce output with compression ratio < 0.9

#### Scenario: Fast path exceeds budget, ML fallback used
- **WHEN** Rust compression exceeds 5ms (configurable via `HEADROOM_FAST_COMPRESS_MS`)
- **THEN** system SHALL attempt ML compression (Kompress) as secondary path

#### Scenario: Both fast and ML fail, heuristic fallback
- **WHEN** both Rust and ML compression exceed budget or produce ratio > 0.95
- **THEN** system SHALL fall back to heuristic TextCompressor with line-based sampling

### Requirement: Dictionary training support

The system SHOULD support zstd dictionary training for domain-specific content types (tool outputs, logs, code). When a dictionary is available for the detected content type, compression ratio SHALL improve by at least 20%.

#### Scenario: Tool output compressed with trained dictionary
- **WHEN** content is identified as tool output and a trained dictionary exists
- **THEN** Rust compression SHALL use the dictionary and achieve at least 20% better ratio than without

### Requirement: PyO3 binding in headroom-workers

The system SHALL expose `compress_text` function via existing `headroom-workers` PyO3 bindings. The function signature SHALL be:

```python
def compress_text(content: str, context_type: str, latency_budget_ms: float) -> dict:
    """
    Returns: {"compressed": str, "ratio": float, "strategy": str, "latency_ms": float}
    """
```

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