# compression-stats

## ADDED Requirements

### Requirement: Per-strategy timing tracking

The system SHALL track timing (avg_ms, max_ms, count) for each compression strategy used: rust_zstd, kompress, text_compressor, passthrough.

#### Scenario: Strategy timing recorded
- **WHEN** a compression operation completes
- **THEN** timing SHALL be recorded in `transform_timing_sum` and `transform_timing_count` keyed by strategy name

### Requirement: Compression ratio visibility

The system SHALL track compression ratio per strategy to identify effectiveness. Data SHALL be surfaced in `/stats` endpoint.

#### Scenario: Ratio tracked per strategy
- **WHEN** compression completes with strategy X and ratio Y
- **THEN** ratio SHALL be aggregated and available in stats output

### Requirement: Fallback chain visibility

The system SHALL track how often each fallback path is taken (rust → ml → heuristic → passthrough) to identify optimization opportunities.

#### Scenario: Fallback path tracked
- **WHEN** compression falls back from rust to ml
- **THEN** fallback counter SHALL be incremented and visible in stats

### Requirement: /stats endpoint integration

Compression stats SHALL be included in the `/stats` JSON response under key `compressor_stats` with structure:

```json
{
  "compressor_stats": {
    "by_strategy": {
      "rust_zstd": {"count": 0, "avg_ms": 0.0, "max_ms": 0.0, "avg_ratio": 0.0},
      "kompress": {"count": 0, "avg_ms": 0.0, "max_ms": 0.0, "avg_ratio": 0.0},
      "text_compressor": {"count": 0, "avg_ms": 0.0, "max_ms": 0.0, "avg_ratio": 0.0},
      "passthrough": {"count": 0, "avg_ms": 0.0, "max_ms": 0.0, "avg_ratio": 0.0}
    },
    "fallbacks": {
      "rust_to_ml": 0,
      "ml_to_heuristic": 0,
      "heuristic_to_passthrough": 0
    },
    "dedup_hits": 0,
    "dedup_misses": 0
  }
}
```

#### Scenario: Stats endpoint returns compression data
- **WHEN** GET /stats is called
- **THEN** response SHALL include `compressor_stats` with all tracked metrics