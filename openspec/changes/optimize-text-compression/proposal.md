## Why

The `compressor:text` transform is the primary pipeline bottleneck at 63ms avg / 338ms max. It relies on Kompress (ModernBERT ML model) running on CPU without GPU acceleration. TurboQuant and similar KV-cache quantization approaches operate at a different layer (inside transformer attention during inference) and are NOT relevant to prompt preprocessing. We need faster text compression that doesn't require expensive ML inference.

## What Changes

1. **Rust-based fast text compression fallback**
   - Add `headroom-compression` crate with zstd/lz4 for sub-millisecond text compression
   - Use as fallback when ML compression exceeds latency budget (e.g., >20ms)
   - Enable dictionary training for domain-specific compression (tool outputs, logs, code)

2. **Compression result caching with LRU**
   - Track compression strategy effectiveness per content pattern
   - Increase hit rate for repeated content types (error messages, stack traces, etc.)

3. **Content similarity deduplication**
   - Use MinHash (datasketch) to detect semantically similar content before recompression
   - Skip compression for content already compressed in session

4. **Performance visibility**
   - Add `compressor_stats` to track which compression strategies are used and their latency
   - Surface hit rate and time saved in `/stats` endpoint

## Capabilities

### New Capabilities

- `rust-text-compressor`: Fast Rust-based text compression using zstd with optional dictionary training. Falls back from ML when latency exceeds budget.
- `compression-deduplication`: MinHash-based detection of similar content to avoid redundant compression.
- `compression-stats`: Per-strategy timing and hit rate tracking for pipeline optimization.

### Modified Capabilities

- None

## Impact

**New crates:**
- `crates/headroom-compression/` - Rust compression library (zstd, lz4)

**Modified files:**
- `headroom/transforms/content_router.py` - Use fast fallback when ML is slow
- `headroom/transforms/pipeline.py` - Track compression strategy effectiveness
- `headroom/proxy/server.py` - Add compression stats to `/stats`

**Dependencies:**
- `zstd` Rust crate (via PyO3)
- Python wrapper via existing `headroom-workers` pattern

**Affected transforms:**
- `compressor:text` - Primary target (63ms → <5ms with Rust fallback)
- `compressor:mixed` - Also benefits from faster compression