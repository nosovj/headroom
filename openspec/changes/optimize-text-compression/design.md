## Context

The `compressor:text` transform currently uses Kompress (ModernBERT ML model) which runs at 63ms avg / 338ms max on CPU. This is the primary pipeline bottleneck. We need a fast fallback that doesn't require ML inference.

The existing `headroom-workers` crate provides Rust workers via PyO3 + tokio. We'll extend this pattern for compression.

## Goals / Non-Goals

**Goals:**
- Reduce `compressor:text` latency from 63ms to <5ms for most content
- Provide transparent fallback when ML exceeds latency budget (20ms)
- Enable dictionary training for domain-specific compression (tool outputs, logs, code)
- Add visibility into compression strategy effectiveness

**Non-Goals:**
- Replace ML compression entirely (ML still provides better ratios for complex content)
- Add KV-cache quantization (operates at different layer - inside transformer attention)
- Modify existing compression ratio algorithms

## Decisions

### 1. Use zstd over lz4 for Rust compression

**Decision:** Use `zstd` as the primary Rust compression library via PyO3.

**Rationale:**
- zstd offers 510 MB/s compression (vs lz4's 675 MB/s) with better ratio (2.9 vs 2.1)
- zstd supports dictionary training for small data (critical for tool outputs/logs)
- Already used by Meta at scale, well-maintained
- PyO3 integration is straightforward

**Alternative considered:** lz4 - faster but worse compression ratio. Not suitable for token reduction goals.

### 2. Fallback chain: ML → Rust → passthrough

**Decision:** Implement a latency-budget fallback chain in `ContentRouter`:

```
if latency_budget_ms exceeded:
    try Rust zstd compression
    if still too slow or poor ratio:
        use TextCompressor (heuristic-based)
        if still poor:
            passthrough (no compression)
```

**Rationale:** ML provides best compression ratio but is slow. Rust is fast (sub-ms) with good-enough ratio. Heuristic fallback catches edge cases.

### 3. Use existing headroom-workers pattern for PyO3

**Decision:** Add compression functions to existing `headroom-workers` crate rather than creating new crate.

**Rationale:**
- Consistent with existing architecture (workers + PyO3 + tokio)
- Shares worker pool management and health checks
- Less code to maintain

**Alternative considered:** Create `headroom-compression` crate - would be cleaner but adds maintenance burden.

### 4. MinHash deduplication via datasketch

**Decision:** Use `datasketch` Python library for MinHash similarity detection before compression.

**Rationale:**
- Well-maintained, 2.9k stars
- Supports LSH for sub-linear lookup
- Already Python-native (no Rust needed for this)

**Alternative considered:** Implement MinHash in Rust - would be faster but adds complexity. Python datasketch is fast enough for deduplication use case.

### 5. Per-content-type compression budgets

**Decision:** Different latency budgets per content type:
- Tool outputs: 5ms budget (frequent, compressible)
- Logs: 10ms budget (moderate)
- General text: 20ms budget (ML fallback threshold)

**Rationale:** Tool outputs are high-frequency and benefit most from fast Rust compression. General text can tolerate ML if it provides better ratio.

## Risks / Trade-offs

[Risk] Rust compression may provide worse ratio than ML  
→ **Mitigation:** ML fallback still available; monitor compression ratio in stats. Alert if ratio drops below threshold.

[Risk] Dictionary training adds complexity  
→ **Mitigation:** Start without training; add per-content-type dictionaries later if needed.

[Risk] MinHash adds memory overhead  
→ **Mitigation:** LRU cache with eviction; limit to session-scoped dedup only.

[Risk] zstd PyO3 binding adds build complexity  
→ **Mitigation:** Use pyo3-zstd or expose raw FFI; test in CI with cross-platform builds.

## Migration Plan

**Phase 1 (Foundation):**
1. Add `compress_text` function to `headroom-workers` PyO3 bindings
2. Wire into `ContentRouter` as fast path (bypass ML if content is compressible)
3. Add `compressor_stats` tracking

**Phase 2 (Fallback):**
1. Implement latency-budget checker before ML compression
2. Add Rust fallback with zstd
3. Add heuristic fallback (TextCompressor) for edge cases

**Phase 3 (Deduplication):**
1. Add MinHash LSH index per session
2. Check similar content before recompression
3. Cache and reuse compression results

**Rollback:** Disable via env var `HEADROOM_USE_RUST_COMPRESSION=false` reverts to ML-only path.

## Open Questions

1. **Dictionary training data**: Should we auto-train from user's own compression history, or use pre-built dictionaries per content type?

2. **Compression ratio threshold**: What minimum ratio justifies compression? Currently considering ratio < 0.8 as "compressing well".

3. **GPU acceleration**: Should we explore ONNX GPU execution for Kompress if available? Could provide 10x speedup without replacing the ML model.