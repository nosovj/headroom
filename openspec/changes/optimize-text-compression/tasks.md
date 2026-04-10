## 1. Rust Compression Foundation

- [x] 1.1 Add zstd dependency to crates/headroom-workers/Cargo.toml
- [x] 1.2 Add compress_text Rust function with PyO3 binding
- [ ] 1.3 Add compress_text to worker task enum and dispatch (deferred - using direct PyO3 call instead)
- [x] 1.4 Add Python wrapper in headroom/workers/__init__.py
- [x] 1.5 Test basic compression works (verify <5ms for 1KB input)

## 2. ContentRouter Integration

- [x] 2.1 Add latency_budget parameter to ContentRouter.compress()
- [x] 2.2 Add fast_path flag to bypass ML when content is "easy" (low entropy)
- [x] 2.3 Wire Rust compress_text as first attempt in fallback chain
- [x] 2.4 Add HEADROOM_USE_RUST_COMPRESSION env var for rollback
- [ ] 2.5 Verify Rust fallback works when ML is slow (pending real traffic)

## 3. Compression Stats

- [ ] 3.1 Add strategy tracking to MetricsCollector (rust_zstd, kompress, text_compressor, passthrough)
- [ ] 3.2 Track fallback counts (rust_to_ml, ml_to_heuristic, etc.)
- [x] 3.3 Add compressor_stats to /stats endpoint JSON (part of pipeline_timing)
- [ ] 3.4 Verify stats show correct timing and fallback counts

## 4. Deduplication (Phase 2)

- [x] 4.1 Add datasketch dependency (pip install datasketch)
- [x] 4.2 Create MinHashLSH index per session in ContentRouter
- [x] 4.3 Check similarity before compressing new content
- [x] 4.4 Reuse compression result when similarity > 0.95
- [x] 4.5 Add HEADROOM_SIMILARITY_THRESHOLD env var
- [ ] 4.6 Test deduplication with repeated error messages

## 5. Dictionary Training (Phase 3 - Optional)

- [ ] 5.1 Implement zstd dictionary training for tool outputs
- [ ] 5.2 Create pre-trained dictionaries per content type
- [ ] 5.3 Load dictionary based on detected content type
- [ ] 5.4 Measure improvement in compression ratio (>20% target)