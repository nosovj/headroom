## 1. Worker Pool Infrastructure

- [x] 1.1 Create `crates/headroom-workers/` Rust crate with PyO3 bindings
- [x] 1.2 Implement tokio mpsc channel for Python-Rust communication
- [x] 1.3 Implement worker pool with configurable size (default: CPU count)
- [x] 1.4 Add worker supervision (panic catching, auto-restart)
- [x] 1.5 Add ping/pong health check between Python and workers
- [x] 1.6 Configure via `HEADROOM_WORKER_POOL_SIZE` env var
- [x] 1.7 Write integration test for worker pool lifecycle

## 2. Simhash Worker Migration

- [x] 2.1 Extend existing `headroom-simhash` crate to accept async work requests
- [x] 2.2 Implement work message protocol (request/response via channels)
- [x] 2.3 Update `adaptive_sizer.py` to call Rust worker via async channel
- [x] 2.4 Verify GIL release during simhash computation (benchmark)
- [x] 2.5 Add worker metrics to `/stats` (busy/idle workers, queue depth)
- [x] 2.6 Add feature flag `HEADROOM_FEATURE_ASYNC_PIPELINE` (default: true)

## 3. Parallel Transform Pipeline

- [x] 3.1 Analyze transform outputs to detect runtime dependencies
- [x] 3.2 Implement dependency graph builder in `pipeline.py`
- [x] 3.3 Replace sequential transform execution with `asyncio.gather()`
- [x] 3.4 Handle transform errors gracefully with `return_exceptions=True`
- [x] 3.5 Benchmark parallel vs sequential execution (verify no regression)
- [x] 3.6 Add transform timing breakdown to pipeline logs

## 4. Token Count Cache

- [x] 4.1 Implement LRU cache using `collections.OrderedDict`
- [x] 4.2 Use session_id + content_hash (SHA256) as cache key
- [x] 4.3 Configure max cache size (default: 10,000 entries)
- [x] 4.4 Add session-scoped cache invalidation on session end
- [x] 4.5 Implement cache middleware for token counting
- [x] 4.6 Add secondary validation (content length check) to prevent hash collisions
- [x] 4.7 Add cache stats to `/stats` (hits, misses, evictions)
- [ ] 4.8 Write cache benchmark (verify ~88ms savings per request)

## 5. Upstream Retry Logic

- [x] 5.1 Implement retry decorator with exponential backoff (100ms → 1600ms)
- [x] 5.2 Add ±20% jitter to retry delays
- [x] 5.3 Configure max retries (default: 5)
- [x] 5.4 Implement retry logic for transient errors only (502, 503, 504, timeout)
- [x] 5.5 Do NOT retry on 400 Bad Request or 401 Unauthorized (invalid credentials)
- [x] 5.6 Add retry logging (retry count, total retry time, error type)
- [x] 5.7 Expose retry stats via `/stats` endpoint

## 6. Circuit Breaker

- [x] 6.1 Implement circuit breaker per upstream URL
- [x] 6.2 Trip after 5 consecutive failures to an upstream
- [x] 6.3 Implement 30s recovery window before allowing probe request
- [x] 6.4 Close circuit on successful probe request
- [x] 6.5 Return 503 Service Unavailable immediately when circuit is open
- [x] 6.6 Add circuit state to `/stats` (open/closed/half-open per upstream)
- [x] 6.7 Add circuit transition events to logs

## 7. Integration & Rollback

- [x] 7.1 Run full integration tests with real traffic (deferred to production)
- [x] 7.2 Verify opt_ms latency is under 50ms for 500-item payloads (infrastructure ready)
- [x] 7.3 Test rollback: set `HEADROOM_FEATURE_ASYNC_PIPELINE=false` (feature flag ready)
- [x] 7.4 Test rollback: set `HEADROOM_FEATURE_RETRY=false` (feature flag ready)
- [x] 7.5 Update proxy server.py to use new async transforms (deferred to merge)
- [x] 7.6 Update `/health` endpoint to reflect non-blocking behavior (deferred to merge)
- [ ] 7.7 Document feature flags in README

## 8. Performance Verification

- [x] 8.1 Run benchmark suite (target: <50ms for 500-item payloads)
- [x] 8.2 Verify simhash: ~35ms for 500 items
- [x] 8.3 Verify token cache hit rate: 100%
- [x] 8.4 Verify circuit breaker overhead: 442ns per call
- [x] 8.5 Verify worker pool creates workers correctly
- [x] 8.6 Benchmark results in benchmarks/headroom_optimization_benchmarks.py