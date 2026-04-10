## Context

The headroom proxy currently processes requests using synchronous Python transforms that block the async event loop during compute-heavy operations (simhash, token counting, content routing). This causes:

1. **Request queuing**: When one request blocks on simhash computation (~60ms), all other requests wait
2. **Health check failures**: The proxy appears unresponsive during blocking operations
3. **Upstream errors cascading**: 401/502 errors from one provider cause retries that further block the event loop

The current architecture uses monkey-patching to add Rust simhash, but the GIL is only released during Rust computation, not during the async/await cycle that orchestrates transforms.

## Goals / Non-Goals

**Goals:**
- Release the Python GIL during all compute-heavy operations
- Enable parallel transform execution (independent transforms run concurrently)
- Eliminate redundant token counting via caching
- Add circuit breakers and retry logic for upstream resilience
- Achieve <50ms optimization latency for 500-item payloads

**Non-Goals:**
- Rewrite entire proxy in Rust
- Modify the public API or client-facing interfaces
- Change the existing compression algorithms (only move them to Rust)
- Implement cross-node distributed worker pools (single-process only)

## Decisions

### 1. Worker Pool Architecture: Rust Async Workers via PyO3

**Decision**: Use a pool of Rust workers communicating with Python via async channels (tokio mpsc).

**Rationale**: 
- PyO3 allows Rust functions to be called from Python with GIL release
- tokio provides efficient async message passing without Python thread blocking
- Worker pool avoids per-request spawn overhead (~1ms savings per request)

**Alternatives considered**:
- `multiprocessing.Pool`: High IPC overhead (pickle serialization) and doesn't integrate well with async
- `concurrent.futures.ThreadPool`: Still holds GIL during Python code execution
- `asyncio.to_thread()`: Only moves to thread pool, doesn't release GIL during Rust computation

### 2. Transform Scheduling: Dependency-Based Parallel Execution

**Decision**: Analyze transform outputs to detect dependencies at runtime; execute independent transforms concurrently.

**Rationale**:
- Current pipeline executes transforms sequentially even when outputs don't depend on each other
- Detecting dependencies at runtime is more maintainable than static analysis
- Use `asyncio.gather()` with `return_exceptions=True` for concurrent execution

**Alternatives considered**:
- Static dependency graph: Requires maintaining a graph; fragile when transforms change
- Pre-sorted topological order: Assumes transforms don't have data-dependent dependencies

### 3. Token Count Cache: LRU with Session-Scoped Invalidation

**Decision**: Use `cachetools.LRUCache` with session_id as part of the key, max 10,000 entries.

**Rationale**:
- Session scope prevents unbounded memory growth while allowing cross-request reuse
- LRU is O(1) and avoids the complexity of more sophisticated eviction policies
- Content hash (SHA256) as key enables fast O(1) lookups

**Alternatives considered**:
- Redis external cache: Adds network round-trip; overkill for single-process sharing
- TTL-based expiration: Doesn't handle long sessions well; evicts entries that are still valid

### 4. Circuit Breaker: Failure Count Threshold with 30s Recovery Window

**Decision**: Trip after 5 consecutive failures; probe after 30s; close on success.

**Rationale**:
- 5 failures balances between detecting real outages and not tripping on transient errors
- 30s recovery window gives upstream time to recover without keeping circuit open indefinitely
- Immediate fail-fast when open prevents resource waste on doomed requests

**Alternatives considered**:
- Error rate threshold (e.g., 50% in 10s): More sensitive to traffic spikes; harder to tune
- Time-based trip (e.g., any 5 errors in 1min): Doesn't distinguish sustained outage from burst

### 5. Retry Backoff: Fixed 2x Exponential with Jitter

**Decision**: Retry delays: 100ms, 200ms, 400ms, 800ms, 1600ms. Add ±20% jitter to prevent thundering herd.

**Rationale**:
- Standard exponential backoff is well-understood and predictable
- Jitter prevents synchronized retries from overwhelming recovered upstream
- 5 retries (max ~3s total) balances resilience against latency impact

**Alternatives considered**:
- Linear backoff: Too slow to recover; doesn't converge fast enough
- Full jitter: Harder to predict latency; some requests wait much longer than others

## Risks / Trade-offs

[Risk] Rust worker panic crashes entire process
→ **Mitigation**: Wrap worker communication in supervision; panics are caught and logged; worker is restarted

[Risk] Channel message serialization overhead
→ **Mitigation**: Use binary serialization (MessagePack) instead of JSON; keep messages small

[Risk] Worker pool size tuning is environment-specific
→ **Mitigation**: Make pool size configurable via `HEADROOM_WORKER_POOL_SIZE` env var; default to CPU count

[Risk] Content hash collisions cause incorrect cache hits
→ **Mitigation**: SHA256 has negligible collision probability for our use case; cache entries include content length as secondary check

[Trade-off] Complexity increase from async Rust workers
→ Acceptable given the performance requirements; isolates GIL release to well-defined boundaries

[Trade-off] Memory overhead from keeping Rust workers alive
→ Pool of N workers × ~2MB each; acceptable for N ≤ CPU count (typically 4-16)

## Migration Plan

### Phase 1: Worker Pool Infrastructure
1. Create `headroom/workers/` Rust crate with tokio channel communication
2. Implement worker pool with configurable size
3. Add basic ping/pong health check between Python and workers

### Phase 2: Move Simhash to Workers
1. Migrate `count_unique_simhash` to Rust worker
2. Update `adaptive_sizer.py` to use async worker calls
3. Verify GIL release during computation

### Phase 3: Parallel Transform Pipeline
1. Add dependency analysis to `pipeline.py`
2. Implement concurrent transform execution via `asyncio.gather`
3. Benchmark to verify parallelism doesn't add overhead

### Phase 4: Token Count Cache
1. Implement LRU cache with session scope
2. Add cache middleware to token counting
3. Add cache stats to `/stats` endpoint

### Phase 5: Upstream Resilience
1. Implement retry logic with exponential backoff
2. Add circuit breaker per upstream URL
3. Add retry/circuit stats to `/stats` endpoint

### Rollback
- Each phase is independently usable; disable via config flags
- `HEADROOM_FEATURE_ASYNC_PIPELINE=false` falls back to synchronous execution
- `HEADROOM_FEATURE_RETRY=false` disables retry/circuit breaker

## Open Questions

1. **Pool sizing**: Should we auto-tune based on request load, or keep fixed at startup?
2. **Worker lifetime**: Should workers be recycled after N requests to prevent memory leaks?
3. **Fallback behavior**: If all workers fail, should we fall back to synchronous Python or return an error?
4. **Cache warming**: Should we pre-populate the token cache on startup with common code patterns?