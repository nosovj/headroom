# Handoff Document: optimize-request-pipeline

## Overview

**Change Name:** `optimize-request-pipeline`
**Location:** `openspec/changes/optimize-request-pipeline/`
**Schema:** spec-driven
**Status:** All 4 artifacts complete (ready for implementation)

## Problem Statement

The headroom proxy has severe latency issues:
- **5-70s upstream errors** (401/502) causing request timeouts
- **227-322ms optimization time** per request (measured via `opt_ms`)
- **~98ms content router overhead** (per timing breakdown)
- **~88ms wasted** on duplicate token counting (happens twice per request)
- **GIL blocking** during Python sync operations blocks async event loop
- **Health check failures** when requests block the event loop

## What This Change Does

1. **Rust async worker pool** - Compute-heavy transforms run in Rust workers that release the GIL
2. **Parallel transform execution** - Independent transforms run concurrently, not sequentially
3. **Token count caching** - Count once, reuse (LRU cache with session scope)
4. **Upstream retry with backoff** - 100ms→1600ms exponential with jitter, max 5 retries
5. **Circuit breaker** - Trip after 5 failures, 30s recovery, fail-fast when open

## Artifacts Created

| Artifact | File | Status |
|----------|------|--------|
| proposal | `proposal.md` | ✅ Complete |
| specs | `specs/async-compression-pipeline/spec.md` | ✅ Complete |
| specs | `specs/token-count-cache/spec.md` | ✅ Complete |
| specs | `specs/upstream-retry-logic/spec.md` | ✅ Complete |
| specs | `specs/rust-content-router/spec.md` | ✅ Complete |
| design | `design.md` | ✅ Complete |
| tasks | `tasks.md` | ✅ Complete |

## Key Decisions (from design.md)

1. **Worker Pool**: Rust workers via PyO3 + tokio channels (NOT multiprocessing.Pool or ThreadPool)
2. **Transform Scheduling**: Runtime dependency detection, `asyncio.gather()` for parallel execution
3. **Token Cache**: `cachetools.LRUCache`, session_id + SHA256(content) as key, 10K max entries
4. **Circuit Breaker**: 5 failures to trip, 30s recovery window, open→probe→close on success
5. **Retry Backoff**: 2x exponential (100ms, 200ms, 400ms, 800ms, 1600ms), ±20% jitter

## Migration Phases (from design.md)

- **Phase 1**: Worker pool infrastructure (crate, channels, supervision)
- **Phase 2**: Simhash to workers (extend existing headroom-simhash crate)
- **Phase 3**: Parallel transform pipeline (dependency analysis, asyncio.gather)
- **Phase 4**: Token count cache (LRU, session scope, middleware)
- **Phase 5**: Upstream resilience (retry, circuit breaker)

## Feature Flags (for rollback)

- `HEADROOM_FEATURE_ASYNC_PIPELINE=false` → Falls back to sync execution
- `HEADROOM_FEATURE_RETRY=false` → Disables retry and circuit breaker
- `HEADROOM_WORKER_POOL_SIZE` → Configurable pool size (default: CPU count)

## Tasks (50 total, in tasks.md)

```
## 1. Worker Pool Infrastructure (7 tasks)
## 2. Simhash Worker Migration (6 tasks)
## 3. Parallel Transform Pipeline (6 tasks)
## 4. Token Count Cache (8 tasks)
## 5. Upstream Retry Logic (7 tasks)
## 6. Circuit Breaker (7 tasks)
## 7. Integration & Rollback (7 tasks)
## 8. Performance Verification (6 tasks)
```

## Current Baseline (from benchmarks/)

- **Before cutover baseline** (`benchmarks/before_cutover_baseline.json`):
  - unified-proxy: 256.5ms avg
  - optimized-fork: 2.6ms avg
  - Speedup ratio: 98.7x

- **Current state** (after Rust simhash patch):
  - count_unique_simhash: ~62ms for 500 items (vs ~400ms Python)
  - compute_optimal_k: ~62ms for 500 items
  - Proxy opt_ms in logs: 70-322ms (still has other overhead)

## Related Files

### Core Optimization Files
- `/home/joe/llm/headroom-fork/headroom/utils.py` - `safe_json_dumps` with orjson
- `/home/joe/llm/headroom-fork/headroom/compression/smart/serialize.py` - orjson JSON
- `/home/joe/llm/headroom-fork/headroom/compression/smart/optimized_simhash.py` - Rust simhash integration
- `/home/joe/llm/headroom-fork/headroom/transforms/adaptive_sizer.py` - count_unique_simhash

### Rust Extension
- `/home/joe/llm/headroom-fork/crates/headroom-simhash/` - PyO3 simhash extension

### Unified Proxy (local, NOT in git)
- `/home/joe/.local/bin/headroom-unified-proxy` - Monkey-patching script running on port 8787

### Logs
- `~/.headroom/logs/watchdog.log` - Request logs, timing
- `~/.headroom/logs/proxy.log` - PERF metrics
- `~/.headroom/logs/unified_proxy.log` - Startup logs

## Git Status

```
Branch: feature/dynamic-upstream-url
Remote: nosovj/headroom
Latest commit: d5f2366 feat: Add simhash benchmark verification script
```

## Running Proxy Status

- Process: 3005628 (python3.11)
- Port: 8787
- Status: Healthy, receiving traffic
- Rust simhash: Patched and working (confirmed at startup)

## Open Questions (from design.md)

1. **Pool sizing**: Auto-tune based on load, or fixed at startup?
2. **Worker lifetime**: Recycle after N requests to prevent memory leaks?
3. **Fallback behavior**: Sync Python or error if all workers fail?
4. **Cache warming**: Pre-populate on startup with common code patterns?

## Next Steps

1. Run `/opsx-apply` to begin implementation
2. Or `/opsx-verify` to check spec coverage
3. Or `/opsx-archive` to save and revisit later

## For Next Session

The next engineer should:
1. Read `proposal.md` for context
2. Read `design.md` for architecture decisions
3. Read all 4 spec files in `specs/` for requirements
4. Start with Phase 1 (Worker Pool Infrastructure) tasks in `tasks.md`
5. Use feature flags to rollback if issues arise