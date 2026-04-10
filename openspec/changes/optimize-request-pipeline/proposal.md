## Why

The headroom proxy is experiencing severe latency issues (5-70s upstream errors, 227-322ms optimization time) caused by synchronous Python operations blocking the async event loop, inefficient token counting, and content routing overhead. The current architecture cannot properly release the GIL during compute-heavy operations, causing request backlogs and health check failures.

## What Changes

- **Rust-based async pipeline**: Move compute-heavy transforms (simhash, token counting, content routing) to async Rust workers that release the GIL
- **Parallel transform execution**: Replace sequential transform pipeline with concurrent task execution
- **Single token counting pass**: Eliminate redundant tokenization by caching and reusing counts
- **Upstream retry with backoff**: Implement intelligent retry logic for 401/502 errors with exponential backoff
- **Worker pool architecture**: Replace synchronous function calls with a pool of async Rust workers

## Capabilities

### New Capabilities

- `async-compression-pipeline`: Async pipeline that executes transforms concurrently in Rust workers, releasing the GIL and avoiding event loop blocking
- `token-count-cache`: Single-pass token counting with in-memory cache to eliminate redundant tokenization (~88ms savings per request)
- `upstream-retry-logic`: Intelligent retry with exponential backoff for transient upstream errors (401, 502, timeouts)
- `rust-content-router`: Rust-based content routing with tree-sitter parsing that runs without GIL contention

### Modified Capabilities

- (none - existing capabilities remain unchanged, implementation changes only)

## Impact

**Affected code:**
- `headroom/proxy/server.py`: Replace sync transform calls with async worker invocations
- `headroom/transforms/adaptive_sizer.py`: Move to Rust worker with GIL-free execution
- `headroom/transforms/content_router.py`: Rewrite in Rust for parallel AST parsing
- `headroom/transforms/pipeline.py`: Replace sequential execution with concurrent task spawning

**Dependencies:**
- Rust toolchain (for PyO3 extensions)
- tokio async runtime (for Rust workers)
- rayon parallel iterators (for content router)

**APIs:**
- No public API changes
- Internal worker communication via async channels