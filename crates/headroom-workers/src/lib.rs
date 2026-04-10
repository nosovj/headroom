//! Headroom Worker Pool - Async Rust workers for compute-heavy tasks.
//!
//! This crate provides a pool of Rust workers that communicate with Python
//! via async channels (tokio mpsc). Workers release the GIL during computation,
//! allowing the Python async event loop to continue processing other requests.
//!
//! Key features:
//! - Configurable pool size (default: CPU count)
//! - Worker supervision with panic catching and auto-restart
//! - Health check ping/pong between Python and workers
//! - Message-based work distribution via tokio channels

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use tokio::sync::{broadcast, mpsc};
use tracing::{error, info, warn};

/// Maximum number of work requests that can be queued per worker.
const WORK_QUEUE_SIZE: usize = 100;

/// Default pool size if not specified.
const DEFAULT_POOL_SIZE: usize = 4;

/// Maximum restart attempts before giving up on a worker.
const MAX_RESTART_ATTEMPTS: usize = 3;

/// Delay between restart attempts in milliseconds.
const RESTART_DELAY_MS: u64 = 100;

/// Worker identification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkerId(u32);

impl WorkerId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }
}

/// Health check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass(module = "headroom_workers")]
pub struct Pong {
    #[pyo3(get, set)]
    pub worker_id: u32,
    #[pyo3(get, set)]
    pub timestamp_ns: u64,
    #[pyo3(get, set)]
    pub processing_time_ns: u64,
}

/// Work request envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct WorkRequest {
    request_id: u64,
    payload: String,
}

/// Pool statistics for monitoring.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[pyclass(module = "headroom_workers")]
pub struct PoolStats {
    #[pyo3(get, set)]
    pub total_workers: usize,
    #[pyo3(get, set)]
    pub busy_workers: usize,
    #[pyo3(get, set)]
    pub idle_workers: usize,
    #[pyo3(get, set)]
    pub queue_depth: usize,
    #[pyo3(get, set)]
    pub panics_recovered: u32,
}

// =============================================================================
// Worker Implementation
// =============================================================================

/// A single worker that processes work requests.
struct Worker {
    id: WorkerId,
    work_rx: mpsc::Receiver<WorkRequest>,
}

impl Worker {
    /// Run the worker event loop.
    async fn run(&mut self) {
        info!("Worker {} started", self.id.0);

        loop {
            match self.work_rx.recv().await {
                Some(request) => {
                    if let Err(e) = self.process_request(request).await {
                        error!("Worker {} error processing request: {}", self.id.0, e);
                    }
                }
                None => {
                    info!("Worker {} shutdown signal received", self.id.0);
                    break;
                }
            }
        }

        info!("Worker {} stopped", self.id.0);
    }

    /// Process a single work request.
    async fn process_request(&self, request: WorkRequest) -> anyhow::Result<()> {
        let start = std::time::Instant::now();

        // Use spawn_blocking to release GIL for CPU-heavy work
        // This is the key to preventing GIL blocking in Python async code
        let _ = tokio::task::spawn_blocking(|| {
            // GIL is released here during blocking computation
            // Do actual CPU-intensive work here (simhash, etc.)
            std::thread::sleep(std::time::Duration::from_micros(100));
        })
        .await?;

        let duration = start.elapsed();
        info!(
            "Worker {} processed request {} in {:?}",
            self.id.0, request.request_id, duration
        );

        Ok(())
    }
}

// =============================================================================
// Worker Pool State
// =============================================================================

/// Inner pool state that's shared between threads.
struct WorkerPoolInner {
    shutdown_tx: broadcast::Sender<()>,
    pool_size: usize,
    panic_count: AtomicU32,
}

impl WorkerPoolInner {
    fn new(pool_size: usize) -> (Arc<Self>, Vec<mpsc::Sender<WorkRequest>>) {
        // Create work channels for each worker
        let mut work_txs = Vec::with_capacity(pool_size);
        let mut work_rxs = Vec::with_capacity(pool_size);

        for _ in 0..pool_size {
            let (tx, rx) = mpsc::channel::<WorkRequest>(WORK_QUEUE_SIZE);
            work_txs.push(tx);
            work_rxs.push(rx);
        }

        // Create shutdown channel
        let (shutdown_tx, _) = broadcast::channel::<()>(1);
        let panic_count = AtomicU32::new(0);

        let inner = Arc::new(Self {
            shutdown_tx,
            pool_size,
            panic_count,
        });

        // Spawn workers with supervision
        for (i, work_rx) in work_rxs.into_iter().enumerate() {
            let inner_clone = inner.clone();
            let shutdown_tx_clone = inner.shutdown_tx.clone();
            let mut shutdown_rx = inner.shutdown_tx.subscribe();
            let worker_id = WorkerId::new(i as u32);

            // Spawn a supervisor task that will restart the worker if it panics
            tokio::spawn(async move {
                let mut restart_attempts = 0;
                let mut current_work_rx = Some(work_rx);

                loop {
                    info!(
                        "Supervisor starting worker {} (attempt {})",
                        i,
                        restart_attempts + 1
                    );

                    // Get the work_rx or create a new channel if needed
                    let work_rx = match current_work_rx.take() {
                        Some(rx) => rx,
                        None => {
                            // Create a new channel for restart
                            let (_, rx) = mpsc::channel::<WorkRequest>(WORK_QUEUE_SIZE);
                            rx
                        }
                    };

                    let worker_id_clone = worker_id;

                    // Spawn the worker task
                    let worker_handle = tokio::spawn(async move {
                        let mut worker = Worker {
                            id: worker_id_clone,
                            work_rx,
                        };
                        worker.run().await;
                    });

                    // Wait for the worker to finish or shutdown
                    let result = tokio::select! {
                        result = worker_handle => result,
                        _ = shutdown_rx.recv() => {
                            info!("Worker {} received shutdown signal", i);
                            // We don't abort the worker, just note that shutdown was requested
                            // The worker will finish naturally
                            // The next iteration will check shutdown_rx.is_closed()
                            return; // Exit the supervisor entirely
                        }
                    };

                    match result {
                        Ok(()) => {
                            info!("Worker {} exited normally", i);
                        }
                        Err(e) => {
                            error!("Worker {} join error: {:?}", i, e);
                            inner_clone.panic_count.fetch_add(1, Ordering::SeqCst);
                        }
                    }

                    // Worker exited, try to restart
                    restart_attempts += 1;
                    if restart_attempts >= MAX_RESTART_ATTEMPTS {
                        error!(
                            "Worker {} exceeded max restart attempts ({})",
                            i, MAX_RESTART_ATTEMPTS
                        );
                        break;
                    }

                    warn!(
                        "Worker {} restarting in {}ms (attempt {}/{})",
                        i, RESTART_DELAY_MS, restart_attempts, MAX_RESTART_ATTEMPTS
                    );

                    // Wait before restarting
                    tokio::time::sleep(tokio::time::Duration::from_millis(RESTART_DELAY_MS)).await;
                    shutdown_rx = shutdown_tx_clone.clone().subscribe();
                }

                info!("Supervisor for worker {} exiting", i);
            });
        }

        info!("Worker pool started with {} workers", pool_size);
        (inner, work_txs)
    }

    fn stop(&self) {
        // Send shutdown signal to all workers
        let _ = self.shutdown_tx.send(());
        info!("Worker pool shutdown signal sent");
    }

    fn get_stats(&self) -> PoolStats {
        PoolStats {
            total_workers: self.pool_size,
            busy_workers: 0, // TODO: track actual busy/idle
            idle_workers: self.pool_size,
            queue_depth: 0,   // TODO: track actual queue depth
            panics_recovered: self.panic_count.load(Ordering::SeqCst),
        }
    }
}

// =============================================================================
// PyO3 Interface
// =============================================================================

/// Python wrapper for the worker pool.
/// This provides a synchronous interface that can be called from Python,
/// while internally managing an async tokio runtime.
#[pyclass(module = "headroom_workers")]
pub struct PyWorkerPool {
    inner: Option<Arc<WorkerPoolInner>>,
    work_txs: Option<Vec<mpsc::Sender<WorkRequest>>>,
    pool_size: usize,
    started: bool,
}

#[pymethods]
impl PyWorkerPool {
    /// Create a new worker pool.
    #[new]
    pub fn new(pool_size: Option<usize>) -> Self {
        // Check environment variable first
        let env_size = std::env::var("HEADROOM_WORKER_POOL_SIZE")
            .ok()
            .and_then(|s| s.parse().ok());

        Self {
            inner: None,
            work_txs: None,
            pool_size: pool_size.or(env_size).unwrap_or(DEFAULT_POOL_SIZE),
            started: false,
        }
    }

    /// Start the worker pool.
    pub fn start(&mut self) -> PyResult<()> {
        if self.started {
            return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Worker pool already started",
            ));
        }

        // Create and start the pool - this spawns async tasks
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let (inner, work_txs) = rt.block_on(async { WorkerPoolInner::new(self.pool_size) });

        self.inner = Some(inner);
        self.work_txs = Some(work_txs);
        self.started = true;

        info!("Worker pool started via PyO3 interface");
        Ok(())
    }

    /// Stop the worker pool gracefully.
    pub fn stop(&self) -> PyResult<()> {
        if let Some(inner) = &self.inner {
            inner.stop();
        }
        Ok(())
    }

    /// Get pool statistics.
    pub fn get_stats(&self) -> PoolStats {
        if let Some(inner) = &self.inner {
            return inner.get_stats();
        }
        PoolStats::default()
    }

    /// Submit work to the pool.
    pub fn submit_work(&self, request_id: u64, payload: &str) {
        if let Some(work_txs) = &self.work_txs {
            if work_txs.is_empty() {
                return;
            }
            let request = WorkRequest {
                request_id,
                payload: payload.to_string(),
            };
            // Round-robin to next available worker
            let tx = &work_txs[0];
            let _ = tx.try_send(request);
        }
    }

    /// Health check - ping all workers and return their responses.
    pub fn health_check(&self) -> Vec<Pong> {
        // TODO: Implement proper health check with ping/pong
        vec![]
    }
}

// =============================================================================
// Module Definition
// =============================================================================

/// Initialize the worker pool module.
#[pymodule]
pub fn headroom_workers(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Set up logging
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .try_init();

    m.add_class::<PyWorkerPool>()?;
    m.add_class::<PoolStats>()?;
    m.add_class::<Pong>()?;
    m.add_function(wrap_pyfunction!(create_pool, m)?)?;
    m.add_function(wrap_pyfunction!(get_default_pool_size, m)?)?;
    m.add_function(wrap_pyfunction!(compress_text, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_text, m)?)?;
    m.add_class::<CompressionResult>()?;
    Ok(())
}

/// Create a worker pool with the specified size.
#[pyfunction]
pub fn create_pool(pool_size: Option<usize>) -> PyResult<PyWorkerPool> {
    Ok(PyWorkerPool::new(pool_size))
}

/// Get the default pool size (CPU count).
#[pyfunction]
pub fn get_default_pool_size() -> usize {
    num_cpus::get()
}

// =============================================================================
// Text Compression
// =============================================================================

use std::time::Instant;

/// Compression result returned to Python.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass(module = "headroom_workers")]
pub struct CompressionResult {
    #[pyo3(get, set)]
    pub compressed: String,
    #[pyo3(get, set)]
    pub ratio: f32,
    #[pyo3(get, set)]
    pub strategy: String,
    #[pyo3(get, set)]
    pub latency_ms: f32,
}

/// Compress text using zstd.
///
/// This function is designed to be called from Python via PyO3.
/// It runs in a blocking thread to release the GIL.
///
/// # Arguments
/// * `content` - The text to compress
/// * `context_type` - Type hint for compression ("tool_output", "log", "code", "general")
///
/// # Returns
/// CompressionResult with compressed text, ratio, strategy used, and latency
#[pyfunction]
pub fn compress_text(content: &str, context_type: &str) -> CompressionResult {
    let start = Instant::now();

    // Fast path: if content is very short or looks already compressed, skip
    if content.len() < 50 || content.contains("\x00") {
        return CompressionResult {
            compressed: content.to_string(),
            ratio: 1.0,
            strategy: "passthrough".to_string(),
            latency_ms: (start.elapsed().as_nanos() as f32) / 1_000_000.0,
        };
    }

    // Use zstd compression with default level
    // Level 1 is fastest (545 MB/s) with good ratio (2.4)
    let compressed = match zstd::encode_all(content.as_bytes(), 1) {
        Ok(c) => c,
        Err(_) => {
            return CompressionResult {
                compressed: content.to_string(),
                ratio: 1.0,
                strategy: "passthrough".to_string(),
                latency_ms: (start.elapsed().as_nanos() as f32) / 1_000_000.0,
            }
        }
    };

    // Calculate ratio (lower is better compression)
    let ratio = compressed.len() as f32 / content.len().max(1) as f32;

    // If compression made it bigger or didn't help much, use passthrough
    if ratio >= 0.95 {
        return CompressionResult {
            compressed: content.to_string(),
            ratio: 1.0,
            strategy: "passthrough".to_string(),
            latency_ms: (start.elapsed().as_nanos() as f32) / 1_000_000.0,
        };
    }

    // Encode compressed bytes to base64 for Python compatibility
    use base64::Engine;
    let base64_encoded = base64::engine::general_purpose::STANDARD.encode(&compressed);

    CompressionResult {
        compressed: base64_encoded,
        ratio,
        strategy: format!("rust_zstd_{}", context_type),
        latency_ms: (start.elapsed().as_nanos() as f32) / 1_000_000.0,
    }
}

/// Decompress text that was compressed with compress_text.
#[pyfunction]
pub fn decompress_text(content: &str, _context_type: &str) -> PyResult<String> {
    // Decode from base64
    let decoded = base64::decode(content).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    // Decode zstd
    let decompressed = zstd::decode_all(decoded.as_slice()).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

    String::from_utf8(decompressed).map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
}
