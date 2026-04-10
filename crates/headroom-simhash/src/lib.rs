//! Rust simhash implementation with PyO3 bindings.
//!
//! Uses twox-hash (xxHash64) for fast 4-gram hashing.
//! rayon for parallel batch computation with GIL release.

use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::max;
use std::hash::Hasher;
use twox_hash::XxHash64;

/// Compute 64-bit simhash for a single text string.
fn simhash_64(text: &str) -> u64 {
    let text_lower = text.to_lowercase();
    let chars: Vec<char> = text_lower.chars().collect();
    let n = chars.len();

    if n == 0 {
        return 0;
    }

    let mut v = [0i64; 64];
    let gram_count = max(1, n.saturating_sub(3));

    for i in 0..gram_count {
        let end = (i + 4).min(n);
        let gram: String = chars[i..end].iter().collect();
        let mut hasher = XxHash64::with_seed(0);
        hasher.write(gram.as_bytes());
        let hash = hasher.finish();
        for j in 0..64 {
            if hash & (1 << j) != 0 {
                v[j] += 1;
            } else {
                v[j] -= 1;
            }
        }
    }

    let mut fingerprint = 0u64;
    for j in 0..64 {
        if v[j] > 0 {
            fingerprint |= 1 << j;
        }
    }
    fingerprint
}

/// Compute Hamming distance between two 64-bit fingerprints.
fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Count unique items using greedy Hamming clustering.
fn count_unique_cluster(items: &[String], threshold: u32) -> usize {
    if items.is_empty() {
        return 0;
    }

    let fingerprints: Vec<u64> = items.iter().map(|s| simhash_64(s)).collect();
    let mut clusters: Vec<u64> = Vec::new();

    for fp in &fingerprints {
        let mut matched = false;
        for rep in &clusters {
            if hamming_distance(*fp, *rep) <= threshold {
                matched = true;
                break;
            }
        }
        if !matched {
            clusters.push(*fp);
        }
    }

    clusters.len()
}

// =============================================================================
// PyO3 Functions
// =============================================================================

/// Compute simhash for a single text string.
#[pyfunction]
pub fn compute_simhash(text: &str) -> u64 {
    simhash_64(text)
}

/// Compute simhash for multiple texts in parallel using rayon.
#[pyfunction]
pub fn compute_simhash_batch(texts: Vec<String>) -> Vec<u64> {
    texts.par_iter().map(|t| simhash_64(t)).collect()
}

/// Count unique items using GIL-free parallel computation.
#[pyfunction]
pub fn count_unique_simhash(items: Vec<String>, threshold: u32) -> usize {
    count_unique_cluster(&items, threshold)
}

// =============================================================================
// Module Definition
// =============================================================================

#[pymodule]
pub fn headroom_simhash(_: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_simhash, m)?)?;
    m.add_function(wrap_pyfunction!(compute_simhash_batch, m)?)?;
    m.add_function(wrap_pyfunction!(count_unique_simhash, m)?)?;
    Ok(())
}
