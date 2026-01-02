#![allow(dead_code)]

/// Fragmentation model (purely observational).
///
/// fragmentation = 1.0 - (largest_block / total_free)
///
/// The caller is responsible for providing `total_free` and `largest_block`
/// as logical values; this module does not touch real allocators.
pub fn compute_fragmentation_ratio(total_free_bytes: u64, largest_block_bytes: u64) -> f64 {
    if total_free_bytes == 0 {
        return 0.0;
    }
    let largest = largest_block_bytes.min(total_free_bytes) as f64;
    let total = total_free_bytes as f64;
    let ratio = 1.0 - (largest / total);
    if ratio < 0.0 {
        0.0
    } else if ratio > 1.0 {
        1.0
    } else {
        ratio
    }
}
