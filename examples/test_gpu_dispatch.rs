//! M6 GPU dispatch sanity test — bypass every gate, call
//! `cuda_matmul_non_pooled` directly on the Llama 2 13B FFN-down
//! shape, and report whether the kernel succeeds.
//!
//! Tells us whether the M6 step 2b regression (if any) is in the
//! gate (the dispatch never reaches the kernel) or in the kernel
//! itself (it reaches but fails / is unexpectedly slow).
//!
//! # Usage
//!
//! ```powershell
//! cargo run --release --example test_gpu_dispatch --features gpu-trace
//! ```
//!
//! With `--features gpu-trace` the stderr log will show every
//! `cuda_malloc` / `cudaMemcpy` / kernel launch / `cuda_free` step
//! with size and per-step timings. Without the feature, only the
//! summary timings printed by this file are emitted.
//!
//! # Shape
//!
//! Llama 2 13B FFN-down: `out = x @ W_down`, where `x` is
//! `[seq=1, intermediate=13824]` and `W_down` is
//! `[intermediate=13824, hidden=5120]`. So the matmul is
//! `[1, 13824] × [13824, 5120] = [1, 5120]`. The largest single
//! buffer is `W_down` at `13824 × 5120 × 4 = 270 MB F32`, well
//! above the 64 MiB pool ceiling. This is the exact shape that
//! step 2b's non-pooled router is supposed to handle.
//!
//! # Output
//!
//! Three timings (allocation+transfer+kernel+download) for the GPU
//! path, one timing for the CPU reference, and a max-abs-diff
//! number to confirm numerical correctness. If the GPU path
//! returns `None` the example reports the failure verbosely and
//! exits with a non-zero status.

use std::time::Instant;

use atenia_engine::cuda::cuda_available;
use atenia_engine::cuda::matmul::cuda_matmul_non_pooled;

const M: usize = 1;
const K: usize = 13824;
const N: usize = 5120;

/// Generate a deterministic but non-trivial f32 pattern.
fn fill_pattern(len: usize, seed: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32) * 0.0001 + seed).sin() * 0.5)
        .collect()
}

/// Reference CPU matmul: `[m, k] × [k, n] = [m, n]`. Triple-loop,
/// no SIMD — small `m` keeps this practical for sanity comparison.
fn cpu_reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn main() {
    println!("=== M6 GPU Dispatch Sanity Test ===");
    println!("Shape: [{}, {}] x [{}, {}] = [{}, {}]", M, K, K, N, M, N);
    println!(
        "Sizes: A={:.1}MB B={:.1}MB out={:.1}MB",
        (M * K * 4) as f64 / (1024.0 * 1024.0),
        (K * N * 4) as f64 / (1024.0 * 1024.0),
        (M * N * 4) as f64 / (1024.0 * 1024.0),
    );
    println!();

    println!("[1/4] cuda_available() = {}", cuda_available());
    if !cuda_available() {
        eprintln!("FATAL: CUDA driver not available; this test is meaningless without a GPU.");
        std::process::exit(1);
    }
    println!();

    println!("[2/4] Allocating host buffers...");
    let t0 = Instant::now();
    let a = fill_pattern(M * K, 0.1);
    let b = fill_pattern(K * N, 0.2);
    println!(
        "      Done in {:.2}s. A.len={}, B.len={}",
        t0.elapsed().as_secs_f64(),
        a.len(),
        b.len()
    );
    println!();

    println!("[3/4] GPU path (cuda_matmul_non_pooled) — TWO consecutive calls, same shape...");
    println!(
        "      Goal: measure whether cuda_malloc cost is one-shot (CUDA context init) or per-call."
    );
    println!();

    println!("      --- Call #1 (cold path, includes any one-shot context init) ---");
    let t_gpu1 = Instant::now();
    let gpu_result1 = cuda_matmul_non_pooled(&a, &b, M, K, N);
    let gpu_elapsed1 = t_gpu1.elapsed().as_secs_f64();
    let gpu_tensor1 = match gpu_result1 {
        Some(t) => {
            println!(
                "      Call #1 SUCCESS in {:.3}s. Output shape: {:?}",
                gpu_elapsed1, t.shape
            );
            t
        }
        None => {
            eprintln!(
                "      Call #1 FAILED — cuda_matmul_non_pooled returned None after {:.3}s",
                gpu_elapsed1
            );
            std::process::exit(2);
        }
    };
    println!();

    println!("      --- Call #2 (warm path, CUDA context already initialised) ---");
    let t_gpu2 = Instant::now();
    let gpu_result2 = cuda_matmul_non_pooled(&a, &b, M, K, N);
    let gpu_elapsed2 = t_gpu2.elapsed().as_secs_f64();
    let gpu_tensor2 = match gpu_result2 {
        Some(t) => {
            println!(
                "      Call #2 SUCCESS in {:.3}s. Output shape: {:?}",
                gpu_elapsed2, t.shape
            );
            t
        }
        None => {
            eprintln!(
                "      Call #2 FAILED — cuda_matmul_non_pooled returned None after {:.3}s",
                gpu_elapsed2
            );
            std::process::exit(2);
        }
    };
    // Use the second tensor as the "definitive" GPU result (warm-path
    // numerics are what production would actually observe).
    let gpu_tensor = gpu_tensor2;
    let _ = gpu_tensor1;
    let gpu_elapsed = gpu_elapsed2;
    println!();

    println!("[4/4] CPU reference matmul (triple-loop)...");
    let t_cpu = Instant::now();
    let cpu_out = cpu_reference(&a, &b, M, K, N);
    let cpu_elapsed = t_cpu.elapsed().as_secs_f64();
    println!("      Done in {:.3}s.", cpu_elapsed);
    println!();

    // Numerical sanity. The CPU reference uses naive accumulation
    // order, GPU uses kernel-internal blocking, so some drift is
    // expected — ADR-004 / M4.7.3 envelope is 0.5 absolute on F32
    // matmul, which is generous enough.
    let gpu_data = gpu_tensor.as_cpu_slice();
    assert_eq!(gpu_data.len(), cpu_out.len());

    let mut max_abs_diff = 0.0_f32;
    let mut argmax = 0_usize;
    for (idx, (g, c)) in gpu_data.iter().zip(cpu_out.iter()).enumerate() {
        let d = (g - c).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
            argmax = idx;
        }
    }

    println!("=== Results ===");
    println!("GPU call #1 (cold): {:.3}s", gpu_elapsed1);
    println!("GPU call #2 (warm): {:.3}s", gpu_elapsed2);
    println!(
        "Cold-vs-warm delta: {:.3}s ({:.2}x cold)",
        gpu_elapsed1 - gpu_elapsed2,
        gpu_elapsed1 / gpu_elapsed2
    );
    println!("CPU path:           {:.3}s", cpu_elapsed);
    println!(
        "Warm GPU vs CPU:    {:.2}x{}",
        cpu_elapsed / gpu_elapsed2,
        if gpu_elapsed2 < cpu_elapsed {
            " (GPU faster)"
        } else {
            " (CPU faster)"
        }
    );
    println!(
        "Max |diff|: {:e} at index {} (gpu={:.6}, cpu={:.6})",
        max_abs_diff, argmax, gpu_data[argmax], cpu_out[argmax]
    );

    if max_abs_diff > 0.5 {
        eprintln!();
        eprintln!(
            "WARNING: max_abs_diff {:e} exceeds the ADR-004 0.5 envelope.",
            max_abs_diff
        );
        eprintln!("This usually indicates a kernel correctness issue, not a dispatch issue.");
        std::process::exit(3);
    }

    println!();
    println!("PASS: GPU dispatch reached the kernel and produced numerically valid output.");
}
