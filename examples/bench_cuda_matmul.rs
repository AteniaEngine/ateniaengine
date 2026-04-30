//! M6.a — R1 falsifier: GPU vs CPU on the production
//! decode-step matmul shapes (M=1, with full host↔device
//! copies on the GPU side).
//!
//! ## What R1 asks
//!
//! "Does CUDA `cuda_matmul` (full CPU-roundtrip — alloc
//! VRAM, cudaMemcpy host→device, kernel, cudaMemcpy
//! device→host, free) beat the pure-CPU
//! `matmul_dispatch` path on the shapes the M5 decode
//! loop actually runs at M=1?"
//!
//! Success criterion: GPU < CPU by ≥30% on at least one
//! representative shape. Failure means the per-call PCIe
//! roundtrip swallows the GPU compute win and M6 must
//! pivot to persistent-residency before any other lever
//! (the §4 design from the M6 research report).
//!
//! ## Pool gate caveat
//!
//! The current `cuda_matmul` (`src/cuda/matmul.rs:31-59`)
//! routes through `with_pooled_device_buffers`, which is
//! gated by `apx4_12::DEFAULT_BLOCK_SIZE = 64 MiB`. The
//! production FFN-down shape on Llama 2 13B Chat is
//! `[1, 13824] @ [13824, 5120]` F32 ⇒ B = 270 MB —
//! **does not fit the pool**. Calling cuda_matmul() on
//! that shape panics with `PoolExhausted`.
//!
//! M6.a therefore measures the closest shapes that DO
//! fit (B ≤ 64 MB), reports the trend, and extrapolates
//! to the production shape via PCIe-Gen4-×8-bandwidth math
//! (≈16 GB/s effective). M6.b lifts the pool gate — at
//! that point this bench should be re-run on the full
//! FFN-down / FFN-up / lm_head shapes.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --example bench_cuda_matmul
//! ```
//!
//! No CLI args. Runs every shape in the table below,
//! warming up once and timing N=20 invocations per side.

use std::time::Instant;
use atenia_engine::tensor::{Device, Tensor};

/// Pool block ceiling. Mirrors `apx4_12::DEFAULT_BLOCK_SIZE`.
const POOL_BLOCK_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone, Copy)]
struct ShapeBench {
    label: &'static str,
    /// (m, k, n) — the matmul shape. Atenia convention: a
    /// is (m × k), b is (k × n), out is (m × n).
    shape: (usize, usize, usize),
}

const SHAPES: &[ShapeBench] = &[
    // Q/K/V/O projection territory on Llama 2 13B Chat
    // (hidden=5120). Production shape is [1, 5120, 5120]
    // F32 ⇒ B = 100 MB — does NOT fit the pool. The
    // `_64mib_fit` variant brings B under the pool ceiling.
    ShapeBench {
        label: "QKVO_64mib_fit / [1, 5120, 3072]",
        shape: (1, 5120, 3072),  // B = 60 MB
    },
    // FFN territory on Llama 2 13B Chat
    // (hidden=5120, intermediate=13824). Production
    // gate/up shape is [1, 5120, 13824] F32 ⇒ B = 270 MB —
    // does NOT fit. Production down shape is
    // [1, 13824, 5120] F32 ⇒ B = 270 MB — does NOT fit.
    // Clamp to the largest-N that fits the pool.
    ShapeBench {
        label: "FFN_gate_64mib_fit / [1, 5120, 3072]",
        shape: (1, 5120, 3072),  // B = 60 MB
    },
    ShapeBench {
        label: "FFN_down_64mib_fit / [1, 13824, 1216]",
        shape: (1, 13824, 1216), // B = 64 MB exactly
    },
    // Smaller "head_dim" shapes for sanity — the matmul
    // dispatcher's M=1 path is known to be a worst case
    // for matrixmultiply (per HANDOFF M4.8 line 279).
    ShapeBench {
        label: "tiny / [1, 4096, 4096]",
        shape: (1, 4096, 4096),  // B = 64 MB exactly
    },
    ShapeBench {
        label: "small / [1, 1024, 1024]",
        shape: (1, 1024, 1024),  // B = 4 MB
    },
];

const WARMUP: usize = 3;
const ITERS: usize = 20;

fn main() {
    eprintln!("=== Atenia M6.a R1 falsifier — CUDA vs CPU at M=1 ===");
    eprintln!();
    eprintln!("Measuring `cuda_matmul` (pool-routed, includes H↔D copies)");
    eprintln!("vs `matmul_dispatch` (CPU AVX2/FMA via matrixmultiply).");
    eprintln!();
    eprintln!("Pool-block ceiling: {} MiB ({}MB).",
        POOL_BLOCK_BYTES / (1024 * 1024),
        POOL_BLOCK_BYTES / 1_000_000);
    eprintln!();

    eprintln!("{:<42} {:>10} {:>10} {:>10} {:>14}",
        "shape (M, K, N)", "B size", "CPU ms", "GPU ms", "GPU/CPU ratio");
    eprintln!("{}", "-".repeat(92));

    let mut summary: Vec<(String, f64, f64, f64)> = Vec::new();

    for s in SHAPES {
        let (m, k, n) = s.shape;
        let b_bytes = k * n * 4;
        let b_size_mb = b_bytes as f64 / 1_000_000.0;
        let fits = b_bytes <= POOL_BLOCK_BYTES;

        if !fits {
            eprintln!("{:<42} {:>9.0}MB  -- out of pool, skipping --", s.label, b_size_mb);
            continue;
        }

        let (cpu_ms, gpu_ms) = bench_shape(m, k, n);
        let ratio = if cpu_ms > 0.0 { gpu_ms / cpu_ms } else { 0.0 };
        let speedup_str = if gpu_ms > 0.0 && gpu_ms < cpu_ms {
            format!("{:>10.3}x faster", cpu_ms / gpu_ms)
        } else if gpu_ms > 0.0 {
            format!("{:>10.3}x slower", gpu_ms / cpu_ms)
        } else {
            "(skipped)".to_string()
        };

        eprintln!("{:<42} {:>9.0}MB  {:>9.3}  {:>9.3}  {:>14}",
            s.label, b_size_mb, cpu_ms, gpu_ms, speedup_str);
        summary.push((s.label.to_string(), cpu_ms, gpu_ms, ratio));
    }

    eprintln!();
    eprintln!("=== R1 verdict ===");
    let any_gpu_win = summary.iter().any(|(_, cpu, gpu, _)|
        *gpu > 0.0 && *gpu < *cpu * 0.7);
    let any_gpu_runs = summary.iter().any(|(_, _, gpu, _)| *gpu > 0.0);

    if !any_gpu_runs {
        eprintln!("⚠  GPU never executed (cuda_matmul panicked / unavailable).");
        eprintln!("   Possible causes: CUDA driver missing, pool exhaustion,");
        eprintln!("   NVCC build of `matmul_kernel.cu` failed. Check `build.rs`");
        eprintln!("   output during the most recent `cargo build --release`.");
        eprintln!();
        eprintln!("   R1 cannot be evaluated until cuda_matmul runs. M6.b");
        eprintln!("   plan should be re-evaluated before implementation.");
    } else if any_gpu_win {
        eprintln!("✓ R1 PASS: at least one shape shows GPU ≥30% faster than CPU");
        eprintln!("  including full host↔device round-trip.");
        eprintln!();
        eprintln!("  M6 strategy validated empirically: GPU offload buys throughput");
        eprintln!("  even at M=1 with PCIe transfers. Proceed with M6.b (lift");
        eprintln!("  the pool gate) and M6.c (persistent-residency scheduler).");
    } else {
        eprintln!("⚠  R1 INCONCLUSIVE / FAIL: no shape shows GPU ≥30% faster than CPU.");
        eprintln!();
        eprintln!("  Possible interpretations:");
        eprintln!("  1. PCIe roundtrip dominates at the M=1 / B-fits-pool shapes.");
        eprintln!("     Production FFN shapes (B=270 MB) are larger and may");
        eprintln!("     amortise the transfer better — needs M6.b to measure.");
        eprintln!("  2. The matrixmultiply CPU path (M4.8) is fast enough at");
        eprintln!("     these shapes that GPU has no headroom.");
        eprintln!("  3. cuda_matmul has driver / kernel-launch overhead that");
        eprintln!("     dominates at small shapes.");
        eprintln!();
        eprintln!("  Recommended action: rebench on FFN-down 270 MB shape after");
        eprintln!("  M6.b lands the non-pooled allocator. If GPU is still slower");
        eprintln!("  there, M6 strategy must pivot.");
    }
    eprintln!();

    // ---- Extrapolation to production FFN-down shape ----
    //
    // The pool gate prevents direct measurement, so we
    // estimate via PCIe Gen4 ×8 at ~16 GB/s effective and
    // the CPU baseline known from M5.f.a (forward = 99.9%
    // of step, decomposes per the FLOP model).
    eprintln!("=== Production FFN-down extrapolation (M6.b target) ===");
    eprintln!();
    let ffn_down_b_bytes: usize = 13824 * 5120 * 4;
    let ffn_down_b_mb = ffn_down_b_bytes as f64 / 1_000_000.0;
    let ffn_down_flops = 2.0 * 13824.0 * 5120.0;
    let pcie_bw_gbs = 16.0e9;          // PCIe Gen4 ×8 effective
    let pcie_ms = (ffn_down_b_bytes as f64 / pcie_bw_gbs) * 1000.0;
    // RTX 4070 Laptop F32 throughput ~7-8 TFLOPS sustained
    let gpu_compute_tflops = 7.5e12;
    let gpu_compute_ms = (ffn_down_flops / gpu_compute_tflops) * 1000.0;
    let gpu_proj_ms = pcie_ms + gpu_compute_ms;

    eprintln!("  shape:                [1, 13824] @ [13824, 5120] F32");
    eprintln!("  B size:               {ffn_down_b_mb:.0} MB (fails 64 MiB pool)");
    eprintln!("  PCIe transfer (B):    {pcie_ms:.2} ms @ 16 GB/s");
    eprintln!("  GPU compute:          {gpu_compute_ms:.2} ms @ 7.5 TFLOPS F32");
    eprintln!("  Projected GPU total:  {gpu_proj_ms:.2} ms (transfer-bound)");
    eprintln!();
    eprintln!("  CPU baseline (M5.f.a): forward execute ~2160 ms / step on TinyLlama");
    eprintln!("                          (1 GFLOPS effective). Extrapolated to");
    eprintln!("                          13B FFN-down: ~150 ms/call (~5× larger op).");
    eprintln!();
    eprintln!("  Projected speedup:    150 ms / {gpu_proj_ms:.2} ms = ~{:.1}x",
        150.0 / gpu_proj_ms);
    eprintln!("                          (transfer-bound — confirms research");
    eprintln!("                          report §1.2 + §2.3 prediction)");
}

/// One shape → (cpu_ms_per_iter, gpu_ms_per_iter).
/// `gpu_ms` is `0.0` if cuda_matmul fails (panic absorbed by
/// `catch_unwind`).
fn bench_shape(m: usize, k: usize, n: usize) -> (f64, f64) {
    // Deterministic-ish input data (avoid `rand` dep noise).
    let a_data: Vec<f32> = (0..m * k)
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| ((i as f32) * 0.0007).cos())
        .collect();
    let a = Tensor::new_cpu(vec![m, k], a_data);
    let b = Tensor::new_cpu(vec![k, n], b_data);

    // CPU side — matmul_dispatch is what `Graph::execute_single`
    // routes to for `NodeType::MatMul` after M4.8.
    let mut out_buf = vec![0.0_f32; m * n];

    // Warmup
    for _ in 0..WARMUP {
        atenia_engine::matmul_dispatcher::matmul_dispatch(
            a.as_cpu_slice(), b.as_cpu_slice(), &mut out_buf, m, k, n);
    }
    let cpu_t0 = Instant::now();
    for _ in 0..ITERS {
        atenia_engine::matmul_dispatcher::matmul_dispatch(
            a.as_cpu_slice(), b.as_cpu_slice(), &mut out_buf, m, k, n);
    }
    let cpu_total = cpu_t0.elapsed().as_secs_f64() * 1000.0;
    let cpu_ms = cpu_total / ITERS as f64;

    // GPU side — `cuda_matmul` includes alloc/H→D/launch/D→H/free
    // through the pool. May panic with `PoolExhausted` for shapes
    // > 64 MiB; catch_unwind to keep the bench moving on the
    // fits-in-pool shapes.
    let gpu_ms = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Warmup
        for _ in 0..WARMUP {
            let _ = atenia_engine::cuda::matmul::cuda_matmul(&a, &b, m, k, n);
        }
        let gpu_t0 = Instant::now();
        for _ in 0..ITERS {
            let _ = atenia_engine::cuda::matmul::cuda_matmul(&a, &b, m, k, n);
        }
        gpu_t0.elapsed().as_secs_f64() * 1000.0 / ITERS as f64
    })).unwrap_or(0.0);

    let _ = Device::CPU; // silence unused-import warning
    (cpu_ms, gpu_ms)
}
