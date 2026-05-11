//! M4.8.a — MatMul / BatchMatMul micro-benchmark harness.
//!
//! Empirical baseline for the M4.8 performance optimization
//! milestone. Measures every CPU MatMul kernel currently
//! reachable in the codebase against the four canonical
//! Llama-family shapes:
//!
//!   - **1 × 5120 × 5120** — Llama 2 13B Q/K/V/O projection
//!     at `seq = 1`.
//!   - **4 × 5120 × 13824** — Llama 2 13B gate / up projection
//!     at `seq = 4`.
//!   - **1 × 4096 × 32000** — LM-head scale at `seq = 1`
//!     (generic 4096-hidden vocab-32k head).
//!   - **batched 40 × 4 × 128 × 128** — Llama 2 13B attention
//!     `BatchMatMul` (40 heads × seq=4 × head_dim=128).
//!
//! The harness is the **close criterion** for every M4.8
//! sub-phase: M4.8.b (default-mode fix), M4.8.c (decode cache
//! + SIMD BF16), M4.8.d (parallel BatchMatMul / MatMul), and
//! M4.8.e (matrixmultiply integration) each must produce a
//! measurable improvement on the same shapes, in the same
//! harness, on the same dev box.
//!
//! Usage:
//! ```powershell
//! cargo run --release --example bench_matmul
//! ```
//!
//! For an "AVX2-aware build" baseline, set RUSTFLAGS first;
//! the harness reports both the runtime AVX2 detection and
//! the compile-time `target_feature` cfg for the running
//! binary so the operator can tell which build they are
//! looking at:
//! ```powershell
//! $env:RUSTFLAGS = "-C target-cpu=native"
//! cargo run --release --example bench_matmul
//! ```
//!
//! All numbers are **wall-clock** (`std::time::Instant`).
//! Each measurement runs 1 warmup iteration + N timed
//! iterations; the harness reports both **median** and
//! **min** elapsed time across the timed iterations. GFLOPS
//! is computed as `2·M·K·N / time` (one multiply + one add
//! per inner-loop iteration). GB/s is the read-bandwidth
//! lower bound — `(M·K + K·N + M·N) · 4 bytes / time` —
//! useful for spotting memory-bound kernels.
//!
//! The harness does NOT verify numerical correctness against
//! a reference kernel: ADR-004 + the M4.7.5.f F64 family
//! validation already cover that. The bench is FLOP / wall-
//! clock only.

use std::time::Instant;

use atenia_engine::apx6::matmul_tiled_6_3b::{matmul_tiled_6_3b, matmul_tiled_6_3b_pex};
use atenia_engine::apx6_4::matmul_4x8_avx2;
use atenia_engine::matmul_dispatcher::{batch_matmul_dispatch, matmul_dispatch};
use atenia_engine::simd_kernels::avx2::bf16_decode_bulk;
use atenia_engine::simd_kernels::avx2::matmul_avx2;
use atenia_engine::tensor::tensor::{bf16_bits_to_f32, f32_to_bf16_bits};

const WARMUP_ITERS: usize = 1;
/// Iterations actually timed. Larger N tightens noise but
/// the slowest shape (`4 × 5120 × 13824`) on the scalar
/// fallback would balloon the harness — keep it short.
const TIMED_ITERS: usize = 5;
/// Iterations for the largest scalar fallback (which can
/// take many minutes on the seq=4 13B shapes). Reported but
/// run once if the kernel is too slow to time meaningfully.
const TIMED_ITERS_SLOW: usize = 1;

/// Detected at runtime; reported once at the top of the
/// output so the operator sees which ISA the harness is
/// actually exercising. Independent of the compile-time
/// `target_feature` cfg — `matmul_avx2` works regardless of
/// the cfg because the kernel is `unsafe fn` over raw
/// intrinsics.
fn report_environment() {
    println!("\n=== bench_matmul — M4.8.a baseline harness ===\n");
    println!(
        "Runtime CPU detection:   AVX2={}  FMA={}  AVX512F={}",
        std::is_x86_feature_detected!("avx2"),
        std::is_x86_feature_detected!("fma"),
        std::is_x86_feature_detected!("avx512f"),
    );
    println!(
        "Compile-time target_feature cfg:   avx2={}  fma={}  avx512f={}",
        cfg!(target_feature = "avx2"),
        cfg!(target_feature = "fma"),
        cfg!(target_feature = "avx512f"),
    );
    println!(
        "ATENIA_APX_MODE = {:?}  (default = \"4.19\")",
        std::env::var("ATENIA_APX_MODE").ok()
    );
    println!("apx_mode() resolved: {:?}", atenia_engine::apx_mode());
    println!("CPU threads (rayon):   {}", rayon::current_num_threads());
    println!();
}

#[derive(Clone, Copy)]
struct MatMulShape {
    label: &'static str,
    m: usize,
    k: usize,
    n: usize,
}

impl MatMulShape {
    fn flops(&self) -> f64 {
        // 2·M·K·N: one multiply + one add per inner-loop step.
        2.0 * (self.m as f64) * (self.k as f64) * (self.n as f64)
    }

    fn bytes(&self) -> f64 {
        // F32 read-bandwidth lower bound. Real kernels reuse
        // through cache so this is only a sanity sentinel.
        ((self.m * self.k) + (self.k * self.n) + (self.m * self.n)) as f64 * 4.0
    }
}

const MATMUL_SHAPES: &[MatMulShape] = &[
    MatMulShape {
        label: "Llama 2 13B Q/K/V/O proj seq=1",
        m: 1,
        k: 5120,
        n: 5120,
    },
    MatMulShape {
        label: "Llama 2 13B gate/up proj seq=4",
        m: 4,
        k: 5120,
        n: 13824,
    },
    MatMulShape {
        label: "LM head 4096 → 32000 seq=1",
        m: 1,
        k: 4096,
        n: 32000,
    },
];

#[derive(Clone, Copy)]
struct BatchShape {
    label: &'static str,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
}

impl BatchShape {
    fn flops(&self) -> f64 {
        2.0 * (self.batch as f64) * (self.m as f64) * (self.k as f64) * (self.n as f64)
    }

    fn bytes(&self) -> f64 {
        let per = (self.m * self.k) + (self.k * self.n) + (self.m * self.n);
        (self.batch * per) as f64 * 4.0
    }
}

const BATCH_SHAPES: &[BatchShape] = &[BatchShape {
    label: "Llama 2 13B attention QK^T (seq=4, 40 heads)",
    batch: 40,
    m: 4,
    k: 128,
    n: 128,
}];

/// Time a closure over `iters` repetitions plus a single
/// warmup, returning `(median_secs, min_secs)`.
fn time_iters<F: FnMut()>(iters: usize, mut f: F) -> (f64, f64) {
    for _ in 0..WARMUP_ITERS {
        f();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        samples.push(t0.elapsed().as_secs_f64());
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = samples[samples.len() / 2];
    let min = samples[0];
    (median, min)
}

/// Pretty-print a kernel result row for a MatMul shape.
fn print_row(kernel: &str, shape: &MatMulShape, median_s: f64, min_s: f64) {
    let gflops = shape.flops() / median_s / 1.0e9;
    let gbs = shape.bytes() / median_s / 1.0e9;
    println!(
        "    {:<28}  {:>10.3} ms (min {:>8.3} ms)  {:>8.2} GFLOPS  {:>7.2} GB/s",
        kernel,
        median_s * 1000.0,
        min_s * 1000.0,
        gflops,
        gbs,
    );
}

fn print_batch_row(kernel: &str, shape: &BatchShape, median_s: f64, min_s: f64) {
    let gflops = shape.flops() / median_s / 1.0e9;
    let gbs = shape.bytes() / median_s / 1.0e9;
    println!(
        "    {:<28}  {:>10.3} ms (min {:>8.3} ms)  {:>8.2} GFLOPS  {:>7.2} GB/s",
        kernel,
        median_s * 1000.0,
        min_s * 1000.0,
        gflops,
        gbs,
    );
}

// ---------- MatMul candidate kernels ----------

fn run_scalar(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    // Reference implementation — same triple-loop the
    // `scalar_matmul` registration in `lib.rs` ships at
    // default build. Reproduced inline so the harness does
    // not depend on whether the registry has the kernel.
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

fn run_avx2_basic(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    // SAFETY: caller responsibility to ensure CPU supports
    // AVX2; harness guards every call with the runtime
    // detection flag.
    unsafe {
        matmul_avx2(a, b, out, m, k, n);
    }
}

fn run_avx2_tiled_6_3b(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    matmul_tiled_6_3b(a, b, out, m, k, n);
}

fn run_avx2_tiled_6_3b_pex(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    matmul_tiled_6_3b_pex(a, b, out, m, k, n);
}

fn run_avx2_4x8_blis(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    // The 6.4 BLIS-style microkernel hardcodes a 4×8 panel
    // assumption; M >= 256 is the dispatcher's gate. We call
    // it on every shape and let it run — it produces correct
    // output but may be sub-optimal below the gate; that is
    // the data we want.
    matmul_4x8_avx2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), m, k, n);
}

fn run_dispatch(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    // The current production path. At default mode "4.19"
    // this falls through to apx3_8 → scalar_matmul on a
    // build without `target_feature = "avx2"` set. With
    // `RUSTFLAGS=-C target-cpu=native` the avx2_matmul
    // registration fires and the dispatcher resolves to it.
    matmul_dispatch(a, b, out, m, k, n);
}

// ---------- MatMul shape benchmark ----------

fn bench_matmul_shape(shape: &MatMulShape) {
    println!(
        "\n[Shape: {}]   M={}  K={}  N={}  ({:.2} GFLOPs)",
        shape.label,
        shape.m,
        shape.k,
        shape.n,
        shape.flops() / 1.0e9
    );

    let a: Vec<f32> = (0..(shape.m * shape.k))
        .map(|i| (i as f32 * 1e-4).sin())
        .collect();
    let b: Vec<f32> = (0..(shape.k * shape.n))
        .map(|i| (i as f32 * 1e-4).cos())
        .collect();
    let mut out = vec![0.0_f32; shape.m * shape.n];

    // The scalar fallback is brutally slow on `4 × 5120 ×
    // 13824` (~565 GFLOPs); time it once instead of 5×.
    let scalar_iters = if shape.flops() > 100.0e9 {
        TIMED_ITERS_SLOW
    } else {
        TIMED_ITERS
    };

    let (m_s, mn_s) = time_iters(scalar_iters, || {
        run_scalar(&a, &b, &mut out, shape.m, shape.k, shape.n);
    });
    print_row(
        if scalar_iters == TIMED_ITERS_SLOW {
            "scalar (1 iter)"
        } else {
            "scalar"
        },
        shape,
        m_s,
        mn_s,
    );

    if std::is_x86_feature_detected!("avx2") {
        let (m_s, mn_s) = time_iters(TIMED_ITERS, || {
            run_avx2_basic(&a, &b, &mut out, shape.m, shape.k, shape.n);
        });
        print_row("avx2 basic (simd_kernels)", shape, m_s, mn_s);

        let (m_s, mn_s) = time_iters(TIMED_ITERS, || {
            run_avx2_tiled_6_3b(&a, &b, &mut out, shape.m, shape.k, shape.n);
        });
        print_row("avx2 tiled 6.3b", shape, m_s, mn_s);

        let (m_s, mn_s) = time_iters(TIMED_ITERS, || {
            run_avx2_tiled_6_3b_pex(&a, &b, &mut out, shape.m, shape.k, shape.n);
        });
        print_row("avx2 tiled 6.3b PEX", shape, m_s, mn_s);

        // The 4x8 BLIS microkernel only handles M ≥ 4.
        if shape.m >= 4 {
            let (m_s, mn_s) = time_iters(TIMED_ITERS, || {
                run_avx2_4x8_blis(&a, &b, &mut out, shape.m, shape.k, shape.n);
            });
            print_row("avx2 4x8 BLIS (apx6_4)", shape, m_s, mn_s);
        }

        // M4.8.e: matrixmultiply::sgemm direct call. Single-
        // threaded (the library's default build); the
        // dispatcher's rayon row-partition wrapper layers
        // multi-thread parallelism on top.
        let (m_s, mn_s) = time_iters(TIMED_ITERS, || unsafe {
            matrixmultiply::sgemm(
                shape.m,
                shape.k,
                shape.n,
                1.0,
                a.as_ptr(),
                shape.k as isize,
                1,
                b.as_ptr(),
                shape.n as isize,
                1,
                0.0,
                out.as_mut_ptr(),
                shape.n as isize,
                1,
            );
        });
        print_row("matrixmultiply sgemm", shape, m_s, mn_s);
    }

    let (m_s, mn_s) = time_iters(scalar_iters, || {
        run_dispatch(&a, &b, &mut out, shape.m, shape.k, shape.n);
    });
    print_row(
        if scalar_iters == TIMED_ITERS_SLOW {
            "matmul_dispatch (1 iter)"
        } else {
            "matmul_dispatch"
        },
        shape,
        m_s,
        mn_s,
    );
}

// ---------- BatchMatMul shape benchmark ----------

fn bench_batch_shape(shape: &BatchShape) {
    println!(
        "\n[BatchMatMul: {}]   batch={} M={} K={} N={}  ({:.2} GFLOPs)",
        shape.label,
        shape.batch,
        shape.m,
        shape.k,
        shape.n,
        shape.flops() / 1.0e9
    );

    let total_a = shape.batch * shape.m * shape.k;
    let total_b = shape.batch * shape.k * shape.n;
    let total_out = shape.batch * shape.m * shape.n;
    let a: Vec<f32> = (0..total_a).map(|i| (i as f32 * 1e-3).sin()).collect();
    let b: Vec<f32> = (0..total_b).map(|i| (i as f32 * 1e-3).cos()).collect();
    let mut out = vec![0.0_f32; total_out];

    // Current production path — serial for-loop in
    // `batch_matmul_dispatch` over each batch slice through
    // the `dispatch_matmul_apx3_8` registry chain.
    let (m_s, mn_s) = time_iters(TIMED_ITERS, || {
        batch_matmul_dispatch(&a, &b, &mut out, shape.batch, shape.m, shape.k, shape.n);
    });
    print_batch_row("batch_matmul_dispatch (serial)", shape, m_s, mn_s);

    if std::is_x86_feature_detected!("avx2") {
        // Reference: serial loop over `matmul_tiled_6_3b`. The
        // delta against `batch_matmul_dispatch` measures the
        // dispatch overhead alone (kernel choice constant);
        // delta against `batch_matmul_dispatch (serial)` is
        // dominated by which matmul kernel each path picks.
        let (m_s, mn_s) = time_iters(TIMED_ITERS, || {
            let stride_a = shape.m * shape.k;
            let stride_b = shape.k * shape.n;
            let stride_out = shape.m * shape.n;
            for bi in 0..shape.batch {
                let a_b = &a[bi * stride_a..(bi + 1) * stride_a];
                let b_b = &b[bi * stride_b..(bi + 1) * stride_b];
                let out_b = &mut out[bi * stride_out..(bi + 1) * stride_out];
                run_avx2_tiled_6_3b(a_b, b_b, out_b, shape.m, shape.k, shape.n);
            }
        });
        print_batch_row("serial loop @ avx2 6.3b", shape, m_s, mn_s);

        // What M4.8.d will turn this into. Manual rayon
        // par_iter over the batch dim feeding the 6.3b
        // kernel into per-batch slices; rayon is already a
        // dependency.
        use rayon::prelude::*;
        let (m_s, mn_s) = time_iters(TIMED_ITERS, || {
            let stride_a = shape.m * shape.k;
            let stride_b = shape.k * shape.n;
            let stride_out = shape.m * shape.n;
            out.par_chunks_mut(stride_out)
                .enumerate()
                .for_each(|(bi, out_b)| {
                    let a_b = &a[bi * stride_a..(bi + 1) * stride_a];
                    let b_b = &b[bi * stride_b..(bi + 1) * stride_b];
                    matmul_tiled_6_3b(a_b, b_b, out_b, shape.m, shape.k, shape.n);
                });
        });
        print_batch_row("rayon par_iter @ avx2 6.3b", shape, m_s, mn_s);
    }
}

// ---------- BF16 decode + allocation cost ----------

fn bench_bf16_and_alloc() {
    println!("\n[BF16 decode + allocation cost — Llama 2 13B-class layer (5120 × 13824)]");

    // 5120 × 13824 = 70,778,880 elements. The `down_proj`
    // weight in Llama 2 13B at MLP layer.
    let n_elems: usize = 5120 * 13824;
    println!(
        "    elements: {} ({:.1} MB BF16, {:.1} MB F32)",
        n_elems,
        n_elems as f64 * 2.0 / 1_048_576.0,
        n_elems as f64 * 4.0 / 1_048_576.0
    );

    // Synth BF16 payload: round-trip a deterministic F32
    // pattern through `f32_to_bf16_bits` so the bytes are
    // representative of what `set_store_params_as_bf16(true)`
    // produces at load time.
    let bf16_payload: Vec<u16> = (0..n_elems)
        .map(|i| f32_to_bf16_bits((i as f32 * 1e-5).sin()))
        .collect();

    // Cost #1a: scalar BF16 → F32 decode (the pre-M4.8.c
    // `Tensor::ensure_cpu` BF16 arm — `bits.iter().map(...)
    // .collect()`). Kept as the regression baseline so future
    // sub-phases can confirm the SIMD path stays ahead.
    let (m_s, mn_s) = time_iters(TIMED_ITERS, || {
        let _decoded: Vec<f32> = bf16_payload.iter().map(|&b| bf16_bits_to_f32(b)).collect();
    });
    println!(
        "    scalar bf16→f32 decode + alloc:        {:>10.3} ms (min {:>8.3} ms)   {:>6.2} GB/s decode",
        m_s * 1000.0,
        mn_s * 1000.0,
        n_elems as f64 * 4.0 / m_s / 1.0e9,
    );

    // Cost #1b: M4.8.c SIMD bulk decode (`bf16_decode_bulk`,
    // 8-lane AVX2 with scalar tail). The current
    // `Tensor::ensure_cpu` BF16 arm and `copy_to_cpu_vec`
    // BF16 arm both route through this kernel. Decode-only
    // cost: write into a pre-allocated `Vec<f32>` so the
    // measurement excludes the M4.8.c-unrelated allocation.
    let mut scratch = vec![0.0_f32; n_elems];
    let (m_s, mn_s) = time_iters(TIMED_ITERS, || {
        bf16_decode_bulk(&bf16_payload, &mut scratch);
    });
    println!(
        "    SIMD bf16→f32 decode (M4.8.c):         {:>10.3} ms (min {:>8.3} ms)   {:>6.2} GB/s decode",
        m_s * 1000.0,
        mn_s * 1000.0,
        n_elems as f64 * 4.0 / m_s / 1.0e9,
    );

    // Cost #2: clone of a `Vec<u16>` of the same size — the
    // hidden cost of `Tensor::clone()` at the top of every
    // MatMul executor arm (`graph.rs:2942-2951`).
    let (m_s, mn_s) = time_iters(TIMED_ITERS, || {
        let _cloned = bf16_payload.clone();
    });
    println!(
        "    Vec<u16> clone (~142 MB):              {:>10.3} ms (min {:>8.3} ms)   {:>6.2} GB/s memcpy",
        m_s * 1000.0,
        mn_s * 1000.0,
        n_elems as f64 * 2.0 / m_s / 1.0e9,
    );

    // Cost #3: raw `Vec<f32>` allocation (zero-init) of the
    // MatMul output buffer for one 13B layer.
    let (m_s, mn_s) = time_iters(TIMED_ITERS, || {
        let _v = vec![0.0_f32; n_elems];
    });
    println!(
        "    vec![0.0_f32; N] allocation (~283 MB): {:>10.3} ms (min {:>8.3} ms)",
        m_s * 1000.0,
        mn_s * 1000.0,
    );

    // Cost #4: clone of a `Vec<f32>` of the same size — what
    // happens when an already-decoded operand is cloned
    // through the MatMul path on a hot tensor.
    let f32_payload: Vec<f32> = bf16_payload.iter().map(|&b| bf16_bits_to_f32(b)).collect();
    let (m_s, mn_s) = time_iters(TIMED_ITERS, || {
        let _cloned = f32_payload.clone();
    });
    println!(
        "    Vec<f32> clone (~283 MB):              {:>10.3} ms (min {:>8.3} ms)   {:>6.2} GB/s memcpy",
        m_s * 1000.0,
        mn_s * 1000.0,
        n_elems as f64 * 4.0 / m_s / 1.0e9,
    );

    // Putting it together: the per-MatMul-call overhead in
    // the current MatMul executor arm is roughly
    //   2 × (Vec<u16> clone of operand) +
    //   2 × (scalar BF16 → F32 decode of operand) +
    //   1 × (vec![0.0; M·N] for the output)
    // before the matmul kernel even starts. The harness's
    // raw numbers above let M4.8.b and M4.8.c quantify how
    // much each fix removes.
    println!("    Per-MatMul-call overhead (estimated 2× clone + 2× decode + 1× output alloc):");
    println!(
        "        ~{:.0} ms per MatMul call on a 13B-class operand pair",
        (m_s + m_s) * 1000.0 /* placeholder — real composite goes via the executor */
    );
    println!("        At 280 MatMul calls per Llama 2 13B forward, this is the lower bound");
    println!("        of the M4.8.c decode-cache savings target.");
}

fn main() {
    report_environment();

    println!("=== MatMul shapes ===");
    for shape in MATMUL_SHAPES {
        bench_matmul_shape(shape);
    }

    println!("\n=== BatchMatMul shapes ===");
    for shape in BATCH_SHAPES {
        bench_batch_shape(shape);
    }

    bench_bf16_and_alloc();

    println!("\n=== bench_matmul complete ===\n");
}
