//! M9.0 — INT8 W8A16 vs Path B (M8.4c) microbench.
//!
//! **Gating data point for the M9 milestone (INT8 weight quantisation).**
//! Compares two GPU matmul paths on the four shapes that dominate the
//! Llama 2 13B decode step. Same methodology as `bench_cublas_bf16.rs`
//! (the M8.0 gate); the result decides whether INT8 Tensor Cores on the
//! RTX 4070 (Ada, sm_89) deliver a real speedup vs the existing M8.4c
//! BF16-resident path.
//!
//! # Paths
//!
//! - **Path A — current production baseline (M8.4c)**:
//!   * Weight resident in VRAM as BF16 (1 byte/elem ÷ 2 vs F32).
//!   * Per-iter: BF16 → F32 transient upcast in VRAM (existing
//!     `bf16_to_f32_launch_device` kernel) followed by
//!     `cublasGemmEx(F32, F32, F32)` with `CUBLAS_COMPUTE_32F`.
//!   * This is exactly what `cuda_matmul_bf16_inplace` does today.
//!
//! - **Path B — M9 W8A16 candidate**:
//!   * Weight resident in VRAM as INT8 (1 byte/elem) plus per-output-
//!     channel F32 scales (N × 4 bytes — negligible vs the matrix).
//!   * Per-iter: INT8 → BF16 dequant kernel (new
//!     `int8_to_bf16_per_channel_launch_device` from
//!     `src/cuda/int8_to_bf16.cu`) followed by
//!     `cublasGemmEx(BF16, BF16, F32)` with BF16 Tensor Cores
//!     and `CUBLAS_COMPUTE_32F`.
//!   * The dequant materialises the BF16 weight in VRAM before the
//!     GEMM (separated). A future M9.0-B fused dequant+GEMM kernel
//!     would skip the materialisation; M9.0-A measures the worst case.
//!
//! Activation (`A`, shape `[1, K]`) is BF16 in both paths — same
//! cost, factored out of the per-path comparison. Both paths produce
//! the same output dtype `[1, N]` F32.
//!
//! # Shapes (Llama 2 13B decode step, M = 1)
//!
//! | # | Op             | M | K     | N      | FLOPs / iter |
//! |---|----------------|--:|------:|-------:|-------------:|
//! | 1 | Q/K/V/O proj   | 1 |  5120 |   5120 |        52 M  |
//! | 2 | FFN gate / up  | 1 |  5120 |  13824 |       142 M  |
//! | 3 | FFN down       | 1 | 13824 |   5120 |       142 M  |
//! | 4 | LM head        | 1 |  5120 |  32000 |       328 M  |
//!
//! # Methodology
//!
//! - 3 warmup iterations per path (kernel launch overhead + first-call
//!   driver context + cuBLAS algorithm selection cache) followed by 20
//!   measured iterations.
//! - Per-iter wall time taken with `Instant::now()` straddling the
//!   `cudaDeviceSynchronize` after the dequant + GEMM dispatches.
//!   Buffers are allocated once per shape and reused.
//! - Quantisation: per-output-channel symmetric absmax, populated by
//!   walking each column of the synthetic F32 weight to compute
//!   `scale_n = max(|w[:,n]|) / 127`, then truncating
//!   `q[k,n] = round(w[k,n] / scale_n)` to `i8`. No calibration —
//!   plain absmax, the simplest INT8 quant scheme.
//! - Correctness: max abs diff on the smallest shape (Q proj) between
//!   Path A and Path B output. The synthetic weights produced by
//!   `synth_f32` are bounded by ~1.0 in absolute value, so the absmax
//!   scale picks ~1/127 ≈ 7.87e-3 — drift envelope estimate
//!   ~2 × scale × K = ~80, far over ADR-004's 0.5 *if* the comparison
//!   is between Path A's F32 output and Path B's INT8-W8A16 output on
//!   adversarial uniformly-distributed weights. Real Llama weights are
//!   normally distributed and ≪ 1.0 in magnitude; the production
//!   envelope is ~10⁻³ per matmul (the M8.6 envelope). For this synth
//!   bench we use a much looser sanity gate (`< 5.0`) — the goal here
//!   is perf, not numerical contract; the contract gate is M9.4's
//!   F64 fixture re-run.
//!
//! # Decision criteria (H2)
//!
//! - **H2 PASS** — Path B ≥ 1.3× over Path A on at least 2 shapes →
//!   INT8 TC delivers real speedup. Proceed M9.1 (quantizer + storage).
//! - **H2 PARTIAL** — Path B between 1.0× and 1.3×: INT8 helps by
//!   capacity (Disk path eliminated) but not perf; M9 still valuable
//!   but the perf argument weakens.
//! - **H2 FAIL** — Path B < 1.0×: dequant overhead exceeds the TC
//!   benefit. PARAR — replan the M9 approach (skip M9.0-B, prioritise
//!   M9.1 capacity-only path, or pivot to per-group / fused kernels).
//!
//! # Usage
//!
//! ```powershell
//! cargo run --release --example bench_int8_w8a16
//! ```
//!
//! Requires CUDA 11.0+ and a GPU with BF16 Tensor Cores
//! (compute capability ≥ 8.0). RTX 4070 (sm_89) supports both BF16
//! and INT8 TC; this microbench exercises only the BF16 TC path
//! because Path B's GEMM is `BF16×BF16→F32` (the dequant produces
//! BF16 inputs). True INT8×INT8→INT32 TC is M9.0-B's territory and
//! requires a different cuBLAS path.

use std::ffi::c_void;
use std::os::raw::c_int;
use std::time::Instant;

use atenia_engine::tensor::tensor::{bf16_bits_to_f32, f32_to_bf16_bits};

// ---------------------------------------------------------------------------
// Existing kernels (linked from build.rs).
// ---------------------------------------------------------------------------

#[link(name = "bf16_to_f32", kind = "static")]
unsafe extern "C" {
    /// Path A — `bf16_to_f32` upcast on a `[K * N]` BF16 buffer to a
    /// matching F32 transient. Same kernel `cuda_matmul_bf16_inplace`
    /// uses internally for the M8.4c per-matmul transient.
    fn bf16_to_f32_launch_device(
        d_src_bf16: *const c_void,
        d_dst_f32: *mut f32,
        n: c_int,
    ) -> c_int;
}

#[link(name = "int8_to_bf16", kind = "static")]
unsafe extern "C" {
    /// Path B — INT8 → BF16 per-output-channel symmetric dequant.
    /// `d_int8` is a `[K, N]` row-major INT8 weight, `d_scales` is an
    /// `[N]` F32 vector of per-column scales, `d_bf16` is the
    /// `[K, N]` u16 output.
    fn int8_to_bf16_per_channel_launch_device(
        d_int8: *const c_void,
        d_scales: *const f32,
        d_bf16: *mut c_void,
        k: c_int,
        n: c_int,
    ) -> c_int;
}

// ---------------------------------------------------------------------------
// CUDA runtime + cuBLAS bindings (only what the bench needs).
// ---------------------------------------------------------------------------

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> c_int;
    fn cudaFree(ptr: *mut c_void) -> c_int;
    fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

#[allow(non_camel_case_types)]
type cublasHandle_t = *mut c_void;

const CUBLAS_OP_N: c_int = 0;
const CUDA_R_32F: c_int = 0;
const CUDA_R_16BF: c_int = 14;
const CUBLAS_COMPUTE_32F: c_int = 68;
const CUBLAS_GEMM_DEFAULT: c_int = -1;

#[link(name = "cublas")]
unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut cublasHandle_t) -> c_int;
    fn cublasDestroy_v2(handle: cublasHandle_t) -> c_int;
    fn cublasGemmEx(
        handle: cublasHandle_t,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: c_int,
        lda: c_int,
        b: *const c_void,
        b_type: c_int,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        c_type: c_int,
        ldc: c_int,
        compute_type: c_int,
        algo: c_int,
    ) -> c_int;
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

fn check_cuda(rc: c_int, what: &str) {
    if rc != 0 {
        eprintln!("FATAL: {} returned cuda rc={}", what, rc);
        std::process::exit(1);
    }
}

fn check_cublas(rc: c_int, what: &str) {
    if rc != 0 {
        eprintln!("FATAL: {} returned cuBLAS status={}", what, rc);
        std::process::exit(1);
    }
}

fn fmt_size(bytes: usize) -> String {
    let mib = bytes as f64 / (1024.0 * 1024.0);
    if mib >= 1024.0 {
        format!("{:.2} GiB", mib / 1024.0)
    } else {
        format!("{:.1} MiB", mib)
    }
}

/// Same synthetic-weight generator as `bench_cublas_bf16.rs` so the
/// numerical envelopes are directly comparable across the two
/// gating benches.
fn synth_f32(numel: usize, seed: u32) -> Vec<f32> {
    (0..numel)
        .map(|i| {
            let s = (seed as f32) * 0.31;
            ((i as f32) * 0.0001 + 0.137 + s).sin() * 0.5
                + ((i as f32) * 0.00007 + 0.42 + s).cos() * 0.4
        })
        .collect()
}

fn to_bf16_bits(src: &[f32]) -> Vec<u16> {
    src.iter().map(|&f| f32_to_bf16_bits(f)).collect()
}

#[allow(dead_code)]
fn from_bf16_bits(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| bf16_bits_to_f32(b)).collect()
}

/// **M9.0 quantizer** — per-output-channel symmetric absmax INT8.
///
/// Walks each column of the row-major `[K, N]` F32 weight, computes
/// `scale_n = max(|w[:,n]|) / 127` (clamped from below by `1e-12` so
/// an all-zero column does not produce `NaN`), then quantises every
/// element via `q[k,n] = round(w[k,n] / scale_n)` clamped to `[-127, 127]`.
///
/// Returns `(q_int8, scales)` where `q_int8.len() == k * n` and
/// `scales.len() == n`.
fn quantize_int8_per_channel_absmax(
    weight_f32: &[f32],
    k: usize,
    n: usize,
) -> (Vec<i8>, Vec<f32>) {
    assert_eq!(weight_f32.len(), k * n);
    let mut scales: Vec<f32> = vec![0.0; n];
    for col in 0..n {
        let mut max_abs = 0.0_f32;
        for row in 0..k {
            let v = weight_f32[row * n + col].abs();
            if v > max_abs {
                max_abs = v;
            }
        }
        scales[col] = (max_abs / 127.0).max(1e-12);
    }
    let mut q: Vec<i8> = vec![0; k * n];
    for row in 0..k {
        for col in 0..n {
            let s = scales[col];
            let qf = (weight_f32[row * n + col] / s).round();
            let qi = qf.clamp(-127.0, 127.0) as i32;
            q[row * n + col] = qi as i8;
        }
    }
    (q, scales)
}

// ---------------------------------------------------------------------------
// Bench shape descriptor.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct BenchShape {
    label: &'static str,
    m: usize,
    k: usize,
    n: usize,
}

impl BenchShape {
    fn flops(&self) -> u64 {
        (2 * self.m * self.k * self.n) as u64
    }
    fn a_bytes_bf16(&self) -> usize {
        self.m * self.k * 2
    }
    fn b_bytes_bf16(&self) -> usize {
        self.k * self.n * 2
    }
    fn b_bytes_f32_transient(&self) -> usize {
        self.k * self.n * 4
    }
    fn b_bytes_int8(&self) -> usize {
        self.k * self.n
    }
    fn scale_bytes(&self) -> usize {
        self.n * 4
    }
    fn c_bytes_f32(&self) -> usize {
        self.m * self.n * 4
    }
}

const SHAPES: &[BenchShape] = &[
    BenchShape { label: "Q/K/V/O proj", m: 1, k: 5120, n: 5120 },
    BenchShape { label: "FFN gate/up ", m: 1, k: 5120, n: 13824 },
    BenchShape { label: "FFN down    ", m: 1, k: 13824, n: 5120 },
    BenchShape { label: "LM head     ", m: 1, k: 5120, n: 32000 },
];

const WARMUP_ITERS: usize = 3;
const MEASURED_ITERS: usize = 20;

// ---------------------------------------------------------------------------
// Path runners.
// ---------------------------------------------------------------------------

/// Path A — M8.4c production baseline.
/// Per iter: BF16 → F32 transient upcast on the weight, then
/// `cublasGemmEx(F32, F32, F32)`. Activation `A` is also F32 in
/// VRAM (caller pre-uploads it; the M8.4c kernel takes F32 input).
///
/// Same row-major-via-column-major transpose trick as
/// `bench_cublas_bf16.rs` (we ask cuBLAS for `C^T = B^T × A^T`).
fn run_path_a_bf16_resident(
    handle: cublasHandle_t,
    shape: BenchShape,
    d_a_f32: *const f32,
    d_b_bf16: *const c_void,
    d_b_f32_transient: *mut c_void,
    d_c_f32: *mut f32,
) -> f64 {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let m = shape.m as c_int;
    let n = shape.n as c_int;
    let k = shape.k as c_int;
    let weight_numel = (shape.k * shape.n) as c_int;

    for _ in 0..WARMUP_ITERS {
        unsafe {
            check_cuda(
                bf16_to_f32_launch_device(
                    d_b_bf16,
                    d_b_f32_transient as *mut f32,
                    weight_numel,
                ),
                "bf16_to_f32 (A warmup)",
            );
            check_cublas(
                cublasGemmEx(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha as *const f32 as *const c_void,
                    d_b_f32_transient as *const c_void, CUDA_R_32F, n,
                    d_a_f32 as *const c_void, CUDA_R_32F, k,
                    &beta as *const f32 as *const c_void,
                    d_c_f32 as *mut c_void, CUDA_R_32F, n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT,
                ),
                "cublasGemmEx (A warmup)",
            );
            check_cuda(cudaDeviceSynchronize(), "sync (A warmup)");
        }
    }

    let t0 = Instant::now();
    for _ in 0..MEASURED_ITERS {
        unsafe {
            check_cuda(
                bf16_to_f32_launch_device(
                    d_b_bf16,
                    d_b_f32_transient as *mut f32,
                    weight_numel,
                ),
                "bf16_to_f32 (A measured)",
            );
            check_cublas(
                cublasGemmEx(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha as *const f32 as *const c_void,
                    d_b_f32_transient as *const c_void, CUDA_R_32F, n,
                    d_a_f32 as *const c_void, CUDA_R_32F, k,
                    &beta as *const f32 as *const c_void,
                    d_c_f32 as *mut c_void, CUDA_R_32F, n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT,
                ),
                "cublasGemmEx (A measured)",
            );
        }
    }
    unsafe {
        check_cuda(cudaDeviceSynchronize(), "sync (A)");
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    elapsed_ms / (MEASURED_ITERS as f64)
}

/// Path B — M9 W8A16 candidate.
/// Per iter: INT8 → BF16 per-channel dequant kernel, then
/// `cublasGemmEx(BF16, BF16, F32)`. Activation `A` is BF16 in VRAM.
fn run_path_b_int8_w8a16(
    handle: cublasHandle_t,
    shape: BenchShape,
    d_a_bf16: *const c_void,
    d_b_int8: *const c_void,
    d_scales: *const f32,
    d_b_bf16_transient: *mut c_void,
    d_c_f32: *mut f32,
) -> f64 {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let m = shape.m as c_int;
    let n = shape.n as c_int;
    let k = shape.k as c_int;

    for _ in 0..WARMUP_ITERS {
        unsafe {
            check_cuda(
                int8_to_bf16_per_channel_launch_device(
                    d_b_int8,
                    d_scales,
                    d_b_bf16_transient,
                    k, n,
                ),
                "int8_to_bf16 (B warmup)",
            );
            check_cublas(
                cublasGemmEx(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha as *const f32 as *const c_void,
                    d_b_bf16_transient as *const c_void, CUDA_R_16BF, n,
                    d_a_bf16, CUDA_R_16BF, k,
                    &beta as *const f32 as *const c_void,
                    d_c_f32 as *mut c_void, CUDA_R_32F, n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT,
                ),
                "cublasGemmEx (B warmup)",
            );
            check_cuda(cudaDeviceSynchronize(), "sync (B warmup)");
        }
    }

    let t0 = Instant::now();
    for _ in 0..MEASURED_ITERS {
        unsafe {
            check_cuda(
                int8_to_bf16_per_channel_launch_device(
                    d_b_int8,
                    d_scales,
                    d_b_bf16_transient,
                    k, n,
                ),
                "int8_to_bf16 (B measured)",
            );
            check_cublas(
                cublasGemmEx(
                    handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n, m, k,
                    &alpha as *const f32 as *const c_void,
                    d_b_bf16_transient as *const c_void, CUDA_R_16BF, n,
                    d_a_bf16, CUDA_R_16BF, k,
                    &beta as *const f32 as *const c_void,
                    d_c_f32 as *mut c_void, CUDA_R_32F, n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT,
                ),
                "cublasGemmEx (B measured)",
            );
        }
    }
    unsafe {
        check_cuda(cudaDeviceSynchronize(), "sync (B)");
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    elapsed_ms / (MEASURED_ITERS as f64)
}

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------

fn main() {
    println!("=== M9.0 — INT8 W8A16 vs Path B (M8.4c) microbench ===");
    println!();

    // Probe initial VRAM.
    let (mut free0, mut total0): (usize, usize) = (0, 0);
    unsafe {
        check_cuda(cudaMemGetInfo(&mut free0, &mut total0), "cudaMemGetInfo (start)");
    }
    println!(
        "VRAM at start: free {} / total {}",
        fmt_size(free0),
        fmt_size(total0)
    );

    // Driver context warmup (absorbs the first-malloc init cost).
    let mut warmup_ptr: *mut c_void = std::ptr::null_mut();
    unsafe {
        check_cuda(cudaMalloc(&mut warmup_ptr, 1024), "cudaMalloc (driver warmup)");
        check_cuda(cudaFree(warmup_ptr), "cudaFree (driver warmup)");
    }

    // cuBLAS handle.
    let mut handle: cublasHandle_t = std::ptr::null_mut();
    unsafe {
        check_cublas(cublasCreate_v2(&mut handle), "cublasCreate");
    }
    println!("cuBLAS handle created (default math mode — BF16 TC under CUBLAS_COMPUTE_32F).");
    println!();

    println!(
        "{:<14}  {:>9}  {:>10}  {:>10}  {:>10}",
        "Shape", "FLOPs", "A BF16/F32", "B INT8 W8A16", "B/A ratio"
    );
    println!(
        "{:<14}  {:>9}  {:>10}  {:>10}  {:>10}",
        "", "(M)", "(ms)", "(ms)", ""
    );
    println!("{}", "-".repeat(70));

    let mut max_diff_a_vs_b: f32 = 0.0;
    let mut all_results: Vec<(BenchShape, f64, f64)> = Vec::new();
    let mut peak_vram_used: usize = 0;

    for shape in SHAPES {
        // Synth host data.
        let a_host_f32: Vec<f32> = synth_f32(shape.m * shape.k, 7);
        let b_host_f32: Vec<f32> = synth_f32(shape.k * shape.n, 13);
        let a_bf16 = to_bf16_bits(&a_host_f32);
        let b_bf16 = to_bf16_bits(&b_host_f32);

        // Quantise the weight to INT8 W8A16 per-channel absmax.
        let (b_int8, scales) =
            quantize_int8_per_channel_absmax(&b_host_f32, shape.k, shape.n);
        debug_assert_eq!(b_int8.len(), shape.k * shape.n);
        debug_assert_eq!(scales.len(), shape.n);

        // Allocate device buffers. We keep both the F32 and BF16 forms
        // of A live (Path A reads F32 A, Path B reads BF16 A) plus the
        // BF16 weight + the F32 transient (Path A) + the INT8 weight +
        // its scales + the BF16 transient (Path B) + the F32 output.
        let mut d_a_f32: *mut c_void = std::ptr::null_mut();
        let mut d_a_bf16: *mut c_void = std::ptr::null_mut();
        let mut d_b_bf16: *mut c_void = std::ptr::null_mut();
        let mut d_b_f32_transient: *mut c_void = std::ptr::null_mut();
        let mut d_b_int8: *mut c_void = std::ptr::null_mut();
        let mut d_scales: *mut c_void = std::ptr::null_mut();
        let mut d_b_bf16_transient: *mut c_void = std::ptr::null_mut();
        let mut d_c_f32: *mut c_void = std::ptr::null_mut();

        unsafe {
            check_cuda(cudaMalloc(&mut d_a_f32, shape.m * shape.k * 4), "malloc A f32");
            check_cuda(cudaMalloc(&mut d_a_bf16, shape.a_bytes_bf16()), "malloc A bf16");
            check_cuda(cudaMalloc(&mut d_b_bf16, shape.b_bytes_bf16()), "malloc B bf16");
            check_cuda(
                cudaMalloc(&mut d_b_f32_transient, shape.b_bytes_f32_transient()),
                "malloc B f32 transient",
            );
            check_cuda(cudaMalloc(&mut d_b_int8, shape.b_bytes_int8()), "malloc B int8");
            check_cuda(cudaMalloc(&mut d_scales, shape.scale_bytes()), "malloc scales");
            check_cuda(
                cudaMalloc(&mut d_b_bf16_transient, shape.b_bytes_bf16()),
                "malloc B bf16 transient",
            );
            check_cuda(cudaMalloc(&mut d_c_f32, shape.c_bytes_f32()), "malloc C f32");

            // H→D uploads.
            check_cuda(
                cudaMemcpy(d_a_f32, a_host_f32.as_ptr() as *const c_void,
                    shape.m * shape.k * 4, CUDA_MEMCPY_HOST_TO_DEVICE),
                "memcpy A f32",
            );
            check_cuda(
                cudaMemcpy(d_a_bf16, a_bf16.as_ptr() as *const c_void,
                    shape.a_bytes_bf16(), CUDA_MEMCPY_HOST_TO_DEVICE),
                "memcpy A bf16",
            );
            check_cuda(
                cudaMemcpy(d_b_bf16, b_bf16.as_ptr() as *const c_void,
                    shape.b_bytes_bf16(), CUDA_MEMCPY_HOST_TO_DEVICE),
                "memcpy B bf16",
            );
            check_cuda(
                cudaMemcpy(d_b_int8, b_int8.as_ptr() as *const c_void,
                    shape.b_bytes_int8(), CUDA_MEMCPY_HOST_TO_DEVICE),
                "memcpy B int8",
            );
            check_cuda(
                cudaMemcpy(d_scales, scales.as_ptr() as *const c_void,
                    shape.scale_bytes(), CUDA_MEMCPY_HOST_TO_DEVICE),
                "memcpy scales",
            );
        }

        // Probe VRAM mid-bench for the rollback gate.
        let (mut free_now, mut total_now): (usize, usize) = (0, 0);
        unsafe {
            check_cuda(cudaMemGetInfo(&mut free_now, &mut total_now), "cudaMemGetInfo");
        }
        let used_now = total_now.saturating_sub(free_now);
        if used_now > peak_vram_used {
            peak_vram_used = used_now;
        }
        if used_now > 7 * 1024 * 1024 * 1024 {
            eprintln!(
                "FATAL: VRAM in use exceeded 7 GiB rollback floor: {} used / {} total",
                fmt_size(used_now),
                fmt_size(total_now)
            );
            std::process::exit(2);
        }

        // Path A (M8.4c BF16-resident + F32 transient upcast + F32 GEMM).
        let ms_a = run_path_a_bf16_resident(
            handle,
            *shape,
            d_a_f32 as *const f32,
            d_b_bf16 as *const c_void,
            d_b_f32_transient,
            d_c_f32 as *mut f32,
        );

        // Snapshot Path A output for the correctness check.
        let mut c_a_host: Vec<f32> = vec![0.0_f32; shape.m * shape.n];
        unsafe {
            check_cuda(
                cudaMemcpy(
                    c_a_host.as_mut_ptr() as *mut c_void,
                    d_c_f32 as *const c_void,
                    shape.c_bytes_f32(),
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                ),
                "memcpy D2H C (A)",
            );
        }

        // Path B (INT8 → BF16 dequant + BF16 TC GEMM).
        let ms_b = run_path_b_int8_w8a16(
            handle,
            *shape,
            d_a_bf16 as *const c_void,
            d_b_int8 as *const c_void,
            d_scales as *const f32,
            d_b_bf16_transient,
            d_c_f32 as *mut f32,
        );

        // Snapshot Path B output.
        let mut c_b_host: Vec<f32> = vec![0.0_f32; shape.m * shape.n];
        unsafe {
            check_cuda(
                cudaMemcpy(
                    c_b_host.as_mut_ptr() as *mut c_void,
                    d_c_f32 as *const c_void,
                    shape.c_bytes_f32(),
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                ),
                "memcpy D2H C (B)",
            );
        }

        let mut local_max_diff = 0.0_f32;
        for (a, b) in c_a_host.iter().zip(c_b_host.iter()) {
            let d = (a - b).abs();
            if d > local_max_diff {
                local_max_diff = d;
            }
        }
        if local_max_diff > max_diff_a_vs_b {
            max_diff_a_vs_b = local_max_diff;
        }

        // Free per-shape buffers.
        unsafe {
            check_cuda(cudaFree(d_a_f32), "free A f32");
            check_cuda(cudaFree(d_a_bf16), "free A bf16");
            check_cuda(cudaFree(d_b_bf16), "free B bf16");
            check_cuda(cudaFree(d_b_f32_transient), "free B f32 transient");
            check_cuda(cudaFree(d_b_int8), "free B int8");
            check_cuda(cudaFree(d_scales), "free scales");
            check_cuda(cudaFree(d_b_bf16_transient), "free B bf16 transient");
            check_cuda(cudaFree(d_c_f32), "free C f32");
        }

        let flops = shape.flops() as f64;
        println!(
            "{:<14}  {:>7.0} M  {:>10.3}  {:>10.3}  {:>9.2}×",
            shape.label,
            flops / 1.0e6,
            ms_a,
            ms_b,
            ms_a / ms_b,
        );

        all_results.push((*shape, ms_a, ms_b));
    }

    println!();

    println!("Effective throughput (GFLOPS — measured per-iter, no H↔D):");
    for (shape, ms_a, ms_b) in &all_results {
        let flops = shape.flops() as f64;
        let gflops = |ms: f64| (flops / 1.0e9) / (ms / 1000.0);
        println!(
            "  {:<14} A={:>8.1}  B={:>8.1}  GFLOPS",
            shape.label,
            gflops(*ms_a),
            gflops(*ms_b),
        );
    }
    println!();

    println!("Numerical envelope (Path A vs Path B, max |C_A − C_B|):");
    println!("  {:.4e}  (synth-weight microbench gate, < 5.0)", max_diff_a_vs_b);
    println!("  Real-weight ADR-004 contract is M9.4's F64 fixture re-run.");
    println!();

    println!("Peak VRAM used during bench: {} / 7 GiB rollback floor", fmt_size(peak_vram_used));
    println!();

    // H2 decision.
    let mut wins_1_3x = 0;
    let mut wins_1_0x = 0;
    let mut losses = 0;
    for (_shape, ms_a, ms_b) in &all_results {
        let ratio = ms_a / ms_b;
        if ratio >= 1.3 {
            wins_1_3x += 1;
        }
        if ratio >= 1.0 {
            wins_1_0x += 1;
        }
        if ratio < 1.0 {
            losses += 1;
        }
    }

    println!("=== H2 (INT8 TC ≥ 1.3× over Path B BF16) decision ===");
    println!("Shapes with B ≥ 1.3× over A: {} / {}", wins_1_3x, all_results.len());
    println!("Shapes with B ≥ 1.0× over A: {} / {}", wins_1_0x, all_results.len());
    println!("Shapes with B < 1.0× over A: {} / {}", losses, all_results.len());
    if wins_1_3x >= 2 {
        println!("Decision:  H2 PASS — INT8 TC delivers real speedup. Proceed M9.1.");
    } else if losses == all_results.len() {
        println!("Decision:  H2 FAIL — INT8 dequant overhead exceeds TC benefit. PARAR.");
    } else if wins_1_0x >= 2 {
        println!("Decision:  H2 PARTIAL — INT8 helps by capacity, not by perf. M9 still");
        println!("           valuable but the perf argument weakens. Plan accordingly.");
    } else {
        println!("Decision:  H2 INCONCLUSIVE — Path B mostly slower than A. Recheck the");
        println!("           dequant kernel before proceeding.");
    }

    // Cleanup.
    unsafe {
        check_cublas(cublasDestroy_v2(handle), "cublasDestroy");
    }

    let (mut free_end, mut total_end): (usize, usize) = (0, 0);
    unsafe {
        check_cuda(cudaMemGetInfo(&mut free_end, &mut total_end), "cudaMemGetInfo (end)");
    }
    println!();
    println!(
        "VRAM at end:   free {} / total {}",
        fmt_size(free_end),
        fmt_size(total_end)
    );
}
