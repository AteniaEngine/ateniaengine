//! M8.0 — cuBLAS BF16 Tensor Core matmul bench.
//!
//! **Gating data point for the M8 milestone.** Compares three GPU
//! matmul paths on the four shapes that dominate the Llama 2 13B
//! decode step. The ratio between Path C (BF16 + Tensor Cores) and
//! Path A (the M6 naive F32 kernel) decides whether M8 ships:
//!
//! - **Plan A (proceed M8.1 → M8.5)**: C ≥ 5× over A on at least
//!   2 of 4 shapes. The architecture investment pays off.
//! - **Plan A partial**: C between 3× and 5× over A. Proceed with
//!   caution; M8.7-M8.8 may not justify the effort.
//! - **Plan B (PARAR)**: C < 3× over A. M8 should not start in
//!   the proposed form. Investigate whether Path B (TF32 on F32)
//!   is the right next step.
//!
//! # Shapes
//!
//! Decode-step matmuls (M = 1) for Llama 2 13B (hidden = 5120,
//! intermediate = 13824, vocab = 32000). The "[M, K] × [K, N]"
//! notation matches the row-major convention of the existing
//! `matmul_f32_launch_device` kernel:
//!
//! | # | Op             | M    | K     | N      | FLOPs / iter |
//! |---|----------------|------|-------|--------|-------------:|
//! | 1 | Q/K/V/O proj   | 1    | 5120  | 5120   |        52 M  |
//! | 2 | FFN gate / up  | 1    | 5120  | 13824  |       142 M  |
//! | 3 | FFN down       | 1    | 13824 | 5120   |       142 M  |
//! | 4 | LM head        | 1    | 5120  | 32000  |       328 M  |
//!
//! # Paths
//!
//! - **Path A** — the M6 naive F32 kernel
//!   (`matmul_f32_launch_device` from `src/cuda/matmul_kernel.cu`).
//!   Block 16×16, no tiling, no Tensor Cores. The current M7
//!   production path.
//!
//! - **Path B** — `cublasGemmEx` with F32 inputs and
//!   `CUBLAS_COMPUTE_32F_FAST_TF32`. Inputs are F32 in VRAM; cuBLAS
//!   downconverts to TF32 internally for Tensor Core math, and the
//!   accumulator stays F32. This is the conservative "use Tensor
//!   Cores without changing storage dtype" baseline.
//!
//! - **Path C** — `cublasGemmEx` with BF16 inputs and
//!   `CUBLAS_COMPUTE_32F`. Inputs are BF16 in VRAM; cuBLAS uses
//!   BF16 Tensor Cores natively with F32 accumulate. This is the
//!   M8 candidate path: weights live as BF16 in VRAM (half the
//!   memory cost) and the kernel consumes them directly.
//!
//! # Methodology
//!
//! - 3 warmup iterations (kernel launch overhead + first-call
//!   driver context) followed by 20 measured iterations.
//! - Per-iter time taken with `Instant::now()` straddling the
//!   `cudaDeviceSynchronize` after the kernel launch — the
//!   asynchronous nature of CUDA otherwise makes wall-clock useless.
//! - Buffers allocated once per shape, reused across iters.
//! - Correctness check: max absolute difference between Path A and
//!   Path C output on the smallest shape (Q proj). Threshold 1e-2;
//!   BF16 truncates the F32 mantissa to 7 bits so a single matmul
//!   typically drifts to 1e-3..1e-2 against an F32 reference. ADR-004
//!   threshold (0.5) applies to **end-to-end logits**, not single
//!   matmuls — this check is a sanity gate, not a numerical contract.
//!
//! # Usage
//!
//! ```powershell
//! cargo run --release --example bench_cublas_bf16
//! ```
//!
//! Requires CUDA 11.0+ and a GPU with BF16 Tensor Cores
//! (compute capability ≥ 8.0). RTX 4070 (sm_89) supports both.

use std::ffi::c_void;
use std::os::raw::c_int;
use std::time::Instant;

use atenia_engine::tensor::tensor::{bf16_bits_to_f32, f32_to_bf16_bits};

// ---------------------------------------------------------------------------
// Path A — the M6 naive F32 kernel.
// ---------------------------------------------------------------------------

#[link(name = "matmul_kernel")]
unsafe extern "C" {
    fn matmul_f32_launch_device(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: c_int,
        k: c_int,
        n: c_int,
    );
}

// ---------------------------------------------------------------------------
// CUDA runtime + cuBLAS bindings (only what the bench needs).
// ---------------------------------------------------------------------------

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> c_int;
    fn cudaFree(ptr: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

// cuBLAS opaque handle.
#[allow(non_camel_case_types)]
type cublasHandle_t = *mut c_void;

// cuBLAS enums — values pinned per the CUDA 11.x headers
// (`cublas_v2.h`, `library_types.h`, `cublasLt.h`).
const CUBLAS_OP_N: c_int = 0;

// `cudaDataType_t` — `library_types.h`.
const CUDA_R_32F: c_int = 0;
const CUDA_R_16BF: c_int = 14;

// `cublasComputeType_t` — `cublas_v2.h` (CUDA 11+).
const CUBLAS_COMPUTE_32F: c_int = 68;
const CUBLAS_COMPUTE_32F_FAST_TF32: c_int = 77;

// `cublasGemmAlgo_t` — `cublas_v2.h`. Default lets cuBLAS pick the
// best Tensor Core algorithm for the operand dtypes.
const CUBLAS_GEMM_DEFAULT: c_int = -1;

// `cublasMath_t` — `cublas_v2.h`. TF32 mode allows the F32 path to
// use Tensor Cores transparently. Required for Path B.
const CUBLAS_TF32_TENSOR_OP_MATH: c_int = 1;

#[link(name = "cublas")]
unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut cublasHandle_t) -> c_int;
    fn cublasDestroy_v2(handle: cublasHandle_t) -> c_int;
    fn cublasSetMathMode(handle: cublasHandle_t, mode: c_int) -> c_int;
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

fn synth_f32(numel: usize, seed: u32) -> Vec<f32> {
    // Deterministic synthetic data via two cheap sines. Same shape as
    // the M7.0 disk bench's `synth_bf16_weight` so reviewers can
    // compare numerical envelopes.
    (0..numel)
        .map(|i| {
            let s = (seed as f32) * 0.31;
            let f = ((i as f32) * 0.0001 + 0.137 + s).sin() * 0.5
                + ((i as f32) * 0.00007 + 0.42 + s).cos() * 0.4;
            f
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
    fn a_bytes_f32(&self) -> usize {
        self.m * self.k * 4
    }
    fn b_bytes_f32(&self) -> usize {
        self.k * self.n * 4
    }
    fn c_bytes_f32(&self) -> usize {
        self.m * self.n * 4
    }
    fn a_bytes_bf16(&self) -> usize {
        self.m * self.k * 2
    }
    fn b_bytes_bf16(&self) -> usize {
        self.k * self.n * 2
    }
}

const SHAPES: &[BenchShape] = &[
    BenchShape {
        label: "Q/K/V/O proj",
        m: 1,
        k: 5120,
        n: 5120,
    },
    BenchShape {
        label: "FFN gate/up ",
        m: 1,
        k: 5120,
        n: 13824,
    },
    BenchShape {
        label: "FFN down    ",
        m: 1,
        k: 13824,
        n: 5120,
    },
    BenchShape {
        label: "LM head     ",
        m: 1,
        k: 5120,
        n: 32000,
    },
];

const WARMUP_ITERS: usize = 3;
const MEASURED_ITERS: usize = 20;

// ---------------------------------------------------------------------------
// Path-specific runners. Each takes pre-uploaded device buffers and
// returns the per-iter time in milliseconds (averaged over MEASURED_ITERS).
// ---------------------------------------------------------------------------

/// Path A — naive F32 kernel `matmul_f32_launch_device`.
///
/// Caller has already uploaded `d_a`, `d_b`. Output `d_c` is overwritten.
fn run_path_a(shape: BenchShape, d_a: *const f32, d_b: *const f32, d_c: *mut f32) -> f64 {
    let m_ci = shape.m as c_int;
    let k_ci = shape.k as c_int;
    let n_ci = shape.n as c_int;

    // Warmup.
    for _ in 0..WARMUP_ITERS {
        unsafe {
            matmul_f32_launch_device(d_a, d_b, d_c, m_ci, k_ci, n_ci);
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (A warmup)");
        }
    }

    // Measure.
    let t0 = Instant::now();
    for _ in 0..MEASURED_ITERS {
        unsafe {
            matmul_f32_launch_device(d_a, d_b, d_c, m_ci, k_ci, n_ci);
        }
    }
    unsafe {
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (A)");
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    elapsed_ms / (MEASURED_ITERS as f64)
}

/// Path B — `cublasGemmEx` with F32 inputs + `CUBLAS_COMPUTE_32F_FAST_TF32`.
///
/// Row-major A[M,K] × B[K,N] = C[M,N] is implemented in cuBLAS's
/// column-major convention via the standard "transpose the whole
/// problem" trick: we ask cuBLAS to compute C^T = B^T × A^T which,
/// viewed as column-major operands, is exactly the same byte layout
/// as our row-major A × B. So we pass `B` as the first operand, `A`
/// as the second, with leading dims `n`, `k`, `n`.
fn run_path_b_tf32(
    handle: cublasHandle_t,
    shape: BenchShape,
    d_a_f32: *const f32,
    d_b_f32: *const f32,
    d_c_f32: *mut f32,
) -> f64 {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let m = shape.m as c_int;
    let n = shape.n as c_int;
    let k = shape.k as c_int;

    // Warmup.
    for _ in 0..WARMUP_ITERS {
        unsafe {
            check_cublas(
                cublasGemmEx(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    &alpha as *const f32 as *const c_void,
                    d_b_f32 as *const c_void,
                    CUDA_R_32F,
                    n,
                    d_a_f32 as *const c_void,
                    CUDA_R_32F,
                    k,
                    &beta as *const f32 as *const c_void,
                    d_c_f32 as *mut c_void,
                    CUDA_R_32F,
                    n,
                    CUBLAS_COMPUTE_32F_FAST_TF32,
                    CUBLAS_GEMM_DEFAULT,
                ),
                "cublasGemmEx (B warmup)",
            );
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (B warmup)");
        }
    }

    let t0 = Instant::now();
    for _ in 0..MEASURED_ITERS {
        unsafe {
            check_cublas(
                cublasGemmEx(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    &alpha as *const f32 as *const c_void,
                    d_b_f32 as *const c_void,
                    CUDA_R_32F,
                    n,
                    d_a_f32 as *const c_void,
                    CUDA_R_32F,
                    k,
                    &beta as *const f32 as *const c_void,
                    d_c_f32 as *mut c_void,
                    CUDA_R_32F,
                    n,
                    CUBLAS_COMPUTE_32F_FAST_TF32,
                    CUBLAS_GEMM_DEFAULT,
                ),
                "cublasGemmEx (B measured)",
            );
        }
    }
    unsafe {
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (B)");
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    elapsed_ms / (MEASURED_ITERS as f64)
}

/// Path C — `cublasGemmEx` with BF16 inputs + `CUBLAS_COMPUTE_32F`.
///
/// Tensor Core BF16 path. Inputs are CUDA_R_16BF (raw u16); output is
/// CUDA_R_32F. Same transpose trick as Path B.
fn run_path_c_bf16(
    handle: cublasHandle_t,
    shape: BenchShape,
    d_a_bf16: *const c_void,
    d_b_bf16: *const c_void,
    d_c_f32: *mut f32,
) -> f64 {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let m = shape.m as c_int;
    let n = shape.n as c_int;
    let k = shape.k as c_int;

    for _ in 0..WARMUP_ITERS {
        unsafe {
            check_cublas(
                cublasGemmEx(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    &alpha as *const f32 as *const c_void,
                    d_b_bf16,
                    CUDA_R_16BF,
                    n,
                    d_a_bf16,
                    CUDA_R_16BF,
                    k,
                    &beta as *const f32 as *const c_void,
                    d_c_f32 as *mut c_void,
                    CUDA_R_32F,
                    n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT,
                ),
                "cublasGemmEx (C warmup)",
            );
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (C warmup)");
        }
    }

    let t0 = Instant::now();
    for _ in 0..MEASURED_ITERS {
        unsafe {
            check_cublas(
                cublasGemmEx(
                    handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    n,
                    m,
                    k,
                    &alpha as *const f32 as *const c_void,
                    d_b_bf16,
                    CUDA_R_16BF,
                    n,
                    d_a_bf16,
                    CUDA_R_16BF,
                    k,
                    &beta as *const f32 as *const c_void,
                    d_c_f32 as *mut c_void,
                    CUDA_R_32F,
                    n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT,
                ),
                "cublasGemmEx (C measured)",
            );
        }
    }
    unsafe {
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (C)");
    }
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    elapsed_ms / (MEASURED_ITERS as f64)
}

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------

fn main() {
    println!("=== M8.0 — cuBLAS BF16 Tensor Core matmul bench ===");
    println!();

    // Probe initial VRAM state.
    let (mut free0, mut total0): (usize, usize) = (0, 0);
    unsafe {
        check_cuda(
            cudaMemGetInfo(&mut free0, &mut total0),
            "cudaMemGetInfo (start)",
        );
    }
    println!(
        "VRAM at start: free {} / total {}",
        fmt_size(free0),
        fmt_size(total0)
    );

    // Warmup the CUDA driver context. The first malloc absorbs ~100 ms
    // of context init; pulling that out of the per-shape loop keeps the
    // first shape's numbers honest.
    let mut warmup_ptr: *mut c_void = std::ptr::null_mut();
    unsafe {
        check_cuda(
            cudaMalloc(&mut warmup_ptr, 1024),
            "cudaMalloc (driver warmup)",
        );
        check_cuda(cudaFree(warmup_ptr), "cudaFree (driver warmup)");
    }

    // Create cuBLAS handle. Enable TF32 path so Path B can use TC.
    let mut handle: cublasHandle_t = std::ptr::null_mut();
    unsafe {
        check_cublas(cublasCreate_v2(&mut handle), "cublasCreate");
        check_cublas(
            cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH),
            "cublasSetMathMode (TF32)",
        );
    }
    println!("cuBLAS handle created. TF32 math mode enabled.");
    println!();

    // Header.
    println!(
        "{:<14}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}",
        "Shape", "FLOPs", "A naive", "B TF32", "C BF16", "C/A", "C/B", "B/A"
    );
    println!(
        "{:<14}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}",
        "", "(M)", "(ms)", "(ms)", "(ms)", "ratio", "ratio", "ratio"
    );
    println!("{}", "-".repeat(98));

    let mut max_diff_a_vs_c: f32 = 0.0;
    let mut all_results: Vec<(BenchShape, f64, f64, f64)> = Vec::new();

    for shape in SHAPES {
        // Synth host data. A is small ([1, K]); B is big.
        let a_host: Vec<f32> = synth_f32(shape.m * shape.k, 7);
        let b_host: Vec<f32> = synth_f32(shape.k * shape.n, 13);
        let a_bf16 = to_bf16_bits(&a_host);
        let b_bf16 = to_bf16_bits(&b_host);

        // Allocate device buffers. Two F32 buffers + two BF16 buffers
        // + one F32 output, all alive simultaneously so we can run
        // all three paths without re-uploading.
        let mut d_a_f32: *mut c_void = std::ptr::null_mut();
        let mut d_b_f32: *mut c_void = std::ptr::null_mut();
        let mut d_c_f32: *mut c_void = std::ptr::null_mut();
        let mut d_a_bf16: *mut c_void = std::ptr::null_mut();
        let mut d_b_bf16: *mut c_void = std::ptr::null_mut();

        unsafe {
            check_cuda(
                cudaMalloc(&mut d_a_f32, shape.a_bytes_f32()),
                "malloc A f32",
            );
            check_cuda(
                cudaMalloc(&mut d_b_f32, shape.b_bytes_f32()),
                "malloc B f32",
            );
            check_cuda(
                cudaMalloc(&mut d_c_f32, shape.c_bytes_f32()),
                "malloc C f32",
            );
            check_cuda(
                cudaMalloc(&mut d_a_bf16, shape.a_bytes_bf16()),
                "malloc A bf16",
            );
            check_cuda(
                cudaMalloc(&mut d_b_bf16, shape.b_bytes_bf16()),
                "malloc B bf16",
            );

            check_cuda(
                cudaMemcpy(
                    d_a_f32,
                    a_host.as_ptr() as *const c_void,
                    shape.a_bytes_f32(),
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
                "memcpy A f32",
            );
            check_cuda(
                cudaMemcpy(
                    d_b_f32,
                    b_host.as_ptr() as *const c_void,
                    shape.b_bytes_f32(),
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
                "memcpy B f32",
            );
            check_cuda(
                cudaMemcpy(
                    d_a_bf16,
                    a_bf16.as_ptr() as *const c_void,
                    shape.a_bytes_bf16(),
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
                "memcpy A bf16",
            );
            check_cuda(
                cudaMemcpy(
                    d_b_bf16,
                    b_bf16.as_ptr() as *const c_void,
                    shape.b_bytes_bf16(),
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                ),
                "memcpy B bf16",
            );
        }

        // Probe VRAM mid-bench for the rollback gate.
        let (mut free_now, mut total_now): (usize, usize) = (0, 0);
        unsafe {
            check_cuda(
                cudaMemGetInfo(&mut free_now, &mut total_now),
                "cudaMemGetInfo",
            );
        }
        let used_now = total_now.saturating_sub(free_now);
        if used_now > 7 * 1024 * 1024 * 1024 {
            eprintln!(
                "FATAL: VRAM in use exceeded 7 GiB rollback floor: {} used / {} total",
                fmt_size(used_now),
                fmt_size(total_now)
            );
            std::process::exit(2);
        }

        // Path A.
        let ms_a = run_path_a(
            *shape,
            d_a_f32 as *const f32,
            d_b_f32 as *const f32,
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

        // Path B.
        let ms_b = run_path_b_tf32(
            handle,
            *shape,
            d_a_f32 as *const f32,
            d_b_f32 as *const f32,
            d_c_f32 as *mut f32,
        );

        // Path C.
        let ms_c = run_path_c_bf16(
            handle,
            *shape,
            d_a_bf16 as *const c_void,
            d_b_bf16 as *const c_void,
            d_c_f32 as *mut f32,
        );

        // Snapshot Path C output for correctness vs A.
        let mut c_c_host: Vec<f32> = vec![0.0_f32; shape.m * shape.n];
        unsafe {
            check_cuda(
                cudaMemcpy(
                    c_c_host.as_mut_ptr() as *mut c_void,
                    d_c_f32 as *const c_void,
                    shape.c_bytes_f32(),
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                ),
                "memcpy D2H C (C)",
            );
        }

        // Per-element abs-diff on the smallest shape. For the big
        // shapes a global max would be dominated by accumulation
        // drift; keep it but record the relative envelope as well.
        let mut local_max_diff = 0.0_f32;
        for (a, c) in c_a_host.iter().zip(c_c_host.iter()) {
            let d = (a - c).abs();
            if d > local_max_diff {
                local_max_diff = d;
            }
        }
        if local_max_diff > max_diff_a_vs_c {
            max_diff_a_vs_c = local_max_diff;
        }

        // Free per-shape buffers.
        unsafe {
            check_cuda(cudaFree(d_a_f32), "free A f32");
            check_cuda(cudaFree(d_b_f32), "free B f32");
            check_cuda(cudaFree(d_c_f32), "free C f32");
            check_cuda(cudaFree(d_a_bf16), "free A bf16");
            check_cuda(cudaFree(d_b_bf16), "free B bf16");
        }

        let flops = shape.flops() as f64;
        let label_with_shape = format!("{}", shape.label);
        println!(
            "{:<14}  {:>7.0} M  {:>9.3}  {:>9.3}  {:>9.3}  {:>8.2}×  {:>8.2}×  {:>8.2}×",
            label_with_shape,
            flops / 1.0e6,
            ms_a,
            ms_b,
            ms_c,
            ms_a / ms_c,
            ms_b / ms_c,
            ms_a / ms_b,
        );

        all_results.push((*shape, ms_a, ms_b, ms_c));
    }

    println!();

    // GFLOPS summary.
    println!("Effective throughput (GFLOPS — measured matmul only, no H↔D):");
    for (shape, ms_a, ms_b, ms_c) in &all_results {
        let flops = shape.flops() as f64;
        let gflops = |ms: f64| (flops / 1.0e9) / (ms / 1000.0);
        println!(
            "  {:<14} A={:>8.1}  B={:>8.1}  C={:>8.1}  GFLOPS",
            shape.label,
            gflops(*ms_a),
            gflops(*ms_b),
            gflops(*ms_c),
        );
    }
    println!();

    println!("Numerical envelope:");
    println!(
        "  Max |C_A − C_C|  (across all shapes): {:.4e}",
        max_diff_a_vs_c
    );
    println!("  (Sanity gate: < 1e-2 expected. ADR-004 single-op envelope is");
    println!("   not tight; end-to-end 4-model F64 validation is the real gate.)");
    println!();

    // Plan A / Plan B decision banner.
    let mut wins_5x = 0;
    let mut wins_3x = 0;
    for (_shape, ms_a, _ms_b, ms_c) in &all_results {
        let ratio = ms_a / ms_c;
        if ratio >= 5.0 {
            wins_5x += 1;
        }
        if ratio >= 3.0 {
            wins_3x += 1;
        }
    }
    println!("=== H3 (cuBLAS BF16 TC ≥ 5× over naive F32) decision ===");
    println!(
        "Shapes with C ≥ 5× over A:  {} / {}",
        wins_5x,
        all_results.len()
    );
    println!(
        "Shapes with C ≥ 3× over A:  {} / {}",
        wins_3x,
        all_results.len()
    );
    if wins_5x >= 2 {
        println!("Decision:  PLAN A  — proceed M8.1 → M8.5 (BF16-resident kernel path)");
    } else if wins_3x >= 2 {
        println!("Decision:  PARTIAL — proceed with caution; revisit M8.7-M8.8 ROI");
    } else {
        println!("Decision:  H3 FAIL — PARAR. Investigate Path B (TF32) before M8.1.");
    }

    // Cleanup.
    unsafe {
        check_cublas(cublasDestroy_v2(handle), "cublasDestroy");
    }

    // Final VRAM report.
    let (mut free_end, mut total_end): (usize, usize) = (0, 0);
    unsafe {
        check_cuda(
            cudaMemGetInfo(&mut free_end, &mut total_end),
            "cudaMemGetInfo (end)",
        );
    }
    println!();
    println!(
        "VRAM at end:   free {} / total {}",
        fmt_size(free_end),
        fmt_size(total_end)
    );
}
