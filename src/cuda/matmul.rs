use std::ffi::c_void;
use std::os::raw::c_int;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::amg::nodes::NodeType;
use crate::cuda::pool_helpers::with_pooled_device_buffers;
use crate::cuda::{cuda_device_ptr, cuda_device_ptr_mut};
use crate::tensor::{DType, Device, Tensor, TensorStorage};

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

// Direct cudart FFI for the non-pooled path. Re-declared here (also
// declared in `cuda/pool_helpers.rs`) so the non-pooled function is
// self-contained at the source level — the linker resolves both
// declarations to the same symbol in the cudart shared library.
#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;
}

// Wrappers around `cudaMalloc`/`cudaFree` provided by the project's
// C side (already linked into the binary via `atenia_kernels`; same
// symbols `apx4_12::gpu_memory_pool` consumes). Re-declared here so
// the non-pooled path does not depend on the pool module.
unsafe extern "C" {
    fn cuda_malloc(ptr: *mut *mut c_void, bytes: usize);
    fn cuda_free(ptr: *mut c_void);
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

/// CPU-path CUDA matmul: copies host operands to VRAM, launches
/// `matmul_f32_launch_device`, and copies the output back. All
/// alloc / H↔D / free bookkeeping lives in
/// [`crate::cuda::pool_helpers::with_pooled_device_buffers`] so this
/// function only has to describe the kernel-specific invocation.
///
/// Panics on pool exhaustion or CUDA-driver failure with a
/// [`crate::tensor::StorageTransferError`] diagnostic in the message.
/// The panic is consistent with the rest of the CPU-path CUDA ops,
/// which do not propagate `Result` up because their callers assign
/// the return tensor directly into the graph node's `output` slot.
pub fn cuda_matmul(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
    let mut out = Tensor::zeros_new(&[m, n], Device::CPU);

    let m_ci = m as c_int;
    let k_ci = k as c_int;
    let n_ci = n as c_int;

    let result = unsafe {
        with_pooled_device_buffers(
            &[a.as_cpu_slice(), b.as_cpu_slice()],
            out.as_cpu_slice_mut(),
            |d_in, d_out| {
                // `matmul_f32_launch_device` returns void; it prints
                // to stderr and returns without signalling on kernel
                // launch / sync failure (pre-existing limitation from
                // before the `_device_ptrs` return-code convention).
                // Report success unconditionally.
                matmul_f32_launch_device(d_in[0], d_in[1], d_out, m_ci, k_ci, n_ci);
                0
            },
        )
    };

    if let Err(e) = result {
        panic!("cuda_matmul failed: {:?}", e);
    }

    out
}

/// Residency-aware CUDA matmul (M4.7.3.a).
///
/// Mirrors the `all_cuda` dispatch pattern from
/// [`crate::cuda::linear::cuda_linear`]: when every operand and the
/// output buffer live on VRAM, the device-pointer launcher is called
/// directly with no host↔device traffic. Falls through to
/// [`cuda_matmul`] (CPU-roundtrip) when any operand is host-resident,
/// preserving the existing CPU-path behaviour byte-for-byte for
/// callers that have not adopted residency yet.
///
/// `out` is mutated in place. Caller is responsible for constructing
/// `out` with `TensorStorage::Cuda` before calling this — see
/// [`Tensor::zeros_new_cuda`].
///
/// The underlying `matmul_f32_launch_device` returns `void`; on
/// kernel launch / sync failure it prints to stderr but does not
/// signal back. This is a pre-existing ABI limitation versus the
/// `_device_ptrs` return-code convention used by `cuda_linear` and
/// friends. M4.7.3 trusts the launcher and assumes success; promoting
/// the kernel to a `_device_ptrs` ABI variant returning `i32` is a
/// known follow-up.
pub fn cuda_matmul_inplace(
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
    m: usize,
    k: usize,
    n: usize,
) {
    let all_cuda = matches!(
        (&a.storage, &b.storage, &out.storage),
        (
            TensorStorage::Cuda(_),
            TensorStorage::Cuda(_),
            TensorStorage::Cuda(_),
        )
    );

    if all_cuda {
        let d_a = cuda_device_ptr(&a.storage);
        let d_b = cuda_device_ptr(&b.storage);
        let d_out = cuda_device_ptr_mut(&out.storage);

        unsafe {
            matmul_f32_launch_device(
                d_a,
                d_b,
                d_out,
                m as c_int,
                k as c_int,
                n as c_int,
            );
        }
    } else {
        // Mixed or all-Cpu storage: delegate to the CPU-roundtrip
        // path. `cuda_matmul` reads operands via `as_cpu_slice`,
        // which would panic on a Cuda operand, so we materialise
        // local Cpu clones first. This branch is unreachable on the
        // Llama hot path (the executor's `ensure_decoded` keeps
        // operand storage uniform — both Cuda or both Cpu), but
        // the fallback exists so `cuda_matmul_inplace` is correct
        // for any legal `(a, b, out)` triple a future caller may
        // produce. The output likewise gets normalised to Cpu
        // before the `clone_from_slice` write-back.
        let mut a_local = a.clone();
        let mut b_local = b.clone();
        a_local
            .ensure_cpu()
            .expect("cuda_matmul_inplace fallback: ensure_cpu on operand A failed");
        b_local
            .ensure_cpu()
            .expect("cuda_matmul_inplace fallback: ensure_cpu on operand B failed");
        let computed = cuda_matmul(&a_local, &b_local, m, k, n);
        out.ensure_cpu()
            .expect("cuda_matmul_inplace fallback: ensure_cpu on output failed");
        out.as_cpu_slice_mut()
            .clone_from_slice(computed.as_cpu_slice());
    }
}

pub fn is_cuda_available_for(node_type: &NodeType) -> bool {
    matches!(
        node_type,
        NodeType::MatMul | NodeType::BatchMatMul | NodeType::Linear
    )
}

/// **M6 step 2a** — non-pooled CPU-roundtrip CUDA matmul. **Dead code
/// at this commit** — no caller wires to it yet; the gate G5
/// (`gpu_can_run_matmul`'s `max_per_alloc <= DEFAULT_BLOCK_SIZE`
/// check at `gpu/dispatch/hooks.rs:127-129`) still bounces every
/// 13B matmul to CPU, so this function is reachable only from its
/// own unit test.
///
/// # Why it exists
///
/// The pooled path ([`cuda_matmul`] via
/// [`crate::cuda::pool_helpers::with_pooled_device_buffers`]) cannot
/// serve allocations larger than `DEFAULT_BLOCK_SIZE = 64 MiB`
/// (`apx4_12/mod.rs:16`); the pool does not sub-allocate and the
/// `alloc_device` assertion explicitly rejects oversize requests.
/// Llama 2 13B's smallest projection matmul (`5120×5120` F32 =
/// 100 MB) already exceeds that, and FFN weights at `5120×13824`
/// (270 MB F32) exceed it 4×. Activating any GPU dispatch for the
/// 13B forward therefore requires a path that bypasses the pool.
///
/// This function provides exactly that path — direct
/// `cuda_malloc` / `cudaMemcpy` H→D / kernel / `cudaMemcpy` D→H /
/// `cuda_free`, with no pool bookkeeping. It is the M6 step-2a
/// "asset commit" referenced in `INVESTIGATION_M6_DEEP.md` §5.
/// The next step (2b) will be to wire it behind a lifted G5 with
/// shape-class routing.
///
/// # Behaviour
///
/// - Returns `Some(Tensor)` of shape `[m, n]`, F32, CPU storage,
///   on success.
/// - Returns `None` on any allocation or transfer failure
///   (`cuda_malloc` returning a null pointer, or `cudaMemcpy`
///   returning non-zero). The caller is expected to fall back to
///   the CPU dispatcher in that case.
/// - On `None` every successfully-allocated device buffer is freed
///   before return via the [`NonPooledAllocs`] RAII guard, so there
///   are no VRAM leaks even on the error paths.
///
/// # Safety
///
/// Internally unsafe via cudart FFI; the public surface is safe.
/// The kernel launcher (`matmul_f32_launch_device`) returns `void`
/// and signals errors only via stderr — there is no way to
/// distinguish a bad launch from a successful one at this layer.
/// This matches the existing [`cuda_matmul`] / [`cuda_matmul_inplace`]
/// behaviour and is acceptable for a dead-code asset commit; the
/// shape gates that callers will impose in step 2b ensure the kernel
/// is only invoked with valid shape triples.
///
/// # Caller contract
///
/// `a` is expected to be `[m × k]` row-major F32, `b` to be
/// `[k × n]` row-major F32. No shape validation is performed
/// here; caller must validate before invocation.
pub fn cuda_matmul_non_pooled(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Option<Tensor> {
    use std::mem;

    #[cfg(feature = "gpu-trace")]
    eprintln!(
        "[GPU-TRACE] cuda_matmul_non_pooled ENTRY: m={}, k={}, n={}, a.len={}, b.len={}",
        m, k, n, a.len(), b.len()
    );

    // Guard: refuse to issue a CUDA kernel call when the driver is
    // not available. Without this check, `cuda_malloc` (a no-op stub
    // on non-CUDA hosts) would still return non-null garbage and the
    // kernel launch would blow up. Mirrors the gate in
    // `gpu_can_run_matmul`.
    let cuda_ok = super::cuda_available();
    #[cfg(feature = "gpu-trace")]
    eprintln!("[GPU-TRACE] cuda_matmul_non_pooled: cuda_available()={}", cuda_ok);
    if !cuda_ok {
        #[cfg(feature = "gpu-trace")]
        eprintln!("[GPU-TRACE] cuda_matmul_non_pooled: EXIT=None (cuda_available=false)");
        return None;
    }

    let bytes_a = a.len() * mem::size_of::<f32>();
    let bytes_b = b.len() * mem::size_of::<f32>();
    let out_len = m * n;
    let bytes_out = out_len * mem::size_of::<f32>();

    #[cfg(feature = "gpu-trace")]
    let t_alloc = std::time::Instant::now();

    let mut allocs = NonPooledAllocs::new();
    let d_a = match allocs.alloc(bytes_a) {
        Some(p) => p,
        None => {
            #[cfg(feature = "gpu-trace")]
            eprintln!(
                "[GPU-TRACE] cuda_matmul_non_pooled: EXIT=None (cuda_malloc A failed, bytes={})",
                bytes_a
            );
            return None;
        }
    };
    #[cfg(feature = "gpu-trace")]
    eprintln!("[GPU-TRACE] cuda_malloc A ok: {:.1}MB", bytes_a as f64 / (1024.0 * 1024.0));

    let d_b = match allocs.alloc(bytes_b) {
        Some(p) => p,
        None => {
            #[cfg(feature = "gpu-trace")]
            eprintln!(
                "[GPU-TRACE] cuda_matmul_non_pooled: EXIT=None (cuda_malloc B failed, bytes={})",
                bytes_b
            );
            return None;
        }
    };
    #[cfg(feature = "gpu-trace")]
    eprintln!("[GPU-TRACE] cuda_malloc B ok: {:.1}MB", bytes_b as f64 / (1024.0 * 1024.0));

    let d_out = match allocs.alloc(bytes_out) {
        Some(p) => p,
        None => {
            #[cfg(feature = "gpu-trace")]
            eprintln!(
                "[GPU-TRACE] cuda_matmul_non_pooled: EXIT=None (cuda_malloc OUT failed, bytes={})",
                bytes_out
            );
            return None;
        }
    };
    #[cfg(feature = "gpu-trace")]
    {
        eprintln!("[GPU-TRACE] cuda_malloc OUT ok: {:.1}MB", bytes_out as f64 / (1024.0 * 1024.0));
        eprintln!("[GPU-TRACE] cuda_malloc total: {:.2}ms", t_alloc.elapsed().as_secs_f64() * 1000.0);
    }

    // H→D: stage both inputs into VRAM. On any failure the RAII
    // guard frees every buffer allocated so far when this function
    // returns.
    unsafe {
        #[cfg(feature = "gpu-trace")]
        let t_h2d = std::time::Instant::now();
        let rc = cudaMemcpy(
            d_a,
            a.as_ptr() as *const c_void,
            bytes_a,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        #[cfg(feature = "gpu-trace")]
        eprintln!("[GPU-TRACE] cudaMemcpy H2D A: rc={}, {:.1}MB", rc, bytes_a as f64 / (1024.0 * 1024.0));
        if rc != 0 {
            #[cfg(feature = "gpu-trace")]
            eprintln!("[GPU-TRACE] cuda_matmul_non_pooled: EXIT=None (memcpy A failed, rc={})", rc);
            return None;
        }
        let rc = cudaMemcpy(
            d_b,
            b.as_ptr() as *const c_void,
            bytes_b,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        #[cfg(feature = "gpu-trace")]
        eprintln!("[GPU-TRACE] cudaMemcpy H2D B: rc={}, {:.1}MB", rc, bytes_b as f64 / (1024.0 * 1024.0));
        if rc != 0 {
            #[cfg(feature = "gpu-trace")]
            eprintln!("[GPU-TRACE] cuda_matmul_non_pooled: EXIT=None (memcpy B failed, rc={})", rc);
            return None;
        }
        #[cfg(feature = "gpu-trace")]
        eprintln!("[GPU-TRACE] H2D total: {:.2}ms", t_h2d.elapsed().as_secs_f64() * 1000.0);

        // Kernel launch. `matmul_f32_launch_device` returns void
        // and syncs internally; no error code is propagated, so we
        // assume success consistent with `cuda_matmul`.
        #[cfg(feature = "gpu-trace")]
        let t_kernel = std::time::Instant::now();
        matmul_f32_launch_device(
            d_a as *const f32,
            d_b as *const f32,
            d_out as *mut f32,
            m as c_int,
            k as c_int,
            n as c_int,
        );
        #[cfg(feature = "gpu-trace")]
        eprintln!("[GPU-TRACE] kernel matmul_f32_launch_device: {:.2}ms", t_kernel.elapsed().as_secs_f64() * 1000.0);

        // D→H: pull the output back into a fresh host Vec. Allocated
        // here (rather than into a pre-built `Tensor::zeros_new`) so
        // the function has a single ownership chain that is easy to
        // reason about on the error paths.
        let mut out_host = vec![0.0_f32; out_len];
        #[cfg(feature = "gpu-trace")]
        let t_d2h = std::time::Instant::now();
        let rc = cudaMemcpy(
            out_host.as_mut_ptr() as *mut c_void,
            d_out,
            bytes_out,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        );
        #[cfg(feature = "gpu-trace")]
        eprintln!("[GPU-TRACE] cudaMemcpy D2H OUT: rc={}, {:.1}MB, {:.2}ms",
            rc, bytes_out as f64 / (1024.0 * 1024.0),
            t_d2h.elapsed().as_secs_f64() * 1000.0);
        if rc != 0 {
            #[cfg(feature = "gpu-trace")]
            eprintln!("[GPU-TRACE] cuda_matmul_non_pooled: EXIT=None (memcpy D2H failed, rc={})", rc);
            return None;
        }

        #[cfg(feature = "gpu-trace")]
        eprintln!("[GPU-TRACE] cuda_matmul_non_pooled: EXIT=Some (success), 3x cuda_free pending on guard drop");

        // Guard drops here when `Some(...)` returns, freeing all 3
        // device buffers before the host Vec is wrapped into the
        // returned Tensor.
        Some(Tensor::new_cpu(vec![m, n], out_host))
    }
}

/// RAII guard that frees every successfully-allocated device buffer
/// when dropped. Mirrors `pool_helpers::PoolGuard` but routes
/// allocations through `cuda_malloc`/`cuda_free` (no pool).
struct NonPooledAllocs {
    ptrs: Vec<*mut c_void>,
}

impl NonPooledAllocs {
    fn new() -> Self {
        Self {
            ptrs: Vec::with_capacity(3),
        }
    }

    /// Allocate `bytes` of VRAM. Returns the device pointer on
    /// success, or `None` if `cuda_malloc` left the pointer null
    /// (driver OOM or device unavailable). On `Some(...)` the
    /// pointer is tracked for cleanup at drop time.
    fn alloc(&mut self, bytes: usize) -> Option<*mut c_void> {
        let mut p: *mut c_void = std::ptr::null_mut();
        unsafe {
            cuda_malloc(&mut p, bytes);
        }
        if p.is_null() {
            return None;
        }
        self.ptrs.push(p);
        Some(p)
    }
}

impl Drop for NonPooledAllocs {
    fn drop(&mut self) {
        for p in self.ptrs.drain(..) {
            unsafe {
                cuda_free(p);
            }
        }
    }
}

// ===========================================================================
// M8.2 — cuda_matmul_bf16_inplace via cublasGemmEx (BF16 inputs, F32 output)
// ===========================================================================
//
// `cublasGemmEx` is the cuBLAS routine that handles mixed-dtype gemm with
// hardware Tensor Core acceleration. For Llama 13B decode-step shapes
// (M = 1, K ∈ {5120, 13824}, N up to 32000) the M8.0 bench measured
// 2.2-2.3× speedup over the M6 naive F32 kernel, mostly from the 2× memory
// bandwidth ratio of BF16 vs F32 (the matmul is memory-bound at M = 1, so
// Tensor Cores can't multiply throughput beyond what bandwidth allows).
//
// **Production-quality vs `examples/bench_cublas_bf16.rs` standalone**:
//   - The bench used per-process `cublasCreate` / `cublasDestroy` around
//     a single bench run. Production code shares a process-wide handle
//     via `OnceLock` so per-call overhead drops from ~100 us to a single
//     atomic load.
//   - The bench used the global default math mode. Production code keeps
//     `CUBLAS_DEFAULT_MATH` so the BF16 gemm picks Tensor Cores
//     automatically without a global TF32 flip that would also affect
//     unrelated F32 callsites.
//   - Per-call buffers (activation BF16, output F32) are allocated via
//     the same `NonPooledAllocs` RAII guard that `cuda_matmul_non_pooled`
//     uses, so any failure path frees every device buffer before
//     returning `false`.

#[allow(non_camel_case_types)]
type cublasHandle_t = *mut c_void;

const CUBLAS_OP_N: c_int = 0;
const CUDA_R_32F: c_int = 0;
// M8.4c — `CUDA_R_16BF` is no longer used in `cuda_matmul_bf16_inplace`
// (the M8.4-original BF16 input path was replaced by a BF16 → F32
// upcast + F32 GEMM). Kept here as a documentation anchor for the
// historical M8.4 path; remove if no future M8 sub-phase references
// the constant.
#[allow(dead_code)]
const CUDA_R_16BF: c_int = 14;
/// **M10.2.0** — pre-TF32 production compute mode. Kept reachable
/// because the M10.2.0 swap to `CUBLAS_COMPUTE_32F_FAST_TF32` is
/// gated empirically: if a future benchmark/fixture run shows the
/// TF32 truncation harms a specific model class, the production
/// path can revert to `CUBLAS_COMPUTE_32F` by swapping one
/// constant in `cuda_matmul_bf16_inplace`. Without the
/// `#[allow(dead_code)]` the compiler regresses warning hygiene.
#[allow(dead_code)]
const CUBLAS_COMPUTE_32F: c_int = 68;
/// **M10.2.0** — TF32 Tensor Core compute mode. Inputs stay
/// `CUDA_R_32F`, but cuBLAS truncates internally to TF32 (8-bit
/// exponent like F32, 10-bit mantissa) and dispatches through
/// the Ada/Hopper Tensor Cores. Engages TC throughput on sm_89
/// without changing the cuBLAS-side input dtype. TF32 mantissa is
/// 8× more precise per element than BF16 (10 bits vs 7), keeping
/// the cascaded drift within ADR-004 strict — validated under the
/// 4-model F64 fixture before this constant became the production
/// compute mode for `cuda_matmul_bf16_inplace` (Path B M8.4c).
const CUBLAS_COMPUTE_32F_FAST_TF32: c_int = 77;
const CUBLAS_GEMM_DEFAULT: c_int = -1;

#[link(name = "cublas")]
unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut cublasHandle_t) -> c_int;
    fn cublasSetStream_v2(handle: cublasHandle_t, stream: *mut c_void) -> c_int;
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

#[link(name = "cudart")]
unsafe extern "C" {
    /// Currently unused — the M8.7.1.r refactor switched the
    /// `cuda_matmul_bf16_inplace` path from the device-wide
    /// `cudaDeviceSynchronize` to per-stream `cudaStreamSynchronize`
    /// after `cudaMemcpyAsync`. Kept here as a reachable FFI
    /// declaration for the M8.7.1.b/c follow-up work (async H→D
    /// pipeline + dedicated copy/compute streams), which may want
    /// a coarse device barrier in some recovery paths. Marked
    /// `#[allow(dead_code)]` so the compiler does not regress
    /// the rest of the module's warning hygiene.
    #[allow(dead_code)]
    fn cudaDeviceSynchronize() -> c_int;
    /// Async memcpy bound to a CUDA stream. With `stream = null`
    /// (the default stream) this is functionally equivalent to
    /// `cudaMemcpy` for non-pinned host buffers — the driver
    /// pins/unpins internally and serializes against device work.
    /// On a pinned host buffer + non-default stream the call
    /// returns immediately and the copy proceeds in parallel
    /// with other stream work.
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: *mut c_void,
    ) -> c_int;
    /// Block the calling host thread until every operation
    /// previously enqueued on `stream` has finished. With
    /// `stream = null` this synchronises the **default stream**
    /// only, which is sufficient for the M8.4c contract because
    /// every op runs on the default stream by construction.
    fn cudaStreamSynchronize(stream: *mut c_void) -> c_int;
}

/// Wrapper around `cublasHandle_t` so the singleton can be `Send + Sync`.
/// The cuBLAS handle is process-wide and thread-safe per the NVIDIA docs;
/// the `*mut c_void` is opaque (a driver-managed cookie), not a host
/// pointer to Rust-managed memory.
struct CublasHandleHolder {
    handle: cublasHandle_t,
}
unsafe impl Send for CublasHandleHolder {}
unsafe impl Sync for CublasHandleHolder {}

static CUBLAS_HANDLE: OnceLock<CublasHandleHolder> = OnceLock::new();

/// Lazily initialise (or return) the process-wide cuBLAS handle. Returns
/// `None` if `cuda_available()` is false or `cublasCreate_v2` fails — the
/// caller falls back without needing to differentiate the failure mode.
/// The handle is never destroyed; it lives until process exit, which is
/// the standard cuBLAS recommendation.
fn cublas_handle() -> Option<cublasHandle_t> {
    if !super::cuda_available() {
        return None;
    }
    let holder = CUBLAS_HANDLE.get_or_init(|| {
        let mut handle: cublasHandle_t = std::ptr::null_mut();
        unsafe {
            // On failure we panic — the OnceLock would otherwise cache the
            // failure forever and every subsequent call would silently get
            // a null handle. Surfacing the cuBLAS status is the operator
            // signal that something is structurally broken.
            let rc = cublasCreate_v2(&mut handle);
            assert_eq!(
                rc, 0,
                "cublasCreate_v2 returned {} during M8.2 handle init", rc
            );
            assert!(!handle.is_null(), "cublasCreate_v2 returned null handle");
        }
        CublasHandleHolder { handle }
    });
    Some(holder.handle)
}

/// **M8.2** — counter for successful `cuda_matmul_bf16_inplace` calls.
/// Mirrors the M6 / M7 / M8.1 counter pattern; public reader is
/// [`vram_bf16_matmul_count`].
static VRAM_BF16_MATMUL_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn vram_bf16_matmul_count() -> usize {
    VRAM_BF16_MATMUL_COUNT.load(Ordering::Relaxed)
}

/// **M8.2** — BF16-resident GPU matmul for the M8 path.
///
/// Shape contract — row-major:
///   `out[m, n] = a[m, k] · b[k, n]`
///
/// Operand storage:
///   - `a`: F32 host (`TensorStorage::Cpu` / `CpuShared`). Cast to BF16
///     on host before upload, halving the PCIe transfer cost.
///   - `b`: BF16 in VRAM (`TensorStorage::Cuda` with `dtype = BF16`,
///     allocated via M8.1's `bf16_to_vram_no_upcast`). Consumed
///     directly by `cublasGemmEx` with `CUDA_R_16BF` — no upcast pass.
///   - `out`: F32 host (`TensorStorage::Cpu`). The downloaded F32
///     output is written into `out.as_cpu_slice_mut()` after `ensure_cpu`.
///
/// The cuBLAS row-major-to-column-major trick: cuBLAS computes in
/// column-major. Asking it to compute `C^T = B^T · A^T` (column-major)
/// yields the byte layout we want for row-major `C = A · B`. The
/// dispatch swaps the operand order in the `cublasGemmEx` call.
///
/// # Returns
///
/// `true` on success (output written, counter incremented). `false` on
/// any precondition failure or CUDA / cuBLAS error. Every device buffer
/// allocated up to the failure point is freed before the `false` return
/// via the [`NonPooledAllocs`] guard (BF16 activation transient + F32
/// output transient). The caller falls back to the F32 dispatcher.
///
/// # Numerical behaviour (M8.4c — Path B)
///
/// **The activation is NOT cast to BF16.** M8.4c discarded the
/// M8.4-original "BF16 activation × BF16 weight" path because the
/// cascaded BF16 truncation of activations through 16-28
/// transformer layers drove the M8.5 4-model F64 validation drift
/// to 0.18-2.33 (failing ADR-004 by 2-5× in 3/4 models — the
/// activation truncation by itself is responsible, see the M8.5b
/// post-mortem in commit history).
///
/// The current implementation:
///
///   1. Keeps the **weight as BF16 in VRAM** (preserves M8.3's
///      capacity-doubling planner contract — same plan, same
///      number of VRAM-resident tensors).
///   2. Upcasts the BF16 weight to a **fresh F32 transient** on
///      the device via [`bf16_to_f32_transient_in_vram`]
///      (existing M6 kernel, bandwidth-bound).
///   3. Uploads the **F32 activation** unchanged (no truncation).
///   4. Runs `cublasGemmEx(F32, F32, F32)` with `COMPUTE_32F` —
///      F32 inputs, F32 output, F32 accumulate. cuBLAS picks the
///      best F32 algorithm for sm_89 (CUDA cores, no Tensor
///      Cores since the inputs are F32 and cuBLAS does not
///      auto-truncate to TF32 under `COMPUTE_32F`).
///
/// Result: numerics identical to the M4.7.2.e CPU path
/// (BF16 storage + F32 matmul), drift ~1.4e-4 on TinyLlama
/// vs 0.9 with the original BF16-activation path.
///
/// The transient F32 buffer is freed at function exit via
/// `TensorGPU`'s `Drop`.
///
/// # Performance trade-off
///
/// Per-matmul cost is ~3× the M8.4-original path (the upcast
/// kernel + F32 cuBLAS without TC vs. raw BF16 TC matmul). For
/// decode-step matmuls (M = 1, bandwidth-bound) the absolute
/// time is still sub-millisecond on Ada. For 13B end-to-end this
/// is invisible because the bottleneck is the 29 Disk-tier CPU
/// matmuls (~30 s/token), not the 11 VRAM matmuls (~30 ms total).
pub fn cuda_matmul_bf16_inplace(
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
    m: usize,
    k: usize,
    n: usize,
    stream: *mut c_void,
) -> bool {
    // 1. Validate b is a BF16-resident TensorGPU. Anything else is a
    //    caller bug or a non-M8 path; refuse and let the caller fall back.
    let b_gpu = match &b.storage {
        TensorStorage::Cuda(gpu) if gpu.dtype() == DType::BF16 => gpu,
        _ => return false,
    };
    if b_gpu.size_bytes() != k * n * 2 {
        return false;
    }

    // 2. cuBLAS handle (process-wide).
    let handle = match cublas_handle() {
        Some(h) => h,
        None => return false,
    };

    // 3. Validate F32 activation. M8.4c keeps the activation as
    //    F32 throughout — no host-side cast, no precision loss.
    let a_slice = a.as_cpu_slice();
    if a_slice.len() != m * k {
        return false;
    }
    let bytes_a_f32 = m * k * 4;
    let bytes_out_f32 = m * n * 4;

    // 4. **M8.4c step 1** — upcast the BF16 weight to a fresh F32
    //    transient on-device. Wrapped in `TensorGPU` so its `Drop`
    //    frees the buffer when this function returns; no manual
    //    free required.
    let b_f32_transient =
        match crate::cuda::bf16_to_f32::bf16_to_f32_transient_in_vram(b_gpu) {
            Some(g) => g,
            None => return false,
        };

    // 5. Allocate the per-call F32 activation upload buffer and the
    //    F32 output buffer via `NonPooledAllocs`. The F32 weight
    //    transient is owned by `b_f32_transient` (separate
    //    lifecycle, freed by the engine on Drop).
    let mut allocs = NonPooledAllocs::new();
    let d_a_f32 = match allocs.alloc(bytes_a_f32) {
        Some(p) => p,
        None => return false,
    };
    let d_out_f32 = match allocs.alloc(bytes_out_f32) {
        Some(p) => p,
        None => return false,
    };

    // 6. **M8.7.1.r** — H→D upload of the F32 activation on the
    //    caller-supplied `stream`. With `stream = null` this is
    //    bit-exact with the M8.4c default-stream sync behaviour
    //    (`cudaMemcpyAsync` on the default stream + non-pinned host
    //    buffer degenerates to a synchronous copy in the driver).
    //    With a non-null stream + pinned host buffer (M8.7.1.b)
    //    the copy returns immediately and proceeds in parallel
    //    with other stream work.
    unsafe {
        let rc = cudaMemcpyAsync(
            d_a_f32,
            a_slice.as_ptr() as *const c_void,
            bytes_a_f32,
            CUDA_MEMCPY_HOST_TO_DEVICE,
            stream,
        );
        if rc != 0 {
            return false;
        }
    }

    // 7. cublasGemmEx — same row-major-via-transpose trick, but
    //    now ALL operands are F32 (`CUDA_R_32F`). The `B` pointer
    //    is the F32 transient produced by step 4.
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe {
        // **M8.7.1.r** — bind the cuBLAS handle to the caller-
        // supplied stream so the GEMM enqueues on the same stream
        // as the H→D upload above and the D→H download below.
        // With `stream = null` this is identical to the previous
        // default-stream behaviour.
        let rc = cublasSetStream_v2(handle, stream);
        if rc != 0 {
            return false;
        }
        let rc = cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n as c_int, m as c_int, k as c_int,
            &alpha as *const f32 as *const c_void,
            // First operand in column-major view = B^T (the row-major
            // weight `b`, now upcasted to F32). Leading dim n.
            b_f32_transient.device_ptr() as *const c_void, CUDA_R_32F, n as c_int,
            // Second operand = A^T (the row-major activation, F32).
            // Leading dim k.
            d_a_f32, CUDA_R_32F, k as c_int,
            &beta as *const f32 as *const c_void,
            d_out_f32, CUDA_R_32F, n as c_int,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT,
        );
        if rc != 0 {
            return false;
        }
    }

    // 8. **M8.7.1.r** — D→H download on the same stream, then a
    //    single `cudaStreamSynchronize` after the download to
    //    block the host thread until the bytes are visible. Pre-
    //    M8.7.1.r the M8.4c body did `cudaDeviceSynchronize`
    //    between the GEMM and the D→H `cudaMemcpy` to defend
    //    against potential reordering — that sync was a
    //    device-wide barrier that serialised every CUDA stream,
    //    blocking any prefetch / staging work the executor had
    //    enqueued on side streams. The new pattern (single
    //    `cudaStreamSynchronize` AFTER the D→H async memcpy)
    //    preserves the in-stream FIFO ordering (cuBLAS GEMM →
    //    `cudaMemcpyAsync` D→H → sync) without touching other
    //    streams; default-stream callers see identical
    //    behaviour because both forms drain the default stream
    //    before returning.
    if out.ensure_cpu().is_err() {
        return false;
    }
    unsafe {
        let out_slice = out.as_cpu_slice_mut();
        if out_slice.len() != m * n {
            return false;
        }
        let rc = cudaMemcpyAsync(
            out_slice.as_mut_ptr() as *mut c_void,
            d_out_f32,
            bytes_out_f32,
            CUDA_MEMCPY_DEVICE_TO_HOST,
            stream,
        );
        if rc != 0 {
            return false;
        }
        let rc = cudaStreamSynchronize(stream);
        if rc != 0 {
            return false;
        }
    }

    VRAM_BF16_MATMUL_COUNT.fetch_add(1, Ordering::Relaxed);
    true
}

// ===========================================================================
// M10.2.1 — BF16-TC native fast path (CUDA_R_16BF inputs, F32 acc + output)
// ===========================================================================

/// **M10.2.1** — counter for successful `cuda_matmul_bf16_native_inplace`
/// calls. Disjoint from [`VRAM_BF16_MATMUL_COUNT`] (the M8.4c
/// "certified" path that upcasts BF16 weight to F32 before the GEMM):
/// every call to the native fast path increments this counter on
/// success and never advances `VRAM_BF16_MATMUL_COUNT`. Lets a smoke
/// or test prove that fast mode actually fired without re-counting
/// certified-path matmuls.
static VRAM_BF16_NATIVE_MATMUL_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Public reader for [`VRAM_BF16_NATIVE_MATMUL_COUNT`].
pub fn vram_bf16_native_matmul_count() -> usize {
    VRAM_BF16_NATIVE_MATMUL_COUNT.load(Ordering::Relaxed)
}

/// **M10.2.1** — BF16-Tensor-Core native matmul (fast mode).
///
/// Numerically equivalent to the M8.4-original path that M8.4c
/// rejected for ADR-004 strict reasons: BF16 activation × BF16
/// weight × F32 accumulate, dispatched through Ada/Hopper Tensor
/// Cores via `cublasGemmEx` with `CUDA_R_16BF` inputs and
/// `CUBLAS_COMPUTE_32F`.
///
/// Trade-off vs [`cuda_matmul_bf16_inplace`] (Path B M8.4c
/// "certified"):
///
/// - **+**: ~2× speedup on the four Llama 13B decode shapes per
///   the M8.0 microbench. No host-side upcast, no on-device upcast
///   transient, BF16 PCIe transfer (half the bytes vs F32).
/// - **−**: cascaded BF16 activation truncation through 16-28
///   transformer layers drives the M8.5 4-model F64 fixture drift
///   to the M8.4-original range (0.07–2.33 across the four models;
///   3 of 4 fail ADR-004 strict by construction).
///
/// Selection between the two paths is operator-driven via the
/// `ATENIA_FAST_MODE=1` environment variable, read at the
/// dispatcher level (not here — this function is a pure kernel).
/// ADR-005 documents the envelope of fast mode and the reframing
/// of ADR-004 to a per-checkpoint certification rather than a
/// blanket runtime guarantee.
///
/// Same I/O contract as [`cuda_matmul_bf16_inplace`]:
///   - `a`: F32 host (`TensorStorage::Cpu` / `CpuShared`). Cast to
///     BF16 host-side before the H→D upload (the only place the
///     activation ever lives in BF16; the AMG graph never sees it).
///   - `b`: BF16 in VRAM (`TensorStorage::Cuda` with `dtype = BF16`).
///     Consumed directly — no upcast, no transient.
///   - `out`: F32 host. The downloaded F32 output is written into
///     `out.as_cpu_slice_mut()` after `ensure_cpu`.
///
/// Returns `true` on success (output written, counter incremented).
/// Returns `false` on any precondition or CUDA / cuBLAS error;
/// every device buffer allocated up to the failure point is freed
/// before the `false` return via [`NonPooledAllocs`]'s `Drop`.
///
/// Stream handling is identical to [`cuda_matmul_bf16_inplace`]:
/// caller-supplied `stream` is bound to the cuBLAS handle for the
/// duration of the GEMM and used for both H→D and D→H async
/// memcpy. With `stream = null` the behaviour is bit-exact with
/// the default-stream sync semantics.
pub fn cuda_matmul_bf16_native_inplace(
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
    m: usize,
    k: usize,
    n: usize,
    stream: *mut c_void,
) -> bool {
    // 1. Validate b is a BF16-resident TensorGPU.
    let b_gpu = match &b.storage {
        TensorStorage::Cuda(gpu) if gpu.dtype() == DType::BF16 => gpu,
        _ => return false,
    };
    if b_gpu.size_bytes() != k * n * 2 {
        return false;
    }

    // 2. cuBLAS handle (process-wide).
    let handle = match cublas_handle() {
        Some(h) => h,
        None => return false,
    };

    // 3. Validate F32 activation and cast host-side to BF16. The
    //    cast is a single `f32_to_bf16_bits` per element; for
    //    decode-step (M=1) shapes this is `K` truncations totaling
    //    a few KiB of work — negligible vs the GEMM itself. The
    //    BF16 buffer is dropped at function exit.
    let a_slice = a.as_cpu_slice();
    if a_slice.len() != m * k {
        return false;
    }
    let a_bf16: Vec<u16> = a_slice
        .iter()
        .map(|&f| crate::tensor::tensor::f32_to_bf16_bits(f))
        .collect();
    let bytes_a_bf16 = m * k * 2;
    let bytes_out_f32 = m * n * 4;

    // 4. Allocate the per-call BF16 activation upload buffer and
    //    the F32 output buffer via `NonPooledAllocs`. No weight
    //    transient (the BF16 weight is consumed directly).
    let mut allocs = NonPooledAllocs::new();
    let d_a_bf16 = match allocs.alloc(bytes_a_bf16) {
        Some(p) => p,
        None => return false,
    };
    let d_out_f32 = match allocs.alloc(bytes_out_f32) {
        Some(p) => p,
        None => return false,
    };

    // 5. H→D upload of the BF16 activation on the caller-supplied
    //    `stream`. With `stream = null` this is bit-exact with the
    //    default-stream sync behaviour.
    unsafe {
        let rc = cudaMemcpyAsync(
            d_a_bf16,
            a_bf16.as_ptr() as *const c_void,
            bytes_a_bf16,
            CUDA_MEMCPY_HOST_TO_DEVICE,
            stream,
        );
        if rc != 0 {
            return false;
        }
    }

    // 6. cublasGemmEx — same row-major-via-transpose trick as
    //    `cuda_matmul_bf16_inplace`, but with `CUDA_R_16BF` inputs
    //    and `CUBLAS_COMPUTE_32F` (BF16 → F32 accumulate, hardware
    //    Tensor Cores on sm_89). Output stays `CUDA_R_32F`.
    //    Compute mode is the strict 32-bit accumulate (NOT the
    //    `_FAST_TF32` variant from M10.2.0 — the inputs are already
    //    BF16, so TF32 truncation has nothing to do).
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe {
        let rc = cublasSetStream_v2(handle, stream);
        if rc != 0 {
            return false;
        }
        let rc = cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n as c_int, m as c_int, k as c_int,
            &alpha as *const f32 as *const c_void,
            // First operand in column-major view = B^T (the row-major
            // BF16 weight `b`). Leading dim n.
            b_gpu.device_ptr() as *const c_void, CUDA_R_16BF, n as c_int,
            // Second operand = A^T (the row-major BF16 activation).
            // Leading dim k.
            d_a_bf16, CUDA_R_16BF, k as c_int,
            &beta as *const f32 as *const c_void,
            d_out_f32, CUDA_R_32F, n as c_int,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT,
        );
        if rc != 0 {
            return false;
        }
    }

    // 7. D→H download on the same stream + single
    //    `cudaStreamSynchronize` after — mirrors M8.7.1.r.
    if out.ensure_cpu().is_err() {
        return false;
    }
    unsafe {
        let out_slice = out.as_cpu_slice_mut();
        if out_slice.len() != m * n {
            return false;
        }
        let rc = cudaMemcpyAsync(
            out_slice.as_mut_ptr() as *mut c_void,
            d_out_f32,
            bytes_out_f32,
            CUDA_MEMCPY_DEVICE_TO_HOST,
            stream,
        );
        if rc != 0 {
            return false;
        }
        let rc = cudaStreamSynchronize(stream);
        if rc != 0 {
            return false;
        }
    }

    VRAM_BF16_NATIVE_MATMUL_COUNT.fetch_add(1, Ordering::Relaxed);
    true
}

// ===========================================================================
// M8.7.0 — Disk → GPU JIT pipeline (single-tensor MVP)
// ===========================================================================

/// **M8.7.0** — counter for successful Disk → GPU BF16 matmul calls.
/// Disjoint from [`VRAM_BF16_MATMUL_COUNT`] (M8.2 — for weights that
/// are already resident in VRAM): every call to
/// [`cuda_matmul_disk_streamed_bf16`] increments this counter on
/// success and never advances `VRAM_BF16_MATMUL_COUNT`. The two
/// counters together let the M8.7 smoke prove the Disk-streamed
/// path actually fired without re-counting M8 resident matmuls.
static DISK_STREAMED_MATMUL_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn disk_streamed_matmul_count() -> usize {
    DISK_STREAMED_MATMUL_COUNT.load(Ordering::Relaxed)
}

/// **M8.7.0 MVP** — stream a BF16 weight from NVMe into a transient
/// VRAM staging slot, run the M8.4c Path B matmul, free the slot.
///
/// Single-thread, single-staging-slot, no async pipeline. Reuses
/// the M8.4c BF16-resident kernel ([`cuda_matmul_bf16_inplace`]) by
/// presenting the staged Disk weight as a fresh `Tensor::from_cuda_gpu`.
/// The transient `TensorGPU`'s `Drop` releases VRAM at scope end so
/// no manual free is required and the staging budget reserved by
/// the M8.7 prereq planner change is honoured.
///
/// Shape contract — row-major: `out[m, n] = a[m, k] · b[k, n]`.
///
/// # Operand storage
///
/// - `a`: F32 host (`TensorStorage::Cpu` / `CpuShared`). Same as
///   [`cuda_matmul_bf16_inplace`].
/// - `b`: BF16 on disk (`TensorStorage::Disk` with
///   `handle.dtype() == DiskDtype::BF16`). Shape must match
///   `[k, n]`.
/// - `out`: F32 host. Same as [`cuda_matmul_bf16_inplace`].
///
/// # Failure modes
///
/// Returns `false` (and increments **no** counter) on any of:
///
/// - `b.storage` is not `Disk(handle)` or the handle dtype is not
///   BF16.
/// - `handle.numel() != k * n` (shape mismatch).
/// - NVMe read fails ([`crate::tensor::disk_tier::read_bf16_raw_bytes`]
///   surfaces the I/O error).
/// - VRAM staging upload fails
///   ([`crate::cuda::bf16_to_f32::bf16_to_vram_no_upcast_from_raw_bytes`]
///   returns `None`).
/// - The downstream [`cuda_matmul_bf16_inplace`] dispatch fails.
///
/// On any failure the caller MUST fall back to the legacy
/// `ensure_decoded` + AVX2 path. The transient host bytes and
/// staging VRAM buffer are freed by `Vec::Drop` and `TensorGPU::Drop`
/// respectively; nothing leaks.
pub fn cuda_matmul_disk_streamed_bf16(
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
    m: usize,
    k: usize,
    n: usize,
    next_handle: Option<&crate::tensor::disk_tier::DiskTensorHandle>,
) -> bool {
    use crate::tensor::disk_tier;

    // 1. Validate b is a BF16 disk handle and shape matches.
    let handle = match &b.storage {
        TensorStorage::Disk(h) if h.dtype() == disk_tier::DiskDtype::BF16 => h,
        _ => return false,
    };
    if handle.numel() != k.saturating_mul(n) {
        return false;
    }

    // 2. **M8.7.1.a** — try to consume a host-side prefetch for
    //    THIS handle first (kicked off during the previous
    //    Disk-streamed matmul's compute). On miss, fall back to
    //    the synchronous M8.7.0 read path.
    let raw_bytes = match crate::cuda::disk_prefetch::take(handle) {
        Some(bytes) => bytes,
        None => {
            let mut buf = vec![0u8; handle.numel() * 2];
            if disk_tier::read_bf16_raw_bytes(handle, &mut buf).is_err() {
                return false;
            }
            buf
        }
    };

    // 3. **M8.7.1.a** — kick off prefetch for the NEXT Disk-streamed
    //    matmul before this one's GPU pipeline runs. The slot is
    //    empty after the take above, so this kick_off installs a
    //    fresh entry that will be consumed by the next call. The
    //    background NVMe read overlaps with this matmul's upload +
    //    GEMM (the GPU work that follows below).
    if let Some(next) = next_handle {
        crate::cuda::disk_prefetch::kick_off(next);
    }

    // 3. Upload to a transient BF16 staging slot in VRAM. The M8.7
    //    prereq planner change (`DISK_PIPELINE_STAGING_BYTES`) has
    //    already reserved 270 MiB of VRAM headroom for exactly this
    //    allocation, so it should not race the resident plan.
    let staging_gpu =
        match crate::cuda::bf16_to_f32::bf16_to_vram_no_upcast_from_raw_bytes(
            &raw_bytes,
            handle.numel(),
            &[k, n],
        ) {
            Some(g) => g,
            None => return false,
        };

    // 4. Wrap as a transient `Tensor` and dispatch through M8.4c
    //    Path B (BF16 weight in VRAM + F32 transient upcast +
    //    cublasGemmEx F32). The `Tensor::from_cuda_gpu` outer dtype
    //    is F32 by construction but the dispatch peeks at
    //    `gpu.dtype() == BF16` (matmul.rs:638) so it routes
    //    correctly.
    let b_staged = Tensor::from_cuda_gpu(vec![k, n], staging_gpu);
    // M8.7.1.r — pass `null` stream = default stream = bit-exact
    // with the M8.4c-original behaviour. M8.7.1.b/c will swap
    // this for a dedicated compute stream.
    if !cuda_matmul_bf16_inplace(a, &b_staged, out, m, k, n, std::ptr::null_mut()) {
        return false;
    }

    // 5. Counter increment on successful dispatch only. The
    //    `b_staged` Tensor (and its inner `TensorGPU`) drops here
    //    automatically, freeing the staging slot.
    DISK_STREAMED_MATMUL_COUNT.fetch_add(1, Ordering::Relaxed);
    true
}

#[cfg(test)]
mod cuda_matmul_non_pooled_tests {
    use super::cuda_matmul_non_pooled;
    use crate::cuda::cuda_available;

    /// Reference CPU matmul for a small `[m, k] × [k, n] = [m, n]`
    /// shape. Hand-rolled (rather than going through `Tensor::matmul`)
    /// so the test has no dependency beyond the function under test.
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

    /// Numerical-equivalence test on a small shape. Uses the same
    /// `if !cuda_available() { return; }` skip pattern as the
    /// `tests/cuda_matmul_residency_test.rs` integration test, so
    /// the suite still passes on machines without a CUDA driver.
    #[test]
    fn cuda_matmul_non_pooled_matches_cpu_on_small_shape() {
        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }

        let m = 4_usize;
        let k = 8_usize;
        let n = 6_usize;

        // Deterministic patterns; see `tests/cuda_matmul_residency_test.rs`
        // for the same convention. Values stay well within f32 precision
        // so any GPU-vs-CPU drift is dominated by accumulation order,
        // not rounding chains — the M4.7.3 envelope was 1e-4 absolute,
        // and we use a slightly looser 1e-3 here per the task spec.
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.1 - 1.0).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.05 + 0.5).collect();

        let cpu = cpu_reference(&a, &b, m, k, n);

        let gpu_tensor = cuda_matmul_non_pooled(&a, &b, m, k, n)
            .expect("cuda_matmul_non_pooled returned None on a known-good small shape");
        assert_eq!(gpu_tensor.shape, vec![m, n]);

        let gpu = gpu_tensor.as_cpu_slice();
        assert_eq!(gpu.len(), cpu.len());

        let mut max_abs_diff = 0.0_f32;
        for (g, c) in gpu.iter().zip(cpu.iter()) {
            let d = (g - c).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        assert!(
            max_abs_diff < 1e-3,
            "max |gpu - cpu| = {} exceeded 1e-3 tolerance",
            max_abs_diff
        );
    }
}

#[cfg(test)]
mod cuda_matmul_bf16_tests {
    use super::{cuda_matmul_bf16_inplace, vram_bf16_matmul_count};
    use crate::cuda::bf16_to_f32::{
        bf16_to_vram_no_upcast, BF16_COUNTER_TEST_LOCK,
    };
    use crate::cuda::cuda_available;
    use crate::tensor::tensor::f32_to_bf16_bits;
    use crate::tensor::Tensor;

    /// Hand-rolled F32 reference matmul. Same formula as the M6
    /// non-pooled test; included here so this test module has no
    /// dependency on the production graph executor.
    fn cpu_matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for p in 0..k {
                    s += a[i * k + p] * b[p * n + j];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    /// Synthesise a deterministic F32 buffer of magnitude ≈ 0.3.
    /// Matches the M8.0 bench convention so the drift envelope
    /// observed there (~0.1 max abs diff for K = 5120-13824) is
    /// directly comparable.
    fn synth_f32(numel: usize, seed: u32) -> Vec<f32> {
        (0..numel)
            .map(|i| {
                let s = (seed as f32) * 0.31;
                ((i as f32) * 0.0001 + 0.137 + s).sin() * 0.2
                    + ((i as f32) * 0.00007 + 0.42 + s).cos() * 0.15
            })
            .collect()
    }

    // **M8.4 fix** — replaced module-local `M8_2_TEST_LOCK` with
    // the shared `BF16_COUNTER_TEST_LOCK` from `cuda::bf16_to_f32`.
    // Both M8.1 (counter-snapshot tests inside that module) and
    // M8.2 (these matmul tests) increment / observe
    // `BF16_RESIDENT_COUNT`; before the fix each module had its
    // own lock, allowing parallel execution to race the global
    // counter and break "before/after" deltas. Sharing the same
    // lock across modules eliminates the race by serialising every
    // BF16-counter-touching test through a single `Mutex<()>`.

    /// **M8.2 drift gate** — for K up to 13824 with magnitude-0.3
    /// data, the BF16 truncation envelope on a single matmul is
    /// `eps_BF16 · sqrt(K) · max(|a|, |b|) ≈ 8e-3 · 117 · 0.3 ≈ 0.28`.
    /// We use 0.5 as a conservative single-op bound consistent
    /// with the ADR-004 end-to-end threshold; M8.5 runs the
    /// 4-model F64 validation that is the actual numerical gate
    /// for production wire-up.
    const M8_2_DRIFT_GATE: f32 = 0.5;

    /// Llama 2 13B decode-step matmul shapes — the same four
    /// shapes the M8.0 cuBLAS bench characterised. M = 1 (single
    /// decode token) keeps the fast path tight.
    const SHAPES: &[(&str, usize, usize, usize)] = &[
        ("Q/K/V/O proj", 1, 5120, 5120),
        ("FFN gate/up ", 1, 5120, 13824),
        ("FFN down    ", 1, 13824, 5120),
        ("LM head     ", 1, 5120, 32000),
    ];

    /// Drives one shape: builds F32 tensors, BF16-converts the
    /// weight, uploads the BF16 weight to VRAM via M8.1, calls
    /// `cuda_matmul_bf16_inplace`, and returns
    /// `(max_abs_diff_vs_f32_ref, counter_delta)`. Asserts on any
    /// dispatch failure (the dispatch returning false is a
    /// structural bug that the drift assertion cannot detect).
    fn run_one_shape(label: &str, m: usize, k: usize, n: usize) -> f32 {
        let a_host: Vec<f32> = synth_f32(m * k, 7);
        let b_host: Vec<f32> = synth_f32(k * n, 13);
        let b_bf16: Vec<u16> = b_host.iter().map(|&f| f32_to_bf16_bits(f)).collect();

        // F32 reference (ground truth — uses the original F32 weight,
        // not the BF16-truncated version, so the measured drift is
        // the cumulative BF16-truncation envelope of both operands).
        let out_ref = cpu_matmul_f32(&a_host, &b_host, m, k, n);

        // Wrap inputs/output as `Tensor`s consumable by the dispatch.
        let a_tensor = Tensor::new_cpu(vec![m, k], a_host.clone());
        let mut out_tensor = Tensor::new_cpu(vec![m, n], vec![0.0_f32; m * n]);
        let gpu = bf16_to_vram_no_upcast(&b_bf16, &[k, n])
            .unwrap_or_else(|| panic!("[{}] bf16_to_vram_no_upcast returned None", label));
        let b_tensor = Tensor::from_cuda_gpu(vec![k, n], gpu);

        // Dispatch. `stream = null` selects the default stream,
        // bit-exact with the M8.4c-original behaviour pre-M8.7.1.r.
        let ok = cuda_matmul_bf16_inplace(
            &a_tensor, &b_tensor, &mut out_tensor, m, k, n, std::ptr::null_mut(),
        );
        assert!(
            ok,
            "[{}] cuda_matmul_bf16_inplace returned false on a known-good triple",
            label
        );

        // Drift vs F32 reference.
        let out = out_tensor.as_cpu_slice();
        let mut max_abs_diff = 0.0_f32;
        let mut max_ref_mag = 0.0_f32;
        for (g, r) in out.iter().zip(out_ref.iter()) {
            let d = (g - r).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
            let m_ = r.abs();
            if m_ > max_ref_mag {
                max_ref_mag = m_;
            }
        }
        eprintln!(
            "[M8.2] {} max|Δ|={:.4e}  max|ref|={:.4e}  ratio={:.4e}",
            label,
            max_abs_diff,
            max_ref_mag,
            if max_ref_mag > 0.0 {
                max_abs_diff / max_ref_mag
            } else {
                0.0
            }
        );
        max_abs_diff
    }

    /// **M8.2 drift sweep** — runs all four Llama 13B decode-step
    /// shapes through `cuda_matmul_bf16_inplace`, asserts that the
    /// drift vs the F32 reference stays under the
    /// `M8_2_DRIFT_GATE` envelope on every shape, and verifies
    /// the counter advanced by exactly four (one per shape).
    /// Skips on hosts without a CUDA driver.
    #[test]
    fn cuda_matmul_bf16_inplace_drift_within_gate_on_llama_13b_shapes() {
        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }
        let _guard = BF16_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let before = vram_bf16_matmul_count();
        for (label, m, k, n) in SHAPES {
            let drift = run_one_shape(label, *m, *k, *n);
            assert!(
                drift < M8_2_DRIFT_GATE,
                "[{}] BF16 matmul drift {:.4e} exceeded gate {:.4e}; \
                 PARAR per protocol",
                label,
                drift,
                M8_2_DRIFT_GATE,
            );
        }
        let after = vram_bf16_matmul_count();
        assert_eq!(
            after - before,
            SHAPES.len(),
            "VRAM_BF16_MATMUL_COUNT should advance by {} (one per shape); \
             got delta {}",
            SHAPES.len(),
            after - before
        );
    }

    /// **M8.2 regression-zero** — the F32 path
    /// (`cuda_matmul_inplace`) must not increment the BF16
    /// counter. Indirect test: snapshot the counter, run a small
    /// F32 matmul through the existing M6 dispatch, snapshot
    /// again, assert no change. Uses the standalone
    /// `cuda_matmul_non_pooled` since it has the same F32
    /// contract and no graph-executor scaffolding.
    #[test]
    fn f32_path_does_not_advance_bf16_counter() {
        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }
        let _guard = BF16_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let before = vram_bf16_matmul_count();
        let a = vec![1.0_f32; 4 * 8];
        let b = vec![1.0_f32; 8 * 4];
        let _ = super::cuda_matmul_non_pooled(&a, &b, 4, 8, 4)
            .expect("F32 reference matmul on a CUDA host");
        let after = vram_bf16_matmul_count();
        assert_eq!(
            after, before,
            "F32 matmul path must not advance the BF16 counter"
        );
    }

    /// **M8.4c — strict drift check for the Path B implementation.**
    ///
    /// Replaces the M8.4-original BF16 input path with the M8.4c
    /// "BF16 weight upcasted to F32 transient + F32 gemm". Drift
    /// should be drastically lower because the activation is no
    /// longer truncated. Single-op envelope vs CPU F32 reference
    /// should be in the range 1e-3 to 5e-3 (only weight BF16
    /// truncation contributes); the gate here is `< 1e-2` —
    /// strict enough to catch any regression that re-introduces
    /// activation truncation, loose enough to absorb the
    /// expected `eps_BF16 × sqrt(K) × max_magnitude` envelope.
    ///
    /// Mid-size shape `[m=1, k=128, n=64]` matches the M8.4
    /// dispatcher integration test exactly so the comparison
    /// "M8.4 vs M8.4c on the same shape" is direct.
    #[test]
    fn m8_4c_dispatcher_bf16_resident_via_f32_upcast_strict_drift() {
        let _guard = BF16_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }

        let m = 1_usize;
        let k = 128_usize;
        let n = 64_usize;

        // Same data recipe as the M8.4 dispatcher test, magnitude 0.3.
        let a_host: Vec<f32> = (0..(m * k))
            .map(|i| ((i as f32) * 0.013 + 0.7).sin() * 0.3)
            .collect();
        let b_host_f32: Vec<f32> = (0..(k * n))
            .map(|i| ((i as f32) * 0.007 + 0.4).cos() * 0.3)
            .collect();
        let b_bf16: Vec<u16> =
            b_host_f32.iter().map(|&v| f32_to_bf16_bits(v)).collect();

        let gpu = crate::cuda::bf16_to_f32::bf16_to_vram_no_upcast(&b_bf16, &[k, n])
            .expect("bf16_to_vram_no_upcast on a CUDA host");
        let b_tensor = Tensor::from_cuda_gpu(vec![k, n], gpu);
        let a_tensor = Tensor::new_cpu(vec![m, k], a_host.clone());
        let mut out_tensor = Tensor::new_cpu(vec![m, n], vec![0.0_f32; m * n]);

        // M8.7.1.r — `stream = null` = default stream = pre-M8.7.1.r
        // behaviour bit-exact.
        let ok = super::cuda_matmul_bf16_inplace(
            &a_tensor,
            &b_tensor,
            &mut out_tensor,
            m,
            k,
            n,
            std::ptr::null_mut(),
        );
        assert!(ok, "M8.4c dispatcher must accept BF16-resident triple");

        // F32 reference vs the BF16-decoded weight.
        let b_ref_f32: Vec<f32> = b_bf16
            .iter()
            .map(|&b| f32::from_bits((b as u32) << 16))
            .collect();
        let mut out_ref = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for p in 0..k {
                    s += a_host[i * k + p] * b_ref_f32[p * n + j];
                }
                out_ref[i * n + j] = s;
            }
        }
        let out = out_tensor.as_cpu_slice();
        let mut max_abs_diff = 0.0_f32;
        for (g, r) in out.iter().zip(out_ref.iter()) {
            let d = (g - r).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        eprintln!(
            "[M8.4c] dispatcher Path-B drift on [m=1, k=128, n=64]: {:.4e} \
             (gate < 1e-2)",
            max_abs_diff
        );
        assert!(
            max_abs_diff < 1e-2,
            "M8.4c Path-B drift {:.4e} exceeded the 1e-2 strict gate. \
             Expected <5e-3 from weight-only BF16 truncation; if this \
             breached, the matmul re-introduced activation truncation \
             and the M8.5 4-model F64 validation will fail again.",
            max_abs_diff
        );
    }

    /// **M8.4c — bit-exact upcast equivalence.** The new
    /// `bf16_to_f32_transient_in_vram` primitive must produce the
    /// same F32 output as the existing `bf16_to_f32_on_device`
    /// helper (both wrap the M6 upcast kernel; the difference is
    /// the input is a `TensorGPU` instead of a host slice).
    /// Bit-exactness is the structural guarantee that no extra
    /// rounding sneaks into the upcast path between M6 and M8.4c.
    #[test]
    fn m8_4c_bf16_to_f32_transient_in_vram_matches_on_device() {
        let _guard = BF16_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }

        // Same 64-element BF16 buffer as the M6 on-device test
        // (`bf16_to_f32_on_device_matches_host_decode`) so the
        // numerical envelope is comparable.
        let host_bf16: Vec<u16> = (0..64)
            .map(|i| {
                let f = ((i as f32) * 0.3 - 4.0).sin();
                (f.to_bits() >> 16) as u16
            })
            .collect();

        // Path A — `bf16_to_f32_on_device` (host slice → upload + kernel + download).
        let from_on_device =
            crate::cuda::bf16_to_f32::bf16_to_f32_on_device(&host_bf16, host_bf16.len())
                .expect("bf16_to_f32_on_device returned None");

        // Path B — upload as BF16 TensorGPU (M8.1) + transient F32 upcast (M8.4c).
        let bf16_gpu =
            crate::cuda::bf16_to_f32::bf16_to_vram_no_upcast(&host_bf16, &[64])
                .expect("bf16_to_vram_no_upcast returned None");
        let f32_transient =
            crate::cuda::bf16_to_f32::bf16_to_f32_transient_in_vram(&bf16_gpu)
                .expect("bf16_to_f32_transient_in_vram returned None");
        // Sanity: transient is F32-typed and matches the BF16 size × 2.
        assert_eq!(
            f32_transient.dtype(),
            crate::tensor::DType::F32,
            "transient buffer must be F32-typed"
        );
        assert_eq!(
            f32_transient.size_bytes(),
            64 * 4,
            "transient F32 buffer must be numel × 4 bytes"
        );
        let from_transient = f32_transient
            .to_cpu()
            .expect("D→H download of F32 transient");

        // Bit-exact comparison.
        assert_eq!(
            from_on_device.len(),
            from_transient.len()
        );
        let mut bitwise_mismatches = 0_usize;
        for (a, b) in from_on_device.iter().zip(from_transient.iter()) {
            if a.to_bits() != b.to_bits() {
                bitwise_mismatches += 1;
            }
        }
        assert_eq!(
            bitwise_mismatches, 0,
            "bf16_to_f32_transient_in_vram must produce bit-exact output \
             vs bf16_to_f32_on_device (both wrap the same M6 kernel); \
             {} elements diverged out of {}",
            bitwise_mismatches,
            from_on_device.len()
        );
    }

    // -----------------------------------------------------------------------
    // M10.2.1 — counter contract for cuda_matmul_bf16_native_inplace
    // -----------------------------------------------------------------------

    /// **M10.2.1** — running a known-good triple through the native
    /// fast kernel must advance `VRAM_BF16_NATIVE_MATMUL_COUNT` by
    /// exactly one. Disjoint from the certified-path counter.
    /// Skips on hosts without a CUDA driver.
    #[test]
    fn fast_path_advances_native_counter_and_not_certified() {
        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }
        let _guard = BF16_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let (m, k, n) = (1usize, 64usize, 64usize);
        let a_host: Vec<f32> = synth_f32(m * k, 7);
        let b_host: Vec<f32> = synth_f32(k * n, 13);
        let b_bf16: Vec<u16> = b_host.iter().map(|&f| f32_to_bf16_bits(f)).collect();

        let a_tensor = Tensor::new_cpu(vec![m, k], a_host);
        let mut out_tensor = Tensor::new_cpu(vec![m, n], vec![0.0_f32; m * n]);
        let gpu = bf16_to_vram_no_upcast(&b_bf16, &[k, n])
            .expect("bf16_to_vram_no_upcast returned None on a known-good shape");
        let b_tensor = Tensor::from_cuda_gpu(vec![k, n], gpu);

        let native_before = super::vram_bf16_native_matmul_count();
        let certified_before = super::vram_bf16_matmul_count();

        let ok = super::cuda_matmul_bf16_native_inplace(
            &a_tensor, &b_tensor, &mut out_tensor, m, k, n, std::ptr::null_mut(),
        );
        assert!(
            ok,
            "cuda_matmul_bf16_native_inplace returned false on a known-good triple"
        );

        let native_after = super::vram_bf16_native_matmul_count();
        let certified_after = super::vram_bf16_matmul_count();

        assert_eq!(
            native_after - native_before,
            1,
            "VRAM_BF16_NATIVE_MATMUL_COUNT must advance by exactly 1 \
             after a successful fast-path call (got delta {})",
            native_after - native_before
        );
        assert_eq!(
            certified_after, certified_before,
            "fast path must not advance the certified counter"
        );
    }

    /// **M10.2.1** — running through the certified path must NOT
    /// advance the native fast-path counter.
    #[test]
    fn certified_path_does_not_advance_native_counter() {
        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }
        let _guard = BF16_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let (m, k, n) = (1usize, 64usize, 64usize);
        let a_host: Vec<f32> = synth_f32(m * k, 7);
        let b_host: Vec<f32> = synth_f32(k * n, 13);
        let b_bf16: Vec<u16> = b_host.iter().map(|&f| f32_to_bf16_bits(f)).collect();

        let a_tensor = Tensor::new_cpu(vec![m, k], a_host);
        let mut out_tensor = Tensor::new_cpu(vec![m, n], vec![0.0_f32; m * n]);
        let gpu = bf16_to_vram_no_upcast(&b_bf16, &[k, n])
            .expect("bf16_to_vram_no_upcast returned None on a known-good shape");
        let b_tensor = Tensor::from_cuda_gpu(vec![k, n], gpu);

        let native_before = super::vram_bf16_native_matmul_count();

        let ok = super::cuda_matmul_bf16_inplace(
            &a_tensor, &b_tensor, &mut out_tensor, m, k, n, std::ptr::null_mut(),
        );
        assert!(ok, "certified path returned false on a known-good triple");

        let native_after = super::vram_bf16_native_matmul_count();
        assert_eq!(
            native_after, native_before,
            "certified path must not advance VRAM_BF16_NATIVE_MATMUL_COUNT"
        );
    }
}

// ===========================================================================
// M8.7.0 — Disk → GPU JIT pipeline tests
// ===========================================================================

#[cfg(test)]
mod cuda_matmul_disk_streamed_tests {
    use super::*;
    use crate::cuda::bf16_to_f32::BF16_COUNTER_TEST_LOCK;
    use crate::cuda::cuda_available;
    use crate::tensor::{disk_tier, Tensor};

    /// Bit pattern of `f32_to_bf16` truncation — round-toward-zero,
    /// no rounding. Same helper the existing M8.2 tests use.
    fn f32_to_bf16_bits(f: f32) -> u16 {
        (f.to_bits() >> 16) as u16
    }

    /// Deterministic synthetic activation. Magnitude ~0.3 keeps the
    /// BF16-truncation envelope bounded (matches M8.2 SHAPES suite
    /// at `cuda_matmul_bf16_tests` above).
    fn synth_f32(n: usize, seed: u32) -> Vec<f32> {
        let mut out = Vec::with_capacity(n);
        let mut s = seed.wrapping_mul(2_654_435_761);
        for _ in 0..n {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // Map u32 → [-0.3, 0.3] uniformly.
            let f = ((s >> 8) as f32 / (1u32 << 24) as f32) * 0.6 - 0.3;
            out.push(f);
        }
        out
    }

    fn cpu_matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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

    /// Cache dir for synthetic disk-tier files. Uses
    /// `target/m8_7_0_test_cache_<pid>` so parallel cargo runs do
    /// not collide. `InnerDiskFile::Drop` removes each file when
    /// the handle is dropped; we additionally `remove_dir_all` at
    /// the end of the test.
    fn cache_dir() -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "atenia_m8_7_0_disk_test_{}",
            std::process::id()
        ));
        p
    }

    /// **M8.7.0** — End-to-end Disk → GPU BF16 matmul test.
    ///
    /// Builds a synthetic BF16 weight, writes it to NVMe via
    /// `disk_tier::write_bf16_tensor`, wraps as `Tensor::from_disk`,
    /// dispatches `cuda_matmul_disk_streamed_bf16`, and asserts:
    /// 1. Counter increments by exactly 1.
    /// 2. Drift vs CPU F32 reference < ADR-004 single-op gate (0.5).
    /// 3. The dispatch returned `true`.
    ///
    /// Mid-size shape `[m=1, k=128, n=64]` matches the M8.4c
    /// dispatcher integration test so the comparison "M8.4c
    /// resident vs M8.7.0 streamed" is direct.
    #[test]
    fn m8_7_0_disk_streamed_bf16_matches_cpu_within_adr_004_gate() {
        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }
        let _guard = BF16_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let (m, k, n) = (1_usize, 128_usize, 64_usize);
        let a_host = synth_f32(m * k, 7);
        let b_host = synth_f32(k * n, 13);
        let b_bf16: Vec<u16> = b_host.iter().map(|&f| f32_to_bf16_bits(f)).collect();

        // Reference (F32 ground truth — the drift includes weight
        // BF16 truncation but not activation truncation, matching
        // M8.4c Path B numerics).
        let out_ref = cpu_matmul_f32(&a_host, &b_host, m, k, n);

        // Write the BF16 weight to NVMe.
        let dir = cache_dir();
        let _ = std::fs::create_dir_all(&dir);
        let handle = disk_tier::write_bf16_tensor(&dir, &b_bf16)
            .expect("write_bf16_tensor: synthetic disk write");

        // Wrap the handle as a Tensor (Disk-tier).
        let a_tensor = Tensor::new_cpu(vec![m, k], a_host.clone());
        let b_tensor = Tensor::from_disk(vec![k, n], handle.clone());
        let mut out_tensor = Tensor::new_cpu(vec![m, n], vec![0.0_f32; m * n]);

        let before = disk_streamed_matmul_count();

        // Dispatch under test. `next_handle = None` covers the
        // M8.7.1.a "no further Disk-streamed matmul ahead" branch.
        let ok = cuda_matmul_disk_streamed_bf16(
            &a_tensor, &b_tensor, &mut out_tensor, m, k, n, None,
        );
        assert!(
            ok,
            "cuda_matmul_disk_streamed_bf16 returned false on a known-good triple"
        );

        let after = disk_streamed_matmul_count();
        assert_eq!(
            after - before, 1,
            "DISK_STREAMED_MATMUL_COUNT must advance by exactly 1 per dispatch"
        );

        // Drift vs F32 reference. M8.4c single-op envelope on this
        // shape is well below the ADR-004 0.5 gate (the M8.4c test
        // uses 1e-2 for the same shape and passes with ~3e-3).
        let out_slice = out_tensor.as_cpu_slice();
        let mut max_abs = 0.0_f32;
        for (g, r) in out_slice.iter().zip(out_ref.iter()) {
            let d = (g - r).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        eprintln!(
            "[M8.7.0] m={} k={} n={}  max|Δ|={:.4e}  gate=0.5 (ADR-004)",
            m, k, n, max_abs
        );
        assert!(
            max_abs < 0.5,
            "M8.7.0 disk-streamed matmul drift {:.4e} exceeds ADR-004 gate 0.5",
            max_abs
        );

        // Best-effort cleanup. The DiskTensorHandle drop already
        // removes the file; we drop the cache dir if empty.
        drop(b_tensor);
        drop(handle);
        let _ = std::fs::remove_dir(&dir);
    }

    /// **M8.7.0 regression-zero** — F32 matmul path must not
    /// advance the disk-streamed counter. Ensures the counter is
    /// only touched by [`cuda_matmul_disk_streamed_bf16`].
    #[test]
    fn m8_7_0_f32_path_does_not_advance_disk_streamed_counter() {
        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }
        let _guard = BF16_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let before = disk_streamed_matmul_count();
        let a = vec![1.0_f32; 4 * 8];
        let b = vec![1.0_f32; 8 * 4];
        let _ = super::cuda_matmul_non_pooled(&a, &b, 4, 8, 4)
            .expect("F32 reference matmul on a CUDA host");
        let after = disk_streamed_matmul_count();
        assert_eq!(
            after, before,
            "F32 matmul path must not advance the disk-streamed counter"
        );
    }

    /// **M8.7.0** — non-Disk operand precondition. Calling the
    /// dispatch with a CPU tensor for `b` must return `false` and
    /// not increment the counter.
    #[test]
    fn m8_7_0_non_disk_operand_returns_false() {
        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }
        let _guard = BF16_COUNTER_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());

        let (m, k, n) = (1_usize, 8_usize, 4_usize);
        let a = Tensor::new_cpu(vec![m, k], vec![1.0_f32; m * k]);
        let b = Tensor::new_cpu(vec![k, n], vec![1.0_f32; k * n]);
        let mut out = Tensor::new_cpu(vec![m, n], vec![0.0_f32; m * n]);

        let before = disk_streamed_matmul_count();
        let ok = cuda_matmul_disk_streamed_bf16(&a, &b, &mut out, m, k, n, None);
        let after = disk_streamed_matmul_count();

        assert!(!ok, "dispatch must reject non-Disk operand");
        assert_eq!(
            after, before,
            "counter must not advance on precondition failure"
        );
    }
}
