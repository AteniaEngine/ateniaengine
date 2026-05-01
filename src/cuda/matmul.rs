use std::ffi::c_void;
use std::os::raw::c_int;

use crate::amg::nodes::NodeType;
use crate::cuda::pool_helpers::with_pooled_device_buffers;
use crate::cuda::{cuda_device_ptr, cuda_device_ptr_mut};
use crate::tensor::{Device, Tensor, TensorStorage};

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

    // Guard: refuse to issue a CUDA kernel call when the driver is
    // not available. Without this check, `cuda_malloc` (a no-op stub
    // on non-CUDA hosts) would still return non-null garbage and the
    // kernel launch would blow up. Mirrors the gate in
    // `gpu_can_run_matmul`.
    if !super::cuda_available() {
        return None;
    }

    let bytes_a = a.len() * mem::size_of::<f32>();
    let bytes_b = b.len() * mem::size_of::<f32>();
    let out_len = m * n;
    let bytes_out = out_len * mem::size_of::<f32>();

    let mut allocs = NonPooledAllocs::new();
    let d_a = allocs.alloc(bytes_a)?;
    let d_b = allocs.alloc(bytes_b)?;
    let d_out = allocs.alloc(bytes_out)?;

    // H→D: stage both inputs into VRAM. On any failure the RAII
    // guard frees every buffer allocated so far when this function
    // returns.
    unsafe {
        let rc = cudaMemcpy(
            d_a,
            a.as_ptr() as *const c_void,
            bytes_a,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        if rc != 0 {
            return None;
        }
        let rc = cudaMemcpy(
            d_b,
            b.as_ptr() as *const c_void,
            bytes_b,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        if rc != 0 {
            return None;
        }

        // Kernel launch. `matmul_f32_launch_device` returns void
        // and syncs internally; no error code is propagated, so we
        // assume success consistent with `cuda_matmul`.
        matmul_f32_launch_device(
            d_a as *const f32,
            d_b as *const f32,
            d_out as *mut f32,
            m as c_int,
            k as c_int,
            n as c_int,
        );

        // D→H: pull the output back into a fresh host Vec. Allocated
        // here (rather than into a pre-built `Tensor::zeros_new`) so
        // the function has a single ownership chain that is easy to
        // reason about on the error paths.
        let mut out_host = vec![0.0_f32; out_len];
        let rc = cudaMemcpy(
            out_host.as_mut_ptr() as *mut c_void,
            d_out,
            bytes_out,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        );
        if rc != 0 {
            return None;
        }

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
