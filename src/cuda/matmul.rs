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

// **M6.b** — `cudaMemcpy` re-binding for the non-pooled
// matmul path. The same FFI is exported from `pool_helpers`
// behind a private visibility; re-declaring here keeps the
// non-pooled call site self-contained without widening the
// pool helper's surface.
#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;
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
/// **M6.b** — non-pooled CUDA matmul (no 64 MiB ceiling).
///
/// Direct `cudaMalloc` / `cudaMemcpy` / kernel launch /
/// `cudaMemcpy` / `cudaFree` per call. Bypasses
/// [`crate::apx4_12::pool_alloc`] entirely so shapes whose
/// largest operand exceeds [`crate::apx4_12::DEFAULT_BLOCK_SIZE`]
/// (the M4.7 decision 34 ceiling) can still reach CUDA.
///
/// The Llama 2 13B Chat decode-step matmuls FFN gate/up
/// (`B = 270 MB` F32), FFN down (same), and lm_head
/// (`B = 655 MB` F32) all need this path; the QKVO
/// projections (`B = 100 MB` F32) too. Only the small
/// attention BMM cells (`< 1 MB` per call) fit the pool —
/// and those don't go through this entrypoint anyway
/// (`NodeType::BatchMatMul` has its own dispatch).
///
/// Costs vs the pooled path:
/// - Per call: one `cudaMalloc`/`cudaFree` cycle for each
///   of A, B, output. CUDA driver overhead ~10-50 μs each;
///   negligible vs the per-matmul compute on FFN-class
///   shapes.
/// - No pre-allocated VRAM commitment — VRAM is consumed
///   only for the duration of the call.
///
/// Falls back to a CPU result (zero-init `out`) and logs
/// to stderr if any CUDA driver call fails. Pre-condition:
/// `a` and `b` are CPU-resident F32 tensors of the right
/// shapes.
///
/// # Panics
/// Never. Failure paths emit a stderr trace and return a
/// CPU-zero output that the caller's `try_gpu_matmul`
/// dispatch will recognise as a failure (it asserts the
/// `ran_gpu` flag the caller sets).
pub fn cuda_matmul_non_pooled(
    a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize,
) -> Option<Tensor> {
    use crate::apx4_12::{cuda_free_raw, cuda_malloc_raw};
    use std::ffi::c_void;

    let a_slice = a.as_cpu_slice();
    let b_slice = b.as_cpu_slice();
    let a_bytes = m * k * std::mem::size_of::<f32>();
    let b_bytes = k * n * std::mem::size_of::<f32>();
    let out_bytes = m * n * std::mem::size_of::<f32>();

    // SAFETY: The FFI surface is straightforward — alloc
    // returns null on failure; cudaMemcpy returns 0 on
    // success. We pair every successful alloc with exactly
    // one free via early-return guards.
    unsafe {
        let d_a = cuda_malloc_raw(a_bytes);
        if d_a.is_null() {
            eprintln!("[M6.b] cuda_matmul_non_pooled: cudaMalloc(A, {} B) returned null", a_bytes);
            return None;
        }
        let d_b = cuda_malloc_raw(b_bytes);
        if d_b.is_null() {
            cuda_free_raw(d_a);
            eprintln!("[M6.b] cuda_matmul_non_pooled: cudaMalloc(B, {} B) returned null", b_bytes);
            return None;
        }
        let d_out = cuda_malloc_raw(out_bytes);
        if d_out.is_null() {
            cuda_free_raw(d_a);
            cuda_free_raw(d_b);
            eprintln!("[M6.b] cuda_matmul_non_pooled: cudaMalloc(out, {} B) returned null", out_bytes);
            return None;
        }

        // Host → Device.
        let r1 = cudaMemcpy(d_a, a_slice.as_ptr() as *const c_void, a_bytes, CUDA_MEMCPY_HOST_TO_DEVICE);
        let r2 = cudaMemcpy(d_b, b_slice.as_ptr() as *const c_void, b_bytes, CUDA_MEMCPY_HOST_TO_DEVICE);
        if r1 != 0 || r2 != 0 {
            cuda_free_raw(d_a);
            cuda_free_raw(d_b);
            cuda_free_raw(d_out);
            eprintln!("[M6.b] cuda_matmul_non_pooled: H→D cudaMemcpy failed (r1={r1}, r2={r2})");
            return None;
        }

        // Kernel.
        matmul_f32_launch_device(
            d_a as *const f32, d_b as *const f32, d_out as *mut f32,
            m as c_int, k as c_int, n as c_int,
        );

        // Device → Host.
        let mut out_data = vec![0.0_f32; m * n];
        let r3 = cudaMemcpy(
            out_data.as_mut_ptr() as *mut c_void,
            d_out as *const c_void,
            out_bytes,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        );
        cuda_free_raw(d_a);
        cuda_free_raw(d_b);
        cuda_free_raw(d_out);
        if r3 != 0 {
            eprintln!("[M6.b] cuda_matmul_non_pooled: D→H cudaMemcpy failed (r3={r3})");
            return None;
        }

        Some(Tensor::new_cpu(vec![m, n], out_data))
    }
}

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
