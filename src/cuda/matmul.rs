use std::ffi::c_void;
use std::os::raw::c_int;

use crate::amg::nodes::NodeType;
use crate::apx4_12::{pool_alloc, pool_free};
use crate::tensor::{Device, Tensor};

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

// Direct FFI to cudart to copy between host and device.
#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1; // cudaMemcpyHostToDevice
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2; // cudaMemcpyDeviceToHost

pub fn cuda_matmul(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
    // For now we treat the buffers as host; the Device is only logical.
    let mut out = Tensor::zeros_new(&[m, n], Device::CPU);

    let size_a_bytes = m
        .checked_mul(k)
        .expect("matmul: overflow in M*K")
        * std::mem::size_of::<f32>();
    let size_b_bytes = k
        .checked_mul(n)
        .expect("matmul: overflow in K*N")
        * std::mem::size_of::<f32>();
    let size_c_bytes = m
        .checked_mul(n)
        .expect("matmul: overflow in M*N")
        * std::mem::size_of::<f32>();

    unsafe {
        // Allocate device buffers from the global pool (APX 4.12).
        //
        // `pool_alloc` returns null when the pool is exhausted or
        // fragmented enough that no block can serve the request. Before
        // this null-check was added, the null pointer flowed into
        // `cudaMemcpy` below and produced a cryptic
        // `cudaErrorInvalidDevicePointer` (code 11) that did not point
        // at the real cause. We now fail fast with an actionable
        // message and free any partial allocation first so the pool
        // can recover.
        //
        // The panic is consistent with the rest of this function, which
        // already panics on `cudaMemcpy` / `cudaDeviceSynchronize`
        // errors. Debt #3 Fase 3.2 will migrate this whole alloc/copy
        // cycle into a shared Rust helper with uniform error handling.
        let d_a = pool_alloc() as *mut f32;
        if d_a.is_null() {
            panic!(
                "cuda_matmul: pool_alloc returned null for buffer A \
                 ({} bytes) — pool exhausted or fragmented",
                size_a_bytes
            );
        }
        let d_b = pool_alloc() as *mut f32;
        if d_b.is_null() {
            pool_free(d_a.cast());
            panic!(
                "cuda_matmul: pool_alloc returned null for buffer B \
                 ({} bytes) — pool exhausted or fragmented",
                size_b_bytes
            );
        }
        let d_c = pool_alloc() as *mut f32;
        if d_c.is_null() {
            pool_free(d_a.cast());
            pool_free(d_b.cast());
            panic!(
                "cuda_matmul: pool_alloc returned null for buffer C \
                 ({} bytes) — pool exhausted or fragmented",
                size_c_bytes
            );
        }

        // Copy host -> device.
        // M3-a: `as_cpu_slice()` panics if the storage is not CPU-resident.
        // When M3-d adds `TensorStorage::Cuda`, callers of `cuda_matmul` must
        // either pass already-GPU tensors (skipping H2D) or call `ensure_cpu`
        // upstream; the current shape of this function assumes CPU-resident
        // inputs.
        let err = cudaMemcpy(
            d_a as *mut c_void,
            a.as_cpu_slice().as_ptr() as *const c_void,
            size_a_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        if err != 0 {
            pool_free(d_a.cast());
            pool_free(d_b.cast());
            pool_free(d_c.cast());
            panic!("cudaMemcpy H2D for A failed with code {}", err);
        }

        let err = cudaMemcpy(
            d_b as *mut c_void,
            b.as_cpu_slice().as_ptr() as *const c_void,
            size_b_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        if err != 0 {
            pool_free(d_a.cast());
            pool_free(d_b.cast());
            pool_free(d_c.cast());
            panic!("cudaMemcpy H2D for B failed with code {}", err);
        }

        // Launch kernel over the already allocated device buffers.
        matmul_f32_launch_device(d_a, d_b, d_c, m as c_int, k as c_int, n as c_int);

        let err = cudaDeviceSynchronize();
        if err != 0 {
            pool_free(d_a.cast());
            pool_free(d_b.cast());
            pool_free(d_c.cast());
            panic!("cudaDeviceSynchronize after matmul failed with code {}", err);
        }

        // Copy device -> host.
        let err = cudaMemcpy(
            out.as_cpu_slice_mut().as_mut_ptr() as *mut c_void,
            d_c as *const c_void,
            size_c_bytes,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        );

        // Always free the pool buffers before propagating any error.
        pool_free(d_a.cast());
        pool_free(d_b.cast());
        pool_free(d_c.cast());

        if err != 0 {
            panic!("cudaMemcpy D2H for C failed with code {}", err);
        }
    }

    out
}

pub fn is_cuda_available_for(node_type: &NodeType) -> bool {
    matches!(
        node_type,
        NodeType::MatMul | NodeType::BatchMatMul | NodeType::Linear
    )
}
