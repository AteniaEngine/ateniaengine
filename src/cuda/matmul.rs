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

// FFI directo a cudart para copiar entre host y device.
#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1; // cudaMemcpyHostToDevice
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2; // cudaMemcpyDeviceToHost

pub fn cuda_matmul(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Tensor {
    // Por ahora tratamos los buffers como host; el Device es sólo lógico.
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
        // Reservar buffers de device desde el pool global (APX 4.12).
        let d_a = pool_alloc() as *mut f32;
        let d_b = pool_alloc() as *mut f32;
        let d_c = pool_alloc() as *mut f32;

        // Copiar host → device.
        let err = cudaMemcpy(
            d_a as *mut c_void,
            a.data.as_ptr() as *const c_void,
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
            b.data.as_ptr() as *const c_void,
            size_b_bytes,
            CUDA_MEMCPY_HOST_TO_DEVICE,
        );
        if err != 0 {
            pool_free(d_a.cast());
            pool_free(d_b.cast());
            pool_free(d_c.cast());
            panic!("cudaMemcpy H2D for B failed with code {}", err);
        }

        // Lanzar kernel sobre los buffers de device ya reservados.
        matmul_f32_launch_device(d_a, d_b, d_c, m as c_int, k as c_int, n as c_int);

        let err = cudaDeviceSynchronize();
        if err != 0 {
            pool_free(d_a.cast());
            pool_free(d_b.cast());
            pool_free(d_c.cast());
            panic!("cudaDeviceSynchronize after matmul failed with code {}", err);
        }

        // Copiar device → host.
        let err = cudaMemcpy(
            out.data.as_mut_ptr() as *mut c_void,
            d_c as *const c_void,
            size_c_bytes,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        );

        // Liberar siempre los buffers del pool antes de propagar cualquier error.
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
