use crate::gpu::{
    ops::matmul::MatMulOp,
    memory::GpuPtr,
};

/// BatchMatMul wrapper.
/// A: [B,H,M,K]
/// B: [B,H,K,N]
/// out: [B,H,M,N]
pub struct BatchMatMulOp;

impl BatchMatMulOp {
    pub fn run(
        a: &GpuPtr,
        b: &GpuPtr,
        out: &GpuPtr,
        batch: usize,
        heads: usize,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let chunk_a = m * k;
        let chunk_b = k * n;
        let chunk_out = m * n;

        for b_i in 0..batch {
            for h_i in 0..heads {
                let idx = b_i * heads + h_i;

                let a_ptr = GpuPtr {
                    ptr: a.ptr + (idx * chunk_a * 4) as u64,
                    size: chunk_a * 4,
                };

                let b_ptr = GpuPtr {
                    ptr: b.ptr + (idx * chunk_b * 4) as u64,
                    size: chunk_b * 4,
                };

                let out_ptr = GpuPtr {
                    ptr: out.ptr + (idx * chunk_out * 4) as u64,
                    size: chunk_out * 4,
                };

                MatMulOp::run(&a_ptr, &b_ptr, &out_ptr, m, k, n);
            }
        }
    }
}
