use crate::gpu::{
    ops::{matmul::MatMulOp, vec_add::VecAddOp},
    memory::GpuPtr,
};

pub struct LinearOp;

impl LinearOp {
    /// Performs: out = x @ W^T + b
    /// x: [M,K]
    /// W: [N,K]
    /// b: [N]
    pub fn run(
        x: &GpuPtr,
        w: &GpuPtr,
        b: &GpuPtr,
        out: &GpuPtr,
        m: usize,
        k: usize,
        n: usize,
    ) {
        // 1) MatMul:  [M,K] x [K,N] => [M,N]
        MatMulOp::run(x, w, out, m, k, n);

        // 2) Add bias (broadcast over rows)
        // bias repeat M times
        for row in 0..m {
            let row_offset = row * n;

            let row_ptr = GpuPtr {
                ptr: out.ptr + (row_offset * 4) as u64,
                size: n * 4,
            };

            VecAddOp::run(&row_ptr, b, &row_ptr, n);
        }
    }
}
