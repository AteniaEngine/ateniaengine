use crate::apx4::gpu_context::gpu_available;
use crate::apx4::gpu_kernels::gpu_matmul;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApxExecTarget {
    CPU,
    GPU,
    Auto,
}

pub fn dispatch_matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
    target: ApxExecTarget,
) {
    match target {
        ApxExecTarget::GPU => {
            if gpu_available() {
                gpu_matmul(a, b, m, k, n, out);
            } else {
                crate::matmul_dispatcher::matmul_dispatch(a, b, out, m, k, n);
            }
        }
        ApxExecTarget::CPU => {
            crate::matmul_dispatcher::matmul_dispatch(a, b, out, m, k, n);
        }
        ApxExecTarget::Auto => {
            if gpu_available() {
                gpu_matmul(a, b, m, k, n, out);
            } else {
                crate::matmul_dispatcher::matmul_dispatch(a, b, out, m, k, n);
            }
        }
    }
}
