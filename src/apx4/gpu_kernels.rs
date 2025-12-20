use crate::cuda::matmul::cuda_matmul;
use crate::tensor::{DType, Device, Layout, Tensor};

pub fn gpu_matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    out: &mut [f32],
) {
    // Build temporary CPU tensors to reuse the existing cuda_matmul path.
    // This does not change MatMul math; it only delegates computation to the
    // CUDA kernel when available.

    // Tensor A with shape [m, k]
    let shape_a = vec![m, k];
    let strides_a = Tensor::compute_strides(&shape_a, &Layout::Contiguous);
    let ta = Tensor {
        shape: shape_a,
        data: a.to_vec(),
        device: Device::CPU,
        dtype: DType::F32,
        layout: Layout::Contiguous,
        strides: strides_a,
        grad: None,
        gpu: None,
        persistence: None,
        op: None,
    };

    // Tensor B with shape [k, n]
    let shape_b = vec![k, n];
    let strides_b = Tensor::compute_strides(&shape_b, &Layout::Contiguous);
    let tb = Tensor {
        shape: shape_b,
        data: b.to_vec(),
        device: Device::CPU,
        dtype: DType::F32,
        layout: Layout::Contiguous,
        strides: strides_b,
        grad: None,
        gpu: None,
        persistence: None,
        op: None,
    };

    let tc = cuda_matmul(&ta, &tb, m, k, n);

    // Copy the result into the flat output buffer.
    assert_eq!(tc.data.len(), out.len(), "gpu_matmul: unexpected output size");
    out.copy_from_slice(&tc.data);
}

pub fn gpu_add(_a: &[f32], _b: &[f32], _out: &mut [f32]) {
    unimplemented!("CUDA add not implemented yet (APX 4.0)");
}
