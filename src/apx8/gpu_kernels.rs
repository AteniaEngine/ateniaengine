// APX 8.6 — GPU Kernels v0
// Fully encapsulated and safe simulated GPU mini-kernels (VecAdd).
// Do not touch backward, CPU kernels, nor critical math.

use crate::tensor::Tensor;

/// Safe simulation of a GPU vector-add kernel.
/// Operates on CPU data but marks the GPU mirror as "dirty" to integrate
/// with the mirroring/persistence layer.
pub fn gpu_vec_add(a: &mut Tensor, b: &Tensor) {
    assert_eq!(a.shape, b.shape, "gpu_vec_add: shape mismatch");

    // Simulate that both live on GPU by creating mirrors if needed.
    a.ensure_gpu_mirror();
    // b is read-only; an optional mirror does not alter its semantics.
    let mut b_clone = b.clone();
    b_clone.ensure_gpu_mirror();

    // Simulated kernel: sum over CPU buffers.
    for (va, vb) in a.data.iter_mut().zip(b.data.iter()) {
        *va += *vb;
    }

    // Mark GPU as "dirty" to indicate the last writer is GPU.
    a.mark_gpu_dirty();
}
