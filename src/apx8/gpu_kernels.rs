// APX 8.6 — GPU Kernels v0 (simulation)
//
// Simulated GPU mini-kernels (VecAdd). Does NOT touch real GPU. The
// pre-Debt-#2 implementation also called into the legacy APX 8.4
// `GPUMirror` metadata layer (`ensure_gpu_mirror`, `mark_gpu_dirty`)
// to track "GPU freshness" as metadata without any real device path.
// The mirror was a pure metadata stub with no `device_ptr` and no
// `Drop`; after the introduction of `TensorStorage::Cuda` in M3-d
// the mirror became redundant and was removed entirely as part of
// Debt #2 cleanup. This kernel now computes the CPU sum directly
// with no metadata side-effects.

use crate::tensor::Tensor;

/// Simulation of a GPU vector-add kernel. Runs on CPU data; the
/// legacy GPU-mirror metadata tracking was removed with Debt #2.
pub fn gpu_vec_add(a: &mut Tensor, b: &Tensor) {
    assert_eq!(a.shape, b.shape, "gpu_vec_add: shape mismatch");

    let b_slice = b.as_cpu_slice();
    for (va, vb) in a.as_cpu_slice_mut().iter_mut().zip(b_slice.iter()) {
        *va += *vb;
    }
}
