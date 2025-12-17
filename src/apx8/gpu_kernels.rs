// APX 8.6 — GPU Kernels v0
// Mini-kernels GPU simulados (VecAdd) totalmente encapsulados y seguros.
// No tocan backward ni kernels CPU ni matemática crítica.

use crate::tensor::Tensor;

/// Simulación segura de un kernel GPU de suma vectorial.
/// Opera sobre los datos CPU pero marca el mirror GPU como "dirty" para
/// integrarse con la capa de mirroring/persistencia.
pub fn gpu_vec_add(a: &mut Tensor, b: &Tensor) {
    assert_eq!(a.shape, b.shape, "gpu_vec_add: shape mismatch");

    // Simular que ambos viven en GPU creando mirrors si es necesario.
    a.ensure_gpu_mirror();
    // b es sólo lectura; un mirror opcional no altera su semántica.
    let mut b_clone = b.clone();
    b_clone.ensure_gpu_mirror();

    // Kernel simulado: suma sobre los buffers CPU.
    for (va, vb) in a.data.iter_mut().zip(b.data.iter()) {
        *va += *vb;
    }

    // Marcar GPU como "dirty" para indicar que el último escritor es GPU.
    a.mark_gpu_dirty();
}
