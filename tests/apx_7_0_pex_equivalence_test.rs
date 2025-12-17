use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::config::get_runtime_flags;

#[test]
fn apx_7_0_pex_matches_seq_in_6_3_mode() {
    // Forzar modo 6.3 para que el dispatcher use la ruta 6.3b
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "6.3");
    }

    let a = Tensor::randn(&[128, 128], Device::CPU);
    let b = Tensor::randn(&[128, 128], Device::CPU);

    // Asegurar que PEX está desactivado para la ruta secuencial
    {
        let mut flags = get_runtime_flags();
        flags.enable_pex = false;
    }

    // Versión secuencial (dispatcher 6.3b clásico)
    let seq = a.matmul(&b);

    // Versión paralela (activa PEX y, en modo 6.3, usa matmul_tiled_6_3b_pex)
    let par = a.matmul_parallel(&b);

    // Comparar máximo error absoluto elemento a elemento.
    let mut max_diff = 0.0f32;
    for (x, y) in seq.data.iter().zip(par.data.iter()) {
        let d = (x - y).abs();
        if d > max_diff {
            max_diff = d;
        }
    }

    assert!(max_diff < 1e-5, "max diff = {}", max_diff);
}
