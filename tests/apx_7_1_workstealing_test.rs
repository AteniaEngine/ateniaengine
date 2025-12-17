use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::config::get_runtime_flags;

#[test]
fn apx_7_1_ws_matmul_matches_seq() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "6.3");
    }

    let a = Tensor::randn(&[128, 128], Device::CPU);
    let b = Tensor::randn(&[128, 128], Device::CPU);

    // Forzar flags a un estado conocido
    {
        let mut flags = get_runtime_flags();
        flags.enable_pex = false;
        flags.enable_workstealing = false;
    }

    let out_seq = a.matmul(&b);
    let out_ws = a.matmul_parallel_ws(&b);

    let mut max_diff = 0.0f32;
    for (x, y) in out_seq.data.iter().zip(out_ws.data.iter()) {
        let d = (x - y).abs();
        if d > max_diff {
            max_diff = d;
        }
    }

    assert!(max_diff < 1e-6, "WS must match sequential, max diff = {}", max_diff);
}
