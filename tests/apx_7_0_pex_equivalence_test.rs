use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::config::get_runtime_flags;

#[test]
fn apx_7_0_pex_matches_seq_in_6_3_mode() {
    // Force mode 6.3 so the dispatcher uses the 6.3b path.
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "6.3");
    }

    let a = Tensor::randn(&[128, 128], Device::CPU);
    let b = Tensor::randn(&[128, 128], Device::CPU);

    // Ensure PEX is disabled for the sequential path.
    {
        let mut flags = get_runtime_flags();
        flags.enable_pex = false;
    }

    // Sequential version (classic 6.3b dispatcher)
    let seq = a.matmul(&b);

    // Parallel version (enables PEX and, in mode 6.3, uses matmul_tiled_6_3b_pex)
    let par = a.matmul_parallel(&b);

    // Compare maximum absolute element-wise error.
    let mut max_diff = 0.0f32;
    for (x, y) in seq.data.iter().zip(par.data.iter()) {
        let d = (x - y).abs();
        if d > max_diff {
            max_diff = d;
        }
    }

    assert!(max_diff < 1e-5, "max diff = {}", max_diff);
}
