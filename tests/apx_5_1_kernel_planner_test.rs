use atenia_engine::apx5::kernel_planner::{KernelPlanner, KernelTarget};

#[test]
fn test_kernel_planner_basic() {
    unsafe {
        std::env::set_var("ATENIA_APX_MODE", "5.1");
    }

    let kp = KernelPlanner::new();

    let small = kp.select_kernel("MatMul", &[8, 8], None);
    assert!(matches!(small.target, KernelTarget::Cpu));

    let big = kp.select_kernel("MatMul", &[512, 512], None);
    assert!(matches!(big.target, KernelTarget::Gpu));
}
