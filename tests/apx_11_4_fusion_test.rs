use atenia_engine::gpu_autodiff::ir_backward::BackwardKernelSpec;
use atenia_engine::gpu_autodiff::fuser::FusionPlanner;

#[test]
fn test_fusion_basic() {
    let a = BackwardKernelSpec::new("op1", "__global__ void op1(){}", vec![], vec![], vec![]);
    let b = BackwardKernelSpec::new("op2", "__global__ void op2(){}", vec![], vec![], vec![]);

    let fused = FusionPlanner::fuse(&[a.clone(), b.clone()]).expect("should fuse");

    assert_eq!(fused.name, "fused_backward");
    assert_eq!(fused.parts.len(), 2);
    assert!(fused.code.contains("op1"));
    assert!(fused.code.contains("op2"));
}
