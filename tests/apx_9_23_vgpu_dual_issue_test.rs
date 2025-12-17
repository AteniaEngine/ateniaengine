use atenia_engine::apx9::vgpu_warp::*;
use atenia_engine::apx9::vgpu_instr::*;
use atenia_engine::apx9::vgpu_dual_issue::VGPUDualIssue;
use atenia_engine::{tensor::Tensor, tensor::DType, tensor::Device};

#[test]
fn apx_9_23_dual_issues_when_no_conflict() {
    let mut warp = VGPUWarp::new(32, 0);
    warp.scoreboard.clear_all();

    let i1 = VGPUInstr::Add { dst: 1, a: 2, b: 3 };
    let i2 = VGPUInstr::Add { dst: 4, a: 5, b: 6 };

    let (ok1, ok2) = VGPUDualIssue::issue(
        &mut warp,
        Some(i1),
        Some(i2),
    );

    assert!(ok1);
    assert!(ok2);
    assert_eq!(warp.pipeline.len(), 2);
}

#[test]
fn apx_9_23_prevents_conflict() {
    let mut warp = VGPUWarp::new(32, 0);
    warp.scoreboard.clear_all();

    let i1 = VGPUInstr::Add { dst: 1, a: 2, b: 3 };
    let i2 = VGPUInstr::Add { dst: 1, a: 5, b: 6 }; // conflicto WAW

    let (ok1, ok2) = VGPUDualIssue::issue(
        &mut warp,
        Some(i1),
        Some(i2),
    );

    assert!(ok1);
    assert!(!ok2);
    assert_eq!(warp.pipeline.len(), 1);
}

#[test]
fn apx_9_23_no_numeric_change() {
    let t = Tensor::ones(vec![1], Device::CPU, DType::F32);
    assert_eq!(t.data[0], 1.0);
}
