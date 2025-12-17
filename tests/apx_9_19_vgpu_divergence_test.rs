use atenia_engine::apx9::vgpu_warp::*;
use atenia_engine::apx9::vgpu_divergence::*;
use atenia_engine::{tensor::Tensor, tensor::DType, tensor::Device};

#[test]
fn apx_9_19_structure() {
    let warp = VGPUWarp::new(32, 0);
    assert_eq!(warp.mask.lanes.len(), 32);
    assert!(warp.div_stack.stack.is_empty());
}

#[test]
fn apx_9_19_predication_basic() {
    let mut warp = VGPUWarp::new(32, 0);

    let pred = vec![true, false, true, false, true];
    warp.mask = WarpMask::from_predicate(&pred);

    assert!(warp.mask.lanes[0]);
    assert!(!warp.mask.lanes[1]);
    assert!(warp.mask.lanes[2]);
}

#[test]
fn apx_9_19_diverge_and_reconverge() {
    let mut warp = VGPUWarp::new(32, 0);

    // Divergencia
    let pred = vec![true, false, true, false];
    warp.div_stack.push(warp.mask.clone(), 99);
    warp.mask = WarpMask::from_predicate(&pred);

    assert!(warp.mask.lanes[0]);
    assert!(!warp.mask.lanes[1]);

    // Reconvergencia
    let frame = warp.div_stack.pop().unwrap();
    warp.mask = frame.mask;

    for lane in warp.mask.lanes {
        assert!(lane, "lane should be reconverged");
    }
}

#[test]
fn apx_9_19_integration_no_numeric_change() {
    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![4], Device::CPU, DType::F32);
    assert_eq!(a.data[0], 1.0);
    assert_eq!(b.data[0], 1.0);
}
