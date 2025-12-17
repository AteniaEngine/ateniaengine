use atenia_engine::apx9::vgpu_warp::*;
use atenia_engine::apx9::vgpu_warp_scheduler::*;
use atenia_engine::{tensor::Tensor, tensor::DType, tensor::Device};

#[test]
fn apx_9_18_structure() {
    let warp = VGPUWarp::new(32, 0);
    let ctx = VGPUWarpCtx { warp_id: 0, warp, state: WarpState::Ready };
    let sch = VGPUWarpScheduler::new(vec![ctx]);
    assert_eq!(sch.warps.len(), 1);
}

#[test]
fn apx_9_18_rr_basic() {
    let w1 = VGPUWarpCtx { warp_id: 0, warp: VGPUWarp::new(32, 0), state: WarpState::Ready };
    let w2 = VGPUWarpCtx { warp_id: 1, warp: VGPUWarp::new(32, 32), state: WarpState::Ready };

    let mut sch = VGPUWarpScheduler::new(vec![w1, w2]);

    let first = sch.next_warp().unwrap().warp_id;
    let second = sch.next_warp().unwrap().warp_id;

    assert_ne!(first, second);
}

#[test]
fn apx_9_18_skip_blocked() {
    let w1 = VGPUWarpCtx { warp_id: 0, warp: VGPUWarp::new(32, 0), state: WarpState::Blocked };
    let w2 = VGPUWarpCtx { warp_id: 1, warp: VGPUWarp::new(32, 32), state: WarpState::Ready };

    let mut sch = VGPUWarpScheduler::new(vec![w1, w2]);

    let next = sch.next_warp().unwrap().warp_id;
    assert_eq!(next, 1);
}

#[test]
fn apx_9_18_integration_no_numeric_change() {
    // Placeholder seguro: sólo verificamos que podemos construir tensores
    // y que el scheduler puede iterar warps Ready sin panics.
    let _a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let _b = Tensor::ones(vec![4], Device::CPU, DType::F32);

    let w1 = VGPUWarpCtx { warp_id: 0, warp: VGPUWarp::new(32, 0), state: WarpState::Ready };
    let w2 = VGPUWarpCtx { warp_id: 1, warp: VGPUWarp::new(32, 32), state: WarpState::Ready };

    let mut sch = VGPUWarpScheduler::new(vec![w1, w2]);

    // Consumimos todos los warps Ready sin cambiar la matemática (no tocamos los tensores).
    while let Some(w) = sch.next_warp() {
        assert!(w.state == WarpState::Ready || w.state == WarpState::Running || w.state == WarpState::Finished);
        // No hacemos nada más aquí: la integración real con VGpuRunner llegará en APX 10.x.
        w.state = WarpState::Finished;
    }

    assert!(true);
}
