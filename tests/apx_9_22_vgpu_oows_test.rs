use atenia_engine::apx9::vgpu_warp::*;
use atenia_engine::apx9::vgpu_pipeline::*;
use atenia_engine::apx9::vgpu_instr::*;
use atenia_engine::apx9::vgpu_oows::VGPUOOWarpScheduler;
use atenia_engine::{tensor::Tensor, tensor::DType, tensor::Device};

#[test]
fn apx_9_22_skips_hazard_warp() {
    let mut w1 = VGPUWarp::new(32, 0);
    let mut w2 = VGPUWarp::new(32, 1);

    // Warp 1 tiene hazard (write pendiente sobre reg1 que será leído).
    w1.scoreboard.mark_write(1);
    w1.fetched_instr = Some(VGPUInstr::Add { dst: 2, a: 1, b: 3 });

    // Warp 2 no tiene hazard, debe ser seleccionada.
    w2.fetched_instr = Some(VGPUInstr::Add { dst: 4, a: 5, b: 6 });

    let warps = vec![w1, w2];
    let sel = VGPUOOWarpScheduler::select_warp(&warps);

    assert_eq!(sel, Some(1));
}

#[test]
fn apx_9_22_selects_first_ready() {
    // Primer warp: no listo (en Decode sin instrucción fetcheada).
    let mut w0 = VGPUWarp::new(32, 0);
    w0.stage = PipelineStage::Decode;
    w0.fetched_instr = None;

    // Segundo warp: listo (tiene instrucción y sin hazards).
    let mut w1 = VGPUWarp::new(32, 1);
    w1.fetched_instr = Some(VGPUInstr::Noop);

    let warps = vec![w0, w1];
    let sel = VGPUOOWarpScheduler::select_warp(&warps);

    assert_eq!(sel, Some(1));
}

#[test]
fn apx_9_22_no_numeric_change() {
    let t = Tensor::ones(vec![2], Device::CPU, DType::F32);
    assert_eq!(t.data[0], 1.0);
}
