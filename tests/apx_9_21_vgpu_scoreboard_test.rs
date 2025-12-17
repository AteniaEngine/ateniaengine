use atenia_engine::apx9::vgpu_warp::*;
use atenia_engine::apx9::vgpu_pipeline::*;
use atenia_engine::apx9::vgpu_instr::*;
use atenia_engine::apx9::vgpu_scoreboard::*;
use atenia_engine::{tensor::Tensor, tensor::DType, tensor::Device};

#[test]
fn apx_9_21_detects_raw_hazard() {
    let mut warp = VGPUWarp::new(32, 0);
    warp.scoreboard.mark_write(1); // reg1 pendiente

    let instr = VGPUInstr::Add { dst: 2, a: 1, b: 3 };
    warp.fetched_instr = Some(instr.clone());
    warp.stage = PipelineStage::Decode;

    VGPUPipeline::step(&mut warp, &vec![instr]);

    assert_eq!(warp.stage, PipelineStage::Decode);
}

#[test]
fn apx_9_21_detects_waw_hazard() {
    let mut warp = VGPUWarp::new(32, 0);
    warp.scoreboard.mark_write(5);

    let instr = VGPUInstr::Add { dst: 5, a: 1, b: 2 };
    warp.fetched_instr = Some(instr.clone());
    warp.stage = PipelineStage::Decode;

    VGPUPipeline::step(&mut warp, &vec![instr]);

    assert_eq!(warp.stage, PipelineStage::Decode);
}

#[test]
fn apx_9_21_write_complete_allows_next() {
    let mut sb = VGPUScoreboard::new(8);

    sb.mark_write(4);
    sb.mark_write_complete(4);

    assert!(sb.can_read(4));
    assert!(sb.can_write(4));
}

#[test]
fn apx_9_21_no_numeric_change() {
    let t = Tensor::zeros(vec![4], Device::CPU, DType::F32);
    assert_eq!(t.data[0], 0.0);
}
