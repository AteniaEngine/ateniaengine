use atenia_engine::apx9::vgpu_warp::*;
use atenia_engine::apx9::vgpu_pipeline::*;
use atenia_engine::apx9::vgpu_instr::*;
use atenia_engine::{tensor::Tensor, tensor::DType, tensor::Device};

#[test]
fn apx_9_20_pipeline_basic_stages() {
    let mut warp = VGPUWarp::new(32, 0);
    let program = vec![VGPUInstr::Noop];

    VGPUPipeline::step(&mut warp, &program);
    assert_eq!(warp.stage, PipelineStage::Decode);

    VGPUPipeline::step(&mut warp, &program);
    assert_eq!(warp.stage, PipelineStage::Execute);

    VGPUPipeline::step(&mut warp, &program);
    assert_eq!(warp.stage, PipelineStage::Fetch);
}

#[test]
fn apx_9_20_pipeline_updates_pc_correctly() {
    let mut warp = VGPUWarp::new(32, 0);
    let program = vec![
        VGPUInstr::Add { dst: 0, a: 0, b: 1 },
        VGPUInstr::Noop,
    ];

    let pc_before = warp.pc;

    for _ in 0..3 {
        VGPUPipeline::step(&mut warp, &program);
    }

    assert!(warp.pc != pc_before);
}

#[test]
fn apx_9_20_pipeline_with_predication() {
    let mut warp = VGPUWarp::new(32, 0);
    let program = vec![
        VGPUInstr::If {
            pred: vec![true, false, true, false],
            then_pc: 1,
            else_pc: 2,
            join_pc: 3,
        },
        VGPUInstr::Noop,
        VGPUInstr::Noop,
        VGPUInstr::Reconverge,
    ];

    // Ejecutar un ciclo completo F→D→E
    for _ in 0..3 {
        VGPUPipeline::step(&mut warp, &program);
    }

    // Se debe haber cambiado la máscara según el predicado.
    assert_eq!(warp.mask.lanes[0], true);
    assert_eq!(warp.mask.lanes[1], false);
}

#[test]
fn apx_9_20_no_numeric_change() {
    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    assert_eq!(a.data[0], 1.0);
}
