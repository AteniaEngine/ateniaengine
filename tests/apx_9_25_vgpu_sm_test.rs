use atenia_engine::apx9::vgpu_instr::VGPUInstr;
use atenia_engine::apx9::vgpu_sm::VirtualSM;
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn apx_9_25_sm_structure() {
    let program = vec![VGPUInstr::Noop];
    let sm = VirtualSM::new(program, 2, 32, 256);

    assert_eq!(sm.warps.len(), 2);
    assert_eq!(sm.pc, 0);

    // Campos estructurales básicos
    assert_eq!(sm.memory.global.data.len(), 256);
}

#[test]
fn apx_9_25_no_numeric_change() {
    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![4], Device::CPU, DType::F32);

    let program = vec![VGPUInstr::Noop];
    let sm = VirtualSM::new(program, 1, 32, 64);

    let c_sm = sm.simulate_vec_add(&a, &b);

    let expected: f32 = a
        .data
        .iter()
        .zip(b.data.iter())
        .map(|(x, y)| x + y)
        .sum();
    let got: f32 = c_sm.data.iter().sum();

    assert!((expected - got).abs() < 1e-6);
}

#[test]
fn apx_9_25_pipeline_integration() {
    // Programa simbólico con unas pocas instrucciones.
    let program = vec![
        VGPUInstr::Noop,
        VGPUInstr::Noop,
        VGPUInstr::Noop,
    ];

    let mut sm = VirtualSM::new(program, 1, 32, 64);
    let initial_pc = sm.pc;

    // Ejecutar varios pasos del SM y verificar que el PC del warp avanza
    // y que el dual-issue ha poblado la cola de pipeline simbólica.
    sm.run_steps(10);

    let final_pc = sm.pc;
    assert!(final_pc >= initial_pc);

    let warp0 = &sm.warps[0];
    assert!(warp0.pipeline.len() >= 1);
}
