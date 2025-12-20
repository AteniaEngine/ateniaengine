use atenia_engine::apx9::gpu_autotuner::*;

#[test]
fn apx_9_9_autotuner_learns_gpu_is_faster() {
    let mut tuner = GpuAutoTuner::new();

    tuner.record(512, 1000, 300);  // GPU faster
    tuner.record(600, 1100, 350);
    tuner.record(550, 1050, 320);

    let decision = tuner.decide(520);
    assert_eq!(decision, GpuDecision::Gpu);
}

#[test]
fn apx_9_9_autotuner_fallback_small_data() {
    let tuner = GpuAutoTuner::new();
    assert_eq!(tuner.decide(100), GpuDecision::Cpu);
}

#[test]
fn apx_9_9_autotuner_structure() {
    let tuner = GpuAutoTuner::new();
    assert!(tuner.samples.len() == 0);
}
