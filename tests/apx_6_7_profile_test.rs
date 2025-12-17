use atenia_engine::apx6_7::runtime_profile::{KernelPerf, RuntimeProfile};
use atenia_engine::apx6_7::auto_bench::{run_initial_bench, estimate_best_kernel};
use atenia_engine::apx5::kernel_planner::KernelTarget;

#[test]
fn profile_builds_for_all_sizes() {
    let mut profile = RuntimeProfile::new();
    run_initial_bench(&mut profile);

    let sizes = [128usize, 256, 512, 1024];
    for s in &sizes {
        assert!(profile.entries.iter().any(|e| e.size == *s));
    }
}

#[test]
fn selector_picks_microkernel_when_faster() {
    let mut profile = RuntimeProfile::new();

    profile.record(KernelPerf {
        size: 256,
        baseline_us: 1000,
        micro64_us: 800,
        selected: "micro64".to_string(),
    });

    let best = estimate_best_kernel(256, &profile).expect("expected a best kernel");
    assert_eq!(best, KernelTarget::CpuFastAvx2);
}

#[test]
fn selector_fallbacks_to_baseline_when_no_profile() {
    let profile = RuntimeProfile::new();
    let best = estimate_best_kernel(256, &profile);
    assert!(best.is_none());
}
