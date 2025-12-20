use atenia_engine::apx6_9::fusion_profiler::FusionProfiler;

#[test]
fn profiler_prefers_fused_when_much_faster() {
    let mut fp = FusionProfiler::new();
    fp.record("FusedQKV", 1000, 800); // fused 20% faster

    let decision = fp.should_use_fused("FusedQKV");
    assert_eq!(decision, Some(true));
}

#[test]
fn profiler_prefers_unfused_when_fused_slower() {
    let mut fp = FusionProfiler::new();
    fp.record("FusedQKV", 1000, 1200); // fused 20% slower

    let decision = fp.should_use_fused("FusedQKV");
    assert_eq!(decision, Some(false));
}

#[test]
fn profiler_returns_none_without_data() {
    let fp = FusionProfiler::new();
    assert_eq!(fp.should_use_fused("FusedQKV"), None);
}
