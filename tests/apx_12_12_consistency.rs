use atenia_engine::profiler::{ConsistencyScanner, GpuConsistency};

#[test]
fn apx_12_12_consistency_basic() {
    // Must never panic
    let rep = ConsistencyScanner::scan();

    // Values must be finite
    assert!(rep.avg_ms.is_finite());
    assert!(rep.max_ms.is_finite());
    assert!(rep.jitter.is_finite());

    // State must be one of the valid ones
    match rep.state {
        GpuConsistency::Stable |
        GpuConsistency::JitterLow |
        GpuConsistency::JitterMedium |
        GpuConsistency::JitterHigh |
        GpuConsistency::Unstable => {}
    }
}
