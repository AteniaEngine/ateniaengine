use atenia_engine::profiler::{GpuHeatmap, PerfBuckets, PerfTier};

#[test]
fn apx_12_11_bucket_classification() {
    let mut hm = GpuHeatmap::new();

    hm.record("fast", 0.01);
    hm.record("cheap", 0.2);
    hm.record("mid", 1.5);
    hm.record("exp", 5.0);
    hm.record("critical", 20.0);

    let buckets = PerfBuckets::from_heatmap(&hm);

    assert_eq!(buckets.get("fast").unwrap(), PerfTier::T0);
    assert_eq!(buckets.get("cheap").unwrap(), PerfTier::T1);
    assert_eq!(buckets.get("mid").unwrap(), PerfTier::T2);
    assert_eq!(buckets.get("exp").unwrap(), PerfTier::T3);
    assert_eq!(buckets.get("critical").unwrap(), PerfTier::T4);
}
