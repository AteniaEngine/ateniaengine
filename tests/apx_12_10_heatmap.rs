use atenia_engine::profiler::heatmap::GpuHeatmap;

#[test]
fn apx_12_10_heatmap_records_values() {
    let mut hm = GpuHeatmap::new();

    hm.record("matmul", 10.0);
    hm.record("matmul", 20.0);

    let entry = hm.get("matmul").unwrap();

    assert_eq!(entry.count, 2);
    assert!((entry.avg_ms - 15.0).abs() < 0.001);
}

#[test]
fn apx_12_10_heatmap_multiple_kernels() {
    let mut hm = GpuHeatmap::new();

    hm.record("matmul", 5.0);
    hm.record("linear", 8.0);

    assert_eq!(hm.total_entries(), 2);
}
