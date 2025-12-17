use atenia_engine::profiler::StabilityScanner;

#[test]
fn apx_12_13_stability_map_basic() {
    let total = 1024 * 1024; // 1 MB
    let step = 64 * 1024;    // 64 KB

    let map = StabilityScanner::scan(total, step)
        .expect("stability scan should never fail");

    assert!(map.entries.len() > 0);
    for e in &map.entries {
        assert!(e.read_ms.is_finite());
        assert!(e.offset < total);
    }
}
