// tests/apx_12_5_timeline.rs

use atenia_engine::engine::timeline::LaunchTimeline;

#[test]
fn apx_12_5_timeline_records_events() {
    let tl = LaunchTimeline::new();

    tl.record("matmul_kernel", (1,1,1), (16,16,1), 0, 3);
    tl.record("relu", (32,1,1), (64,1,1), 0, 1);

    assert_eq!(tl.len(), 2);

    let last = tl.last().unwrap();
    assert_eq!(last.name, "relu");
    assert_eq!(last.params, 1);
}
