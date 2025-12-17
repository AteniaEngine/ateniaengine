use atenia_engine::profiler::exec_record::*;
use atenia_engine::gpu::fingerprint::KernelFingerprint;

#[test]
fn apx_12_7_recorder_basic() {
    let mut rec = ExecutionRecorder::new();

    let dummy_fp = KernelFingerprint::new(
        9999,
        (1, 1, 1),
        (1, 1, 1),
        0,
        4,
        0,
        32,
        "TEST",
    );

    rec.record("fake_kernel", 1.23, dummy_fp.clone());
    rec.record("fake_kernel2", 2.34, dummy_fp);

    assert_eq!(rec.records.len(), 2);
    assert_eq!(rec.records[0].id, 0);
    assert_eq!(rec.records[1].id, 1);
    assert_eq!(rec.records[0].kernel_name, "fake_kernel");
}
