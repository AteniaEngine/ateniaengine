use atenia_engine::validator::*;
use atenia_engine::gpu::fingerprint::KernelFingerprint;
use atenia_engine::profiler::exec_record::ExecutionRecord;

#[test]
fn apx_12_8_kvh_basic() {
    let kvh = KernelValidationHarness::new(true);

    // FAKE CPU reference
    let cpu = vec![1.0f32, 2.0, 3.0];

    // FAKE identical GPU output
    let gpu = vec![1.0f32, 2.0, 3.0];

    // FAKE execution record
    let fp = KernelFingerprint::new(
        123,
        (1,1,1),
        (1,1,1),
        0,
        4,
        0,
        32,
        "KVH_TEST"
    );

    let rec = ExecutionRecord::new(
        0,
        "fake",
        1.0,
        fp,
    );

    let r = kvh.validate(&rec, &cpu, &gpu);
    assert!(r.ok);
}
