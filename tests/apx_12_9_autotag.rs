use atenia_engine::gpu::fingerprint::KernelFingerprint;

#[test]
fn apx_12_9_tags_basic() {
    let mut fp = KernelFingerprint::new(
        999,
        (1, 1, 1),
        (1, 1, 1),
        0,
        0,
        0,
        32,
        "TST",
    );

    fp.tags = fp.tags.with("matmul").with("autogen");

    assert!(fp.tags.contains("matmul"));
    assert!(fp.tags.contains("autogen"));
}
