use atenia_engine::engine::fingerprint::ExecFingerprint;
use atenia_engine::gpu::fingerprint::KernelFingerprint;

#[test]
fn apx_12_6_fingerprint_stable() {
    let fp1 = ExecFingerprint::new(
        "matmul_kernel",
        (32, 32, 1),
        (16, 16, 1),
        2048,
        4,
    );

    let fp2 = ExecFingerprint::new(
        "matmul_kernel",
        (32, 32, 1),
        (16, 16, 1),
        2048,
        4,
    );

    // Deben ser idénticos
    assert_eq!(fp1.hash64(), fp2.hash64());

    // Cambiando algo  cambia la huella
    let fp3 = ExecFingerprint::new(
        "matmul_kernel",
        (16, 16, 1),
        (16, 16, 1),
        2048,
        4,
    );

    assert_ne!(fp1.hash64(), fp3.hash64());
}

#[test]
fn apx_12_6_fingerprint_basic_kernel() {
    let fp = KernelFingerprint::new(
        12345,
        (16, 16, 1),
        (8, 8, 1),
        2048,
        32,
        20,
        32,
        "TEST",
    );

    assert_eq!(fp.ptx_hash, 12345);
    assert_eq!(fp.grid.0, 16);
    assert_eq!(fp.block.0, 8);
    assert!(fp.timestamp_ms > 0);
}
