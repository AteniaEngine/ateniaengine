//! APX v18 milestone 3: real RAM-pressure detection test.
//!
//! Parallel to `apx_memory_pressure_detection_test.rs` but for system
//! RAM instead of VRAM. Allocates a real `Vec<u8>`, writes to it to
//! force physical commit, and observes the forecaster's response.

use atenia_engine::amm::forecaster::MemoryForecaster;

const MIB: u64 = 1024 * 1024;

#[test]
fn test_engine_detects_real_ram_pressure() {
    let forecaster = MemoryForecaster::new();

    // Step 1-3: baseline available RAM.
    let avail_before = match forecaster.available_ram_bytes() {
        Some(b) => b,
        None => {
            eprintln!(
                "SKIPPED: RAM probe unavailable \
                 (test_engine_detects_real_ram_pressure)"
            );
            return;
        }
    };
    assert!(avail_before > 0);

    // Step 4: pressure_size = min(2% of available, 200 MiB), floor 50 MiB.
    // More conservative than VRAM: RAM has more background noise from
    // other processes.
    let pressure_size: u64 = (avail_before / 50).min(200 * MIB).max(50 * MIB);
    assert!(
        pressure_size < avail_before,
        "cannot plan a pressure allocation smaller than available RAM \
         (avail_before = {} MiB)",
        avail_before / MIB
    );

    // Step 6: allocate and touch every byte to force physical commit.
    let mut buf: Vec<u8> = vec![0u8; pressure_size as usize];
    for b in buf.iter_mut() {
        *b = 1;
    }
    // Prevent the optimizer from eliding the write.
    std::hint::black_box(&buf);

    // Step 7-8: verify available RAM dropped by ~pressure_size.
    let avail_after_alloc = forecaster
        .available_ram_bytes()
        .expect("probe must still work after allocation");
    let actual_drop = avail_before.saturating_sub(avail_after_alloc);
    let tolerance = 100 * MIB;
    let drop_diff = actual_drop.abs_diff(pressure_size);
    assert!(
        drop_diff < tolerance,
        "RAM drop mismatch: expected ~{} MiB, got {} MiB (diff {} MiB, tol {} MiB)",
        pressure_size / MIB,
        actual_drop / MIB,
        drop_diff / MIB,
        tolerance / MIB
    );

    // Step 9: forecaster should report pressure for a request that does not fit.
    let required = avail_after_alloc.saturating_add(100 * MIB);
    assert_eq!(
        forecaster.is_under_ram_pressure(required, 0),
        Some(true),
        "forecaster must report RAM pressure when required={} MiB > avail={} MiB",
        required / MIB,
        avail_after_alloc / MIB
    );

    // Step 10: drop the buffer.
    drop(buf);

    // Step 11: available RAM should recover.
    let avail_after_free = forecaster
        .available_ram_bytes()
        .expect("probe must still work after drop");
    let recovered = avail_after_free.saturating_sub(avail_after_alloc);
    let expected_recovery = pressure_size.saturating_sub(tolerance);
    assert!(
        recovered > expected_recovery,
        "RAM did not recover after drop: after_alloc={} MiB, \
         after_free={} MiB, recovered={} MiB (expected > {} MiB)",
        avail_after_alloc / MIB,
        avail_after_free / MIB,
        recovered / MIB,
        expected_recovery / MIB
    );

    // Step 12: trivial request must not be pressured.
    assert_eq!(
        forecaster.is_under_ram_pressure(1 * MIB, 1 * MIB),
        Some(false),
        "forecaster must NOT report RAM pressure for 2 MiB when avail is {} MiB",
        avail_after_free / MIB
    );
}
