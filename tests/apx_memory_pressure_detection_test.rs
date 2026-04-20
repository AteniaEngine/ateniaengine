//! APX v18 milestone 2: real memory-pressure detection test.
//!
//! Replaces `apx_predictive_fallback_test.rs`, which used a hardcoded
//! `panic!()` to simulate failure and a test-side `if total >= X` to
//! simulate detection. Here the engine both allocates real VRAM via
//! `GpuMemoryEngine` and detects pressure via `MemoryForecaster`. The
//! test only observes.

use atenia_engine::amm::forecaster::MemoryForecaster;
use atenia_engine::gpu::memory::GpuMemoryEngine;

const MIB: u64 = 1024 * 1024;

#[test]
fn test_engine_detects_real_vram_pressure() {
    let forecaster = MemoryForecaster::new();

    // Skip early if the probe is unavailable, before committing to creating
    // a CUDA context.
    if forecaster.available_vram_bytes().is_none() {
        eprintln!(
            "SKIPPED: VRAM probe unavailable \
             (test_engine_detects_real_vram_pressure)"
        );
        return;
    }

    // Initialize CUDA first. Creating a context reserves driver workspace
    // (~100 MiB on typical setups). We want to measure the effect of a
    // concrete allocation, not the cost of spinning up the runtime itself,
    // so the baseline is taken AFTER the context exists.
    let mem = match GpuMemoryEngine::new() {
        Ok(m) => m,
        Err(_) => {
            eprintln!(
                "SKIPPED: CUDA driver unavailable \
                 (GpuMemoryEngine::new failed)"
            );
            return;
        }
    };

    // Step 1-2: baseline free VRAM with CUDA context already initialized.
    let free_before = forecaster
        .available_vram_bytes()
        .expect("probe worked moments ago, must still work");
    assert!(free_before > 0);

    // Step 3: pressure_size = min(10% of free, 500 MiB), floor 64 MiB so
    // the delta is clearly above the 50 MiB tolerance below.
    let pressure_size: u64 = (free_before / 10).min(500 * MIB).max(64 * MIB);
    assert!(
        pressure_size < free_before,
        "cannot plan a pressure allocation smaller than available VRAM \
         (free_before = {} MiB)",
        free_before / MIB
    );

    // Step 4: allocate real VRAM through the engine's CUDA memory API.
    let gpu = mem
        .alloc(pressure_size as usize)
        .expect("CUDA allocation must succeed for a size well under free VRAM");

    // Step 5-6: free VRAM should drop by approximately pressure_size.
    let free_after_alloc = forecaster
        .available_vram_bytes()
        .expect("probe must still work after allocation");
    let actual_drop = free_before.saturating_sub(free_after_alloc);
    let tolerance = 50 * MIB;
    let drop_diff = actual_drop.abs_diff(pressure_size);
    assert!(
        drop_diff < tolerance,
        "VRAM drop mismatch: expected ~{} MiB, got {} MiB (diff {} MiB, tol {} MiB)",
        pressure_size / MIB,
        actual_drop / MIB,
        drop_diff / MIB,
        tolerance / MIB
    );

    // Step 7-8: engine should report pressure for a request that does not fit.
    let required = free_after_alloc.saturating_add(50 * MIB);
    assert_eq!(
        forecaster.is_under_memory_pressure(required, 0),
        Some(true),
        "forecaster must report pressure when required={} MiB > free={} MiB",
        required / MIB,
        free_after_alloc / MIB
    );

    // Step 9: release VRAM.
    mem.free(&gpu).expect("free must succeed");

    // Step 10: VRAM should recover by approximately pressure_size.
    let free_after_free = forecaster
        .available_vram_bytes()
        .expect("probe must still work after free");
    let recovered = free_after_free.saturating_sub(free_after_alloc);
    let expected_recovery = pressure_size.saturating_sub(tolerance);
    assert!(
        recovered > expected_recovery,
        "VRAM did not recover after free: after_alloc={} MiB, \
         after_free={} MiB, recovered={} MiB (expected > {} MiB)",
        free_after_alloc / MIB,
        free_after_free / MIB,
        recovered / MIB,
        expected_recovery / MIB
    );

    // Step 11: trivial request must not be pressured.
    assert_eq!(
        forecaster.is_under_memory_pressure(1 * MIB, 1 * MIB),
        Some(false),
        "forecaster must NOT report pressure for 2 MiB when free VRAM is {} MiB",
        free_after_free / MIB
    );
}
