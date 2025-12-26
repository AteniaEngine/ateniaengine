use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::memory_types::{MemorySnapshot, MemoryTier, TierStatus};
use atenia_engine::v13::offload_engine::{OffloadAction, SmartOffloadEngine};

fn snapshot_with_pressures(vram: f32, ram: f32) -> MemorySnapshot {
    MemorySnapshot {
        vram: TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: Some(vram),
        },
        ram: TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: Some(ram),
        },
        ssd: TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: Some(0.0),
        },
    }
}

fn make_cache_dir(name: &str) -> String {
    let _ = std::fs::remove_dir_all(name);
    name.to_string()
}

fn make_mem_with_tier(id: &str, tier: MemoryTier) -> HybridMemoryManager {
    let cache_dir = make_cache_dir("./.atenia_cache_test_offload_hys");
    let mut mem = HybridMemoryManager::new(&cache_dir);
    mem.register_tensor(id, 0, tier);
    mem
}

#[test]
fn stable_band_does_not_trigger() {
    let id = "t_stable";
    let mem = make_mem_with_tier(id, MemoryTier::Vram);

    let snapshot = snapshot_with_pressures(0.90, 0.10); // between low (0.85) and high (0.95)
    let mut engine = SmartOffloadEngine::default();

    let plan = engine.plan_with_tick(&snapshot, &[id], &mem, 10);

    assert!(plan.actions.is_empty());
}

#[test]
fn triggers_on_high() {
    let id = "t_high";
    let mem = make_mem_with_tier(id, MemoryTier::Vram);

    let snapshot = snapshot_with_pressures(0.99, 0.10);
    let mut engine = SmartOffloadEngine::default();

    let plan = engine.plan_with_tick(&snapshot, &[id], &mem, 5);

    assert_eq!(plan.actions.len(), 1);
    match &plan.actions[0] {
        OffloadAction::MoveToRam { tensor_id } => {
            assert_eq!(tensor_id, id);
        }
        other => panic!("unexpected action: {:?}", other),
    }

    assert!(plan.reason.contains("VRAM pressure high"));
}

#[test]
fn cooldown_skips_repeated_moves() {
    let id = "t_cooldown";
    let mem = make_mem_with_tier(id, MemoryTier::Vram);

    let snapshot = snapshot_with_pressures(0.99, 0.10);
    let mut engine = SmartOffloadEngine::default();

    // First tick: should plan a move.
    let plan1 = engine.plan_with_tick(&snapshot, &[id], &mem, 10);
    assert_eq!(plan1.actions.len(), 1);

    // Second tick within cooldown window: no new actions.
    let plan2 = engine.plan_with_tick(&snapshot, &[id], &mem, 12);
    assert!(plan2.actions.is_empty());
    assert!(plan2.reason.contains("cooldown") || plan2.reason.contains("No new offloading"));

    // After cooldown has passed: actions allowed again.
    let plan3 = engine.plan_with_tick(&snapshot, &[id], &mem, 16);
    assert_eq!(plan3.actions.len(), 1);
}

#[test]
fn below_low_turns_off_pressure() {
    let id = "t_low";
    let mem = make_mem_with_tier(id, MemoryTier::Vram);

    let snapshot = snapshot_with_pressures(0.80, 0.10); // below vram_low (0.85)
    let mut engine = SmartOffloadEngine::default();

    let plan = engine.plan_with_tick(&snapshot, &[id], &mem, 0);

    assert!(plan.actions.is_empty());
    assert!(
        plan.reason.contains("below low threshold")
            || plan.reason.contains("No offloading needed")
            || plan.reason.contains("No new offloading")
    );
}
