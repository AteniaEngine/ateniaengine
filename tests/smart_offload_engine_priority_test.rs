use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::memory_types::{MemorySnapshot, MemoryTier, TierStatus};
use atenia_engine::v13::offload_engine::{OffloadAction, SmartOffloadEngine};

fn snapshot_with_ram_pressure(ram: f32) -> MemorySnapshot {
    MemorySnapshot {
        vram: TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: Some(0.0),
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

fn make_mem() -> HybridMemoryManager {
    let cache_dir = make_cache_dir("./.atenia_cache_test_offload_pri");
    HybridMemoryManager::new(&cache_dir)
}

#[test]
fn selects_largest_tensors_first_under_ram_pressure() {
    let mut mem = make_mem();
    mem.register_tensor("t1", 10, MemoryTier::Ram);
    mem.register_tensor("t2", 100, MemoryTier::Ram);
    mem.register_tensor("t3", 50, MemoryTier::Ram);

    let snapshot = snapshot_with_ram_pressure(0.99);
    let mut engine = SmartOffloadEngine::default();
    engine.max_actions_per_tick = 2;

    let plan = engine.plan_with_tick(&snapshot, &["t1", "t2", "t3"], &mem, 0);

    assert_eq!(plan.actions.len(), 2);
    match (&plan.actions[0], &plan.actions[1]) {
        (OffloadAction::MoveToSsd { tensor_id: a }, OffloadAction::MoveToSsd { tensor_id: b }) => {
            assert_eq!(a, "t2");
            assert_eq!(b, "t3");
        }
        other => panic!("unexpected actions: {:?}", other),
    }
}

#[test]
fn tie_breaks_by_id_when_same_size() {
    let mut mem = make_mem();
    mem.register_tensor("tA", 50, MemoryTier::Ram);
    mem.register_tensor("tB", 50, MemoryTier::Ram);

    let snapshot = snapshot_with_ram_pressure(0.99);
    let mut engine = SmartOffloadEngine::default();
    engine.max_actions_per_tick = 1;

    let plan = engine.plan_with_tick(&snapshot, &["tA", "tB"], &mem, 0);

    assert_eq!(plan.actions.len(), 1);
    match &plan.actions[0] {
        OffloadAction::MoveToSsd { tensor_id } => {
            assert_eq!(tensor_id, "tA");
        }
        other => panic!("unexpected action: {:?}", other),
    }
}

#[test]
fn respects_cooldown_even_if_large() {
    let mut mem = make_mem();
    mem.register_tensor("t_small", 10, MemoryTier::Ram);
    mem.register_tensor("t_large", 100, MemoryTier::Ram);

    let snapshot = snapshot_with_ram_pressure(0.99);
    let mut engine = SmartOffloadEngine::default();
    engine.max_actions_per_tick = 1;

    // First tick: should move the large tensor first.
    let plan1 = engine.plan_with_tick(&snapshot, &["t_small", "t_large"], &mem, 10);
    assert!(!plan1.actions.is_empty());

    // Second tick within cooldown: large tensor should be skipped, small can be selected.
    let plan2 = engine.plan_with_tick(&snapshot, &["t_small", "t_large"], &mem, 12);
    assert_eq!(plan2.actions.len(), 1);
    match &plan2.actions[0] {
        OffloadAction::MoveToSsd { tensor_id } => {
            assert_eq!(tensor_id, "t_small");
        }
        other => panic!("unexpected action: {:?}", other),
    }
}

#[test]
fn reason_includes_priority_summary() {
    let mut mem = make_mem();
    mem.register_tensor("t1", 10, MemoryTier::Ram);
    mem.register_tensor("t2", 20, MemoryTier::Ram);

    let snapshot = snapshot_with_ram_pressure(0.99);
    let mut engine = SmartOffloadEngine::default();
    engine.max_actions_per_tick = 1;

    let plan = engine.plan_with_tick(&snapshot, &["t1", "t2"], &mem, 0);

    assert!(plan.reason.contains("Priority offloading enabled"));
    assert!(plan.reason.contains("selected"));
}
