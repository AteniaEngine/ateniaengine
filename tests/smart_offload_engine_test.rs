use std::collections::HashMap;
use std::sync::Mutex;

use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::memory_types::{MemorySnapshot, MemoryTier, MoveError, TierStatus};
use atenia_engine::v13::offload_engine::{OffloadAction, SmartOffloadEngine};
use atenia_engine::v13::vram_adapter::VramAdapter;

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

fn neutral_snapshot() -> MemorySnapshot {
    snapshot_with_pressures(0.1, 0.1)
}

fn make_cache_dir(name: &str) -> String {
    let _ = std::fs::remove_dir_all(name);
    name.to_string()
}

struct FakeVramAdapter {
    storage: Mutex<HashMap<String, Vec<u8>>>,
}

impl FakeVramAdapter {
    fn new() -> Self {
        FakeVramAdapter {
            storage: Mutex::new(HashMap::new()),
        }
    }
}

impl VramAdapter for FakeVramAdapter {
    fn is_available(&self) -> bool {
        true
    }

    fn upload(&self, id: &str, data: &[u8]) -> Result<(), MoveError> {
        let mut guard = match self.storage.lock() {
            Ok(g) => g,
            Err(_) => {
                return Err(MoveError::BackendUnavailable(
                    "Failed to lock FakeVramAdapter storage".to_string(),
                ))
            }
        };
        guard.insert(id.to_string(), data.to_vec());
        Ok(())
    }

    fn download(&self, id: &str) -> Result<Vec<u8>, MoveError> {
        let guard = match self.storage.lock() {
            Ok(g) => g,
            Err(_) => {
                return Err(MoveError::BackendUnavailable(
                    "Failed to lock FakeVramAdapter storage".to_string(),
                ))
            }
        };
        match guard.get(id) {
            Some(bytes) => Ok(bytes.clone()),
            None => Err(MoveError::BackendUnavailable(
                "VRAM handle not found in FakeVramAdapter".to_string(),
            )),
        }
    }

    fn free(&self, id: &str) -> Result<(), MoveError> {
        let mut guard = match self.storage.lock() {
            Ok(g) => g,
            Err(_) => {
                return Err(MoveError::BackendUnavailable(
                    "Failed to lock FakeVramAdapter storage".to_string(),
                ))
            }
        };
        guard.remove(id);
        Ok(())
    }
}

#[test]
fn vram_high_moves_vram_tensors_to_ram() {
    let cache_dir = make_cache_dir("./.atenia_cache_test_offload_vram");
    let vram = Box::new(FakeVramAdapter::new());
    let mut mem = HybridMemoryManager::new_with_vram(&cache_dir, vram);

    // Register tensor in RAM with real data, then move it to VRAM.
    let id = "t1";
    let data = vec![1u8, 2, 3, 4];
    match mem.register_tensor_with_data(id, data.clone(), MemoryTier::Ram) {
        Ok(()) => {}
        Err(e) => panic!("register_tensor_with_data failed: {:?}", e),
    }

    let neutral = neutral_snapshot();
    let plan_to_vram = match mem.plan_move(id, MemoryTier::Vram, &neutral) {
        Ok(p) => p,
        Err(e) => panic!("plan_move to Vram failed: {:?}", e),
    };
    if let Err(e) = mem.apply_move(id, &plan_to_vram) {
        panic!("apply_move to Vram failed: {:?}", e);
    }
    assert_eq!(mem.get_tier(id), Some(MemoryTier::Vram));

    let snapshot = snapshot_with_pressures(0.99, 0.1);
    let engine = SmartOffloadEngine::default();
    let plan = engine.plan(&snapshot, &[id], &mem);

    assert_eq!(plan.actions.len(), 1);
    match &plan.actions[0] {
        OffloadAction::MoveToRam { tensor_id } => {
            assert_eq!(tensor_id, id);
        }
        other => panic!("unexpected action: {:?}", other),
    }
    assert!(plan.reason.contains("VRAM pressure high"));

    if let Err(e) = engine.apply(&snapshot, &plan, &mut mem) {
        panic!("apply offload plan failed: {:?}", e);
    }

    assert_eq!(mem.get_tier(id), Some(MemoryTier::Ram));
}

#[test]
fn ram_high_moves_ram_tensors_to_ssd() {
    let cache_dir = make_cache_dir("./.atenia_cache_test_offload_ram");
    let mut mem = HybridMemoryManager::new(&cache_dir);

    let id = "t2";
    let data = vec![5u8, 6, 7, 8];
    match mem.register_tensor_with_data(id, data, MemoryTier::Ram) {
        Ok(()) => {}
        Err(e) => panic!("register_tensor_with_data failed: {:?}", e),
    }

    let snapshot = snapshot_with_pressures(0.1, 0.99);
    let engine = SmartOffloadEngine::default();
    let plan = engine.plan(&snapshot, &[id], &mem);

    assert_eq!(plan.actions.len(), 1);
    match &plan.actions[0] {
        OffloadAction::MoveToSsd { tensor_id } => {
            assert_eq!(tensor_id, id);
        }
        other => panic!("unexpected action: {:?}", other),
    }

    if let Err(e) = engine.apply(&snapshot, &plan, &mut mem) {
        panic!("apply offload plan failed: {:?}", e);
    }

    assert_eq!(mem.get_tier(id), Some(MemoryTier::Ssd));
}

#[test]
fn both_high_prefers_ssd() {
    let cache_dir = make_cache_dir("./.atenia_cache_test_offload_both");
    let vram = Box::new(FakeVramAdapter::new());
    let mut mem = HybridMemoryManager::new_with_vram(&cache_dir, vram);

    // t3: start in RAM with data, then move to VRAM.
    let t3 = "t3";
    match mem.register_tensor_with_data(t3, vec![9u8, 10, 11, 12], MemoryTier::Ram) {
        Ok(()) => {}
        Err(e) => panic!("register_tensor_with_data failed: {:?}", e),
    }
    let neutral = neutral_snapshot();
    let plan_to_vram = match mem.plan_move(t3, MemoryTier::Vram, &neutral) {
        Ok(p) => p,
        Err(e) => panic!("plan_move to Vram failed: {:?}", e),
    };
    if let Err(e) = mem.apply_move(t3, &plan_to_vram) {
        panic!("apply_move to Vram failed: {:?}", e);
    }
    assert_eq!(mem.get_tier(t3), Some(MemoryTier::Vram));

    // t4: stays in RAM with data.
    let t4 = "t4";
    match mem.register_tensor_with_data(t4, vec![13u8, 14, 15, 16], MemoryTier::Ram) {
        Ok(()) => {}
        Err(e) => panic!("register_tensor_with_data failed: {:?}", e),
    }

    let snapshot = snapshot_with_pressures(0.99, 0.99);
    let engine = SmartOffloadEngine::default();
    let plan = engine.plan(&snapshot, &[t3, t4], &mem);

    assert_eq!(plan.actions.len(), 2);

    let mut saw_t3 = false;
    let mut saw_t4 = false;

    for action in &plan.actions {
        match action {
            OffloadAction::MoveToSsd { tensor_id } => {
                if tensor_id == t3 {
                    saw_t3 = true;
                } else if tensor_id == t4 {
                    saw_t4 = true;
                } else {
                    panic!("unexpected tensor id in MoveToSsd: {}", tensor_id);
                }
            }
            other => panic!("unexpected action: {:?}", other),
        }
    }

    assert!(saw_t3 && saw_t4);
    assert!(plan.reason.contains("VRAM and RAM pressure high"));

    if let Err(e) = engine.apply(&snapshot, &plan, &mut mem) {
        panic!("apply offload plan failed: {:?}", e);
    }

    assert_eq!(mem.get_tier(t3), Some(MemoryTier::Ssd));
    assert_eq!(mem.get_tier(t4), Some(MemoryTier::Ssd));
}

#[test]
fn no_offload_when_pressures_low() {
    let cache_dir = make_cache_dir("./.atenia_cache_test_offload_none");
    let mut mem = HybridMemoryManager::new(&cache_dir);

    let id = "t5";
    match mem.register_tensor_with_data(id, vec![42u8, 43, 44, 45], MemoryTier::Ram) {
        Ok(()) => {}
        Err(e) => panic!("register_tensor_with_data failed: {:?}", e),
    }

    let snapshot = snapshot_with_pressures(0.1, 0.1);
    let engine = SmartOffloadEngine::default();
    let plan = engine.plan(&snapshot, &[id], &mem);

    assert!(plan.actions.is_empty());
    assert!(plan.reason.contains("No offloading needed"));
}
