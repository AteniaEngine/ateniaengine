use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::memory_types::{MemoryTier, MoveError, StorageBacking};
use atenia_engine::v13::vram_adapter::VramAdapter;

use std::collections::HashMap;
use std::sync::Mutex;

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

fn make_bytes() -> Vec<u8> {
    (0u8..16u8).collect()
}

#[test]
fn ram_to_vram_and_back_roundtrip() {
    let cache_dir = "./.atenia_cache_test_vram_ram_roundtrip";
    let _ = std::fs::remove_dir_all(cache_dir);

    let vram = Box::new(FakeVramAdapter::new());
    let mut mgr = HybridMemoryManager::new_with_vram(cache_dir, vram);

    let data = make_bytes();
    let id = "tensor_ram_vram_roundtrip";

    if let Err(e) = mgr.register_tensor_with_data(id, data.clone(), MemoryTier::Ram) {
        panic!("register_tensor_with_data should succeed: {:?}", e);
    }

    let snapshot = atenia_engine::v13::memory_types::MemorySnapshot {
        vram: atenia_engine::v13::memory_types::TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
        ram: atenia_engine::v13::memory_types::TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
        ssd: atenia_engine::v13::memory_types::TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
    };

    // RAM -> VRAM
    let plan_to_vram = match mgr.plan_move(id, MemoryTier::Vram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move to VRAM should succeed: {:?}", e),
    };

    if let Err(e) = mgr.apply_move(id, &plan_to_vram) {
        panic!("apply_move to VRAM should succeed: {:?}", e);
    }

    assert_eq!(mgr.get_tier(id), Some(MemoryTier::Vram));
    match mgr.backing_for_test(id) {
        Some(StorageBacking::VramHandle { .. }) => {}
        other => panic!("Expected VramHandle backing, got {:?}", other),
    }

    // VRAM -> RAM
    let plan_to_ram = match mgr.plan_move(id, MemoryTier::Ram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move to RAM should succeed: {:?}", e),
    };

    if let Err(e) = mgr.apply_move(id, &plan_to_ram) {
        panic!("apply_move to RAM should succeed: {:?}", e);
    }

    assert_eq!(mgr.get_tier(id), Some(MemoryTier::Ram));
    match mgr.backing_for_test(id) {
        Some(StorageBacking::Ram(bytes)) => {
            assert_eq!(bytes, &data);
        }
        other => panic!("Expected RAM backing after roundtrip, got {:?}", other),
    }

    let _ = std::fs::remove_dir_all(cache_dir);
}

#[test]
fn ssd_to_vram_via_fake_adapter() {
    let cache_dir = "./.atenia_cache_test_vram_ssd_roundtrip";
    let _ = std::fs::remove_dir_all(cache_dir);

    let vram = Box::new(FakeVramAdapter::new());
    let mut mgr = HybridMemoryManager::new_with_vram(cache_dir, vram);

    let data = make_bytes();
    let id = "tensor_ssd_to_vram";

    if let Err(e) = mgr.register_tensor_with_data(id, data.clone(), MemoryTier::Ssd) {
        panic!("register_tensor_with_data should succeed: {:?}", e);
    }

    let snapshot = atenia_engine::v13::memory_types::MemorySnapshot {
        vram: atenia_engine::v13::memory_types::TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
        ram: atenia_engine::v13::memory_types::TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
        ssd: atenia_engine::v13::memory_types::TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
    };

    // SSD -> VRAM
    let plan_to_vram = match mgr.plan_move(id, MemoryTier::Vram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move to VRAM should succeed: {:?}", e),
    };

    if let Err(e) = mgr.apply_move(id, &plan_to_vram) {
        panic!("apply_move to VRAM should succeed: {:?}", e);
    }

    assert_eq!(mgr.get_tier(id), Some(MemoryTier::Vram));
    match mgr.backing_for_test(id) {
        Some(StorageBacking::VramHandle { .. }) => {}
        other => panic!("Expected VramHandle backing after SSD->VRAM, got {:?}", other),
    }

    // VRAM -> RAM
    let plan_to_ram = match mgr.plan_move(id, MemoryTier::Ram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move to RAM should succeed: {:?}", e),
    };

    if let Err(e) = mgr.apply_move(id, &plan_to_ram) {
        panic!("apply_move to RAM should succeed: {:?}", e);
    }

    assert_eq!(mgr.get_tier(id), Some(MemoryTier::Ram));
    match mgr.backing_for_test(id) {
        Some(StorageBacking::Ram(bytes)) => {
            assert_eq!(bytes, &data);
        }
        other => panic!("Expected RAM backing after SSD->VRAM->RAM, got {:?}", other),
    }

    let _ = std::fs::remove_dir_all(cache_dir);
}

#[test]
fn vram_unavailable_degrades_to_ram() {
    let cache_dir = "./.atenia_cache_test_vram_unavailable";
    let _ = std::fs::remove_dir_all(cache_dir);

    // Default constructor uses NullVramAdapter, which is unavailable.
    let mut mgr = HybridMemoryManager::new(cache_dir);

    let data = make_bytes();
    let id = "tensor_vram_unavailable";

    if let Err(e) = mgr.register_tensor_with_data(id, data.clone(), MemoryTier::Ram) {
        panic!("register_tensor_with_data should succeed: {:?}", e);
    }

    let snapshot = atenia_engine::v13::memory_types::MemorySnapshot {
        vram: atenia_engine::v13::memory_types::TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
        ram: atenia_engine::v13::memory_types::TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
        ssd: atenia_engine::v13::memory_types::TierStatus {
            total_bytes: None,
            free_bytes: None,
            pressure: None,
        },
    };

    let plan = match mgr.plan_move(id, MemoryTier::Vram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move to VRAM should not fail even if VRAM is unavailable: {:?}", e),
    };

    // Plan should degrade to RAM.
    assert_eq!(plan.to, MemoryTier::Ram);
    assert!(plan.reason.contains("VRAM unavailable"));

    if let Err(e) = mgr.apply_move(id, &plan) {
        panic!("apply_move should succeed even when VRAM is unavailable: {:?}", e);
    }

    // Tensor should remain in RAM with RAM backing.
    assert_eq!(mgr.get_tier(id), Some(MemoryTier::Ram));
    match mgr.backing_for_test(id) {
        Some(StorageBacking::Ram(bytes)) => {
            assert_eq!(bytes, &data);
        }
        other => panic!("Expected RAM backing when VRAM is unavailable, got {:?}", other),
    }

    let _ = std::fs::remove_dir_all(cache_dir);
}
