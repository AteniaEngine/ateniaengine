use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::memory_types::{MemoryTier, MoveError, StorageBacking};
use atenia_engine::v13::persistent_cache::{CacheKind, PersistentHybridCache};
use atenia_engine::v13::vram_adapter::VramAdapter;

struct FakeVramAdapter;

impl VramAdapter for FakeVramAdapter {
    fn is_available(&self) -> bool {
        false
    }

    fn upload(&self, _id: &str, _data: &[u8]) -> Result<(), MoveError> {
        Err(MoveError::BackendUnavailable(
            "FakeVramAdapter does not support upload".to_string(),
        ))
    }

    fn download(&self, _id: &str) -> Result<Vec<u8>, MoveError> {
        Err(MoveError::BackendUnavailable(
            "FakeVramAdapter does not support download".to_string(),
        ))
    }

    fn free(&self, _id: &str) -> Result<(), MoveError> {
        Ok(())
    }
}

fn make_mgr_with_cache(root: &str) -> (HybridMemoryManager, PersistentHybridCache) {
    let _ = std::fs::remove_dir_all(root);

    let vram = Box::new(FakeVramAdapter);
    let mut mgr = HybridMemoryManager::new_with_vram(root, vram);

    let cache_root = format!("{}__persistent", root);
    let cache = PersistentHybridCache::new(&cache_root);
    mgr.attach_persistent_cache(cache.clone());

    (mgr, cache)
}

#[test]
fn ram_to_ssd_uses_persistent_cache_when_attached() {
    let root = "./.atenia_cache_test_persistent_integration_ram_to_ssd";
    let (mut mgr, cache) = make_mgr_with_cache(root);

    let bytes = vec![1u8, 2, 3, 4];
    let id = "t1";

    if let Err(e) = mgr.register_tensor_with_data(id, bytes.clone(), MemoryTier::Ram) {
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

    let plan = match mgr.plan_move(id, MemoryTier::Ssd, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move RAM->SSD should succeed: {:?}", e),
    };

    if let Err(e) = mgr.apply_move(id, &plan) {
        panic!("apply_move RAM->SSD should succeed: {:?}", e);
    }

    assert_eq!(mgr.get_tier(id), Some(MemoryTier::Ssd));

    match mgr.backing_for_test(id) {
        Some(StorageBacking::SsdFile { .. }) => {
            // Backing is on SSD as expected; cache presence is checked below.
        }
        other => panic!("expected SsdFile backing, got {:?}", other),
    };

    // Derive the deterministic cache key we expect from the specification.
    let expected_key = format!("tensor:{}:len{}", id, bytes.len());

    // When cache is attached, the blob must exist under the Tensor namespace.
    assert!(cache.exists(CacheKind::Tensor, &expected_key));

    let loaded = match cache.get_blob(CacheKind::Tensor, &expected_key) {
        Ok(b) => b,
        Err(e) => panic!("cache.get_blob should succeed: {:?}", e),
    };

    assert_eq!(loaded, bytes);

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn ssd_to_ram_reads_from_cache_and_restores_bytes() {
    let root = "./.atenia_cache_test_persistent_integration_ssd_to_ram";
    let (mut mgr, _cache) = make_mgr_with_cache(root);

    let bytes = vec![1u8, 2, 3, 4];
    let id = "t1";

    if let Err(e) = mgr.register_tensor_with_data(id, bytes.clone(), MemoryTier::Ram) {
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

    let plan_to_ssd = match mgr.plan_move(id, MemoryTier::Ssd, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move RAM->SSD should succeed: {:?}", e),
    };

    if let Err(e) = mgr.apply_move(id, &plan_to_ssd) {
        panic!("apply_move RAM->SSD should succeed: {:?}", e);
    }

    assert_eq!(mgr.get_tier(id), Some(MemoryTier::Ssd));

    // Now move back to RAM using SSD->RAM.
    let plan_to_ram = match mgr.plan_move(id, MemoryTier::Ram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move SSD->RAM should succeed: {:?}", e),
    };

    if let Err(e) = mgr.apply_move(id, &plan_to_ram) {
        panic!("apply_move SSD->RAM should succeed: {:?}", e);
    }

    assert_eq!(mgr.get_tier(id), Some(MemoryTier::Ram));

    // Ensure bytes are restored.
    match mgr.backing_for_test(id) {
        Some(StorageBacking::Ram(restored)) => {
            assert_eq!(restored, &bytes);
        }
        other => panic!("expected RAM backing after SSD->RAM, got {:?}", other),
    }

    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn works_without_cache_attached() {
    let root = "./.atenia_cache_test_persistent_integration_legacy";
    let _ = std::fs::remove_dir_all(root);

    let vram = Box::new(FakeVramAdapter);
    let mut mgr = HybridMemoryManager::new_with_vram(root, vram);

    let bytes = vec![9u8, 8, 7, 6];
    let id = "t2";

    if let Err(e) = mgr.register_tensor_with_data(id, bytes.clone(), MemoryTier::Ram) {
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

    let plan_to_ssd = match mgr.plan_move(id, MemoryTier::Ssd, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move RAM->SSD should succeed: {:?}", e),
    };

    if let Err(e) = mgr.apply_move(id, &plan_to_ssd) {
        panic!("apply_move RAM->SSD should succeed: {:?}", e);
    }

    assert_eq!(mgr.get_tier(id), Some(MemoryTier::Ssd));

    // Move back to RAM using legacy behavior.
    let plan_to_ram = match mgr.plan_move(id, MemoryTier::Ram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move SSD->RAM should succeed: {:?}", e),
    };

    if let Err(e) = mgr.apply_move(id, &plan_to_ram) {
        panic!("apply_move SSD->RAM should succeed: {:?}", e);
    }

    assert_eq!(mgr.get_tier(id), Some(MemoryTier::Ram));

    match mgr.backing_for_test(id) {
        Some(StorageBacking::Ram(restored)) => {
            assert_eq!(restored, &bytes);
        }
        other => panic!("expected RAM backing after SSD->RAM, got {:?}", other),
    }

    let _ = std::fs::remove_dir_all(root);
}
