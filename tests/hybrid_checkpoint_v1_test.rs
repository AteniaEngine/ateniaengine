use std::fs;

use atenia_engine::v13::checkpoint::{restore_checkpoint, save_checkpoint};
use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::memory_types::{MemoryTier, MoveError};
use atenia_engine::v13::persistent_cache::{CacheKind, PersistentHybridCache};
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

fn make_mem_and_cache(root_cache: &str) -> (HybridMemoryManager, PersistentHybridCache) {
    let _ = fs::remove_dir_all(root_cache);
    let vram = Box::new(FakeVramAdapter::new());
    let mut mem = HybridMemoryManager::new_with_vram(root_cache, vram);
    let cache = PersistentHybridCache::new(root_cache);
    mem.attach_persistent_cache(cache.clone());
    (mem, cache)
}

#[test]
fn save_and_restore_checkpoint_ram_entries() {
    let cache_root = "./.atenia_cache_test_checkpoint_cache_ram";
    let ckpt_root = "./.atenia_checkpoint_test_v1_ram";

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);

    let (mut mem, cache) = make_mem_and_cache(cache_root);

    let id = "t1";
    let bytes = vec![1u8, 2, 3];

    if let Err(e) = mem.register_tensor_with_data(id, bytes.clone(), MemoryTier::Ram) {
        panic!("register_tensor_with_data should succeed: {:?}", e);
    }

    let key = format!("tensor:{}:len{}", id, bytes.len());
    if let Err(e) = cache.put_blob(CacheKind::Tensor, &key, &bytes, 1, true) {
        panic!("put_blob should succeed: {:?}", e);
    }

    let result = save_checkpoint(ckpt_root, 1, &mem);
    let checkpoint = match result {
        Ok(c) => c,
        Err(e) => panic!("save_checkpoint should succeed: {:?}", e),
    };

    assert_eq!(checkpoint.entries.len(), 1);
    assert_eq!(checkpoint.entries[0].id, id.to_string());

    let vram2 = Box::new(FakeVramAdapter::new());
    let mut mem2 = HybridMemoryManager::new_with_vram(cache_root, vram2);
    mem2.attach_persistent_cache(cache.clone());

    let result_restore = restore_checkpoint(ckpt_root, &mut mem2);
    let restored = match result_restore {
        Ok(c) => c,
        Err(e) => panic!("restore_checkpoint should succeed: {:?}", e),
    };

    assert_eq!(restored.entries.len(), 1);

    assert_eq!(mem2.get_tier(id), Some(MemoryTier::Ram));

    match mem2.backing_for_test(id) {
        Some(atenia_engine::v13::memory_types::StorageBacking::Ram(restored_bytes)) => {
            assert_eq!(restored_bytes, &bytes);
        }
        other => panic!("expected RAM backing after restore, got {:?}", other),
    }

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);
}

#[test]
fn restore_ssd_entry_as_reference_without_loading() {
    let cache_root = "./.atenia_cache_test_checkpoint_cache_ssd";
    let ckpt_root = "./.atenia_checkpoint_test_v1_ssd";

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);

    let (mut mem, _) = make_mem_and_cache(cache_root);

    let id = "t2";
    let bytes = vec![5u8, 6, 7, 8];

    if let Err(e) = mem.register_tensor_with_data(id, bytes.clone(), MemoryTier::Ram) {
        panic!("register_tensor_with_data should succeed: {:?}", e);
    }

    // Move RAM -> SSD so it becomes SSD-backed and persisted in the cache.
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

    let plan = match mem.plan_move(id, MemoryTier::Ssd, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move RAM->SSD should succeed: {:?}", e),
    };

    if let Err(e) = mem.apply_move(id, &plan) {
        panic!("apply_move RAM->SSD should succeed: {:?}", e);
    }

    assert_eq!(mem.get_tier(id), Some(MemoryTier::Ssd));

    let result = save_checkpoint(ckpt_root, 2, &mem);
    let checkpoint = match result {
        Ok(c) => c,
        Err(e) => panic!("save_checkpoint should succeed: {:?}", e),
    };

    assert_eq!(checkpoint.entries.len(), 1);

    let vram2 = Box::new(FakeVramAdapter::new());
    let mut mem2 = HybridMemoryManager::new_with_vram(cache_root, vram2);
    mem2.attach_persistent_cache(PersistentHybridCache::new(cache_root));

    let result_restore = restore_checkpoint(ckpt_root, &mut mem2);
    let restored = match result_restore {
        Ok(c) => c,
        Err(e) => panic!("restore_checkpoint should succeed: {:?}", e),
    };

    assert_eq!(restored.entries.len(), 1);
    assert_eq!(mem2.get_tier(id), Some(MemoryTier::Ssd));

    // Now move SSD -> RAM and verify the bytes match.
    let plan_back = match mem2.plan_move(id, MemoryTier::Ram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move SSD->RAM should succeed: {:?}", e),
    };

    if let Err(e) = mem2.apply_move(id, &plan_back) {
        panic!("apply_move SSD->RAM should succeed: {:?}", e);
    }

    match mem2.backing_for_test(id) {
        Some(atenia_engine::v13::memory_types::StorageBacking::Ram(restored_bytes)) => {
            assert_eq!(restored_bytes, &bytes);
        }
        other => panic!("expected RAM backing after SSD->RAM, got {:?}", other),
    }

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);
}

#[test]
fn vram_entries_restore_safely_to_ram() {
    let cache_root = "./.atenia_cache_test_checkpoint_cache_vram";
    let ckpt_root = "./.atenia_checkpoint_test_v1_vram";

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);

    let vram = Box::new(FakeVramAdapter::new());
    let mut mem = HybridMemoryManager::new_with_vram(cache_root, vram);
    let cache = PersistentHybridCache::new(cache_root);
    mem.attach_persistent_cache(cache.clone());

    let id = "t3";
    let bytes = vec![9u8, 9];

    // Register in RAM first, then move to VRAM so that bytes are associated
    // with a VRAM handle via the adapter.
    if let Err(e) = mem.register_tensor_with_data(id, bytes.clone(), MemoryTier::Ram) {
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

    let plan_to_vram = match mem.plan_move(id, MemoryTier::Vram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move RAM->VRAM should succeed: {:?}", e),
    };

    if let Err(e) = mem.apply_move(id, &plan_to_vram) {
        panic!("apply_move RAM->VRAM should succeed: {:?}", e);
    }

    assert_eq!(mem.get_tier(id), Some(MemoryTier::Vram));

    // Persist VRAM gradient/tensor into the cache via a direct put_blob.
    let key = format!("tensor:{}:len{}", id, bytes.len());
    if let Err(e) = cache.put_blob(CacheKind::Tensor, &key, &bytes, 3, true) {
        panic!("put_blob should succeed: {:?}", e);
    }

    let result = save_checkpoint(ckpt_root, 3, &mem);
    let checkpoint = match result {
        Ok(c) => c,
        Err(e) => panic!("save_checkpoint should succeed: {:?}", e),
    };

    assert_eq!(checkpoint.entries.len(), 1);

    // Restore into a new manager that may or may not have real VRAM support;
    // checkpointing logic must be safe and restore to RAM.
    let vram2 = Box::new(FakeVramAdapter::new());
    let mut mem2 = HybridMemoryManager::new_with_vram(cache_root, vram2);
    mem2.attach_persistent_cache(cache.clone());

    let result_restore = restore_checkpoint(ckpt_root, &mut mem2);
    let restored = match result_restore {
        Ok(c) => c,
        Err(e) => panic!("restore_checkpoint should succeed: {:?}", e),
    };

    assert_eq!(restored.entries.len(), 1);
    assert_eq!(mem2.get_tier(id), Some(MemoryTier::Ram));

    match mem2.backing_for_test(id) {
        Some(atenia_engine::v13::memory_types::StorageBacking::Ram(restored_bytes)) => {
            assert_eq!(restored_bytes, &bytes);
        }
        other => panic!("expected RAM backing after VRAM-safe restore, got {:?}", other),
    }

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);
}
