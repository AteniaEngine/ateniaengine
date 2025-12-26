use std::fs;

use atenia_engine::v13::autograd::{
    persist_grads_after_backward, warm_grads_before_backward,
};
use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::memory_types::{MemoryTier, MoveError, StorageBacking};
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

fn make_mgr_and_cache(root: &str) -> (HybridMemoryManager, PersistentHybridCache) {
    let _ = fs::remove_dir_all(root);

    let vram = Box::new(FakeVramAdapter::new());
    let mut mgr = HybridMemoryManager::new_with_vram(root, vram);

    let cache_root = format!("{}__grad_cache", root);
    let cache = PersistentHybridCache::new(&cache_root);
    mgr.attach_persistent_cache(cache.clone());

    (mgr, cache)
}

#[test]
fn persist_and_restore_gradient_roundtrip() {
    let root = "./.atenia_cache_test_gradients_roundtrip";
    let (mut mgr, cache) = make_mgr_and_cache(root);

    let bytes = vec![9u8, 8, 7];
    let id = "g1";

    if let Err(e) = mgr.register_tensor_with_data(id, bytes.clone(), MemoryTier::Ram) {
        panic!("register_tensor_with_data should succeed: {:?}", e);
    }

    let grad_ids = vec![id.to_string()];

    let report = match persist_grads_after_backward(&mut mgr, &grad_ids, 1) {
        Ok(r) => r,
        Err(e) => panic!("persist_grads_after_backward should succeed: {:?}", e),
    };

    assert_eq!(report.saved, 1);

    let key = format!("grad:{}:len{}", id, bytes.len());
    assert!(cache.exists(CacheKind::Gradient, &key));

    // Simulate missing gradient in memory by removing it from the manager,
    // then warming it from the persistent cache attached to the same manager.
    mgr.remove_for_test(id);

    let restored = warm_grads_before_backward(&mut mgr, &grad_ids);
    assert_eq!(restored, 1);

    assert_eq!(mgr.get_tier(id), Some(MemoryTier::Ram));

    match mgr.backing_for_test(id) {
        Some(StorageBacking::Ram(restored_bytes)) => {
            assert_eq!(restored_bytes, &bytes);
        }
        other => panic!("expected RAM backing for restored gradient, got {:?}", other),
    }

    let _ = fs::remove_dir_all(root);
}

#[test]
fn gradients_use_separate_namespace_from_tensors() {
    let root = "./.atenia_cache_test_gradients_namespace";
    let _ = fs::remove_dir_all(root);

    let cache_root = format!("{}__grad_cache", root);
    let _ = fs::remove_dir_all(&cache_root);
    let cache = PersistentHybridCache::new(&cache_root);

    let t_bytes = vec![1u8, 2, 3];
    let g_bytes = vec![4u8, 5, 6];

    if let Err(e) = cache.put_blob(CacheKind::Tensor, "k", &t_bytes, 0, false) {
        panic!("tensor put_blob should succeed: {:?}", e);
    }

    if let Err(e) = cache.put_blob(CacheKind::Gradient, "k", &g_bytes, 0, false) {
        panic!("gradient put_blob should succeed: {:?}", e);
    }

    assert!(cache.exists(CacheKind::Tensor, "k"));
    assert!(cache.exists(CacheKind::Gradient, "k"));

    let loaded_t = match cache.get_blob(CacheKind::Tensor, "k") {
        Ok(b) => b,
        Err(e) => panic!("get_blob tensor should succeed: {:?}", e),
    };

    let loaded_g = match cache.get_blob(CacheKind::Gradient, "k") {
        Ok(b) => b,
        Err(e) => panic!("get_blob gradient should succeed: {:?}", e),
    };

    assert_eq!(loaded_t, t_bytes);
    assert_eq!(loaded_g, g_bytes);

    let _ = fs::remove_dir_all(root);
}

#[test]
fn persist_moves_vram_grad_to_ram_before_saving() {
    let root = "./.atenia_cache_test_gradients_vram";
    let (mut mgr, cache) = make_mgr_and_cache(root);

    let id = "g2";
    let bytes = vec![1u8, 1, 1, 1];

    // Register gradient in RAM with backing data.
    if let Err(e) = mgr.register_tensor_with_data(id, bytes.clone(), MemoryTier::Ram) {
        panic!("register_tensor_with_data should succeed: {:?}", e);
    }

    // Move to VRAM to simulate a gradient residing in VRAM.
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

    let plan_to_vram = match mgr.plan_move(id, MemoryTier::Vram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move RAM->VRAM should succeed: {:?}", e),
    };

    if let Err(e) = mgr.apply_move(id, &plan_to_vram) {
        panic!("apply_move RAM->VRAM should succeed: {:?}", e);
    }

    let grad_ids = vec![id.to_string()];

    let report = match persist_grads_after_backward(&mut mgr, &grad_ids, 2) {
        Ok(r) => r,
        Err(e) => panic!("persist_grads_after_backward should succeed: {:?}", e),
    };

    assert_eq!(report.saved, 1);

    let key = format!("grad:{}:len{}", id, bytes.len());
    assert!(cache.exists(CacheKind::Gradient, &key));

    let _ = fs::remove_dir_all(root);
}
