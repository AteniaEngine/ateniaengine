use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::memory_types::{
    MemoryTier, MoveError,
};
use atenia_engine::v13::ssd_cache::SsdCache;

use std::fs;

fn make_bytes() -> Vec<u8> {
    (0u8..16u8).collect()
}

fn cache_dir() -> &'static str {
    "./.atenia_cache_test_backed"
}

#[test]
fn ram_to_ssd_moves_bytes() {
    let _ = fs::remove_dir_all(cache_dir());

    let mut mgr = HybridMemoryManager::new(cache_dir());
    let data = make_bytes();
    let id = "tensor_ram_to_ssd";

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

    let plan = match mgr.plan_move(id, MemoryTier::Ssd, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move to SSD should succeed: {:?}", e),
    };

    if let Err(e) = mgr.apply_move(id, &plan) {
        panic!("apply_move should succeed: {:?}", e);
    }

    let tier = mgr.get_tier(id);
    assert_eq!(tier, Some(MemoryTier::Ssd));

    let cache = SsdCache::new(cache_dir());
    let path = cache.blob_path(id);
    assert!(std::path::Path::new(&path).exists());

    let file_bytes = match cache.read_blob(&path) {
        Ok(b) => b,
        Err(e) => panic!("read_blob should succeed: {:?}", e),
    };
    assert_eq!(file_bytes, data);

    let _ = fs::remove_dir_all(cache_dir());
}

#[test]
fn ssd_to_ram_moves_bytes_and_deletes_file() {
    let _ = fs::remove_dir_all(cache_dir());

    let mut mgr = HybridMemoryManager::new(cache_dir());
    let data = make_bytes();
    let id = "tensor_ssd_to_ram";

    // Start by registering in SSD.
    if let Err(e) = mgr.register_tensor_with_data(id, data.clone(), MemoryTier::Ssd) {
        panic!("register_tensor_with_data should succeed: {:?}", e);
    }

    let cache = SsdCache::new(cache_dir());
    let path = cache.blob_path(id);
    assert!(std::path::Path::new(&path).exists());

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

    let plan = match mgr.plan_move(id, MemoryTier::Ram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move to RAM should succeed: {:?}", e),
    };

    if let Err(e) = mgr.apply_move(id, &plan) {
        panic!("apply_move should succeed: {:?}", e);
    }

    let tier = mgr.get_tier(id);
    assert_eq!(tier, Some(MemoryTier::Ram));

    // Inspect backing via helper.
    match mgr.backing_for_test(id) {
        Some(atenia_engine::v13::memory_types::StorageBacking::Ram(bytes)) => {
            assert_eq!(bytes, &data);
        }
        other => panic!("Expected RAM backing, got {:?}", other),
    }

    // File should be deleted best-effort.
    let exists_after = std::path::Path::new(&path).exists();
    assert!(!exists_after);

    let _ = fs::remove_dir_all(cache_dir());
}

#[test]
fn cannot_move_none_backing_to_ssd() {
    let _ = fs::remove_dir_all(cache_dir());

    let mut mgr = HybridMemoryManager::new(cache_dir());
    let id = "tensor_none_to_ssd";

    mgr.register_tensor(id, 16, MemoryTier::Ram);

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

    let result = mgr.plan_move(id, MemoryTier::Ssd, &snapshot);
    match result {
        Err(MoveError::Unsupported(msg)) => {
            assert_eq!(msg, "Cannot move tensor to SSD without backing data");
        }
        other => panic!("Expected Unsupported error, got {:?}", other),
    }

    let _ = fs::remove_dir_all(cache_dir());
}

#[test]
fn length_mismatch_is_error() {
    let _ = fs::remove_dir_all(cache_dir());

    let mut mgr = HybridMemoryManager::new(cache_dir());

    // Register properly with data and then tamper the footprint using the
    // dedicated test helper.
    let data = make_bytes();
    let id = "tensor_length_mismatch";
    if let Err(e) = mgr.register_tensor_with_data(id, data.clone(), MemoryTier::Ssd) {
        panic!("register_tensor_with_data should succeed: {:?}", e);
    }

    // Now manually shrink the footprint to force a mismatch.
    mgr.set_footprint_bytes_for_test(id, 1);

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

    let plan = match mgr.plan_move(id, MemoryTier::Ram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan_move should succeed: {:?}", e),
    };

    let result = mgr.apply_move(id, &plan);
    match result {
        Err(MoveError::Unsupported(msg)) => {
            assert!(msg.contains("Byte length mismatch"));
        }
        other => panic!("Expected Unsupported error due to length mismatch, got {:?}", other),
    }

    let _ = fs::remove_dir_all(cache_dir());
}
