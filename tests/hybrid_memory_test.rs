use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::memory_types::{
    MemorySnapshot, MemoryTier, MoveError, TierStatus,
};

use std::fs;

fn empty_snapshot() -> MemorySnapshot {
    let tier = TierStatus {
        total_bytes: None,
        free_bytes: None,
        pressure: None,
    };
    MemorySnapshot {
        vram: tier,
        ram: tier,
        ssd: tier,
    }
}

#[test]
fn register_and_query_tier() {
    let mut mgr = HybridMemoryManager::new(".atenia_cache_test");
    mgr.register_tensor("t1", 1024, MemoryTier::Vram);

    let tier = mgr.get_tier("t1");
    assert_eq!(tier, Some(MemoryTier::Vram));
}

#[test]
fn plan_and_apply_move_ram() {
    let mut mgr = HybridMemoryManager::new(".atenia_cache_test");
    mgr.register_tensor("t1", 1024, MemoryTier::Vram);

    let snapshot = empty_snapshot();
    let plan = match mgr.plan_move("t1", MemoryTier::Ram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan should succeed, got error: {:?}", e),
    };

    assert_eq!(plan.from, MemoryTier::Vram);
    assert_eq!(plan.to, MemoryTier::Ram);

    if let Err(e) = mgr.apply_move("t1", &plan) {
        panic!("apply should succeed, got error: {:?}", e);
    }
    let tier = mgr.get_tier("t1");
    assert_eq!(tier, Some(MemoryTier::Ram));
}

#[test]
fn plan_to_ssd_ensures_cache_dir() {
    let cache_dir = "./.atenia_cache_test";
    // Best-effort cleanup before the test.
    let _ = fs::remove_dir_all(cache_dir);

    let mut mgr = HybridMemoryManager::new(cache_dir);
    mgr.register_tensor("t1", 1024, MemoryTier::Vram);

    let snapshot = empty_snapshot();
    let plan = match mgr.plan_move("t1", MemoryTier::Ssd, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan to SSD should succeed, got error: {:?}", e),
    };

    assert_eq!(plan.to, MemoryTier::Ssd);

    // Directory should exist after planning the move.
    assert!(std::path::Path::new(cache_dir).exists());

    // Best-effort cleanup after the test.
    let _ = fs::remove_dir_all(cache_dir);
}

#[test]
fn already_in_target_tier_is_ok() {
    let mut mgr = HybridMemoryManager::new(".atenia_cache_test");
    mgr.register_tensor("t1", 1024, MemoryTier::Ram);

    let snapshot = empty_snapshot();
    let plan = match mgr.plan_move("t1", MemoryTier::Ram, &snapshot) {
        Ok(p) => p,
        Err(e) => panic!("plan should succeed, got error: {:?}", e),
    };

    assert_eq!(plan.from, MemoryTier::Ram);
    assert_eq!(plan.to, MemoryTier::Ram);
    assert_eq!(plan.reason, "Already in target tier");
}

#[test]
fn unknown_tensor_returns_error() {
    let mgr = HybridMemoryManager::new(".atenia_cache_test");

    let snapshot = empty_snapshot();
    let result = mgr.plan_move("unknown", MemoryTier::Ram, &snapshot);

    match result {
        Err(MoveError::Unsupported(msg)) => {
            assert_eq!(msg, "Tensor not registered");
        }
        other => panic!("Unexpected result: {:?}", other),
    }
}
