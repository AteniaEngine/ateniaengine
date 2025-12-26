use std::fs;

use atenia_engine::v13::checkpoint::{drift, lazy};
use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::memory_types::MemoryTier;
use atenia_engine::v13::persistent_cache::{CacheKind, PersistentHybridCache};

fn write_manifest(
    ckpt_root: &str,
    id: &str,
    tier: &str,
    len: usize,
    key: &str,
    desired_tier: &str,
    plan_summary: &str,
) {
    if let Err(e) = fs::create_dir_all(ckpt_root) {
        panic!("create_dir_all should succeed: {:?}", e);
    }

    let manifest = format!(
        "version=1\ncreated_unix=1\nentry_count=1\n\n\
         id={id}\n\
         is_grad=0\n\
         tier={tier}\n\
         len={len}\n\
         cache_kind=tensor\n\
         cache_key={key}\n\
         desired_tier={desired_tier}\n\
         plan_summary={plan_summary}\n\n",
    );

    let manifest_path = format!("{}/checkpoint.meta", ckpt_root);
    if let Err(e) = fs::write(&manifest_path, manifest) {
        panic!("writing manifest should succeed: {:?}", e);
    }
}

#[test]
fn lazy_restore_does_not_load_unused_entries() {
    let cache_root = "./.atenia_cache_test_lazy_restore_unused_cache";
    let ckpt_root = "./.atenia_checkpoint_test_lazy_restore_unused";

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);

    let cache = PersistentHybridCache::new(cache_root);

    let id = "t_lazy";
    let bytes = vec![1u8, 2, 3, 4];
    let key = format!("tensor:{}:len{}", id, bytes.len());

    if let Err(e) = cache.put_blob(CacheKind::Tensor, &key, &bytes, 1, true) {
        panic!("put_blob should succeed: {:?}", e);
    }

    write_manifest(ckpt_root, id, "ram", bytes.len(), &key, "ram", "CPU plan");

    let mut mem = HybridMemoryManager::new(cache_root);
    mem.attach_persistent_cache(cache);

    drift::take_all_for_test();
    lazy::clear_for_test();

    let result = lazy::restore_checkpoint_lazy(ckpt_root, &mut mem);
    match result {
        Ok(checkpoint) => {
            assert_eq!(checkpoint.entries.len(), 1);
        }
        Err(e) => panic!("restore_checkpoint_lazy should succeed: {:?}", e),
    }

    // Entry should be registered logically.
    assert_eq!(mem.get_tier(id), Some(MemoryTier::Ram));

    // Lazy backing must exist and be unmaterialized.
    let state = lazy::state_for_test(id).expect("lazy state should be present");
    assert_eq!(state, lazy::LazyState::Unmaterialized);

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);
}

#[test]
fn on_demand_materialization_is_correct() {
    let cache_root = "./.atenia_cache_test_lazy_restore_materialize_cache";
    let ckpt_root = "./.atenia_checkpoint_test_lazy_restore_materialize";

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);

    let cache = PersistentHybridCache::new(cache_root);

    let id = "t_lazy_mat";
    let bytes = vec![9u8, 8, 7];
    let key = format!("tensor:{}:len{}", id, bytes.len());

    if let Err(e) = cache.put_blob(CacheKind::Tensor, &key, &bytes, 2, true) {
        panic!("put_blob should succeed: {:?}", e);
    }

    write_manifest(ckpt_root, id, "ram", bytes.len(), &key, "ram", "CPU plan");

    let mut mem = HybridMemoryManager::new(cache_root);
    mem.attach_persistent_cache(cache);

    drift::take_all_for_test();
    lazy::clear_for_test();

    let result = lazy::restore_checkpoint_lazy(ckpt_root, &mut mem);
    match result {
        Ok(checkpoint) => {
            assert_eq!(checkpoint.entries.len(), 1);
        }
        Err(e) => panic!("restore_checkpoint_lazy should succeed: {:?}", e),
    }

    let state = lazy::state_for_test(id).expect("lazy state should be present");
    assert_eq!(state, lazy::LazyState::Unmaterialized);

    // First materialization should load bytes into RAM backing.
    if let Err(e) = lazy::ensure_materialized(&mut mem, id) {
        panic!("ensure_materialized should succeed: {:?}", e);
    }

    let state_after = lazy::state_for_test(id).expect("lazy state should be present after materialization");
    assert_eq!(state_after, lazy::LazyState::Materialized);

    match mem.backing_for_test(id) {
        Some(atenia_engine::v13::memory_types::StorageBacking::Ram(restored_bytes)) => {
            assert_eq!(restored_bytes, &bytes);
        }
        other => panic!("expected RAM backing after lazy materialization, got {:?}", other),
    }

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);
}

#[test]
fn lazy_restore_preserves_hints_and_drift() {
    let cache_root = "./.atenia_cache_test_lazy_restore_hints_cache";
    let ckpt_root = "./.atenia_checkpoint_test_lazy_restore_hints";

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);

    let cache = PersistentHybridCache::new(cache_root);

    let id = "t_lazy_hints";
    let bytes = vec![5u8, 5, 5];
    let key = format!("tensor:{}:len{}", id, bytes.len());

    if let Err(e) = cache.put_blob(CacheKind::Tensor, &key, &bytes, 3, true) {
        panic!("put_blob should succeed: {:?}", e);
    }

    // desired_tier=vram, plan mentions GPU to trigger both drift kinds.
    write_manifest(
        ckpt_root,
        id,
        "vram",
        bytes.len(),
        &key,
        "vram",
        "GPU preferred for heavy op",
    );

    let mut mem = HybridMemoryManager::new(cache_root);
    mem.attach_persistent_cache(cache);

    drift::take_all_for_test();
    lazy::clear_for_test();

    let result = lazy::restore_checkpoint_lazy(ckpt_root, &mut mem);
    match result {
        Ok(checkpoint) => {
            assert_eq!(checkpoint.entries.len(), 1);
        }
        Err(e) => panic!("restore_checkpoint_lazy should succeed: {:?}", e),
    }

    // Hints must be present on the manager.
    assert_eq!(mem.get_desired_tier_hint(id), Some(MemoryTier::Vram));
    let summary = mem.get_last_plan_summary(id).expect("plan summary should be present");
    assert!(summary.contains("GPU"));

    // Drift must be recorded and remain valid after materialization.
    let reports = drift::take_all_for_test();
    assert_eq!(reports.len(), 1);

    let report = &reports[0];
    assert_eq!(report.entry_id.0, id.to_string());

    let mut saw_missing_backend = false;
    let mut saw_tier_downgrade = false;
    let mut saw_plan_mismatch = false;

    for d in &report.drifts {
        match d {
            atenia_engine::v13::checkpoint::drift::CheckpointDrift::MissingBackend { desired } => {
                assert_eq!(*desired, MemoryTier::Vram);
                saw_missing_backend = true;
            }
            atenia_engine::v13::checkpoint::drift::CheckpointDrift::TierDowngrade { desired, restored } => {
                assert_eq!(*desired, MemoryTier::Vram);
                assert_eq!(*restored, MemoryTier::Ram);
                saw_tier_downgrade = true;
            }
            atenia_engine::v13::checkpoint::drift::CheckpointDrift::PlanMismatch { summary } => {
                assert!(summary.contains("GPU"));
                saw_plan_mismatch = true;
            }
        }
    }

    assert!(saw_missing_backend);
    assert!(saw_tier_downgrade);
    assert!(saw_plan_mismatch);

    // Materialize afterwards and ensure hints and semantic state remain.
    if let Err(e) = lazy::ensure_materialized(&mut mem, id) {
        panic!("ensure_materialized should succeed: {:?}", e);
    }

    assert_eq!(mem.get_desired_tier_hint(id), Some(MemoryTier::Vram));
    let summary_after = mem.get_last_plan_summary(id).expect("plan summary should be present after materialization");
    assert!(summary_after.contains("GPU"));

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);
}
