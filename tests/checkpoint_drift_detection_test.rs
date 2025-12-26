use std::fs;

use atenia_engine::v13::checkpoint::restore_checkpoint;
use atenia_engine::v13::checkpoint::drift::take_all_for_test;
use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::memory_types::MemoryTier;
use atenia_engine::v13::persistent_cache::{CacheKind, PersistentHybridCache};

#[test]
fn checkpoint_warns_on_tier_downgrade() {
    let cache_root = "./.atenia_cache_test_checkpoint_drift_downgrade_cache";
    let ckpt_root = "./.atenia_checkpoint_test_drift_downgrade";

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);

    let cache = PersistentHybridCache::new(cache_root);

    let id = "t_drift";
    let bytes = vec![1u8, 2, 3];
    let key = format!("tensor:{}:len{}", id, bytes.len());

    if let Err(e) = cache.put_blob(CacheKind::Tensor, &key, &bytes, 1, true) {
        panic!("put_blob should succeed: {:?}", e);
    }

    if let Err(e) = fs::create_dir_all(ckpt_root) {
        panic!("create_dir_all should succeed: {:?}", e);
    }

    let manifest = format!(
        "version=1\ncreated_unix=1\nentry_count=1\n\n\
         id={id}\n\
         is_grad=0\n\
         tier=vram\n\
         len=3\n\
         cache_kind=tensor\n\
         cache_key={key}\n\
         desired_tier=vram\n\
         plan_summary=Prefer GPU for heavy op\n\n",
    );

    let manifest_path = format!("{}/checkpoint.meta", ckpt_root);
    if let Err(e) = fs::write(&manifest_path, manifest) {
        panic!("writing manifest should succeed: {:?}", e);
    }

    let mut mem = HybridMemoryManager::new(cache_root);
    mem.attach_persistent_cache(cache);

    let _ = take_all_for_test();

    let result = restore_checkpoint(ckpt_root, &mut mem);
    match result {
        Ok(_) => {}
        Err(e) => panic!("restore_checkpoint should succeed: {:?}", e),
    }

    let reports = take_all_for_test();
    assert_eq!(reports.len(), 1);

    let report = &reports[0];
    assert_eq!(report.entry_id.0, id.to_string());

    let mut found_downgrade = false;
    for d in &report.drifts {
        if let atenia_engine::v13::checkpoint::drift::CheckpointDrift::TierDowngrade { desired, restored } = *d {
            assert_eq!(desired, MemoryTier::Vram);
            assert_eq!(restored, MemoryTier::Ram);
            found_downgrade = true;
        }
    }

    assert!(found_downgrade);

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);
}

#[test]
fn checkpoint_detects_missing_backend() {
    let cache_root = "./.atenia_cache_test_checkpoint_drift_missing_backend_cache";
    let ckpt_root = "./.atenia_checkpoint_test_drift_missing_backend";

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);

    let cache = PersistentHybridCache::new(cache_root);

    let id = "t_missing";
    let bytes = vec![4u8, 5, 6];
    let key = format!("tensor:{}:len{}", id, bytes.len());

    if let Err(e) = cache.put_blob(CacheKind::Tensor, &key, &bytes, 2, true) {
        panic!("put_blob should succeed: {:?}", e);
    }

    if let Err(e) = fs::create_dir_all(ckpt_root) {
        panic!("create_dir_all should succeed: {:?}", e);
    }

    let manifest = format!(
        "version=1\ncreated_unix=2\nentry_count=1\n\n\
         id={id}\n\
         is_grad=0\n\
         tier=vram\n\
         len=3\n\
         cache_kind=tensor\n\
         cache_key={key}\n\
         desired_tier=vram\n\
         plan_summary=Prefer GPU backend\n\n",
    );

    let manifest_path = format!("{}/checkpoint.meta", ckpt_root);
    if let Err(e) = fs::write(&manifest_path, manifest) {
        panic!("writing manifest should succeed: {:?}", e);
    }

    let mut mem = HybridMemoryManager::new(cache_root);
    mem.attach_persistent_cache(cache);

    let _ = take_all_for_test();

    let result = restore_checkpoint(ckpt_root, &mut mem);
    match result {
        Ok(_) => {}
        Err(e) => panic!("restore_checkpoint should succeed: {:?}", e),
    }

    let reports = take_all_for_test();
    assert_eq!(reports.len(), 1);

    let report = &reports[0];
    assert_eq!(report.entry_id.0, id.to_string());

    let mut found_missing = false;
    for d in &report.drifts {
        if let atenia_engine::v13::checkpoint::drift::CheckpointDrift::MissingBackend { desired } = *d {
            assert_eq!(desired, MemoryTier::Vram);
            found_missing = true;
        }
    }

    assert!(found_missing);

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);
}

#[test]
fn checkpoint_plan_summary_mismatch() {
    let cache_root = "./.atenia_cache_test_checkpoint_drift_plan_mismatch_cache";
    let ckpt_root = "./.atenia_checkpoint_test_drift_plan_mismatch";

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);

    let cache = PersistentHybridCache::new(cache_root);

    let id = "t_plan";
    let bytes = vec![7u8, 8];
    let key = format!("tensor:{}:len{}", id, bytes.len());

    if let Err(e) = cache.put_blob(CacheKind::Tensor, &key, &bytes, 3, true) {
        panic!("put_blob should succeed: {:?}", e);
    }

    if let Err(e) = fs::create_dir_all(ckpt_root) {
        panic!("create_dir_all should succeed: {:?}", e);
    }

    let manifest = format!(
        "version=1\ncreated_unix=3\nentry_count=1\n\n\
         id={id}\n\
         is_grad=0\n\
         tier=ram\n\
         len=2\n\
         cache_kind=tensor\n\
         cache_key={key}\n\
         desired_tier=ram\n\
         plan_summary=GPU preferred for this op\n\n",
    );

    let manifest_path = format!("{}/checkpoint.meta", ckpt_root);
    if let Err(e) = fs::write(&manifest_path, manifest) {
        panic!("writing manifest should succeed: {:?}", e);
    }

    let mut mem = HybridMemoryManager::new(cache_root);
    mem.attach_persistent_cache(cache);

    let _ = take_all_for_test();

    let result = restore_checkpoint(ckpt_root, &mut mem);
    match result {
        Ok(_) => {}
        Err(e) => panic!("restore_checkpoint should succeed: {:?}", e),
    }

    let reports = take_all_for_test();
    assert_eq!(reports.len(), 1);

    let report = &reports[0];
    assert_eq!(report.entry_id.0, id.to_string());

    let mut found_mismatch = false;
    for d in &report.drifts {
        if let atenia_engine::v13::checkpoint::drift::CheckpointDrift::PlanMismatch { summary } = d {
            assert!(summary.contains("GPU"));
            found_mismatch = true;
        }
    }

    assert!(found_mismatch);

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);
}
