use std::fs;

use atenia_engine::v13::checkpoint::{restore_checkpoint, save_checkpoint};
use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::memory_types::MemoryTier;
use atenia_engine::v13::persistent_cache::{CacheKind, PersistentHybridCache};

#[test]
fn save_and_restore_preserves_hints() {
    let cache_root = "./.atenia_cache_test_checkpoint_hints_cache_ram";
    let ckpt_root = "./.atenia_checkpoint_test_hints_ram";

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);

    let mut mem = HybridMemoryManager::new(cache_root);
    let cache = PersistentHybridCache::new(cache_root);
    mem.attach_persistent_cache(cache.clone());

    let id = "t1";
    let bytes = vec![1u8, 2, 3];

    if let Err(e) = mem.register_tensor_with_data(id, bytes.clone(), MemoryTier::Ram) {
        panic!("register_tensor_with_data should succeed: {:?}", e);
    }

    // Ensure tensor data is present in the persistent cache.
    let key = format!("tensor:{}:len{}", id, bytes.len());
    if let Err(e) = cache.put_blob(CacheKind::Tensor, &key, &bytes, 1, true) {
        panic!("put_blob should succeed: {:?}", e);
    }

    mem.set_desired_tier_hint(id, Some(MemoryTier::Vram));
    mem.set_last_plan_summary(id, Some("Prefer VRAM for compute-heavy node".to_string()));

    // Sanity check: hints must be visible on the original manager before
    // saving the checkpoint.
    match mem.get_last_plan_summary(id) {
        Some(s) => assert!(s.contains("Prefer VRAM")),
        None => panic!("hint was not stored in HybridMemoryManager before save"),
    }

    let checkpoint = match save_checkpoint(ckpt_root, 1, &mem) {
        Ok(c) => c,
        Err(e) => panic!("save_checkpoint should succeed: {:?}", e),
    };

    assert_eq!(checkpoint.entries.len(), 1);

    // Restore into a fresh manager using the same cache.
    let mut mem2 = HybridMemoryManager::new(cache_root);
    mem2.attach_persistent_cache(cache.clone());

    let restored = match restore_checkpoint(ckpt_root, &mut mem2) {
        Ok(c) => c,
        Err(e) => panic!("restore_checkpoint should succeed: {:?}", e),
    };

    assert_eq!(restored.entries.len(), 1);

    // Verify that plan_summary was persisted and restored into the checkpoint
    // structure itself.
    match &restored.entries[0].last_plan_summary {
        Some(s) => {
            assert!(s.contains("Prefer VRAM"));
        }
        None => panic!("expected last_plan_summary to be restored"),
    }

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);
}

#[test]
fn restore_old_manifest_without_hints_is_ok() {
    let cache_root = "./.atenia_cache_test_checkpoint_hints_cache_legacy";
    let ckpt_root = "./.atenia_checkpoint_test_hints_legacy";

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);

    let manifest = "version=1\ncreated_unix=1\nentry_count=1\n\n\
                    id=t_old\n\
                    is_grad=0\n\
                    tier=cpu\n\
                    len=0\n\
                    cache_kind=none\n\
                    cache_key=none\n\n";

    if let Err(e) = fs::create_dir_all(ckpt_root) {
        panic!("create_dir_all should succeed: {:?}", e);
    }

    let manifest_path = format!("{}/checkpoint.meta", ckpt_root);
    if let Err(e) = fs::write(&manifest_path, manifest) {
        panic!("writing legacy manifest should succeed: {:?}", e);
    }

    let mut mem = HybridMemoryManager::new(cache_root);
    let cache = PersistentHybridCache::new(cache_root);
    mem.attach_persistent_cache(cache);

    let restored = match restore_checkpoint(ckpt_root, &mut mem) {
        Ok(c) => c,
        Err(e) => panic!("restore_checkpoint on legacy manifest should succeed: {:?}", e),
    };

    assert_eq!(restored.entries.len(), 1);

    assert_eq!(mem.get_desired_tier_hint("t_old"), None);
    assert_eq!(mem.get_last_plan_summary("t_old"), None);

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);
}

#[test]
fn vram_hint_survives_safe_restore_to_ram() {
    let cache_root = "./.atenia_cache_test_checkpoint_hints_cache_vram";
    let ckpt_root = "./.atenia_checkpoint_test_hints_vram";

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);

    // Prepare cache with a tensor blob for t3.
    let cache = PersistentHybridCache::new(cache_root);

    let id = "t3";
    let bytes = vec![9u8, 9];
    let key = format!("tensor:{}:len{}", id, bytes.len());

    if let Err(e) = cache.put_blob(CacheKind::Tensor, &key, &bytes, 3, true) {
        panic!("put_blob should succeed: {:?}", e);
    }

    // Manually construct a manifest entry marking the tensor as VRAM with a
    // desired_tier hint of VRAM.
    let manifest = format!(
        "version=1\ncreated_unix=3\nentry_count=1\n\n\
         id={id}\n\
         is_grad=0\n\
         tier=vram\n\
         len={}\n\
         cache_kind=tensor\n\
         cache_key={}\n\
         desired_tier=vram\n\
         plan_summary=none\n\n",
        bytes.len(),
        key,
    );

    if let Err(e) = fs::create_dir_all(ckpt_root) {
        panic!("create_dir_all should succeed: {:?}", e);
    }

    let manifest_path = format!("{}/checkpoint.meta", ckpt_root);
    if let Err(e) = fs::write(&manifest_path, manifest) {
        panic!("writing manifest should succeed: {:?}", e);
    }

    // Restore into a manager without explicit GPU requirements; the tensor
    // must be materialized in RAM but keep its VRAM desired_tier hint.
    let mut mem = HybridMemoryManager::new(cache_root);
    mem.attach_persistent_cache(cache);

    let restored = match restore_checkpoint(ckpt_root, &mut mem) {
        Ok(c) => c,
        Err(e) => panic!("restore_checkpoint should succeed: {:?}", e),
    };

    assert_eq!(restored.entries.len(), 1);

    assert_eq!(mem.get_tier(id), Some(MemoryTier::Ram));
    assert_eq!(mem.get_desired_tier_hint(id), Some(MemoryTier::Vram));

    let _ = fs::remove_dir_all(cache_root);
    let _ = fs::remove_dir_all(ckpt_root);
}
