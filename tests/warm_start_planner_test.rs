use atenia_engine::v13::checkpoint::drift::{CheckpointDrift, DriftReport};
use atenia_engine::v13::checkpoint::{HybridCheckpoint, WarmStartAction};
use atenia_engine::v13::hybrid_memory::HybridMemoryManager;
use atenia_engine::v13::memory_types::{MemoryTier, TensorId};
use atenia_engine::v13::warm_start::build_warm_start_plan;

fn make_checkpoint_entry(
    id: &str,
    is_grad: bool,
    tier: MemoryTier,
    desired_tier: Option<MemoryTier>,
) -> atenia_engine::v13::checkpoint::CheckpointEntry {
    atenia_engine::v13::checkpoint::CheckpointEntry {
        id: id.to_string(),
        is_grad,
        tier,
        cache_kind: None,
        cache_key: None,
        len_bytes: 0,
        desired_tier,
        last_plan_summary: None,
    }
}

fn make_checkpoint(entries: Vec<atenia_engine::v13::checkpoint::CheckpointEntry>) -> HybridCheckpoint {
    HybridCheckpoint {
        version: 1,
        created_unix: 1,
        entries,
    }
}

#[test]
fn warm_start_prefers_vram_when_gpu_available() {
    let mut mem = HybridMemoryManager::new("./.atenia_cache_test_warm_start_vram_gpu");

    let entry = make_checkpoint_entry("t1", false, MemoryTier::Ram, Some(MemoryTier::Vram));
    let checkpoint = make_checkpoint(vec![entry]);

    // Register logical tensor so get_tier works.
    mem.register_tensor("t1", 0, MemoryTier::Ram);

    let drift_reports: Vec<DriftReport> = Vec::new();

    let plan = build_warm_start_plan(&mem, &checkpoint, &drift_reports, true);

    assert_eq!(plan.decisions.len(), 1);

    let d = &plan.decisions[0];
    match &d.action {
        WarmStartAction::HintPromote { to } => {
            assert_eq!(*to, MemoryTier::Vram);
            assert!(d.reason.contains("Desired VRAM") && d.reason.contains("GPU available"));
        }
        other => panic!("expected HintPromote to Vram, got {:?}", other),
    }

    assert_eq!(plan.summary, "warm_start: keep=0 promote=1 degrade=0");
}

#[test]
fn warm_start_degrades_when_gpu_missing() {
    let mut mem = HybridMemoryManager::new("./.atenia_cache_test_warm_start_vram_nogpu");

    let entry = make_checkpoint_entry("t2", false, MemoryTier::Ram, Some(MemoryTier::Vram));
    let checkpoint = make_checkpoint(vec![entry]);

    mem.register_tensor("t2", 0, MemoryTier::Ram);

    let drift_reports: Vec<DriftReport> = Vec::new();

    let plan = build_warm_start_plan(&mem, &checkpoint, &drift_reports, false);

    assert_eq!(plan.decisions.len(), 1);

    let d = &plan.decisions[0];
    match &d.action {
        WarmStartAction::DegradeSafe { to } => {
            assert_eq!(*to, MemoryTier::Ram);
            assert!(d.reason.contains("GPU unavailable"));
        }
        other => panic!("expected DegradeSafe to Ram, got {:?}", other),
    }
}

#[test]
fn warm_start_respects_drift_downgrade_reason() {
    let mut mem = HybridMemoryManager::new("./.atenia_cache_test_warm_start_drift");

    let entry = make_checkpoint_entry("t3", false, MemoryTier::Ram, Some(MemoryTier::Vram));
    let checkpoint = make_checkpoint(vec![entry]);

    mem.register_tensor("t3", 0, MemoryTier::Ram);

    let drift = DriftReport {
        entry_id: TensorId("t3".to_string()),
        drifts: vec![CheckpointDrift::TierDowngrade {
            desired: MemoryTier::Vram,
            restored: MemoryTier::Ram,
        }],
    };

    let drift_reports = vec![drift];

    let plan = build_warm_start_plan(&mem, &checkpoint, &drift_reports, false);

    assert_eq!(plan.decisions.len(), 1);

    let d = &plan.decisions[0];
    match &d.action {
        WarmStartAction::DegradeSafe { to } => {
            assert_eq!(*to, MemoryTier::Ram);
            assert!(d.reason.contains("downgraded") || d.reason.contains("drift"));
        }
        other => panic!("expected DegradeSafe with downgrade reason, got {:?}", other),
    }
}

#[test]
fn warm_start_does_not_materialize_lazy_entries() {
    let cache_root = "./.atenia_cache_test_warm_start_lazy_cache";
    let ckpt_root = "./.atenia_checkpoint_test_warm_start_lazy";

    let _ = std::fs::remove_dir_all(cache_root);
    let _ = std::fs::remove_dir_all(ckpt_root);

    let cache = atenia_engine::v13::persistent_cache::PersistentHybridCache::new(cache_root);

    let id = "t_lazy";
    let bytes = vec![1u8, 2, 3];
    let key = format!("tensor:{}:len{}", id, bytes.len());

    if let Err(e) = cache.put_blob(atenia_engine::v13::persistent_cache::CacheKind::Tensor, &key, &bytes, 1, true) {
        panic!("put_blob should succeed: {:?}", e);
    }

    // Build a minimal manifest for lazy restore.
    let manifest = format!(
        "version=1\ncreated_unix=1\nentry_count=1\n\n\
         id={id}\n\
         is_grad=0\n\
         tier=ssd\n\
         len={}\n\
         cache_kind=tensor\n\
         cache_key={key}\n\
         desired_tier=ssd\n\
         plan_summary=SSD preferred\n\n",
        bytes.len()
    );

    if let Err(e) = std::fs::create_dir_all(ckpt_root) {
        panic!("create_dir_all should succeed: {:?}", e);
    }

    let manifest_path = format!("{}/checkpoint.meta", ckpt_root);
    if let Err(e) = std::fs::write(&manifest_path, manifest) {
        panic!("writing manifest should succeed: {:?}", e);
    }

    let mut mem = HybridMemoryManager::new(cache_root);
    mem.attach_persistent_cache(cache);

    atenia_engine::v13::checkpoint::drift::take_all_for_test();
    atenia_engine::v13::checkpoint::lazy::clear_for_test();

    let result = atenia_engine::v13::checkpoint::lazy::restore_checkpoint_lazy(ckpt_root, &mut mem);
    match result {
        Ok(checkpoint) => {
            assert_eq!(checkpoint.entries.len(), 1);
        }
        Err(e) => panic!("restore_checkpoint_lazy should succeed: {:?}", e),
    }

    // Build warm-start plan; must not materialize lazy entries.
    let drift_reports: Vec<DriftReport> = Vec::new();
    let checkpoint = make_checkpoint(vec![make_checkpoint_entry(
        id,
        false,
        MemoryTier::Ssd,
        Some(MemoryTier::Ssd),
    )]);

    let _plan = build_warm_start_plan(&mem, &checkpoint, &drift_reports, false);

    let state = atenia_engine::v13::checkpoint::lazy::state_for_test(id)
        .expect("lazy state should be present");
    assert_eq!(state, atenia_engine::v13::checkpoint::lazy::LazyState::Unmaterialized);

    let _ = std::fs::remove_dir_all(cache_root);
    let _ = std::fs::remove_dir_all(ckpt_root);
}
