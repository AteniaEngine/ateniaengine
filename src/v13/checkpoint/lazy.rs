use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

use crate::v13::checkpoint::{drift, CheckpointError, HybridCheckpoint};
use crate::v13::hybrid_memory::HybridMemoryManager;
use crate::v13::memory_types::{MemoryTier, TensorId};
use crate::v13::persistent_cache::{CacheError as PCacheError, CacheKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LazyState {
    Unmaterialized,
    Materialized,
}

#[derive(Debug, Clone)]
pub enum LazySource {
    PersistentCache { kind: CacheKind, key: String },
    SsdPath { path: String },
    RamSnapshot { bytes: Vec<u8> },
}

#[derive(Debug, Clone)]
pub struct LazyBacking {
    pub state: LazyState,
    pub tier: MemoryTier,
    pub length: usize,
    pub source: LazySource,
}

static LAZY_REGISTRY: OnceLock<Mutex<HashMap<TensorId, LazyBacking>>> = OnceLock::new();

fn registry() -> &'static Mutex<HashMap<TensorId, LazyBacking>> {
    LAZY_REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn insert_lazy(id: TensorId, backing: LazyBacking) {
    if let Ok(mut guard) = registry().lock() {
        guard.insert(id, backing);
    }
}

fn get_lazy(id: &TensorId) -> Option<LazyBacking> {
    match registry().lock() {
        Ok(guard) => guard.get(id).cloned(),
        Err(_) => None,
    }
}

fn set_state(id: &TensorId, state: LazyState) {
    if let Ok(mut guard) = registry().lock() {
        if let Some(entry) = guard.get_mut(id) {
            entry.state = state;
        }
    }
}

pub fn state_for_test(id: &str) -> Option<LazyState> {
    let tid = TensorId(id.to_string());
    get_lazy(&tid).map(|b| b.state)
}

pub fn clear_for_test() {
    if let Ok(mut guard) = registry().lock() {
        guard.clear();
    }
}

fn map_pcache_error(msg: &str, err: PCacheError) -> CheckpointError {
    match err {
        PCacheError::Io(e) => CheckpointError::Io(format!("{}: {}", msg, e)),
        PCacheError::NotFound => CheckpointError::MissingBlob(msg.to_string()),
        PCacheError::Corrupt(e) => CheckpointError::InvalidFormat(format!("{}: {}", msg, e)),
        PCacheError::AlreadyExists => CheckpointError::InvalidFormat(format!(
            "Unexpected AlreadyExists error from persistent cache: {}",
            msg,
        )),
    }
}

pub fn restore_checkpoint_lazy(
    root: impl Into<PathBuf>,
    mem: &mut HybridMemoryManager,
) -> Result<HybridCheckpoint, CheckpointError> {
    let root_path = root.into();

    let checkpoint = super::read_manifest(&root_path)?;

    if mem.persistent_cache().is_none() {
        return Err(CheckpointError::Io(
            "Persistent cache not attached to HybridMemoryManager".to_string(),
        ));
    }

    drift::clear_reports();

    // Hybrid checkpointing v1 is hardware-agnostic and safe on CPU-only hosts.
    // Drift detection assumes `gpu_available = false` and only observes.
    let gpu_available = false;

    for entry in &checkpoint.entries {
        let desired_tier = entry.desired_tier;
        let last_plan_summary = entry.last_plan_summary.clone();

        // Effective tier: VRAM entries are restored as RAM for safety.
        let effective_tier = match entry.tier {
            MemoryTier::Vram => MemoryTier::Ram,
            other => other,
        };

        // Register logical tensor with length and tier, but without backing.
        mem.register_tensor(&entry.id, entry.len_bytes as u64, effective_tier);

        if let (Some(kind), Some(key)) = (entry.cache_kind, entry.cache_key.clone()) {
            let backing = LazyBacking {
                state: LazyState::Unmaterialized,
                tier: effective_tier,
                length: entry.len_bytes,
                source: LazySource::PersistentCache {
                    kind,
                    key,
                },
            };

            insert_lazy(TensorId(entry.id.clone()), backing);
        }

        // Drift detection mirrors the eager restore logic but does not load
        // bytes or change behavior.
        let mut drifts: Vec<drift::CheckpointDrift> = Vec::new();

        if let Some(desired) = desired_tier {
            if !gpu_available && matches!(desired, MemoryTier::Vram) {
                drifts.push(drift::CheckpointDrift::MissingBackend { desired });
            }

            if matches!(desired, MemoryTier::Vram) && matches!(effective_tier, MemoryTier::Ram) {
                drifts.push(drift::CheckpointDrift::TierDowngrade {
                    desired,
                    restored: effective_tier,
                });
            }
        }

        if let Some(ref summary) = last_plan_summary {
            let summary_upper = summary.to_uppercase();
            if !gpu_available && summary_upper.contains("GPU") {
                drifts.push(drift::CheckpointDrift::PlanMismatch {
                    summary: summary.clone(),
                });
            }
        }

        if !drifts.is_empty() {
            let report = drift::DriftReport {
                entry_id: TensorId(entry.id.clone()),
                drifts,
            };

            drift::push_report(report);
        }

        mem.set_desired_tier_hint(&entry.id, desired_tier);
        mem.set_last_plan_summary(&entry.id, last_plan_summary.clone());
    }

    Ok(checkpoint)
}

pub fn ensure_materialized(mem: &mut HybridMemoryManager, id: &str) -> Result<(), CheckpointError> {
    let tid = TensorId(id.to_string());

    let backing_opt = get_lazy(&tid);
    let backing = match backing_opt {
        Some(b) => b,
        None => {
            // Not a lazy entry; assume already materialized or managed
            // through the normal eager path.
            return Ok(());
        }
    };

    if backing.state == LazyState::Materialized {
        return Ok(());
    }

    match backing.source {
        LazySource::PersistentCache { kind, ref key } => {
            let cache = match mem.persistent_cache() {
                Some(c) => c,
                None => {
                    return Err(CheckpointError::Io(
                        "Persistent cache not attached to HybridMemoryManager".to_string(),
                    ));
                }
            };

            let bytes = match cache.get_blob(kind, key) {
                Ok(b) => b,
                Err(e) => {
                    return Err(map_pcache_error(
                        &format!("Failed to materialize lazy entry {} from cache", id),
                        e,
                    ));
                }
            };

            if let Err(e) = mem.register_tensor_with_data(id, bytes, backing.tier) {
                return Err(CheckpointError::Io(format!(
                    "Failed to materialize tensor {} from lazy backing: {:?}",
                    id, e
                )));
            }

            set_state(&tid, LazyState::Materialized);
        }
        LazySource::SsdPath { .. } => {
            // Not used in this subversion; placeholder for future extensions.
        }
        LazySource::RamSnapshot { .. } => {
            // Not used in this subversion; placeholder for future extensions.
        }
    }

    Ok(())
}
