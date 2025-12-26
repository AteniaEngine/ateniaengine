use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::v13::hybrid_memory::HybridMemoryManager;
use crate::v13::memory_types::{MemoryTier, TensorId};
use crate::v13::persistent_cache::{CacheError as PCacheError, CacheKind, PersistentHybridCache};

pub mod drift;
pub mod lazy;

#[derive(Debug, Clone)]
pub struct CheckpointEntry {
    pub id: String,
    pub is_grad: bool,
    pub tier: MemoryTier,
    pub cache_kind: Option<CacheKind>,
    pub cache_key: Option<String>,
    pub len_bytes: usize,
    pub desired_tier: Option<MemoryTier>,
    pub last_plan_summary: Option<String>,
}

#[derive(Debug, Clone)]
pub struct HybridCheckpoint {
    pub version: u32,
    pub created_unix: u64,
    pub entries: Vec<CheckpointEntry>,
}

#[derive(Debug, Clone)]
pub enum WarmStartAction {
    Keep,
    HintPromote { to: MemoryTier },
    DegradeSafe { to: MemoryTier },
}

#[derive(Debug, Clone)]
pub struct WarmStartDecision {
    pub id: String,
    pub is_grad: bool,
    pub current: MemoryTier,
    pub desired: Option<MemoryTier>,
    pub action: WarmStartAction,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct WarmStartPlan {
    pub decisions: Vec<WarmStartDecision>,
    pub summary: String,
}

#[derive(Debug, Clone)]
pub enum CheckpointError {
    Io(String),
    InvalidFormat(String),
    MissingBlob(String),
}

fn tier_to_str(t: MemoryTier) -> &'static str {
    match t {
        MemoryTier::Ram => "ram",
        MemoryTier::Ssd => "ssd",
        MemoryTier::Vram => "vram",
        MemoryTier::Cpu => "cpu",
    }
}

fn str_to_tier(s: &str) -> Option<MemoryTier> {
    match s {
        "ram" => Some(MemoryTier::Ram),
        "ssd" => Some(MemoryTier::Ssd),
        "vram" => Some(MemoryTier::Vram),
        "cpu" => Some(MemoryTier::Cpu),
        _ => None,
    }
}

fn cache_kind_to_str(kind: CacheKind) -> &'static str {
    match kind {
        CacheKind::Tensor => "tensor",
        CacheKind::Gradient => "gradient",
        CacheKind::KernelMeta => "kernel_meta",
    }
}

fn str_to_cache_kind(s: &str) -> Option<CacheKind> {
    match s {
        "tensor" => Some(CacheKind::Tensor),
        "gradient" => Some(CacheKind::Gradient),
        "kernel_meta" => Some(CacheKind::KernelMeta),
        _ => None,
    }
}

fn map_pcache_error(msg: &str, err: PCacheError) -> CheckpointError {
    match err {
        PCacheError::Io(e) => CheckpointError::Io(format!("{}: {}", msg, e)),
        PCacheError::NotFound => CheckpointError::MissingBlob(msg.to_string()),
        PCacheError::Corrupt(e) => CheckpointError::InvalidFormat(format!("{}: {}", msg, e)),
        PCacheError::AlreadyExists => CheckpointError::InvalidFormat(format!(
            "Unexpected AlreadyExists error from persistent cache: {}",
            msg
        )),
    }
}

fn manifest_path(root: &Path) -> PathBuf {
    root.join("checkpoint.meta")
}

fn write_manifest(root: &Path, checkpoint: &HybridCheckpoint) -> Result<(), CheckpointError> {
    if let Err(e) = fs::create_dir_all(root) {
        return Err(CheckpointError::Io(format!("Failed to create checkpoint root {:?}: {}", root, e)));
    }

    let path = manifest_path(root);
    let mut file = match File::create(&path) {
        Ok(f) => f,
        Err(e) => {
            return Err(CheckpointError::Io(format!(
                "Failed to create checkpoint manifest {:?}: {}",
                path, e
            )));
        }
    };

    let header = format!(
        "version={}\ncreated_unix={}\nentry_count={}\n\n",
        checkpoint.version,
        checkpoint.created_unix,
        checkpoint.entries.len(),
    );

    if let Err(e) = file.write_all(header.as_bytes()) {
        return Err(CheckpointError::Io(format!(
            "Failed to write checkpoint header {:?}: {}",
            path, e
        )));
    }

    for entry in &checkpoint.entries {
        let cache_kind_str = match entry.cache_kind {
            Some(k) => cache_kind_to_str(k),
            None => "none",
        };
        let cache_key_str = match &entry.cache_key {
            Some(k) => k.as_str(),
            None => "none",
        };
        let is_grad_int = if entry.is_grad { 1 } else { 0 };

        let desired_tier_str = match entry.desired_tier {
            Some(t) => tier_to_str(t),
            None => "none",
        };

        let mut plan_summary_str = match &entry.last_plan_summary {
            Some(s) => s.clone(),
            None => "none".to_string(),
        };

        if plan_summary_str.contains('\n') || plan_summary_str.contains('\r') {
            plan_summary_str = plan_summary_str
                .replace('\n', " ")
                .replace('\r', " ");
        }

        let block = format!(
            "id={}\nis_grad={}\ntier={}\nlen={}\ncache_kind={}\ncache_key={}\ndesired_tier={}\nplan_summary={}\n\n",
            entry.id,
            is_grad_int,
            tier_to_str(entry.tier),
            entry.len_bytes,
            cache_kind_str,
            cache_key_str,
            desired_tier_str,
            plan_summary_str,
        );

        if let Err(e) = file.write_all(block.as_bytes()) {
            return Err(CheckpointError::Io(format!(
                "Failed to write checkpoint entry {:?}: {}",
                path, e
            )));
        }
    }

    Ok(())
}

pub(crate) fn read_manifest(root: &Path) -> Result<HybridCheckpoint, CheckpointError> {
    let path = manifest_path(root);
    let mut file = match File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            return Err(CheckpointError::Io(format!(
                "Failed to open checkpoint manifest {:?}: {}",
                path, e
            )));
        }
    };

    let mut contents = String::new();
    if let Err(e) = file.read_to_string(&mut contents) {
        return Err(CheckpointError::Io(format!(
            "Failed to read checkpoint manifest {:?}: {}",
            path, e
        )));
    }

    let mut lines = contents.lines();

    let version_line = match lines.next() {
        Some(l) => l,
        None => {
            return Err(CheckpointError::InvalidFormat(
                "Missing version line in checkpoint manifest".to_string(),
            ));
        }
    };
    let created_line = match lines.next() {
        Some(l) => l,
        None => {
            return Err(CheckpointError::InvalidFormat(
                "Missing created_unix line in checkpoint manifest".to_string(),
            ));
        }
    };
    let count_line = match lines.next() {
        Some(l) => l,
        None => {
            return Err(CheckpointError::InvalidFormat(
                "Missing entry_count line in checkpoint manifest".to_string(),
            ));
        }
    };

    let parse_kv = |line: &str| -> Result<(String, String), CheckpointError> {
        let pos = match line.find('=') {
            Some(p) => p,
            None => {
                return Err(CheckpointError::InvalidFormat(format!(
                    "Invalid line (no '='): {}",
                    line
                )));
            }
        };
        let (k, v) = line.split_at(pos);
        Ok((k.to_string(), v[1..].to_string()))
    };

    let (vk, vv) = parse_kv(version_line)?;
    if vk.as_str() != "version" {
        return Err(CheckpointError::InvalidFormat(format!(
            "Expected version key, found {}",
            vk
        )));
    }
    let version: u32 = match vv.parse() {
        Ok(n) => n,
        Err(e) => {
            return Err(CheckpointError::InvalidFormat(format!(
                "Invalid version value {} ({})",
                vv, e
            )));
        }
    };

    let (ck, cv) = parse_kv(created_line)?;
    if ck.as_str() != "created_unix" {
        return Err(CheckpointError::InvalidFormat(format!(
            "Expected created_unix key, found {}",
            ck
        )));
    }
    let created_unix: u64 = match cv.parse() {
        Ok(n) => n,
        Err(e) => {
            return Err(CheckpointError::InvalidFormat(format!(
                "Invalid created_unix value {} ({})",
                cv, e
            )));
        }
    };

    let (ek, ev) = parse_kv(count_line)?;
    if ek.as_str() != "entry_count" {
        return Err(CheckpointError::InvalidFormat(format!(
            "Expected entry_count key, found {}",
            ek
        )));
    }
    let expected_entries: usize = match ev.parse() {
        Ok(n) => n,
        Err(e) => {
            return Err(CheckpointError::InvalidFormat(format!(
                "Invalid entry_count value {} ({})",
                ev, e
            )));
        }
    };

    // Skip optional blank line after header.
    if let Some(l) = lines.next() {
        if !l.trim().is_empty() {
            // Treat non-empty line as start of first entry block.
            // Push it back by building an iterator that starts with this line.
            let mut rest = String::new();
            rest.push_str(l);
            rest.push('\n');
            for remaining in lines {
                rest.push_str(remaining);
                rest.push('\n');
            }
            return parse_entries(version, created_unix, expected_entries, &rest);
        }
    }

    let mut rest = String::new();
    for remaining in lines {
        rest.push_str(remaining);
        rest.push('\n');
    }

    parse_entries(version, created_unix, expected_entries, &rest)
}

fn parse_entries(
    version: u32,
    created_unix: u64,
    expected_entries: usize,
    body: &str,
) -> Result<HybridCheckpoint, CheckpointError> {
    let mut entries: Vec<CheckpointEntry> = Vec::new();
    let mut current: Vec<&str> = Vec::new();

    for line in body.lines() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                let entry = parse_entry_block(&current)?;
                entries.push(entry);
                current.clear();
            }
            continue;
        }
        current.push(line);
    }

    if !current.is_empty() {
        let entry = parse_entry_block(&current)?;
        entries.push(entry);
    }

    if entries.len() != expected_entries {
        return Err(CheckpointError::InvalidFormat(format!(
            "Entry count mismatch: expected {}, found {}",
            expected_entries,
            entries.len()
        )));
    }

    Ok(HybridCheckpoint {
        version,
        created_unix,
        entries,
    })
}

fn parse_entry_block(lines: &[&str]) -> Result<CheckpointEntry, CheckpointError> {
    let mut id: Option<String> = None;
    let mut is_grad: Option<bool> = None;
    let mut tier: Option<MemoryTier> = None;
    let mut len_bytes: Option<usize> = None;
    let mut cache_kind: Option<Option<CacheKind>> = None;
    let mut cache_key: Option<Option<String>> = None;
    let mut desired_tier: Option<MemoryTier> = None;
    let mut last_plan_summary: Option<String> = None;

    for line in lines {
        let pos = match line.find('=') {
            Some(p) => p,
            None => {
                return Err(CheckpointError::InvalidFormat(format!(
                    "Invalid entry line (no '='): {}",
                    line
                )));
            }
        };
        let (k, v) = line.split_at(pos);
        let v = &v[1..];

        match k {
            "id" => {
                id = Some(v.to_string());
            }
            "is_grad" => {
                let val: u32 = match v.parse() {
                    Ok(n) => n,
                    Err(e) => {
                        return Err(CheckpointError::InvalidFormat(format!(
                            "Invalid is_grad value {} ({})",
                            v, e
                        )));
                    }
                };
                is_grad = Some(val == 1);
            }
            "tier" => {
                let t = match str_to_tier(v) {
                    Some(t) => t,
                    None => {
                        return Err(CheckpointError::InvalidFormat(format!(
                            "Unknown tier value {}",
                            v
                        )));
                    }
                };
                tier = Some(t);
            }
            "len" => {
                let n: usize = match v.parse() {
                    Ok(n) => n,
                    Err(e) => {
                        return Err(CheckpointError::InvalidFormat(format!(
                            "Invalid len value {} ({})",
                            v, e
                        )));
                    }
                };
                len_bytes = Some(n);
            }
            "cache_kind" => {
                if v == "none" {
                    cache_kind = Some(None);
                } else {
                    let knd = match str_to_cache_kind(v) {
                        Some(k) => k,
                        None => {
                            return Err(CheckpointError::InvalidFormat(format!(
                                "Unknown cache_kind value {}",
                                v
                            )));
                        }
                    };
                    cache_kind = Some(Some(knd));
                }
            }
            "cache_key" => {
                if v == "none" {
                    cache_key = Some(None);
                } else {
                    cache_key = Some(Some(v.to_string()));
                }
            }
            "desired_tier" => {
                if v == "none" {
                    desired_tier = None;
                } else {
                    let t = match str_to_tier(v) {
                        Some(t) => t,
                        None => {
                            return Err(CheckpointError::InvalidFormat(format!(
                                "Unknown desired_tier value {}",
                                v
                            )));
                        }
                    };
                    desired_tier = Some(t);
                }
            }
            "plan_summary" => {
                if v == "none" {
                    last_plan_summary = None;
                } else {
                    last_plan_summary = Some(v.to_string());
                }
            }
            _ => {
                return Err(CheckpointError::InvalidFormat(format!(
                    "Unknown entry key {}",
                    k
                )));
            }
        }
    }

    let id = match id {
        Some(v) => v,
        None => {
            return Err(CheckpointError::InvalidFormat(
                "Missing id in entry".to_string(),
            ));
        }
    };
    let is_grad = match is_grad {
        Some(v) => v,
        None => {
            return Err(CheckpointError::InvalidFormat(format!(
                "Missing is_grad in entry {}",
                id
            )));
        }
    };
    let tier = match tier {
        Some(v) => v,
        None => {
            return Err(CheckpointError::InvalidFormat(format!(
                "Missing tier in entry {}",
                id
            )));
        }
    };
    let len_bytes = match len_bytes {
        Some(v) => v,
        None => {
            return Err(CheckpointError::InvalidFormat(format!(
                "Missing len in entry {}",
                id
            )));
        }
    };

    let cache_kind = match cache_kind {
        Some(v) => v,
        None => {
            return Err(CheckpointError::InvalidFormat(format!(
                "Missing cache_kind in entry {}",
                id
            )));
        }
    };
    let cache_key = match cache_key {
        Some(v) => v,
        None => {
            return Err(CheckpointError::InvalidFormat(format!(
                "Missing cache_key in entry {}",
                id
            )));
        }
    };

    if cache_kind.is_none() && cache_key.is_some() {
        return Err(CheckpointError::InvalidFormat(format!(
            "cache_key present but cache_kind none in entry {}",
            id
        )));
    }
    if cache_kind.is_some() && cache_key.is_none() {
        return Err(CheckpointError::InvalidFormat(format!(
            "cache_kind present but cache_key none in entry {}",
            id
        )));
    }

    Ok(CheckpointEntry {
        id,
        is_grad,
        tier,
        cache_kind,
        cache_key,
        len_bytes,
        desired_tier,
        last_plan_summary,
    })
}

fn build_entries_from_mem(mem: &HybridMemoryManager, created_unix: u64) -> Vec<CheckpointEntry> {
    let mut entries: Vec<CheckpointEntry> = Vec::new();

    let ids = mem.ids_for_checkpoint();

    let persistent: Option<PersistentHybridCache> = mem.persistent_cache().cloned();

    for id in ids {
        let tier = match mem.get_tier(&id) {
            Some(t) => t,
            None => {
                continue;
            }
        };

        let len_bytes = match mem.tensor_len_bytes(&id) {
            Some(n) => n,
            None => {
                continue;
            }
        };

        let is_grad = mem.is_grad_id(&id);

        let (cache_kind, cache_key) = match &persistent {
            Some(cache) => {
                let (kind, base_key) = if is_grad {
                    (CacheKind::Gradient, format!("grad:{}:len{}", id, len_bytes))
                } else {
                    (CacheKind::Tensor, format!("tensor:{}:len{}", id, len_bytes))
                };

                if cache.exists(kind, &base_key) {
                    (Some(kind), Some(base_key))
                } else {
                    (None, None)
                }
            }
            None => (None, None),
        };

        let desired_tier = mem.get_desired_tier_hint(&id);
        let last_plan_summary = mem.get_last_plan_summary(&id);

        let entry = CheckpointEntry {
            id: id.clone(),
            is_grad,
            tier,
            cache_kind,
            cache_key,
            len_bytes,
            desired_tier,
            last_plan_summary,
        };

        let _ = created_unix; // unused in entries but part of HybridCheckpoint.

        entries.push(entry);
    }

    // Deterministic ordering: sort by (is_grad, id).
    entries.sort_by(|a, b| {
        let ag = if a.is_grad { 1u8 } else { 0u8 };
        let bg = if b.is_grad { 1u8 } else { 0u8 };
        ag.cmp(&bg).then_with(|| a.id.cmp(&b.id))
    });

    entries
}

pub fn save_checkpoint(
    root: impl Into<PathBuf>,
    created_unix: u64,
    mem: &HybridMemoryManager,
) -> Result<HybridCheckpoint, CheckpointError> {
    let root_path = root.into();

    let mut checkpoint = HybridCheckpoint {
        version: 1,
        created_unix,
        entries: build_entries_from_mem(mem, created_unix),
    };

    // Ensure hints are fully synchronized from the live HybridMemoryManager,
    // even if build_entries_from_mem missed them due to future changes.
    for entry in &mut checkpoint.entries {
        entry.desired_tier = mem.get_desired_tier_hint(&entry.id);
        entry.last_plan_summary = mem.get_last_plan_summary(&entry.id);
    }

    write_manifest(&root_path, &checkpoint)?;

    Ok(checkpoint)
}

pub fn restore_checkpoint(
    root: impl Into<PathBuf>,
    mem: &mut HybridMemoryManager,
) -> Result<HybridCheckpoint, CheckpointError> {
    let root_path = root.into();

    let checkpoint = read_manifest(&root_path)?;

    let persistent = match mem.persistent_cache() {
        Some(c) => c.clone(),
        None => {
            return Err(CheckpointError::Io(
                "Persistent cache not attached to HybridMemoryManager".to_string(),
            ));
        }
    };

    drift::clear_reports();

    // For now we conservatively treat restore as running without a real GPU
    // backend. Hybrid checkpointing v1 is hardware-agnostic and always safe
    // to restore on CPU-only hosts, so drift detection assumes
    // `gpu_available = false` and only reports, never alters behavior.
    let gpu_available = false;

    for entry in &checkpoint.entries {
        let kind = entry.cache_kind;
        let key_opt = entry.cache_key.clone();
        let desired_tier = entry.desired_tier;
        let last_plan_summary = entry.last_plan_summary.clone();

        let mut actual_tier = entry.tier;

        match entry.tier {
            MemoryTier::Ram => {
                let kind = match kind {
                    Some(k) => k,
                    None => {
                        return Err(CheckpointError::MissingBlob(format!(
                            "RAM entry without cache reference: {}",
                            entry.id
                        )));
                    }
                };
                let key = match key_opt {
                    Some(ref k) => k.clone(),
                    None => {
                        return Err(CheckpointError::MissingBlob(format!(
                            "RAM entry without cache key: {}",
                            entry.id
                        )));
                    }
                };

                let bytes = match persistent.get_blob(kind, &key) {
                    Ok(b) => b,
                    Err(e) => {
                        return Err(map_pcache_error(
                            &format!("Failed to load RAM entry {} from cache", entry.id),
                            e,
                        ));
                    }
                };

                if let Err(e) = mem.register_tensor_with_data(&entry.id, bytes, MemoryTier::Ram) {
                    return Err(CheckpointError::Io(format!(
                        "Failed to register RAM tensor {} from checkpoint: {:?}",
                        entry.id, e
                    )));
                }
                actual_tier = MemoryTier::Ram;
            }
            MemoryTier::Ssd => {
                let kind = match kind {
                    Some(k) => k,
                    None => {
                        return Err(CheckpointError::MissingBlob(format!(
                            "SSD entry missing cache reference: {}",
                            entry.id
                        )));
                    }
                };
                let key = match key_opt {
                    Some(ref k) => k.clone(),
                    None => {
                        return Err(CheckpointError::MissingBlob(format!(
                            "SSD entry missing cache key: {}",
                            entry.id
                        )));
                    }
                };

                let bytes = match persistent.get_blob(kind, &key) {
                    Ok(b) => b,
                    Err(e) => {
                        return Err(map_pcache_error(
                            &format!("Failed to load SSD entry {} from cache", entry.id),
                            e,
                        ));
                    }
                };

                if let Err(e) = mem.register_tensor_with_data(&entry.id, bytes, MemoryTier::Ssd) {
                    return Err(CheckpointError::Io(format!(
                        "Failed to register SSD tensor {} from checkpoint: {:?}",
                        entry.id, e
                    )));
                }
                actual_tier = MemoryTier::Ssd;
            }
            MemoryTier::Vram => {
                // Hardware-agnostic: restore as RAM, even if VRAM was used before.
                let kind = match kind {
                    Some(k) => k,
                    None => {
                        return Err(CheckpointError::MissingBlob(format!(
                            "VRAM entry without cache reference: {}",
                            entry.id
                        )));
                    }
                };
                let key = match key_opt {
                    Some(ref k) => k.clone(),
                    None => {
                        return Err(CheckpointError::MissingBlob(format!(
                            "VRAM entry without cache key: {}",
                            entry.id
                        )));
                    }
                };

                let bytes = match persistent.get_blob(kind, &key) {
                    Ok(b) => b,
                    Err(e) => {
                        return Err(map_pcache_error(
                            &format!("Failed to load VRAM entry {} from cache", entry.id),
                            e,
                        ));
                    }
                };

                if let Err(e) = mem.register_tensor_with_data(&entry.id, bytes, MemoryTier::Ram) {
                    return Err(CheckpointError::Io(format!(
                        "Failed to register VRAM->RAM tensor {} from checkpoint: {:?}",
                        entry.id, e
                    )));
                }
                actual_tier = MemoryTier::Ram;
            }
            MemoryTier::Cpu => {
                // No defined policy for Cpu tier in this version.
                // Still allow hints to be restored.
            }
        }

        // Drift detection: this version only observes and reports, it does not
        // change behavior or return types.
        let mut drifts: Vec<drift::CheckpointDrift> = Vec::new();

        if let Some(desired) = desired_tier {
            // Missing backend: desired VRAM but no GPU backend attached.
            if !gpu_available && matches!(desired, MemoryTier::Vram) {
                drifts.push(drift::CheckpointDrift::MissingBackend { desired });
            }

            // Tier downgrade: desired VRAM but restored as RAM.
            if matches!(desired, MemoryTier::Vram) && matches!(actual_tier, MemoryTier::Ram) {
                drifts.push(drift::CheckpointDrift::TierDowngrade {
                    desired,
                    restored: actual_tier,
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

        // Apply placement hints after materializing the tensor.
        mem.set_desired_tier_hint(&entry.id, desired_tier);
        mem.set_last_plan_summary(&entry.id, last_plan_summary.clone());
    }

    Ok(checkpoint)
}
