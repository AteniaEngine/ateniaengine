use std::collections::HashMap;

use super::memory_types::{
    CompressionKind, MemoryFootprint, MemorySnapshot, MemoryTier, MoveError, MovePlan,
    StorageBacking, TensorResidence,
};
use super::persistent_cache::{CacheError, CacheKind, PersistentHybridCache};
use super::ssd_cache::SsdCache;
use super::vram_adapter::{NullVramAdapter, VramAdapter};

pub struct HybridMemoryManager {
    tensors: HashMap<String, TensorResidence>,
    cache: SsdCache,
    vram: Box<dyn VramAdapter + Send + Sync>,
    persistent: Option<PersistentHybridCache>,
    grad_cache: HashMap<String, String>,
    hints: HashMap<String, PlacementHints>,
}

#[derive(Debug, Clone, Default)]
struct PlacementHints {
    desired_tier: Option<MemoryTier>,
    last_plan_summary: Option<String>,
}

impl HybridMemoryManager {
    pub fn new(cache_dir: &str) -> Self {
        HybridMemoryManager::new_with_vram(cache_dir, Box::new(NullVramAdapter))
    }

    pub fn new_with_vram(cache_dir: &str, vram: Box<dyn VramAdapter + Send + Sync>) -> Self {
        HybridMemoryManager {
            tensors: HashMap::new(),
            cache: SsdCache::new(cache_dir),
            vram,
            persistent: None,
            grad_cache: HashMap::new(),
            hints: HashMap::new(),
        }
    }

    pub fn attach_persistent_cache(&mut self, cache: PersistentHybridCache) {
        self.persistent = Some(cache);
    }

    pub fn persistent_cache(&self) -> Option<&PersistentHybridCache> {
        self.persistent.as_ref()
    }

    pub fn register_tensor_with_data(
        &mut self,
        id: &str,
        data: Vec<u8>,
        initial: MemoryTier,
    ) -> Result<(), MoveError> {
        let footprint = MemoryFootprint {
            bytes: data.len() as u64,
        };

        let backing = match initial {
            MemoryTier::Ram => StorageBacking::Ram(data),
            MemoryTier::Ssd => {
                self.cache.ensure_dir()?;
                let path = self.cache.blob_path(id);
                // TODO: introduce a real compression policy; for now we always
                // write uncompressed data to keep behavior identical.
                let meta = self
                    .cache
                    .write_blob(&path, &data, CompressionKind::None)?;
                StorageBacking::SsdFile {
                    path,
                    compression: Some(meta),
                }
            }
            MemoryTier::Cpu | MemoryTier::Vram => StorageBacking::None,
        };

        let residence = TensorResidence {
            id: super::memory_types::TensorId(id.to_string()),
            tier: initial,
            footprint,
            backing,
        };

        self.tensors.insert(id.to_string(), residence);
        Ok(())
    }

    pub fn register_tensor(&mut self, id: &str, bytes: u64, initial: MemoryTier) {
        let residence = TensorResidence {
            id: super::memory_types::TensorId(id.to_string()),
            tier: initial,
            footprint: MemoryFootprint { bytes },
            backing: StorageBacking::None,
        };
        self.tensors.insert(id.to_string(), residence);
    }

    pub fn get_tier(&self, id: &str) -> Option<MemoryTier> {
        self.tensors.get(id).map(|r| r.tier)
    }

    pub fn tensor_len_bytes(&self, id: &str) -> Option<usize> {
        self.tensors
            .get(id)
            .map(|r| r.footprint.bytes as usize)
    }

    pub fn ids_for_checkpoint(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    pub fn is_grad_id(&self, id: &str) -> bool {
        self.grad_cache.contains_key(id)
    }

    pub fn set_desired_tier_hint(&mut self, id: &str, tier: Option<MemoryTier>) {
        let entry = self
            .hints
            .entry(id.to_string())
            .or_insert_with(PlacementHints::default);
        entry.desired_tier = tier;
    }

    pub fn get_desired_tier_hint(&self, id: &str) -> Option<MemoryTier> {
        self.hints
            .get(id)
            .and_then(|h| h.desired_tier)
    }

    pub fn set_last_plan_summary(&mut self, id: &str, summary: Option<String>) {
        let entry = self
            .hints
            .entry(id.to_string())
            .or_insert_with(PlacementHints::default);
        entry.last_plan_summary = summary;
    }

    pub fn get_last_plan_summary(&self, id: &str) -> Option<String> {
        match self.hints.get(id) {
            Some(h) => h.last_plan_summary.clone(),
            None => None,
        }
    }

    pub fn persist_gradient_to_ssd_cache(
        &mut self,
        grad_id: &str,
        bytes: &[u8],
        created_unix: u64,
        overwrite: bool,
    ) -> Result<(), CacheError> {
        let cache = match &self.persistent {
            Some(c) => c,
            None => {
                return Err(CacheError::Io(
                    "Persistent cache not attached to HybridMemoryManager".to_string(),
                ))
            }
        };

        let key = format!("grad:{}:len{}", grad_id, bytes.len());
        cache.put_blob(CacheKind::Gradient, &key, bytes, created_unix, overwrite)?;
        self.grad_cache.insert(grad_id.to_string(), key);
        Ok(())
    }

    pub fn restore_gradient_from_ssd_cache(
        &mut self,
        grad_id: &str,
    ) -> Result<Vec<u8>, CacheError> {
        let cache = match &self.persistent {
            Some(c) => c,
            None => {
                return Err(CacheError::Io(
                    "Persistent cache not attached to HybridMemoryManager".to_string(),
                ))
            }
        };

        let key = match self.grad_cache.get(grad_id) {
            Some(k) => k.clone(),
            None => {
                return Err(CacheError::NotFound);
            }
        };

        cache.get_blob(CacheKind::Gradient, &key)
    }

    pub fn plan_move(
        &self,
        id: &str,
        target: MemoryTier,
        _snapshot: &MemorySnapshot,
    ) -> Result<MovePlan, MoveError> {
        let residence = match self.tensors.get(id) {
            Some(r) => r,
            None => {
                return Err(MoveError::Unsupported(
                    "Tensor not registered".to_string(),
                ))
            }
        };

        if residence.tier == target {
            return Ok(MovePlan {
                from: residence.tier,
                to: target,
                reason: "Already in target tier".to_string(),
            });
        }

        if matches!(target, MemoryTier::Ssd) {
            // For tensors logically in RAM, we require concrete backing bytes
            // before planning a move to SSD. This protects against planning
            // impossible moves for tensors registered without data.
            if residence.tier == MemoryTier::Ram
                && matches!(residence.backing, StorageBacking::None)
            {
                return Err(MoveError::Unsupported(
                    "Cannot move tensor to SSD without backing data".to_string(),
                ));
            }

            // Ensure SSD cache directory exists before planning a move to SSD.
            // The actual data transfer and validation are handled in apply_move;
            // planning remains best-effort and only enforces that the cache
            // location is ready.
            self.cache.ensure_dir()?;
        }

        // If VRAM is requested but the adapter is not available, degrade the
        // plan to RAM to keep behavior stable on machines without GPUs.
        if matches!(target, MemoryTier::Vram) && !self.vram.is_available() {
            return Ok(MovePlan {
                from: residence.tier,
                to: MemoryTier::Ram,
                reason: "VRAM unavailable; degrading placement to RAM".to_string(),
            });
        }

        Ok(MovePlan {
            from: residence.tier,
            to: target,
            reason: "Planned logical move between tiers".to_string(),
        })
    }

    pub fn apply_move(&mut self, id: &str, plan: &MovePlan) -> Result<(), MoveError> {
        let residence = match self.tensors.get_mut(id) {
            Some(r) => r,
            None => {
                return Err(MoveError::Unsupported(
                    "Tensor not registered".to_string(),
                ))
            }
        };

        if residence.tier == plan.to {
            // No-op move.
            return Ok(());
        }

        match (residence.tier, &mut residence.backing, plan.to) {
            // RAM -> SSD: write blob (optionally compressed), persist in cache if attached,
            // and drop in-memory bytes.
            (MemoryTier::Ram, StorageBacking::Ram(data), MemoryTier::Ssd) => {
                self.cache.ensure_dir()?;
                let path = self.cache.blob_path(id);
                // TODO: introduce a compression policy; default is no compression.
                let meta = self
                    .cache
                    .write_blob(&path, data, CompressionKind::None)?;

                if let Some(pcache) = &self.persistent {
                    let logical_len = data.len();
                    let key = format!("tensor:{}:len{}", id, logical_len);
                    let _ = pcache.put_blob(
                        CacheKind::Tensor,
                        &key,
                        data,
                        0,
                        true,
                    );
                }

                *data = Vec::new();
                residence.backing = StorageBacking::SsdFile {
                    path,
                    compression: Some(meta),
                };
                residence.tier = MemoryTier::Ssd;
                Ok(())
            }
            // SSD -> RAM: read blob (with optional decompression) and best-effort delete file.
            (MemoryTier::Ssd, StorageBacking::SsdFile { path, compression }, MemoryTier::Ram) => {
                let bytes = match compression {
                    Some(meta) => self.cache.read_blob_with_meta(path, meta)?,
                    None => self.cache.read_blob(path)?,
                };
                residence.footprint.validate_len(bytes.len())?;
                self.cache.delete_blob(path)?;
                residence.backing = StorageBacking::Ram(bytes);
                residence.tier = MemoryTier::Ram;
                Ok(())
            }
            // RAM -> VRAM: best-effort upload, fallback-first.
            (MemoryTier::Ram, StorageBacking::Ram(data), MemoryTier::Vram) => {
                if !self.vram.is_available() {
                    // Keep tensor in RAM; placement was degraded at planning time.
                    return Ok(());
                }

                // Upload bytes to VRAM; on success, drop RAM copy.
                self.vram.upload(id, data)?;
                *data = Vec::new();
                residence.backing = StorageBacking::VramHandle {
                    key: id.to_string(),
                };
                residence.tier = MemoryTier::Vram;
                Ok(())
            }
            // VRAM -> RAM: download back to system memory.
            (MemoryTier::Vram, StorageBacking::VramHandle { key }, MemoryTier::Ram) => {
                if !self.vram.is_available() {
                    return Err(MoveError::BackendUnavailable(
                        "VRAM not available for download".to_string(),
                    ));
                }

                let bytes = self.vram.download(key)?;
                residence.footprint.validate_len(bytes.len())?;
                let _ = self.vram.free(key);
                residence.backing = StorageBacking::Ram(bytes);
                residence.tier = MemoryTier::Ram;
                Ok(())
            }
            // SSD -> VRAM: read from SSD (no decompression yet), upload to VRAM, optionally remove blob.
            (MemoryTier::Ssd, StorageBacking::SsdFile { path, .. }, MemoryTier::Vram) => {
                if !self.vram.is_available() {
                    // Planning should have degraded to RAM, but be defensive.
                    return Ok(());
                }

                let bytes = self.cache.read_blob(path)?;
                residence.footprint.validate_len(bytes.len())?;
                self.vram.upload(id, &bytes)?;
                let _ = self.cache.delete_blob(path);
                residence.backing = StorageBacking::VramHandle {
                    key: id.to_string(),
                };
                residence.tier = MemoryTier::Vram;
                Ok(())
            }
            // VRAM -> SSD: download from VRAM, write to SSD (uncompressed), free VRAM.
            (MemoryTier::Vram, StorageBacking::VramHandle { key }, MemoryTier::Ssd) => {
                if !self.vram.is_available() {
                    return Err(MoveError::BackendUnavailable(
                        "VRAM not available for download".to_string(),
                    ));
                }

                let bytes = self.vram.download(key)?;
                residence.footprint.validate_len(bytes.len())?;
                self.cache.ensure_dir()?;
                let path = self.cache.blob_path(id);
                let meta = self
                    .cache
                    .write_blob(&path, &bytes, CompressionKind::None)?;

                if let Some(pcache) = &self.persistent {
                    let key = format!("tensor:{}:len{}", id, residence.footprint.bytes as usize);
                    let _ = pcache.put_blob(
                        CacheKind::Tensor,
                        &key,
                        &bytes,
                        0,
                        true,
                    );
                }

                let _ = self.vram.free(key);
                residence.backing = StorageBacking::SsdFile {
                    path,
                    compression: Some(meta),
                };
                residence.tier = MemoryTier::Ssd;
                Ok(())
            }
            // Logical moves involving CPU/VRAM: keep backing as-is, update tier.
            (_, StorageBacking::None, new_tier) => {
                residence.tier = new_tier;
                Ok(())
            }
            (_, StorageBacking::Ram(_), MemoryTier::Cpu | MemoryTier::Vram)
            | (_, StorageBacking::SsdFile { .. }, MemoryTier::Cpu | MemoryTier::Vram) => {
                // Backing is kept while the logical tier changes; real VRAM
                // transfers will be introduced in later subversions.
                residence.tier = plan.to;
                Ok(())
            }
            // Other combinations (e.g. RAM<->RAM, SSD<->SSD) are simple tier updates.
            _ => {
                residence.tier = plan.to;
                Ok(())
            }
        }
    }

    // Helper primarily intended for tests to inspect the current backing.
    pub fn backing_for_test(&self, id: &str) -> Option<&StorageBacking> {
        self.tensors.get(id).map(|r| &r.backing)
    }

    // Helper intended for tests to force a footprint change and validate
    // negative paths like length mismatches.
    pub fn set_footprint_bytes_for_test(&mut self, id: &str, bytes: u64) {
        if let Some(r) = self.tensors.get_mut(id) {
            r.footprint.bytes = bytes;
        }
    }

    pub fn remove_for_test(&mut self, id: &str) {
        let _ = self.tensors.remove(id);
    }
}
