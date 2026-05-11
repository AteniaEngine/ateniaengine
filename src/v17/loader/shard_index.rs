//! HuggingFace `model.safetensors.index.json` parser (M4.7.1.a).
//!
//! Sharded safetensors checkpoints (Llama 2 13B, Mistral 7B,
//! Qwen 2.5 7B, …) ship as N files plus a single
//! `model.safetensors.index.json` that maps each tensor name to the
//! file it lives in. This module parses that index into a typed
//! `ShardIndex` and resolves shard filenames to absolute paths
//! relative to the index file's directory.
//!
//! Format (verified against Mistral 7B v0.3 and Qwen 2.5 7B):
//! ```json
//! {
//!   "metadata": { "total_size": 14496047104 },
//!   "weight_map": {
//!     "model.embed_tokens.weight": "model-00001-of-00003.safetensors",
//!     "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00003.safetensors",
//!     ...
//!   }
//! }
//! ```
//!
//! Per the safetensors sharded spec, every tensor lives in exactly
//! one shard — `weight_map` is injective by construction. This
//! parser rejects duplicate tensor keys defensively (a corrupted or
//! hand-edited index could violate the invariant).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::v17::loader::loader_errors::LoaderError;

/// Parsed `model.safetensors.index.json`.
///
/// `weight_map` keys are tensor names (HuggingFace convention,
/// e.g. `model.layers.0.self_attn.q_proj.weight`). Values are bare
/// shard filenames as they appear in the JSON, **not** absolute
/// paths — call [`Self::shard_path`] to resolve a shard to its
/// absolute location.
///
/// Stored as `BTreeMap` for deterministic iteration order, which
/// matters for tests that compare `LoadReport.missing` lists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardIndex {
    /// Total size in bytes of all shards combined, as declared by
    /// the index. Useful as a sanity check against
    /// `sum(fs::metadata(shard).len())`. `0` when the field is
    /// absent (older indexes).
    pub total_size: u64,
    /// Tensor name → shard filename (bare, not absolute).
    pub weight_map: BTreeMap<String, String>,
    /// Directory containing the index file. Shard filenames in
    /// `weight_map` are resolved relative to this path.
    pub base_dir: PathBuf,
}

impl ShardIndex {
    /// Parse an index from a JSON string. Use [`Self::from_file`]
    /// when working with on-disk indexes — it captures the parent
    /// directory automatically for path resolution.
    pub fn from_json_str(s: &str, base_dir: PathBuf) -> Result<Self, LoaderError> {
        let v: Value = serde_json::from_str(s).map_err(|e| {
            LoaderError::InvalidFormat(format!("shard index JSON parse error: {}", e))
        })?;

        let total_size = v
            .get("metadata")
            .and_then(|m| m.get("total_size"))
            .and_then(|n| n.as_u64())
            .unwrap_or(0);

        let weight_map_value = v.get("weight_map").ok_or_else(|| {
            LoaderError::InvalidFormat(
                "shard index missing required `weight_map` field".to_string(),
            )
        })?;

        let weight_map_obj = weight_map_value.as_object().ok_or_else(|| {
            LoaderError::InvalidFormat("shard index `weight_map` must be a JSON object".to_string())
        })?;

        if weight_map_obj.is_empty() {
            return Err(LoaderError::InvalidFormat(
                "shard index `weight_map` is empty".to_string(),
            ));
        }

        let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
        for (tensor_name, shard_value) in weight_map_obj {
            let shard_filename = shard_value.as_str().ok_or_else(|| {
                LoaderError::InvalidFormat(format!(
                    "shard index `weight_map` value for `{}` is not a string",
                    tensor_name
                ))
            })?;
            // BTreeMap::insert returns Some(prev) on duplicate; the
            // JSON object syntax allows duplicate keys (last wins),
            // which is silent corruption. Reject explicitly.
            if let Some(prior) = weight_map.insert(tensor_name.clone(), shard_filename.to_string())
            {
                return Err(LoaderError::InvalidFormat(format!(
                    "shard index has duplicate entry for tensor `{}`: \
                     first mapped to `{}`, then to `{}`",
                    tensor_name, prior, shard_filename
                )));
            }
        }

        Ok(Self {
            total_size,
            weight_map,
            base_dir,
        })
    }

    /// Read and parse an index from a file path. The file's parent
    /// directory is captured for shard path resolution; if the path
    /// has no parent (root file system), the current working
    /// directory is used.
    pub fn from_file(path: &Path) -> Result<Self, LoaderError> {
        let s = fs::read_to_string(path).map_err(|e| {
            LoaderError::IoError(format!("failed to read {}: {}", path.display(), e))
        })?;
        let base_dir = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        Self::from_json_str(&s, base_dir)
    }

    /// Number of distinct shard files referenced by the index.
    pub fn shard_count(&self) -> usize {
        let mut shards: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for shard in self.weight_map.values() {
            shards.insert(shard.as_str());
        }
        shards.len()
    }

    /// Iterator over distinct shard filenames (bare, not absolute),
    /// in deterministic alphabetical order.
    pub fn shard_filenames(&self) -> Vec<String> {
        let mut shards: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for shard in self.weight_map.values() {
            shards.insert(shard.clone());
        }
        shards.into_iter().collect()
    }

    /// Resolve a shard filename to its absolute path under
    /// `base_dir`.
    pub fn shard_path(&self, shard_filename: &str) -> PathBuf {
        self.base_dir.join(shard_filename)
    }

    /// Group tensor names by the shard they belong to. Useful for
    /// the per-shard load loop in `ShardedSafetensorsReader`.
    /// Returned map preserves deterministic iteration order.
    pub fn tensors_by_shard(&self) -> BTreeMap<String, Vec<String>> {
        let mut by_shard: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (tensor_name, shard) in &self.weight_map {
            by_shard
                .entry(shard.clone())
                .or_default()
                .push(tensor_name.clone());
        }
        by_shard
    }

    /// Look up the shard filename for a single tensor.
    pub fn shard_for(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(String::as_str)
    }
}
