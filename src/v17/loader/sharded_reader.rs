//! Sharded safetensors reader (M4.7.1.b).
//!
//! Loads multi-file HuggingFace safetensors checkpoints
//! (`model-NNNNN-of-NNNNN.safetensors` + `model.safetensors.index.json`)
//! by orchestrating one [`SafetensorsReader`] at a time and dropping
//! it before the next opens. Peak RAM during load stays bounded by
//! `max(shard_size) + max(per_tensor_F32_transient)` instead of the
//! sum of all shards.
//!
//! # Design
//!
//! Atenia's `TensorEntry<'a>` borrows from `SafetensorsReader::bytes`
//! by design (zero-copy decode). That makes a "single virtual reader
//! that rotates shards" approach incompatible with the borrow
//! checker without changing entry lifetimes — a breaking change for
//! the four M4.6 single-file production checkpoints.
//!
//! Instead, this module exposes a thin driver around the existing
//! [`SafetensorsReader`] and the new [`WeightMapper::load_one_shard_into`]
//! per-shard primitive. Each shard is opened, its entries are
//! decoded and copied into the graph, and the shard's
//! `Vec<u8>` is freed before the next shard opens. The single-file
//! [`SafetensorsReader::open`] / [`WeightMapper::load_into`] paths
//! are not touched — sharded loading is a pure addition.

use std::path::{Path, PathBuf};

use crate::amg::graph::Graph;

use super::loader_errors::LoaderError;
use super::safetensors_reader::SafetensorsReader;
use super::shard_index::ShardIndex;
use super::weight_mapper::{LoadReport, WeightMapper};

/// Driver that loads a sharded HuggingFace safetensors checkpoint
/// into an Atenia graph. Holds only the parsed
/// [`model.safetensors.index.json`](ShardIndex) at construction
/// time — shard files are opened on demand during
/// [`Self::load_into`] and dropped before the next shard opens.
pub struct ShardedSafetensorsReader {
    index: ShardIndex,
}

impl ShardedSafetensorsReader {
    /// Open a sharded checkpoint by parsing its index file.
    /// `index_path` should point at `model.safetensors.index.json`
    /// inside the model directory; shard filenames are resolved
    /// relative to its parent directory.
    pub fn open(index_path: &Path) -> Result<Self, LoaderError> {
        let index = ShardIndex::from_file(index_path)?;
        Ok(Self { index })
    }

    /// Construct directly from an already-parsed index. Useful in
    /// tests where the index is synthesised in memory.
    pub fn from_index(index: ShardIndex) -> Self {
        Self { index }
    }

    /// Borrow the parsed index.
    pub fn index(&self) -> &ShardIndex {
        &self.index
    }

    /// Number of distinct shard files this checkpoint comprises.
    pub fn shard_count(&self) -> usize {
        self.index.shard_count()
    }

    /// Number of tensors declared by the index across all shards.
    pub fn tensor_count(&self) -> usize {
        self.index.weight_map.len()
    }

    /// Distinct shard filenames in deterministic order.
    pub fn shard_filenames(&self) -> Vec<String> {
        self.index.shard_filenames()
    }

    /// Resolve a shard filename to its absolute path.
    pub fn shard_path(&self, shard_filename: &str) -> PathBuf {
        self.index.shard_path(shard_filename)
    }

    /// Load every tensor referenced by the index into the graph
    /// using `mapper`. Shards are processed one at a time; each
    /// shard's `Vec<u8>` raw buffer is freed before the next
    /// shard's file is read.
    ///
    /// The aggregated [`LoadReport`] mirrors the single-file
    /// semantics: `loaded` is the total count across shards;
    /// `skipped` is the union of per-shard skipped entries
    /// (file tensors not in the mapper); `missing` is the set of
    /// mapper names not satisfied by any shard.
    pub fn load_into(
        &self,
        graph: &mut Graph,
        mapper: &WeightMapper,
    ) -> Result<LoadReport, LoaderError> {
        let mut report = LoadReport::default();
        let mut satisfied: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        for shard_filename in self.index.shard_filenames() {
            let path = self.index.shard_path(&shard_filename);
            let reader = SafetensorsReader::open(&path)?;
            let outcome = mapper.load_one_shard_into(graph, &reader, &mut satisfied)?;
            report.loaded += outcome.loaded;
            report.skipped.extend(outcome.skipped);
            // `reader` (and its `Vec<u8>`) drops here before the
            // next shard opens — the load-bearing line of this
            // module.
            drop(reader);
        }

        report.missing = mapper.collect_missing(&satisfied);
        Ok(report)
    }
}
