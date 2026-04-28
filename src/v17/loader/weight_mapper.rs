//! Weight mapper (M4-c, extended for M4.5-b1).
//!
//! Formalizes the "safetensors tensor name → graph parameter node_id"
//! mapping that M4-b validated ad-hoc via `HashMap::from_iter`, and
//! adds the validations that the raw iterator approach did not perform:
//!
//! - **Shape**: every tensor loaded from the safetensors file must
//!   match the shape of the parameter tensor already materialized in
//!   the graph at build time, **after any registered transformations
//!   have been applied**. Mismatch surfaces as
//!   [`LoaderError::ShapeMismatch`] with both shapes attached.
//! - **Dtype**: only F32 is supported in M4-c. BF16 / F16 / FP8 fall
//!   through as [`LoaderError::UnsupportedDType`] via the M4-a
//!   `TensorEntry::to_vec_f32` path. M4-d extends decode coverage.
//!
//! Mismatches between the mapper's name set and the reader's name
//! set are **not** fatal in M4-c. `load_into` reports them through
//! [`LoadReport::skipped`] (tensors in the file but not in the
//! mapper — e.g. producer metadata tensors) and [`LoadReport::missing`]
//! (tensors the mapper expected but the file did not provide —
//! e.g. partial checkpoints). Callers decide whether to treat them
//! as errors. A future strict-mode entry point can be added without
//! changing the loose-mode surface.
//!
//! # M4.5-b1 extension: `LoadTransform`
//!
//! HuggingFace and Atenia disagree on Linear-weight layout
//! (`[out, in]` vs `[in, out]`), and TinyLlama-style models use
//! Grouped Query Attention which Atenia's M4.5 graph treats as MHA
//! by tiling K/V projections at load time. To keep the graph
//! free of model-specific reshaping plumbing, the mapper applies
//! a small per-tensor transform pipeline between decode and copy:
//! `decode_f32 → transforms → shape_check → copy`.
//!
//! Existing call sites that build a mapper via
//! [`WeightMapper::from_param_names_and_ids`] without configuring
//! transforms get the original M4-c behavior bit-exact: empty
//! transform list ≡ direct copy.
//!
//! # Design notes
//!
//! The mapper is intentionally decoupled from `MiniFluxHandles` or
//! any specific architecture struct. Its constructor accepts two
//! index-aligned slices (names + node_ids) that any `build_*` helper
//! in `src/nn/` can produce. TinyLlama-specific transform wiring
//! lives in `src/nn/tinyllama/weight_loading.rs`.

use std::collections::HashMap;

use crate::amg::graph::Graph;

use super::loader_errors::LoaderError;
use super::safetensors_reader::SafetensorsReader;

/// A single transformation applied to a tensor's decoded float values
/// between safetensors decode and graph copy. Multiple transforms are
/// applied in the order they appear in [`WeightMapping::transforms`].
#[derive(Debug, Clone, PartialEq)]
pub enum LoadTransform {
    /// 2D matrix transpose `[a, b] -> [b, a]`. Used to convert
    /// HuggingFace Linear weights (stored as `[out_features, in_features]`)
    /// into Atenia's `[in_features, out_features]` convention.
    Transpose2D,

    /// Repeat each consecutive `group_size`-block along `dim`
    /// `repeats` times. For GQA K/V expansion to MHA-equivalent
    /// shape: applied to a K/V projection weight in HF layout
    /// `[n_kv_heads * head_dim, hidden]` with
    /// `dim=0, group_size=head_dim, repeats=kv_groups`, this produces
    /// `[n_q_heads * head_dim, hidden]`. The result is bit-exact to
    /// what `torch.repeat_interleave(K, dim=2, repeats=kv_groups)`
    /// would produce on the runtime tensor `[b, s, n_kv, d]` after
    /// the projection.
    TileGroupedDim {
        dim: usize,
        group_size: usize,
        repeats: usize,
    },

    /// Multiply every element by `factor`. Used to pre-fold the
    /// `1/sqrt(head_dim)` attention scale into K_proj weights so
    /// the graph does not need a separate scaling node.
    Scale { factor: f32 },

    /// Reshape (metadata only) the loaded tensor to `target`. The
    /// new shape must have the same numel as the current shape;
    /// data is left untouched (Layout::Contiguous preserved).
    /// Use case: aligning a 1D RMSNorm `[hidden]` gamma with the
    /// graph parameter `[1, 1, hidden]` that `BroadcastMul`
    /// expects (M4.5-b1).
    Reshape { target: Vec<usize> },
}

/// One entry in [`WeightMapper`]: the graph parameter node that
/// receives this tensor's values and the optional list of transforms
/// applied between decode and copy.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightMapping {
    pub node_id: usize,
    pub transforms: Vec<LoadTransform>,
}

/// Bidirectional view from a tensor's logical name (as used inside
/// `register_weight`-style builders) to the graph parameter node that
/// holds its values, plus the transforms applied to that tensor on
/// load. Construct with [`WeightMapper::from_param_names_and_ids`]
/// and configure transforms with [`WeightMapper::set_transforms`].
#[derive(Debug, Clone)]
pub struct WeightMapper {
    mapping: HashMap<String, WeightMapping>,
    /// When `true`, every parameter loaded into the graph is
    /// down-converted from F32 to BF16 just before being copied
    /// into the destination `Tensor` (M4.7.2). The
    /// `LoadTransform` pipeline still runs in F32 (Scale,
    /// TileGroupedDim, Transpose2D, Reshape need F32 working
    /// values), so the BF16 down-convert is a single-pass
    /// truncation of the post-transform F32 buffer. Default
    /// `false` for full backward compatibility with the M4.6
    /// numerical-validation fixtures and every existing test.
    store_params_as_bf16: bool,
}

/// Summary returned by [`WeightMapper::load_into`] after a
/// loose-mode load. `skipped` and `missing` are informational in
/// M4-c; callers can choose to enforce emptiness at their own layer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadReport {
    pub loaded: usize,
    pub skipped: Vec<String>,
    pub missing: Vec<String>,
}

impl WeightMapper {
    /// Build a mapper from two index-aligned slices. Each entry
    /// starts with an empty transform list (bit-exact backward
    /// compatibility with M4-c). Configure transforms with
    /// [`Self::set_transforms`].
    ///
    /// Fails with [`LoaderError::InvalidFormat`] if the slices have
    /// different lengths, or if the name slice has duplicates (the
    /// resulting `HashMap` would silently drop collisions — surfacing
    /// the duplicate as an error is preferable to a hard-to-debug
    /// load).
    pub fn from_param_names_and_ids(
        names: &[String],
        ids: &[usize],
    ) -> Result<Self, LoaderError> {
        if names.len() != ids.len() {
            return Err(LoaderError::InvalidFormat(format!(
                "WeightMapper: names.len()={} differs from ids.len()={}; \
                 the two slices must be index-aligned",
                names.len(),
                ids.len()
            )));
        }

        let mut mapping: HashMap<String, WeightMapping> =
            HashMap::with_capacity(names.len());
        for (name, id) in names.iter().zip(ids.iter()) {
            let entry = WeightMapping {
                node_id: *id,
                transforms: Vec::new(),
            };
            if mapping.insert(name.clone(), entry).is_some() {
                return Err(LoaderError::InvalidFormat(format!(
                    "WeightMapper: duplicate parameter name '{}'; \
                     a safetensors checkpoint cannot disambiguate a \
                     duplicated name, so refusing to build a mapper \
                     that would silently prefer one node over another",
                    name
                )));
            }
        }

        Ok(Self {
            mapping,
            store_params_as_bf16: false,
        })
    }

    /// Toggle the BF16 storage path (M4.7.2). When `true`, every
    /// parameter is down-converted from F32 to `Vec<u16>` after the
    /// `LoadTransform` pipeline runs, and stored as
    /// `TensorStorage::CpuBf16` instead of `TensorStorage::Cpu`.
    /// Halves the persistent RAM footprint of model parameters.
    /// See module-level documentation and the M4.7.2.a commit for
    /// the BF16 storage contract; the precision impact was
    /// validated empirically by the spike (commit `a786837`).
    ///
    /// Returns `&mut Self` to allow fluent configuration.
    pub fn set_store_params_as_bf16(&mut self, enabled: bool) -> &mut Self {
        self.store_params_as_bf16 = enabled;
        self
    }

    /// Whether the BF16 storage path is currently active. Used by
    /// tests and diagnostic logging.
    pub fn store_params_as_bf16(&self) -> bool {
        self.store_params_as_bf16
    }

    /// Attach a transform pipeline to a parameter name. The transforms
    /// are applied in the order given between decode and copy. Returns
    /// [`LoaderError::InvalidFormat`] if the name is not in the mapper
    /// (defensive: caller likely intended a different name).
    pub fn set_transforms(
        &mut self,
        name: &str,
        transforms: Vec<LoadTransform>,
    ) -> Result<(), LoaderError> {
        match self.mapping.get_mut(name) {
            Some(entry) => {
                entry.transforms = transforms;
                Ok(())
            }
            None => Err(LoaderError::InvalidFormat(format!(
                "WeightMapper::set_transforms: parameter name '{}' is not in \
                 the mapper; check the name against the parameter list",
                name
            ))),
        }
    }

    /// Number of entries in the mapper.
    pub fn len(&self) -> usize {
        self.mapping.len()
    }

    /// Whether the mapper contains a given parameter name.
    pub fn contains(&self, name: &str) -> bool {
        self.mapping.contains_key(name)
    }

    /// Look up a parameter name. Returns the node_id of the graph
    /// parameter that holds its tensor, or `None` if the name is not
    /// mapped.
    pub fn get(&self, name: &str) -> Option<usize> {
        self.mapping.get(name).map(|m| m.node_id)
    }

    /// Borrow the full mapping entry (node_id + transforms) for a
    /// name. Useful in tests and in the TinyLlama helper to verify
    /// the configured transform list.
    pub fn get_mapping(&self, name: &str) -> Option<&WeightMapping> {
        self.mapping.get(name)
    }

    /// Load every tensor from `reader` whose name is present in the
    /// mapper into the corresponding graph parameter node. Decodes,
    /// applies transforms, validates shape post-transform, then copies.
    ///
    /// Loose-mode: missing entries (in mapper, absent from reader)
    /// and skipped entries (in reader, absent from mapper) are
    /// reported through [`LoadReport`] rather than raising errors.
    /// Shape mismatch and dtype-unsupported are hard errors — the
    /// load aborts on the first occurrence.
    pub fn load_into(
        &self,
        graph: &mut Graph,
        reader: &SafetensorsReader,
    ) -> Result<LoadReport, LoaderError> {
        // Single-shard load: thin wrapper over the per-shard inner
        // loop. Behaviour is bit-identical to the original M4-c
        // implementation; the refactor exists so
        // `ShardedSafetensorsReader` (M4.7.1.b) can reuse the inner
        // loop across multiple shard files without duplicating the
        // decode-transform-copy logic.
        let mut report = LoadReport::default();
        let mut satisfied: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        let outcome = self.load_one_shard_into(graph, reader, &mut satisfied)?;
        report.loaded = outcome.loaded;
        report.skipped = outcome.skipped;
        report.missing = self.collect_missing(&satisfied);
        Ok(report)
    }

    /// Per-shard load primitive. Iterates `reader`, decodes each
    /// tensor, applies transforms, copies into the graph parameter,
    /// and records every mapper name it satisfied into
    /// `satisfied_names`. Tensors present in the file but absent
    /// from the mapper are returned in `ShardLoadOutcome.skipped`;
    /// the caller aggregates `skipped` across shards.
    ///
    /// `satisfied_names` is the cross-shard accumulator that lets
    /// the sharded driver compute the global `missing` list at the
    /// end (see [`Self::collect_missing`]). For a single-shard load
    /// the caller may pass an empty set and discard it.
    pub fn load_one_shard_into(
        &self,
        graph: &mut Graph,
        reader: &SafetensorsReader,
        satisfied_names: &mut std::collections::HashSet<String>,
    ) -> Result<ShardLoadOutcome, LoaderError> {
        let mut outcome = ShardLoadOutcome::default();

        for entry in reader.iter() {
            let Some(mapping) = self.mapping.get(entry.name) else {
                outcome.skipped.push(entry.name.to_string());
                continue;
            };

            let node_id = mapping.node_id;

            // Decode raw values from the safetensors entry. Shape
            // here is the file shape (pre-transform).
            let mut values = entry.to_vec_f32()?;
            let mut current_shape: Vec<usize> = entry.shape.to_vec();

            // Apply registered transforms in order. Each transform
            // updates `values` and `current_shape` in lockstep.
            for transform in &mapping.transforms {
                match transform {
                    LoadTransform::Transpose2D => {
                        if current_shape.len() != 2 {
                            return Err(LoaderError::InvalidFormat(format!(
                                "tensor '{}': Transpose2D requires a 2D tensor, \
                                 got rank {} (shape {:?})",
                                entry.name,
                                current_shape.len(),
                                current_shape
                            )));
                        }
                        let rows = current_shape[0];
                        let cols = current_shape[1];
                        values = transpose_2d_flat(&values, rows, cols);
                        current_shape = vec![cols, rows];
                    }
                    LoadTransform::TileGroupedDim {
                        dim,
                        group_size,
                        repeats,
                    } => {
                        values = tile_grouped_dim(
                            &values,
                            &current_shape,
                            *dim,
                            *group_size,
                            *repeats,
                        )
                        .map_err(|msg| {
                            LoaderError::InvalidFormat(format!(
                                "tensor '{}': {}",
                                entry.name, msg
                            ))
                        })?;
                        current_shape[*dim] *= *repeats;
                    }
                    LoadTransform::Scale { factor } => {
                        for v in values.iter_mut() {
                            *v *= *factor;
                        }
                    }
                    LoadTransform::Reshape { target } => {
                        let target_numel: usize = target.iter().product();
                        let current_numel: usize = current_shape.iter().product();
                        if target_numel != current_numel {
                            return Err(LoaderError::InvalidFormat(format!(
                                "tensor '{}': Reshape target {:?} (numel {}) does not match \
                                 current shape {:?} (numel {})",
                                entry.name,
                                target,
                                target_numel,
                                current_shape,
                                current_numel
                            )));
                        }
                        // Pure metadata change — data layout is unchanged.
                        current_shape = target.clone();
                    }
                }
            }

            // Look up the graph parameter and validate the
            // post-transform shape against it.
            let tensor = graph
                .nodes
                .get_mut(node_id)
                .and_then(|n| n.output.as_mut())
                .ok_or_else(|| {
                    LoaderError::InvalidFormat(format!(
                        "mapper points '{}' at node_id {} but that node has no \
                         materialized tensor in the graph",
                        entry.name, node_id
                    ))
                })?;

            if current_shape != tensor.shape.as_slice() {
                return Err(LoaderError::ShapeMismatch {
                    tensor_name: entry.name.to_string(),
                    expected: tensor.shape.clone(),
                    actual: current_shape,
                });
            }

            // Element-count check, valid for both Cpu (`as_cpu_slice_mut`)
            // and CpuBf16 (`numel()`) destinations. We cannot pre-emptively
            // borrow the F32 slice here because the BF16 path will mutate
            // `tensor.storage` to a different variant.
            let dest_numel = tensor.numel();
            if dest_numel != values.len() {
                return Err(LoaderError::InvalidFormat(format!(
                    "tensor '{}': element count {} after transforms does not match \
                     graph parameter capacity {}",
                    entry.name,
                    values.len(),
                    dest_numel
                )));
            }

            // M4.7.2 spike — BF16 precision-floor simulation.
            //
            // When the env var `ATENIA_BF16_PRECISION_FLOOR=1` is
            // set, every loaded parameter is round-tripped through
            // BF16 quantisation just before being copied into the
            // graph parameter. This faithfully simulates the
            // precision impact of native BF16 storage (every
            // parameter value is forced onto the BF16 grid) without
            // changing `TensorStorage` or the engine's F32 ops. The
            // transforms (Scale, TileGroupedDim, …) run in full F32
            // first; the floor applies to their output, mirroring
            // the "decode → transform → quantise → store as BF16"
            // pipeline a real BF16 storage layer would implement.
            //
            // BF16 retains the 8 F32 exponent bits and the upper 7
            // F32 mantissa bits; the lower 16 mantissa bits are
            // zeroed. Implemented inline to avoid a half-crate
            // dependency for the spike.
            if std::env::var("ATENIA_BF16_PRECISION_FLOOR")
                .ok()
                .as_deref()
                == Some("1")
            {
                for v in values.iter_mut() {
                    let bits = v.to_bits() & 0xFFFF_0000_u32;
                    *v = f32::from_bits(bits);
                }
            }

            if self.store_params_as_bf16 {
                // M4.7.2: native BF16 storage path. The post-transform
                // F32 working buffer is down-converted to `Vec<u16>` via
                // truncation (`(v.to_bits() >> 16) as u16`) and assigned
                // to the destination tensor as `TensorStorage::CpuBf16`.
                // The persistent footprint is half the F32 path; the
                // precision impact was validated by the spike
                // (commit `a786837`) and is bit-exact equivalent because
                // the precision-floor simulation above and this
                // down-convert apply the same operation. Round-trip on
                // decode-on-access is exactly the same value.
                let bits: Vec<u16> =
                    values.iter().map(|&v| (v.to_bits() >> 16) as u16).collect();
                tensor.set_cpu_bf16_bits(bits);
            } else {
                let slice = tensor.as_cpu_slice_mut();
                slice.copy_from_slice(&values);
            }
            outcome.loaded += 1;
            satisfied_names.insert(entry.name.to_string());
        }

        Ok(outcome)
    }

    /// Compute the cross-shard `missing` list: mapper names that
    /// were never satisfied by any shard. Used by both the
    /// single-shard `load_into` and the sharded driver to fill the
    /// final `LoadReport.missing` field.
    pub fn collect_missing(
        &self,
        satisfied_names: &std::collections::HashSet<String>,
    ) -> Vec<String> {
        self.mapping
            .keys()
            .filter(|name| !satisfied_names.contains(name.as_str()))
            .cloned()
            .collect()
    }
}

/// Outcome of a single-shard load. Aggregated across shards by
/// [`crate::v17::loader::sharded_reader::ShardedSafetensorsReader`]
/// to produce a final [`LoadReport`].
///
/// `skipped` is per-shard (tensors in this shard that the mapper
/// does not know about). `loaded` is per-shard count. `missing` is
/// **not** computed at the per-shard level — it is meaningless
/// without seeing every shard, since a tensor missing from shard 0
/// might be present in shard 1. The caller computes it once at the
/// end via [`WeightMapper::collect_missing`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShardLoadOutcome {
    pub loaded: usize,
    pub skipped: Vec<String>,
}

// ---------------------------------------------------------------------
// Transform helpers (free functions, also used by tests)
// ---------------------------------------------------------------------

/// 2D matrix transpose on a flat row-major buffer.
/// `values` has logical shape `[rows, cols]`; the result has logical
/// shape `[cols, rows]`.
pub(crate) fn transpose_2d_flat(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    debug_assert_eq!(values.len(), rows * cols);
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = values[r * cols + c];
        }
    }
    out
}

/// Repeat each consecutive `group_size`-block along `dim` `repeats`
/// times. Returns a new flat buffer in row-major contiguous layout
/// for the post-tile shape.
///
/// Returns `Err(message)` if `shape[dim]` is not a multiple of
/// `group_size` or if `dim` is out of range.
pub(crate) fn tile_grouped_dim(
    values: &[f32],
    shape: &[usize],
    dim: usize,
    group_size: usize,
    repeats: usize,
) -> Result<Vec<f32>, String> {
    if dim >= shape.len() {
        return Err(format!(
            "TileGroupedDim: dim {} out of range for shape {:?}",
            dim, shape
        ));
    }
    if group_size == 0 {
        return Err("TileGroupedDim: group_size must be positive".into());
    }
    if repeats == 0 {
        return Err("TileGroupedDim: repeats must be positive".into());
    }
    let current_d = shape[dim];
    if current_d % group_size != 0 {
        return Err(format!(
            "TileGroupedDim: shape[{}]={} is not divisible by group_size={}",
            dim, current_d, group_size
        ));
    }
    let expected_in: usize = shape.iter().product();
    if values.len() != expected_in {
        return Err(format!(
            "TileGroupedDim: input length {} does not match shape {:?} (product {})",
            values.len(),
            shape,
            expected_in
        ));
    }

    let outer: usize = shape[..dim].iter().product();
    let inner: usize = shape[dim + 1..].iter().product();
    let n_groups = current_d / group_size;
    let new_d = current_d * repeats;
    let mut out = vec![0.0_f32; outer * new_d * inner];

    for o in 0..outer {
        let in_outer = o * current_d * inner;
        let out_outer = o * new_d * inner;
        for g in 0..n_groups {
            let in_group = in_outer + g * group_size * inner;
            let out_group_base = out_outer + (g * repeats) * group_size * inner;
            for r in 0..repeats {
                let out_block = out_group_base + r * group_size * inner;
                let block_len = group_size * inner;
                out[out_block..out_block + block_len]
                    .copy_from_slice(&values[in_group..in_group + block_len]);
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_param_names_and_ids_rejects_length_mismatch() {
        let names = vec!["a".to_string(), "b".to_string()];
        let ids = vec![1usize];
        let err = WeightMapper::from_param_names_and_ids(&names, &ids)
            .expect_err("length mismatch must fail");
        match err {
            LoaderError::InvalidFormat(msg) => {
                assert!(msg.contains("index-aligned"));
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    #[test]
    fn from_param_names_and_ids_rejects_duplicates() {
        let names = vec!["dup".to_string(), "dup".to_string()];
        let ids = vec![1usize, 2usize];
        let err = WeightMapper::from_param_names_and_ids(&names, &ids)
            .expect_err("duplicate name must fail");
        match err {
            LoaderError::InvalidFormat(msg) => {
                assert!(msg.contains("duplicate"));
            }
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }

    #[test]
    fn from_param_names_and_ids_accepts_empty() {
        let mapper = WeightMapper::from_param_names_and_ids(&[], &[])
            .expect("empty is legal (zero-param model)");
        assert_eq!(mapper.len(), 0);
        assert!(!mapper.contains("anything"));
        assert_eq!(mapper.get("anything"), None);
    }

    #[test]
    fn fresh_mapping_has_no_transforms() {
        let names = vec!["w".to_string()];
        let ids = vec![42usize];
        let mapper = WeightMapper::from_param_names_and_ids(&names, &ids).unwrap();
        let entry = mapper.get_mapping("w").unwrap();
        assert_eq!(entry.node_id, 42);
        assert!(entry.transforms.is_empty());
    }

    #[test]
    fn set_transforms_attaches_pipeline() {
        let names = vec!["w".to_string()];
        let ids = vec![7usize];
        let mut mapper = WeightMapper::from_param_names_and_ids(&names, &ids).unwrap();
        mapper
            .set_transforms("w", vec![LoadTransform::Transpose2D])
            .unwrap();
        let entry = mapper.get_mapping("w").unwrap();
        assert_eq!(entry.transforms, vec![LoadTransform::Transpose2D]);
    }

    #[test]
    fn set_transforms_rejects_unknown_name() {
        let mut mapper = WeightMapper::from_param_names_and_ids(&[], &[]).unwrap();
        let err = mapper
            .set_transforms("nope", vec![LoadTransform::Transpose2D])
            .expect_err("unknown name must fail");
        match err {
            LoaderError::InvalidFormat(msg) => assert!(msg.contains("not in the mapper")),
            other => panic!("expected InvalidFormat, got {:?}", other),
        }
    }
}
