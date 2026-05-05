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

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::amg::graph::Graph;
use crate::amg::weight_store::{SharedParam, WeightStore, WeightStoreError};
use crate::gpu::tier_plan::{Tier, TierPlan};
use crate::tensor::DType;

use super::loader_errors::LoaderError;
use super::safetensors_reader::SafetensorsReader;

/// **M6 replan sub-fase 2** — counters that record which
/// upload sub-path each `Tier::Vram` parameter took during
/// `load_into_with_residency_plan`.
///
/// Two paths exist:
/// - **Fast** — BF16 source + zero `LoadTransform`s. The raw
///   safetensors bytes are shipped to VRAM with a single H→D
///   memcpy via
///   [`crate::cuda::bf16_to_f32::bf16_to_f32_resident_in_vram_from_raw_bytes`].
///   No host-side F32 transient.
/// - **Slow** — F32 source, or BF16 + at least one transform.
///   The transforms must run on F32 on the host
///   (`Transpose2D`, `TileGroupedDim`, `Scale`); this path
///   materialises a per-entry `Vec<f32>`, performs the
///   transforms, down-converts to BF16 bits, and uploads the
///   bits via the standard
///   [`crate::cuda::bf16_to_f32::bf16_to_f32_resident_in_vram`]
///   wrapper. The F32 transient drops at the end of the
///   iteration.
///
/// Tests assert that an "all-BF16, no-transforms" plan goes
/// 100% through the fast path — confirming no F32 transient
/// was allocated for any Vram-tier upload.
static VRAM_FAST_PATH_COUNT: AtomicUsize = AtomicUsize::new(0);
static VRAM_SLOW_PATH_COUNT: AtomicUsize = AtomicUsize::new(0);
/// **M8.4** — counts `Tier::Vram + dtype = BF16 + no-transforms`
/// loads that took the **BF16-resident** path (no upcast to F32
/// at load time, the BF16 buffer stays as BF16 in VRAM and is
/// consumed directly by `cuda_matmul_bf16_inplace` at dispatch
/// time).
///
/// Strictly disjoint from `VRAM_FAST_PATH_COUNT`: a single load
/// of a BF16 + no-transforms tensor under `Tier::Vram` will
/// increment exactly one of the two counters, depending on the
/// `ATENIA_M8_BF16_KERNEL` flag. This makes the M8.4 wire-up
/// auditable from a smoke log without instrumenting the
/// dispatcher.
static VRAM_BF16_FAST_PATH_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Returns the number of `Tier::Vram` uploads that took the
/// raw-bytes fast path since process start.
pub fn vram_fast_path_count() -> usize {
    VRAM_FAST_PATH_COUNT.load(Ordering::Relaxed)
}

/// Returns the number of `Tier::Vram` uploads that took the
/// transforms / F32-source slow path since process start.
pub fn vram_slow_path_count() -> usize {
    VRAM_SLOW_PATH_COUNT.load(Ordering::Relaxed)
}

/// **M8.4** — Returns the number of `Tier::Vram` uploads that
/// took the BF16-resident path (M8 kernel enabled) since process
/// start. See [`VRAM_BF16_FAST_PATH_COUNT`].
pub fn vram_bf16_fast_path_count() -> usize {
    VRAM_BF16_FAST_PATH_COUNT.load(Ordering::Relaxed)
}

/// **M8.4b** — counts `Tier::Vram + ATENIA_M8_BF16_KERNEL=1` loads
/// that ran through the **slow path** (i.e. the parameter has
/// `LoadTransform`s registered, so the raw-bytes fast path can't
/// handle it). The slow path materialises an F32 working buffer,
/// applies the transforms (Transpose2D / TileGroupedDim / Scale /
/// Reshape), then under M8 re-encodes the transformed F32 to
/// BF16 and uploads via [`bf16_to_vram_no_upcast`] — the M8.4b
/// fix that closed the gap discovered by the M8.5 4-model
/// validation (every Llama-family `_proj.weight` has at least
/// `LoadTransform::Transpose2D`, so the M8.4-original fast-path
/// arm never fired in production).
///
/// Strictly disjoint from `VRAM_FAST_PATH_COUNT`,
/// `VRAM_SLOW_PATH_COUNT`, and `VRAM_BF16_FAST_PATH_COUNT`. A
/// single load increments exactly one of those four.
static VRAM_BF16_SLOW_PATH_COUNT: AtomicUsize = AtomicUsize::new(0);

/// **M8.4b** — Returns the number of `Tier::Vram` slow-path loads
/// that produced a BF16-resident TensorGPU. See
/// [`VRAM_BF16_SLOW_PATH_COUNT`].
pub fn vram_bf16_slow_path_count() -> usize {
    VRAM_BF16_SLOW_PATH_COUNT.load(Ordering::Relaxed)
}

/// **M9.2** — counts `Tier::Vram + ATENIA_M9_INT8=1` loads that
/// took the INT8 W8A16 path. Each increment corresponds to a
/// `_proj.weight` tensor whose F32 working buffer was quantised
/// per-channel-symmetric absmax (M9.1 quantizer) and uploaded
/// to VRAM via [`crate::cuda::int8_to_bf16::int8_to_bf16_in_vram`].
/// The dispatch path consumes the resulting BF16 device buffer
/// directly via the M8.4c kernel — the INT8 source bytes are
/// freed once the dequant kernel returns.
///
/// Strictly disjoint from the four BF16/F32 VRAM counters above:
/// a single load increments exactly one of the five.
static VRAM_INT8_PATH_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Read-only accessor for [`VRAM_INT8_PATH_COUNT`]. Test gates use
/// this snapshot to confirm the M9.2 routing (operator can audit
/// from a smoke log without instrumenting the dispatcher).
pub fn vram_int8_path_count() -> usize {
    VRAM_INT8_PATH_COUNT.load(Ordering::Relaxed)
}

/// **M7.0** — counters that record which Disk-tier write
/// sub-path each parameter took during
/// `load_one_shard_into_with_residency_plan`. The Disk arm
/// has the same fast-path / slow-path structure that the Vram
/// arm pioneered in M6 sub-fase 2 — but at this commit
/// (M7.0) the **fast path is not yet implemented**: every
/// Disk-tier entry currently goes through the slow path
/// (decode to F32, then optionally re-encode to BF16, then
/// `disk_tier::write_*_tensor`). The fast-path counter is
/// reserved for M7.1 which will add a raw-bytes BF16 write
/// path mirroring `bf16_to_f32_resident_in_vram_from_raw_bytes`.
///
/// Today's expectation: `disk_slow_path_count` increments
/// once per Disk-tier entry; `disk_fast_path_count` stays
/// at zero until M7.1 lands.
static DISK_FAST_PATH_COUNT: AtomicUsize = AtomicUsize::new(0);
static DISK_SLOW_PATH_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Returns the number of `Tier::Disk` writes that took the
/// raw-bytes fast path. Reserved for M7.1; returns 0 today.
pub fn disk_fast_path_count() -> usize {
    DISK_FAST_PATH_COUNT.load(Ordering::Relaxed)
}

/// Returns the number of `Tier::Disk` writes that took the
/// F32-transient slow path since process start. The M6 Disk
/// arm always goes through this path; M7.1 will introduce a
/// raw-bytes BF16-source-no-transforms fast path that bypasses
/// the F32 transient entirely.
pub fn disk_slow_path_count() -> usize {
    DISK_SLOW_PATH_COUNT.load(Ordering::Relaxed)
}

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
    /// **M8.4c** — explicit override for the M8 BF16-resident
    /// VRAM path. When `Some(true)` / `Some(false)`, the slow
    /// and fast path arms of `load_one_shard_into_with_residency_plan`
    /// route to the BF16-resident upload (M8.1 primitive) /
    /// the M6 F32-resident upload (M6 primitive) regardless of
    /// the `ATENIA_M8_BF16_KERNEL` env var.
    ///
    /// `None` (default) preserves M8.4-original behaviour: the
    /// loader reads the env var directly. This keeps tests that
    /// set the env var via an `M8FlagGuard` working unchanged,
    /// while letting `pipeline.rs` compute a conditional
    /// (env_requested AND model_dominates_ram) and pass the
    /// result explicitly via [`Self::set_bf16_kernel_active`].
    bf16_kernel_active: Option<bool>,
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
            bf16_kernel_active: None,
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

    /// **M8.4c** — explicit override for the M8 BF16-resident
    /// VRAM path. `Some(true)` forces the BF16 upload regardless
    /// of `ATENIA_M8_BF16_KERNEL`; `Some(false)` forces the F32
    /// (M6) upload regardless. `None` reverts to the default
    /// behaviour of reading the env var.
    ///
    /// The pipeline (`src/nn/llama/pipeline.rs`) uses this setter
    /// to gate M8 activation by the adaptive heuristic
    /// `model_total_bytes > 0.7 × free_ram_bytes`: 7B models that
    /// fit comfortably in RAM keep the M6 F32 path even if the
    /// operator set the env var, avoiding the per-matmul upcast
    /// overhead the M8 path costs in the BF16-resident world.
    pub fn set_bf16_kernel_active(&mut self, enabled: Option<bool>) -> &mut Self {
        self.bf16_kernel_active = enabled;
        self
    }

    /// Internal helper: resolves the active M8 BF16 kernel flag
    /// for one load. Explicit override (`Self::set_bf16_kernel_active`)
    /// wins; otherwise falls back to the `ATENIA_M8_BF16_KERNEL`
    /// env var. Centralised here so the fast-path and slow-path
    /// arms agree on the same source of truth without two
    /// independent reads of process-global state.
    fn m8_bf16_kernel_active(&self) -> bool {
        self.bf16_kernel_active.unwrap_or_else(|| {
            std::env::var("ATENIA_M8_BF16_KERNEL").as_deref() == Ok("1")
        })
    }

    /// **M9.2** — gate for the INT8 W8A16 loader path. Reads
    /// `ATENIA_M9_INT8=1` once per load. The path only fires when
    /// the parameter name ends with `_proj.weight` (caller
    /// re-checks at the use site so a future config flip does not
    /// silently widen the predicate).
    fn m9_int8_active(&self) -> bool {
        std::env::var("ATENIA_M9_INT8").as_deref() == Ok("1")
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

    /// **M6 replan sub-fase 2** — tier-aware load entry point.
    ///
    /// Iterates `reader`, applies transforms in F32 as the
    /// classic [`Self::load_into`] does, then dispatches each
    /// loaded parameter to one of three destinations based on
    /// `plan.get(name)`:
    ///
    /// - [`Tier::Vram`] — the parameter is uploaded to VRAM and
    ///   inserted into the returned [`WeightStore`] as
    ///   [`SharedParam::Cuda`]. The graph slot is **left
    ///   untouched**; the bytes flow through
    ///   `cuda::bf16_to_f32::bf16_to_f32_resident_in_vram*`.
    ///   For the common case of a BF16 source with no
    ///   transforms, the raw safetensors bytes go directly to
    ///   `bf16_to_f32_resident_in_vram_from_raw_bytes` and **no
    ///   F32 transient is materialised on the host** — the
    ///   "fast path." Otherwise the F32 working buffer is built
    ///   exactly once per entry, transformed, down-converted to
    ///   BF16 bits in place, uploaded, and dropped — the "slow
    ///   path" (still no permanent F32 RAM footprint).
    ///
    /// - [`Tier::Ram`] — the parameter is written into the
    ///   graph slot via the same `Cpu` / `CpuBf16` path that
    ///   [`Self::load_one_shard_into`] uses. After the entry
    ///   loop completes, a pass-2 hoist via
    ///   [`WeightStore::extract_from_graph`] (filtered to the
    ///   Ram-tier `param_ids`) lifts the slots into Arc-shared
    ///   `SharedParam::F32` / `SharedParam::Bf16` entries.
    ///
    /// - [`Tier::Disk`] — the parameter's transformed bytes are
    ///   serialised to NVMe via
    ///   `disk_tier::write_{f32,bf16}_tensor` and inserted into
    ///   the store as [`SharedParam::Disk`]. The graph slot is
    ///   left untouched.
    ///
    /// The returned [`WeightStore`] therefore owns Arcs for Ram
    /// entries (graph slots also hold a clone via `extract_from_graph`),
    /// `TensorGPU` references for Vram entries (no host-side
    /// remnant beyond the safetensors reader's owned bytes), and
    /// `DiskTensorHandle`s for Disk entries (RAM is reclaimed
    /// once the F32 transient drops at the end of the iteration).
    ///
    /// # Parameters
    ///
    /// - `param_ids` / `param_names` — index-aligned slices
    ///   describing the graph's parameter slot layout, exactly
    ///   as in [`WeightStore::extract_from_graph`]. Sourced from
    ///   `LlamaHandles` in the production caller.
    ///
    /// # Errors
    ///
    /// Same surface as [`Self::load_into`] for the F32 path
    /// (`ShapeMismatch`, `InvalidFormat`, `UnsupportedDType`,
    /// `IoError`). Adds the Vram failure mode: a `cuda_malloc`
    /// or kernel error during upload returns `InvalidFormat`
    /// with the failing parameter name. Adds the Disk failure
    /// mode: a `disk_tier::write_*_tensor` error returns
    /// `IoError`.
    ///
    /// # Atomicity
    ///
    /// On error after some Vram or Disk entries have already
    /// been inserted, those entries stay in the partially-built
    /// store but the function returns `Err`. The caller is
    /// expected to drop both the partial store and the graph;
    /// `Drop` impls reclaim VRAM and disk-tier files. RAM-tier
    /// entries are not yet hoisted at error time (they live in
    /// the graph slot until pass 2) so dropping the graph
    /// reclaims them automatically.
    pub fn load_into_with_residency_plan(
        &self,
        graph: &mut Graph,
        reader: &SafetensorsReader,
        plan: &TierPlan,
        param_ids: &[usize],
        param_names: &[String],
    ) -> Result<(WeightStore, LoadReport), LoaderError> {
        let mut report = LoadReport::default();
        let mut satisfied: HashSet<String> = HashSet::new();
        let mut store = WeightStore::new();
        let mut already_inserted: HashSet<String> = HashSet::new();

        let outcome = self.load_one_shard_into_with_residency_plan(
            graph,
            reader,
            plan,
            &mut store,
            &mut already_inserted,
            &mut satisfied,
        )?;

        report.loaded = outcome.loaded;
        report.skipped = outcome.skipped;
        report.missing = self.collect_missing(&satisfied);

        finalize_ram_extract(graph, &mut store, &already_inserted, param_ids, param_names)?;

        Ok((store, report))
    }

    /// **M6 replan sub-fase 3** — per-shard primitive used by both
    /// the single-shard `load_into_with_residency_plan` and the
    /// multi-shard `ShardedSafetensorsReader::load_into_with_residency_plan`.
    /// Mutates the caller-owned `store`, `already_inserted`, and
    /// `satisfied` accumulators in place; does **not** perform the
    /// final Ram-tier extract — the caller is responsible for that
    /// step (typically once after every shard has been processed).
    pub fn load_one_shard_into_with_residency_plan(
        &self,
        graph: &mut Graph,
        reader: &SafetensorsReader,
        plan: &TierPlan,
        store: &mut WeightStore,
        already_inserted: &mut HashSet<String>,
        satisfied: &mut HashSet<String>,
    ) -> Result<ShardLoadOutcome, LoaderError> {
        let mut outcome = ShardLoadOutcome::default();

        for entry in reader.iter() {
            let Some(mapping) = self.mapping.get(entry.name) else {
                outcome.skipped.push(entry.name.to_string());
                continue;
            };
            let node_id = mapping.node_id;
            let tier = plan.get(entry.name).unwrap_or(Tier::Ram);

            // ----- VRAM fast path: BF16 source + no transforms.
            //
            // Direct H→D memcpy of the raw safetensors bytes —
            // no `Vec<f32>`, no `Vec<u16>`, no host transient
            // beyond what the reader already paid for. This is
            // the path the M6 replan was designed to enable; the
            // peak-RAM contract is "BF16 source bytes only."
            if tier == Tier::Vram
                && entry.dtype == DType::BF16
                && mapping.transforms.is_empty()
            {
                let numel: usize = entry.shape.iter().product();
                let tensor = graph
                    .nodes
                    .get(node_id)
                    .and_then(|n| n.output.as_ref())
                    .ok_or_else(|| {
                        LoaderError::InvalidFormat(format!(
                            "mapper points '{}' at node_id {} but that node has no \
                             materialized tensor in the graph",
                            entry.name, node_id
                        ))
                    })?;
                if entry.shape != tensor.shape.as_slice() {
                    return Err(LoaderError::ShapeMismatch {
                        tensor_name: entry.name.to_string(),
                        expected: tensor.shape.clone(),
                        actual: entry.shape.to_vec(),
                    });
                }

                // **M8.4** — gate between the M6 F32-resident path
                // and the new M8 BF16-resident path. Default
                // (env unset or != "1") keeps the M6 contract
                // bit-identical: BF16 source → F32 in VRAM via
                // the upcast kernel. Flag on routes through the
                // M8.1 primitive that keeps BF16 in VRAM and
                // halves the per-weight footprint, ready for
                // `cuda_matmul_bf16_inplace` (M8.2) to consume
                // directly. The two paths increment **disjoint**
                // counters so the operator can audit which path
                // ran from the smoke log.
                let m8_bf16_kernel = self.m8_bf16_kernel_active();

                let gpu = if m8_bf16_kernel {
                    crate::cuda::bf16_to_f32::bf16_to_vram_no_upcast_from_raw_bytes(
                        entry.raw_bytes,
                        numel,
                        entry.shape,
                    )
                    .ok_or_else(|| {
                        LoaderError::InvalidFormat(format!(
                            "BF16-resident VRAM upload failed for '{}' \
                             (M8.4 path)",
                            entry.name
                        ))
                    })?
                } else {
                    crate::cuda::bf16_to_f32::bf16_to_f32_resident_in_vram_from_raw_bytes(
                        entry.raw_bytes,
                        numel,
                        entry.shape,
                    )
                    .ok_or_else(|| {
                        LoaderError::InvalidFormat(format!(
                            "BF16→VRAM fast-path upload failed for '{}'",
                            entry.name
                        ))
                    })?
                };

                store.params.push(SharedParam::Cuda {
                    shape: entry.shape.to_vec(),
                    gpu,
                });
                store.names.push(entry.name.to_string());
                already_inserted.insert(entry.name.to_string());

                if m8_bf16_kernel {
                    VRAM_BF16_FAST_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
                } else {
                    VRAM_FAST_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
                }
                outcome.loaded += 1;
                satisfied.insert(entry.name.to_string());
                continue;
            }

            // ----- M7.1 — DISK fast path: BF16 source + no transforms.
            //
            // Direct write of the raw safetensors bytes to NVMe via
            // `disk_tier::write_bf16_from_raw_bytes`. No `Vec<f32>`,
            // no `Vec<u16>`, no host transient beyond what the
            // reader already paid for. Closes risk R3 (BSOD-class
            // peak-RAM regression on the Disk-tier loader path):
            // peak host RAM during this write is exactly the
            // safetensors reader's owned bytes — no per-tensor
            // 540 MB F32 transient that the slow path would
            // materialise via `entry.to_vec_f32()`.
            //
            // Only applies when the mapper is configured to store
            // params as BF16 (`store_params_as_bf16=true`). With
            // `store_params_as_bf16=false`, the slow path's F32
            // semantic on disk takes over (the source must be
            // upcast to F32 before writing, which requires the
            // F32 transient anyway).
            if tier == Tier::Disk
                && entry.dtype == DType::BF16
                && mapping.transforms.is_empty()
                && self.store_params_as_bf16
            {
                let numel: usize = entry.shape.iter().product();
                let tensor = graph
                    .nodes
                    .get(node_id)
                    .and_then(|n| n.output.as_ref())
                    .ok_or_else(|| {
                        LoaderError::InvalidFormat(format!(
                            "mapper points '{}' at node_id {} but that node has no \
                             materialized tensor in the graph",
                            entry.name, node_id
                        ))
                    })?;
                if entry.shape != tensor.shape.as_slice() {
                    return Err(LoaderError::ShapeMismatch {
                        tensor_name: entry.name.to_string(),
                        expected: tensor.shape.clone(),
                        actual: entry.shape.to_vec(),
                    });
                }

                let cache_dir = crate::tensor::disk_tier::default_cache_dir();
                let handle = crate::tensor::disk_tier::write_bf16_from_raw_bytes(
                    &cache_dir,
                    entry.raw_bytes,
                    numel,
                )
                .map_err(|e| LoaderError::IoError(e.to_string()))?;

                store.params.push(SharedParam::Disk {
                    shape: entry.shape.to_vec(),
                    handle,
                });
                store.names.push(entry.name.to_string());
                already_inserted.insert(entry.name.to_string());

                DISK_FAST_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
                outcome.loaded += 1;
                satisfied.insert(entry.name.to_string());
                continue;
            }

            // ----- Slow path: F32 working buffer + transforms.
            //
            // Mirrors `load_one_shard_into`'s decode-transform
            // pipeline. The F32 vec exists only for the duration
            // of this iteration; it is consumed by either
            // `bf16_to_f32_resident_in_vram` (Vram slow path),
            // `disk_tier::write_*_tensor` (Disk), or
            // `tensor.set_cpu_bf16_bits` / `as_cpu_slice_mut`
            // (Ram).
            let mut values = entry.to_vec_f32()?;
            let mut current_shape: Vec<usize> = entry.shape.to_vec();

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
                        current_shape = target.clone();
                    }
                }
            }

            // Validate post-transform shape against the graph
            // parameter slot. Same check as `load_one_shard_into`.
            let tensor_shape = {
                let tensor = graph
                    .nodes
                    .get(node_id)
                    .and_then(|n| n.output.as_ref())
                    .ok_or_else(|| {
                        LoaderError::InvalidFormat(format!(
                            "mapper points '{}' at node_id {} but that node has no \
                             materialized tensor in the graph",
                            entry.name, node_id
                        ))
                    })?;
                tensor.shape.clone()
            };
            if current_shape != tensor_shape.as_slice() {
                return Err(LoaderError::ShapeMismatch {
                    tensor_name: entry.name.to_string(),
                    expected: tensor_shape,
                    actual: current_shape,
                });
            }

            let dest_numel: usize = current_shape.iter().product();
            if dest_numel != values.len() {
                return Err(LoaderError::InvalidFormat(format!(
                    "tensor '{}': element count {} after transforms does not match \
                     graph parameter capacity {}",
                    entry.name,
                    values.len(),
                    dest_numel
                )));
            }

            // BF16 precision floor — same as `load_one_shard_into`.
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

            match tier {
                Tier::Vram => {
                    // **M9.2 — INT8 W8A16 branch.** When
                    // `ATENIA_M9_INT8=1` is set and the parameter is
                    // a `_proj.weight` tensor, quantise the post-
                    // transform F32 buffer per-output-channel
                    // symmetric absmax (M9.1 quantizer), upload
                    // INT8 + scales to VRAM, and dispatch the
                    // dequant kernel which materialises a BF16
                    // device buffer ready for the M8.4c BF16
                    // matmul. This is the load-time path; the
                    // dispatch contract (BF16 in VRAM consumed by
                    // `cuda_matmul_bf16_inplace`) is unchanged. The
                    // INT8 source bytes are freed once the dequant
                    // returns; M9.3 will switch to keeping INT8
                    // resident with a recycled BF16 staging slot
                    // for the true capacity-doubling win.
                    let m9_int8 = self.m9_int8_active()
                        && entry.name.ends_with("_proj.weight");

                    if m9_int8 {
                        // M9.4 — per-group (Q8_0-style) quantisation.
                        // Group size 128 along the K axis. The M9.4
                        // F64 fixture under per-channel produced
                        // drift 1.10–8.88 across the 4 production
                        // models; per-group localises the scale to
                        // 128-element blocks, recovering the ADR-004
                        // margin lost to per-column outliers. The
                        // CUDA dispatch path uses the matching
                        // `int8_to_bf16_per_group` kernel.
                        const M9_4_GROUP_SIZE: usize = 128;
                        let (q, scales) =
                            crate::tensor::quantizer::absmax_per_group_symmetric(
                                &values,
                                &current_shape,
                                M9_4_GROUP_SIZE,
                            );
                        drop(values);

                        let gpu = crate::cuda::int8_to_bf16::int8_per_group_to_bf16_in_vram(
                            &q,
                            &scales,
                            &current_shape,
                            M9_4_GROUP_SIZE,
                        )
                        .ok_or_else(|| {
                            LoaderError::InvalidFormat(format!(
                                "INT8 per-group → BF16 VRAM upload failed for '{}' \
                                 (M9.4 path, group_size = {})",
                                entry.name, M9_4_GROUP_SIZE,
                            ))
                        })?;

                        store.params.push(SharedParam::Cuda {
                            shape: current_shape.clone(),
                            gpu,
                        });
                        store.names.push(entry.name.to_string());
                        already_inserted.insert(entry.name.to_string());
                        VRAM_INT8_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
                    } else {
                        // Slow path: F32 transforms have already been
                        // applied to `values: Vec<f32>` — the BF16
                        // re-encode + upload happens here.
                        //
                        // **M8.4b** — gate between the M6 F32-resident
                        // path and the M8 BF16-resident path,
                        // analogous to the fast-path arm above. Every
                        // Llama-family `_proj.weight` has at least
                        // `LoadTransform::Transpose2D` registered, so
                        // until this fix the slow path always took
                        // the M6 F32 upload regardless of the M8
                        // flag — that's the gap M8.5 surfaced. With
                        // this branch, transforms run in F32
                        // (preserving precision during cascaded ops),
                        // the result re-encodes to BF16, and the
                        // BF16 buffer goes to VRAM via
                        // `bf16_to_vram_no_upcast` for direct
                        // consumption by `cuda_matmul_bf16_inplace`.
                        let bits: Vec<u16> = values
                            .iter()
                            .map(|&v| (v.to_bits() >> 16) as u16)
                            .collect();
                        drop(values);

                        let m8_bf16_kernel = self.m8_bf16_kernel_active();

                        let gpu = if m8_bf16_kernel {
                            crate::cuda::bf16_to_f32::bf16_to_vram_no_upcast(
                                &bits,
                                &current_shape,
                            )
                            .ok_or_else(|| {
                                LoaderError::InvalidFormat(format!(
                                    "BF16-resident VRAM slow-path upload failed for '{}' \
                                     (M8.4b path)",
                                    entry.name
                                ))
                            })?
                        } else {
                            crate::cuda::bf16_to_f32::bf16_to_f32_resident_in_vram(
                                &bits,
                                &current_shape,
                            )
                            .ok_or_else(|| {
                                LoaderError::InvalidFormat(format!(
                                    "BF16→VRAM slow-path upload failed for '{}'",
                                    entry.name
                                ))
                            })?
                        };

                        store.params.push(SharedParam::Cuda {
                            shape: current_shape.clone(),
                            gpu,
                        });
                        store.names.push(entry.name.to_string());
                        already_inserted.insert(entry.name.to_string());

                        if m8_bf16_kernel {
                            VRAM_BF16_SLOW_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
                        } else {
                            VRAM_SLOW_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                Tier::Disk => {
                    let cache_dir = crate::tensor::disk_tier::default_cache_dir();
                    let handle = if self.store_params_as_bf16 {
                        let bits: Vec<u16> = values
                            .iter()
                            .map(|&v| (v.to_bits() >> 16) as u16)
                            .collect();
                        drop(values);
                        crate::tensor::disk_tier::write_bf16_tensor(&cache_dir, &bits)
                            .map_err(|e| LoaderError::IoError(e.to_string()))?
                    } else {
                        let h = crate::tensor::disk_tier::write_f32_tensor(&cache_dir, &values)
                            .map_err(|e| LoaderError::IoError(e.to_string()))?;
                        drop(values);
                        h
                    };
                    store.params.push(SharedParam::Disk {
                        shape: current_shape.clone(),
                        handle,
                    });
                    store.names.push(entry.name.to_string());
                    already_inserted.insert(entry.name.to_string());
                    // **M7.0** — counter tag. Today the Disk arm
                    // always uses the slow path (F32 transient
                    // materialised by `entry.to_vec_f32()` before
                    // this match block). M7.1 will add a fast-path
                    // BF16 raw-bytes branch that bypasses the
                    // transient and increments the matching
                    // `DISK_FAST_PATH_COUNT` instead.
                    DISK_SLOW_PATH_COUNT.fetch_add(1, Ordering::Relaxed);
                }
                Tier::Ram => {
                    // Existing graph-slot write path — kept
                    // bit-exact with `load_one_shard_into`.
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
                    if self.store_params_as_bf16 {
                        let bits: Vec<u16> = values
                            .iter()
                            .map(|&v| (v.to_bits() >> 16) as u16)
                            .collect();
                        tensor.set_cpu_bf16_bits(bits);
                    } else {
                        let slice = tensor.as_cpu_slice_mut();
                        slice.copy_from_slice(&values);
                    }
                }
            }

            outcome.loaded += 1;
            satisfied.insert(entry.name.to_string());
        }

        Ok(outcome)
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

/// **M6 replan sub-fase 3** — pass-2 helper for the tier-aware
/// loaders. Given a partially-populated `store` (already holding
/// the Vram and Disk entries inserted by the per-shard primitive)
/// and the set of names already inserted, this function:
///
/// 1. Computes the complement of `param_ids` / `param_names` —
///    the entries that should land on Ram.
/// 2. Calls [`WeightStore::extract_from_graph`] on that filtered
///    list. Each Ram-tier graph slot is hoisted to a `CpuShared`
///    or `CpuBf16Shared` view, and the matching `SharedParam::F32`
///    or `SharedParam::Bf16` is appended to a fresh helper store.
/// 3. Merges the fresh helper store into `store`, preserving the
///    Vram/Disk entries that are already there.
///
/// Used by both
/// [`WeightMapper::load_into_with_residency_plan`] (single-shard)
/// and
/// [`crate::v17::loader::sharded_reader::ShardedSafetensorsReader::load_into_with_residency_plan`]
/// (multi-shard) — those callers diverge on the per-shard loop
/// but share this final extraction step.
pub fn finalize_ram_extract(
    graph: &mut Graph,
    store: &mut WeightStore,
    already_inserted: &HashSet<String>,
    param_ids: &[usize],
    param_names: &[String],
) -> Result<(), LoaderError> {
    let ram_indices: Vec<usize> = param_ids
        .iter()
        .zip(param_names.iter())
        .filter(|(_, name)| !already_inserted.contains(*name))
        .map(|(id, _)| *id)
        .collect();
    let ram_names: Vec<String> = param_ids
        .iter()
        .zip(param_names.iter())
        .filter(|(_, name)| !already_inserted.contains(*name))
        .map(|(_, name)| name.clone())
        .collect();

    let ram_store =
        WeightStore::extract_from_graph(graph, &ram_indices, &ram_names).map_err(|e| {
            LoaderError::InvalidFormat(format!(
                "Ram-tier extract_from_graph failed: {}",
                e
            ))
        })?;

    for (i, p) in ram_store.params.into_iter().enumerate() {
        store.params.push(p);
        store.names.push(ram_store.names[i].clone());
    }

    let _ = std::marker::PhantomData::<WeightStoreError>;
    Ok(())
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

    /// M9.2 — `m9_int8_active` reads the `ATENIA_M9_INT8` env var.
    /// Serialised on the same lock as the tier_plan tests because
    /// the env var is process-global and parallel tests would race.
    static M9_INT8_LOADER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn m9_int8_active_reads_atenia_m9_int8_flag() {
        let _g = M9_INT8_LOADER_LOCK.lock().unwrap();
        let mapper = WeightMapper::from_param_names_and_ids(&[], &[]).unwrap();

        unsafe { std::env::remove_var("ATENIA_M9_INT8"); }
        assert!(!mapper.m9_int8_active(),
            "flag unset → m9_int8_active() must be false");

        unsafe { std::env::set_var("ATENIA_M9_INT8", "1"); }
        assert!(mapper.m9_int8_active(),
            "flag = '1' → m9_int8_active() must be true");

        unsafe { std::env::set_var("ATENIA_M9_INT8", "0"); }
        assert!(!mapper.m9_int8_active(),
            "flag = '0' → m9_int8_active() must be false (only '1' enables)");

        unsafe { std::env::remove_var("ATENIA_M9_INT8"); }
    }

    /// M9.2 — `vram_int8_path_count()` reads the static counter
    /// without panicking. Without a CUDA-equipped host we can't
    /// drive the increment from a unit test; the counter advance
    /// is exercised by the (`#[ignore]`d) M9.4 end-to-end test
    /// once it lands. This test pins the accessor's existence and
    /// the read-only contract.
    #[test]
    fn vram_int8_path_count_accessor_is_callable() {
        let before = vram_int8_path_count();
        // Reading twice in a row without intervening loads must
        // return the same value (no hidden mutation in the
        // accessor).
        let again = vram_int8_path_count();
        assert_eq!(before, again);
    }
}
