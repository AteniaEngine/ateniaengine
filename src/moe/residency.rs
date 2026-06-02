//! **MOE-FULL-8** — experimental tiered **expert residency** for MoE.
//!
//! The MoE block certified through MOE-FULL-7 keeps *every* expert's weights
//! materialised in RAM (`RealMoeLayer.routed.experts: Vec<MoeDenseExpert>`).
//! That is fine for the tiny fixtures but is the headline blocker called out
//! in `docs/MOE_OVERVIEW.md` / `docs/MOE_FULL_PATH_AUDIT.md`: it does not scale
//! to Mixtral-8x7B / large Qwen-MoE / DeepSeek-MoE, where the experts dominate
//! the parameter count and cannot all live in RAM at once.
//!
//! This module demonstrates that the MoE architecture can sit on top of
//! Atenia's **real residency infrastructure** — the [`SharedParam`] tiers
//! (`F32`/`Bf16` in RAM, `Disk` on NVMe via [`crate::tensor::disk_tier`],
//! `Cuda` in VRAM) used by the productive `WeightStore` — *without*
//! materialising all experts. Each expert's three projection weights live in a
//! chosen tier; on a forward only the **router-selected top-k experts** are
//! resolved (materialised → executed → dropped). The rest never leave their
//! tier (NVMe experts cost **zero RAM** until requested).
//!
//! ```text
//!   route(x)  ──► softmax weights          (router stays RAM-resident)
//!   top_k(weights, k)  ──► selected indices + renormalised weights
//!   for e in selected:                     (NOT all experts)
//!        resolve expert e from its tier  ──► transient MoeDenseExpert
//!        y_e = expert.forward(x)  ;  out += w · y_e
//!        drop the materialised weights
//!   (+ optional shared expert per convention)
//! ```
//!
//! ## Correctness by construction
//!
//! The forward performs the **exact** operations of the certified
//! [`RealMoeLayer::forward_with`] — same [`MoeDenseLayer::route`], same
//! [`top_k_routing_with`] selection/renormalisation, same
//! [`MoeDenseExpert::forward`] SwiGLU, same weighted combine, same shared-
//! expert convention. Only *where the expert weights live* and *when they are
//! materialised* differ. The output is therefore bit-identical to
//! `RealMoeLayer::forward_auto`, locked by an equality test.
//!
//! ## Scope (experimental, honest, bounded)
//!
//! * CPU-only, test/opt-in only. No productive loader / runtime / Adapter
//!   Toolkit / CLI / pipeline / fail-loud change. Real MoE checkpoints still
//!   fail loud (MOE-2). This is a *residency mechanism* demo, not Mixtral
//!   support.
//! * Reuses `SharedParam` + `disk_tier` unchanged — it **consumes** the
//!   residency infrastructure, it does not modify it.
//! * No GQA, no quantised experts, no VRAM tier exercised here (the `Cuda`
//!   variant requires a device; the demo stays CPU/NVMe). The design admits
//!   them — `SharedParam` already has the `Cuda` arm — but they are out of
//!   this milestone's scope.

use std::collections::HashMap;
use std::io;
use std::path::PathBuf;

use crate::amg::weight_store::SharedParam;
use crate::tensor::disk_tier::{self, DiskTensorHandle};
use std::sync::Arc;

use super::dense::{MoeDenseExpert, MoeDenseError};
use super::layer::{MoeExecutionConvention, MoeLayerConfig, MoeLayerError, RealMoeLayer};
use super::sparse::{top_k_routing_with, MoeSparseError};

/// Which residency tier an expert's weights are placed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertTier {
    /// RAM-resident F32 (`SharedParam::F32`). Cheap to resolve (Arc read);
    /// still counts against host RAM.
    Ram,
    /// NVMe-resident (`SharedParam::Disk` via `disk_tier`). **Zero RAM** until
    /// the expert is requested; resolving reads the file back to F32.
    Disk,
}

/// One expert's three projection weights held in the residency store. Each is
/// a [`SharedParam`] so it can live in any tier independently.
#[derive(Debug, Clone)]
struct ResidentExpert {
    d_model: usize,
    d_ff: usize,
    gate: SharedParam, // [d_ff, d_model]
    up: SharedParam,   // [d_ff, d_model]
    down: SharedParam, // [d_model, d_ff]
}

/// Per-forward residency evidence: which experts were materialised and how
/// many weight bytes that cost. Proves only the top-k experts (not all N) are
/// resolved per token.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ResidencyInfo {
    /// Routed experts materialised this forward (sorted ascending).
    pub materialized_experts: Vec<usize>,
    /// Total weight bytes materialised this forward (routed + shared).
    pub materialized_bytes: usize,
}

/// An MoE layer whose expert weights live in a residency tier and are resolved
/// on demand. Built from a certified [`RealMoeLayer`].
#[derive(Debug)]
pub struct ResidentExpertLayer {
    pub config: MoeLayerConfig,
    pub convention: MoeExecutionConvention,
    pub tier: ExpertTier,
    /// Router weight `[num_experts, d_model]`, always RAM-resident (small).
    w_router: Vec<f32>,
    experts: Vec<ResidentExpert>,
    shared: Option<ResidentExpert>,
    shared_gate: Option<Vec<f32>>,
}

/// Place an `f32` weight buffer into the chosen tier as a `SharedParam`.
fn place(tier: ExpertTier, shape: Vec<usize>, data: &[f32]) -> io::Result<SharedParam> {
    match tier {
        ExpertTier::Ram => Ok(SharedParam::F32 { shape, arc: Arc::new(data.to_vec()) }),
        ExpertTier::Disk => {
            let handle: DiskTensorHandle =
                disk_tier::write_f32_tensor(&disk_tier::default_cache_dir(), data)?;
            Ok(SharedParam::Disk { shape, handle })
        }
    }
}

/// **MOE-PROD-4** — one tiered tensor recorded in the on-disk tier manifest.
#[derive(Debug, Clone)]
pub struct TierEntry {
    /// Logical key (e.g. `L3.e17.gate`) — also the bare file name minus `.bin`.
    pub key: String,
    /// Bare file name (`<key>.bin`).
    pub file: String,
    /// Element count (f32).
    pub numel: usize,
}

/// **MOE-PROD-4** — accumulates the **persistent** expert tier for one model
/// load: the on-disk directory, the per-tensor entries, and how many tensors
/// were **reused** from an existing tier vs **written** fresh.
#[derive(Debug, Default)]
pub struct TierContext {
    pub dir: PathBuf,
    pub entries: Vec<TierEntry>,
    pub reused: usize,
    pub written: usize,
}

/// Place an f32 weight buffer into the **persistent** disk tier at a
/// deterministic path `dir/<key>.bin`. If that file already exists with the
/// expected byte length, **reuse** it (no write); otherwise write it. Records
/// the entry and bumps the reused/written counter.
fn place_at(
    ctx: &mut TierContext,
    key: &str,
    shape: Vec<usize>,
    data: &[f32],
) -> io::Result<SharedParam> {
    let file = format!("{key}.bin");
    let path = ctx.dir.join(&file);
    let expected_bytes = std::mem::size_of_val(data) as u64;
    let valid = std::fs::metadata(&path).map(|m| m.len() == expected_bytes).unwrap_or(false);
    let handle = if valid {
        ctx.reused += 1;
        disk_tier::open_existing_f32(&path, data.len())
    } else {
        ctx.written += 1;
        disk_tier::write_f32_tensor_named(&path, data)?
    };
    ctx.entries.push(TierEntry { key: key.to_string(), file, numel: data.len() });
    Ok(SharedParam::Disk { shape, handle })
}

/// Materialise a stored param to an owned `Vec<f32>` (reads NVMe for the
/// `Disk` tier; copies the Arc for the `Ram` tier).
fn materialize(p: &SharedParam) -> Vec<f32> {
    let mut t = p.to_tensor();
    t.ensure_cpu().expect("residency: ensure_cpu failed materialising expert weight");
    t.copy_to_cpu_vec()
}

/// Host-RAM bytes a stored param occupies (Disk/Cuda → 0; they live off-RAM).
fn ram_bytes(p: &SharedParam) -> usize {
    match p {
        SharedParam::F32 { arc, .. } => arc.len() * 4,
        SharedParam::Bf16 { arc, .. } => arc.len() * 2,
        // Disk lives on NVMe, Cuda in VRAM — neither costs host RAM here.
        _ => 0,
    }
}

impl ResidentExpert {
    fn from_dense(tier: ExpertTier, e: &MoeDenseExpert) -> io::Result<Self> {
        Ok(Self {
            d_model: e.d_model,
            d_ff: e.d_ff,
            gate: place(tier, vec![e.d_ff, e.d_model], &e.w_gate)?,
            up: place(tier, vec![e.d_ff, e.d_model], &e.w_up)?,
            down: place(tier, vec![e.d_model, e.d_ff], &e.w_down)?,
        })
    }

    /// **MOE-PROD-4** — like [`Self::from_dense`] but into the **persistent**
    /// disk tier under deterministic keys `{prefix}.{gate,up,down}` (reusing
    /// existing files when present).
    fn from_dense_at(
        ctx: &mut TierContext,
        prefix: &str,
        e: &MoeDenseExpert,
    ) -> io::Result<Self> {
        Ok(Self {
            d_model: e.d_model,
            d_ff: e.d_ff,
            gate: place_at(ctx, &format!("{prefix}.gate"), vec![e.d_ff, e.d_model], &e.w_gate)?,
            up: place_at(ctx, &format!("{prefix}.up"), vec![e.d_ff, e.d_model], &e.w_up)?,
            down: place_at(ctx, &format!("{prefix}.down"), vec![e.d_model, e.d_ff], &e.w_down)?,
        })
    }

    /// Materialise the three projections into a transient `MoeDenseExpert`.
    fn resolve(&self) -> Result<(MoeDenseExpert, usize), MoeDenseError> {
        let g = materialize(&self.gate);
        let u = materialize(&self.up);
        let d = materialize(&self.down);
        let bytes = (g.len() + u.len() + d.len()) * 4;
        let expert = MoeDenseExpert::new(self.d_model, self.d_ff, g, u, d)?;
        Ok((expert, bytes))
    }

    fn ram_bytes(&self) -> usize {
        ram_bytes(&self.gate) + ram_bytes(&self.up) + ram_bytes(&self.down)
    }

    fn full_bytes(&self) -> usize {
        (self.d_ff * self.d_model * 2 + self.d_model * self.d_ff) * 4
    }
}

impl ResidentExpertLayer {
    /// Build a residency-backed layer from a certified [`RealMoeLayer`],
    /// relocating every expert's weights into `tier`. The router weight and
    /// optional shared-expert gate stay RAM-resident. The execution convention
    /// is inherited via [`RealMoeLayer::resolve_convention`].
    pub fn from_real_layer(layer: &RealMoeLayer, tier: ExpertTier) -> io::Result<Self> {
        let experts = layer
            .routed
            .experts
            .iter()
            .map(|e| ResidentExpert::from_dense(tier, e))
            .collect::<io::Result<Vec<_>>>()?;
        let shared = match &layer.shared {
            Some(se) => Some(ResidentExpert::from_dense(tier, se)?),
            None => None,
        };
        Ok(Self {
            config: layer.config,
            convention: layer.resolve_convention(),
            tier,
            w_router: layer.routed.w_router.clone(),
            experts,
            shared,
            shared_gate: layer.shared_gate.clone(),
        })
    }

    /// **MOE-PROD-4** — build a disk-tier residency layer whose expert weights
    /// land in the **persistent** tier (`ctx.dir`) under deterministic keys
    /// `L{layer_id}.e{i}.*` / `L{layer_id}.shared.*`, reusing existing tier
    /// files when present (skips the write). Identical to
    /// [`Self::from_real_layer`] with `Disk` otherwise — the on-disk bytes are
    /// the same whether written or reused, so output is bit-identical. Router +
    /// shared-expert gate stay RAM-resident (small; re-read each load).
    pub fn from_real_layer_at(
        layer: &RealMoeLayer,
        ctx: &mut TierContext,
        layer_id: usize,
    ) -> io::Result<Self> {
        let experts = layer
            .routed
            .experts
            .iter()
            .enumerate()
            .map(|(i, e)| ResidentExpert::from_dense_at(ctx, &format!("L{layer_id}.e{i}"), e))
            .collect::<io::Result<Vec<_>>>()?;
        let shared = match &layer.shared {
            Some(se) => Some(ResidentExpert::from_dense_at(ctx, &format!("L{layer_id}.shared"), se)?),
            None => None,
        };
        Ok(Self {
            config: layer.config,
            convention: layer.resolve_convention(),
            tier: ExpertTier::Disk,
            w_router: layer.routed.w_router.clone(),
            experts,
            shared,
            shared_gate: layer.shared_gate.clone(),
        })
    }

    /// Number of routed experts.
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Host-RAM bytes currently resident across router + all expert weights +
    /// shared expert. For the `Disk` tier the expert weights contribute **0**
    /// (they live on NVMe), so this collapses to ~router size — the headline
    /// residency win.
    pub fn resident_ram_bytes(&self) -> usize {
        let router = self.w_router.len() * 4;
        let experts: usize = self.experts.iter().map(|e| e.ram_bytes()).sum();
        let shared = self.shared.as_ref().map(|e| e.ram_bytes()).unwrap_or(0);
        let gate = self.shared_gate.as_ref().map(|g| g.len() * 4).unwrap_or(0);
        router + experts + shared + gate
    }

    /// Bytes that the **current** RAM-resident MoE block would cost if every
    /// expert were materialised in RAM at once (the pre-MOE-FULL-8 footprint).
    /// Useful as the denominator for a residency-saving ratio.
    pub fn full_materialization_bytes(&self) -> usize {
        let router = self.w_router.len() * 4;
        let experts: usize = self.experts.iter().map(|e| e.full_bytes()).sum();
        let shared = self.shared.as_ref().map(|e| e.full_bytes()).unwrap_or(0);
        router + experts + shared
    }

    /// Route `x`, select top-k, and run **only** the selected experts, each
    /// resolved on demand from its tier. Returns the output plus
    /// [`ResidencyInfo`] evidence. Bit-identical to
    /// [`RealMoeLayer::forward_with`] under the same convention.
    pub fn forward(&self, x: &[f32]) -> Result<(Vec<f32>, ResidencyInfo), MoeLayerError> {
        // 1. Router (RAM-resident) → softmax weights.
        let weights = router_softmax(&self.w_router, self.config.num_experts, self.config.d_model, x)?;

        // 2. Top-k selection (certified), convention-aware renormalisation.
        let renormalize = matches!(self.convention, MoeExecutionConvention::Atenia);
        let selection = top_k_routing_with(&weights, self.config.experts_per_token, renormalize)
            .map_err(MoeLayerError::Sparse)?;

        // 3. Resolve + run ONLY the selected experts.
        let mut out = vec![0.0_f32; self.config.d_model];
        let mut materialized_bytes = 0usize;
        for (slot, &e) in selection.indices.iter().enumerate() {
            let (expert, bytes) = self.experts[e]
                .resolve()
                .map_err(|err| MoeLayerError::Binding(super::binding::MoeBindingError::Dense(err)))?;
            materialized_bytes += bytes;
            let y = expert
                .forward(x)
                .map_err(|err| MoeLayerError::Binding(super::binding::MoeBindingError::Dense(err)))?;
            let w = selection.weights[slot];
            for d in 0..self.config.d_model {
                out[d] += w * y[d];
            }
            // `expert` (and its materialised weights) drops here.
        }

        // 4. Shared expert (runs on every token), per convention.
        if let Some(se) = &self.shared {
            let (expert, bytes) = se
                .resolve()
                .map_err(|err| MoeLayerError::Binding(super::binding::MoeBindingError::Dense(err)))?;
            materialized_bytes += bytes;
            let s = expert
                .forward(x)
                .map_err(|err| MoeLayerError::Binding(super::binding::MoeBindingError::Dense(err)))?;
            let scale = match self.convention {
                MoeExecutionConvention::Atenia => 1.0_f32,
                MoeExecutionConvention::HuggingFaceQwen => match &self.shared_gate {
                    Some(g) => sigmoid_dot(g, x),
                    None => 1.0,
                },
            };
            for d in 0..self.config.d_model {
                out[d] += scale * s[d];
            }
        }

        Ok((
            out,
            ResidencyInfo {
                materialized_experts: selection.indices,
                materialized_bytes,
            },
        ))
    }
}

// ============================================================================
// **MOE-FULL-9** — expert LRU cache (avoid constant NVMe reads).
//
// On-demand residency (MOE-FULL-8) re-reads an expert from its tier on every
// forward that routes to it. For the NVMe tier that is an I/O hit per token.
// The `ExpertCache` keeps recently-resolved experts materialised in RAM under
// a bounded LRU budget, so repeated routing (the common case — a handful of
// experts dominate) is served from RAM. Supports prefetch (warm specific
// experts) and reuse (a cache hit skips the NVMe read entirely).
// ============================================================================

/// Cache instrumentation. `misses` equals the number of tier reads
/// (NVMe/RAM resolutions) actually performed; `hits` are served from cache.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Lookups served from the cache (no tier read).
    pub hits: usize,
    /// Lookups that required a tier read (NVMe/RAM resolution).
    pub misses: usize,
    /// Entries evicted under the LRU budget.
    pub evictions: usize,
    /// Experts loaded by an explicit `prefetch` call.
    pub prefetched: usize,
    /// Total weight bytes read from the tier (misses + prefetch).
    pub tier_bytes_read: usize,
}

/// Bounded LRU cache of materialised experts, keyed by routed-expert index.
/// Capacity 0 disables caching (every lookup is a miss). Owned by the caller
/// and threaded across forwards so residency is amortised over a generation.
#[derive(Debug)]
pub struct ExpertCache {
    capacity: usize,
    clock: u64,
    entries: HashMap<usize, (MoeDenseExpert, u64)>,
    stats: CacheStats,
}

impl ExpertCache {
    /// New cache holding at most `capacity` experts (0 = no caching).
    pub fn new(capacity: usize) -> Self {
        Self { capacity, clock: 0, entries: HashMap::new(), stats: CacheStats::default() }
    }

    /// Current resident entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    /// Cumulative statistics.
    pub fn stats(&self) -> CacheStats {
        self.stats
    }
    /// Whether expert `idx` is currently resident.
    pub fn contains(&self, idx: usize) -> bool {
        self.entries.contains_key(&idx)
    }
    /// Host-RAM bytes the cached experts occupy.
    pub fn resident_bytes(&self) -> usize {
        self.entries
            .values()
            .map(|(e, _)| (e.w_gate.len() + e.w_up.len() + e.w_down.len()) * 4)
            .sum()
    }

    fn bump(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    /// Evict the least-recently-used entry if over budget.
    fn evict_if_needed(&mut self) {
        while self.capacity > 0 && self.entries.len() > self.capacity {
            if let Some((&lru, _)) =
                self.entries.iter().min_by_key(|(_, (_, used))| *used).map(|(k, v)| (k, v))
            {
                self.entries.remove(&lru);
                self.stats.evictions += 1;
            } else {
                break;
            }
        }
    }

    /// Insert (or refresh) a materialised expert. Used by `prefetch` and the
    /// miss path. Counts neither hit nor miss by itself.
    fn put(&mut self, idx: usize, expert: MoeDenseExpert) {
        let t = self.bump();
        self.entries.insert(idx, (expert, t));
        // Capacity 0 means "do not retain": drop immediately after use.
        self.evict_if_needed();
    }
}

impl ResidentExpertLayer {
    /// Resolve expert `idx` through `cache`: a hit reuses the RAM copy (no tier
    /// read), a miss reads the tier, records the cost, and caches the result
    /// (subject to the LRU budget). Returns an owned `MoeDenseExpert` clone for
    /// the forward to consume (cheap relative to the tier read it avoids).
    fn resolve_cached(
        &self,
        cache: &mut ExpertCache,
        idx: usize,
    ) -> Result<MoeDenseExpert, MoeLayerError> {
        if let Some((expert, used)) = cache.entries.get_mut(&idx) {
            cache.clock += 1;
            *used = cache.clock;
            cache.stats.hits += 1;
            return Ok(expert.clone());
        }
        let (expert, bytes) = self.experts[idx]
            .resolve()
            .map_err(|e| MoeLayerError::Binding(super::binding::MoeBindingError::Dense(e)))?;
        cache.stats.misses += 1;
        cache.stats.tier_bytes_read += bytes;
        if cache.capacity > 0 {
            cache.put(idx, expert.clone());
        }
        Ok(expert)
    }

    /// Warm `cache` with the given routed experts (e.g. the hot set identified
    /// by profiling). Each prefetched expert is read once from its tier and
    /// retained under the LRU budget. Idempotent for already-resident experts.
    pub fn prefetch(
        &self,
        cache: &mut ExpertCache,
        indices: &[usize],
    ) -> Result<(), MoeLayerError> {
        for &idx in indices {
            if idx >= self.experts.len() || cache.contains(idx) {
                continue;
            }
            let (expert, bytes) = self.experts[idx]
                .resolve()
                .map_err(|e| MoeLayerError::Binding(super::binding::MoeBindingError::Dense(e)))?;
            cache.stats.prefetched += 1;
            cache.stats.tier_bytes_read += bytes;
            cache.put(idx, expert);
        }
        Ok(())
    }

    /// Like [`Self::forward`], but routed experts are resolved through `cache`
    /// (LRU + prefetch + reuse). Bit-identical output to [`Self::forward`];
    /// the only difference is that repeated experts are served from RAM
    /// instead of re-read from the tier. The `cache` accumulates stats across
    /// calls — thread it through a whole generation to amortise residency.
    pub fn forward_cached(
        &self,
        cache: &mut ExpertCache,
        x: &[f32],
    ) -> Result<(Vec<f32>, ResidencyInfo), MoeLayerError> {
        let weights =
            router_softmax(&self.w_router, self.config.num_experts, self.config.d_model, x)?;
        let renormalize = matches!(self.convention, MoeExecutionConvention::Atenia);
        let selection = top_k_routing_with(&weights, self.config.experts_per_token, renormalize)
            .map_err(MoeLayerError::Sparse)?;

        let mut out = vec![0.0_f32; self.config.d_model];
        let mut materialized_bytes = 0usize;
        for (slot, &e) in selection.indices.iter().enumerate() {
            let before = cache.stats.tier_bytes_read;
            let expert = self.resolve_cached(cache, e)?;
            materialized_bytes += cache.stats.tier_bytes_read - before;
            let y = expert
                .forward(x)
                .map_err(|err| MoeLayerError::Binding(super::binding::MoeBindingError::Dense(err)))?;
            let w = selection.weights[slot];
            for d in 0..self.config.d_model {
                out[d] += w * y[d];
            }
        }

        // Shared expert (Mixtral has none) — resolved directly each token.
        if let Some(se) = &self.shared {
            let (expert, bytes) = se
                .resolve()
                .map_err(|err| MoeLayerError::Binding(super::binding::MoeBindingError::Dense(err)))?;
            materialized_bytes += bytes;
            let s = expert
                .forward(x)
                .map_err(|err| MoeLayerError::Binding(super::binding::MoeBindingError::Dense(err)))?;
            let scale = match self.convention {
                MoeExecutionConvention::Atenia => 1.0_f32,
                MoeExecutionConvention::HuggingFaceQwen => match &self.shared_gate {
                    Some(g) => sigmoid_dot(g, x),
                    None => 1.0,
                },
            };
            for d in 0..self.config.d_model {
                out[d] += scale * s[d];
            }
        }

        Ok((out, ResidencyInfo { materialized_experts: selection.indices, materialized_bytes }))
    }
}

/// Router softmax: `softmax(W_router · x)`, identical to
/// [`MoeDenseLayer::route`]. f64 accumulation.
fn router_softmax(
    w_router: &[f32],
    num_experts: usize,
    d_model: usize,
    x: &[f32],
) -> Result<Vec<f32>, MoeLayerError> {
    if x.len() != d_model {
        return Err(MoeLayerError::Sparse(MoeSparseError::Dense(
            MoeDenseError::DimMismatch { what: "router input", expected: d_model, actual: x.len() },
        )));
    }
    let mut logits = vec![0.0_f32; num_experts];
    for r in 0..num_experts {
        let base = r * d_model;
        let mut acc = 0.0_f64;
        for c in 0..d_model {
            acc += (w_router[base + c] as f64) * (x[c] as f64);
        }
        logits[r] = acc as f32;
    }
    Ok(super::dense::softmax(&logits))
}

/// `sigmoid(g · x)` (f64 accumulation) — mirrors `layer.rs::sigmoid_dot`.
fn sigmoid_dot(g: &[f32], x: &[f32]) -> f32 {
    let n = g.len().min(x.len());
    let mut acc = 0.0_f64;
    for i in 0..n {
        acc += (g[i] as f64) * (x[i] as f64);
    }
    (1.0 / (1.0 + (-acc).exp())) as f32
}

// ============================================================================
// Tests (synthetic + small real-style fixtures; CPU + NVMe only).
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::data_plane::MoeWeightMap;
    use std::collections::HashMap;

    fn seeded(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as u32;
            out.push((u as f32 / u32::MAX as f32) * 2.0 - 1.0);
        }
        out
    }

    /// Build a Mixtral-style real MoE layer with `n` experts (no shared).
    fn build_real(n: usize, d_model: usize, d_ff: usize) -> RealMoeLayer {
        let mut ns: Vec<(String, Vec<usize>)> = Vec::new();
        let mut store: HashMap<String, Vec<f32>> = HashMap::new();
        let router = "model.layers.0.block_sparse_moe.gate.weight".to_string();
        ns.push((router.clone(), vec![n, d_model]));
        store.insert(router, seeded(1, n * d_model));
        for e in 0..n {
            let base = 100 + e as u64;
            let g = format!("model.layers.0.block_sparse_moe.experts.{e}.w1.weight");
            let u = format!("model.layers.0.block_sparse_moe.experts.{e}.w3.weight");
            let d = format!("model.layers.0.block_sparse_moe.experts.{e}.w2.weight");
            ns.push((g.clone(), vec![d_ff, d_model]));
            ns.push((u.clone(), vec![d_ff, d_model]));
            ns.push((d.clone(), vec![d_model, d_ff]));
            store.insert(g, seeded(base * 10 + 1, d_ff * d_model));
            store.insert(u, seeded(base * 10 + 2, d_ff * d_model));
            store.insert(d, seeded(base * 10 + 3, d_model * d_ff));
        }
        let map = MoeWeightMap::from_tensors(ns.iter().map(|(n, s)| (n.as_str(), s.clone())));
        let resolve = move |name: &str| store.get(name).cloned();
        let cfg = MoeLayerConfig::new(n, 2, false, d_model, d_ff).unwrap();
        RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap()
    }

    #[test]
    fn ram_tier_matches_real_layer_bit_for_bit() {
        let real = build_real(4, 8, 16);
        let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Ram).unwrap();
        let x = seeded(7, 8);
        let (got, info) = res.forward(&x).unwrap();
        let want = real.forward_auto(&x).unwrap();
        assert_eq!(got, want, "RAM-tier residency forward must equal RealMoeLayer");
        assert_eq!(info.materialized_experts.len(), 2);
    }

    #[test]
    fn disk_tier_matches_real_layer_bit_for_bit() {
        let real = build_real(6, 8, 16);
        let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();
        let x = seeded(9, 8);
        let (got, info) = res.forward(&x).unwrap();
        let want = real.forward_auto(&x).unwrap();
        assert_eq!(got, want, "NVMe-tier residency forward must equal RealMoeLayer");
        assert_eq!(info.materialized_experts.len(), 2);
    }

    #[test]
    fn disk_tier_keeps_experts_out_of_ram() {
        // Headline residency evidence: with experts on NVMe, host RAM holds
        // ~router only, while full materialisation would be far larger.
        let n = 32;
        let real = build_real(n, 16, 32);
        let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();
        let resident = res.resident_ram_bytes();
        let full = res.full_materialization_bytes();
        // Router only: n * d_model * 4 = 32 * 16 * 4 = 2048 bytes.
        assert_eq!(resident, n * 16 * 4);
        // Experts dominate: full >> resident.
        assert!(full > resident * 20, "expected experts to dominate: full={full} resident={resident}");
    }

    #[test]
    fn ram_tier_keeps_experts_in_ram() {
        let n = 8;
        let real = build_real(n, 16, 32);
        let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Ram).unwrap();
        // RAM tier: resident == full (experts are in RAM).
        assert_eq!(res.resident_ram_bytes(), res.full_materialization_bytes());
    }

    #[test]
    fn only_top_k_experts_materialized_per_forward() {
        // With 64 experts and top-k=2, each forward must resolve exactly 2
        // experts (not 64) — the per-token residency property.
        let real = build_real(64, 8, 16);
        let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();
        let per_expert = (16 * 8 * 2 + 8 * 16) * 4; // gate+up+down bytes
        for s in 0..5u64 {
            let x = seeded(1000 + s, 8);
            let (_, info) = res.forward(&x).unwrap();
            assert_eq!(info.materialized_experts.len(), 2);
            assert_eq!(info.materialized_bytes, 2 * per_expert);
            assert!(info.materialized_experts.iter().all(|&e| e < 64));
        }
    }

    #[test]
    fn forward_is_deterministic_across_tiers() {
        let real = build_real(5, 8, 16);
        let x = seeded(42, 8);
        let ram = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Ram).unwrap();
        let disk = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();
        let (a, ai) = ram.forward(&x).unwrap();
        let (b, _) = ram.forward(&x).unwrap();
        let (c, ci) = disk.forward(&x).unwrap();
        assert_eq!(a, b, "RAM forward deterministic");
        assert_eq!(a, c, "RAM and NVMe tiers agree bit-for-bit");
        assert_eq!(ai.materialized_experts, ci.materialized_experts);
    }

    // ---- MOE-FULL-9: expert cache ----

    #[test]
    fn cached_forward_matches_uncached() {
        let real = build_real(8, 8, 16);
        let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();
        let mut cache = ExpertCache::new(4);
        for s in 0..6u64 {
            let x = seeded(500 + s, 8);
            let (cached, _) = res.forward_cached(&mut cache, &x).unwrap();
            let (plain, _) = res.forward(&x).unwrap();
            assert_eq!(cached, plain, "cached forward must equal uncached (seed {s})");
        }
    }

    #[test]
    fn cache_reuse_avoids_tier_reads() {
        // The SAME token routed twice: the second forward must be all hits
        // (zero new tier reads).
        let real = build_real(8, 8, 16);
        let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();
        let mut cache = ExpertCache::new(8);
        let x = seeded(77, 8);

        let (_, _) = res.forward_cached(&mut cache, &x).unwrap();
        let after_first = cache.stats();
        assert_eq!(after_first.misses, 2, "first forward reads top-k=2 experts");
        assert_eq!(after_first.hits, 0);

        let (_, _) = res.forward_cached(&mut cache, &x).unwrap();
        let after_second = cache.stats();
        assert_eq!(after_second.misses, 2, "no NEW tier reads on the repeat");
        assert_eq!(after_second.hits, 2, "the repeat is served from cache");
    }

    #[test]
    fn prefetch_warms_cache_to_zero_misses() {
        let real = build_real(6, 8, 16);
        let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();
        let mut cache = ExpertCache::new(6);
        // Prefetch ALL experts → subsequent forwards never touch the tier.
        res.prefetch(&mut cache, &[0, 1, 2, 3, 4, 5]).unwrap();
        assert_eq!(cache.stats().prefetched, 6);
        assert_eq!(cache.len(), 6);
        for s in 0..5u64 {
            let x = seeded(900 + s, 8);
            let (_, _) = res.forward_cached(&mut cache, &x).unwrap();
        }
        let st = cache.stats();
        assert_eq!(st.misses, 0, "prefetched cache must serve every forward (0 misses)");
        assert!(st.hits >= 10);
    }

    #[test]
    fn lru_evicts_under_budget() {
        // Capacity 1: each new distinct expert evicts the previous.
        let real = build_real(8, 8, 16);
        let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();
        let mut cache = ExpertCache::new(1);
        for s in 0..6u64 {
            let x = seeded(1234 + s * 7, 8);
            let _ = res.forward_cached(&mut cache, &x).unwrap();
        }
        // Never exceeds the budget; evictions happened.
        assert!(cache.len() <= 1);
        assert!(cache.stats().evictions > 0, "capacity-1 cache must evict");
    }

    #[test]
    fn capacity_zero_disables_caching() {
        let real = build_real(4, 8, 16);
        let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();
        let mut cache = ExpertCache::new(0);
        let x = seeded(55, 8);
        let _ = res.forward_cached(&mut cache, &x).unwrap();
        let _ = res.forward_cached(&mut cache, &x).unwrap();
        // Every lookup is a miss; nothing retained.
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 4);
    }
}
