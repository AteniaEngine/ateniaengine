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
use crate::tensor::disk_tier::{self, DiskDtype, DiskTensorHandle};
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
    /// Element count.
    pub numel: usize,
    /// **MOE-PROD-6 / NUMERIC-POLICY-2** — on-disk dtype: F32 (`numel*4` B), BF16
    /// (`numel*2` B, lossless for bf16-source), or QInt8 (`rows*4 + numel` B,
    /// lossy per-row int8). Backend tensors stay F32/BF16; only routed + shared
    /// experts may be QInt8.
    pub dtype: DiskDtype,
    /// **NUMERIC-POLICY-2** — actual on-disk byte length (dtype-dependent; QInt8
    /// is not a clean per-element width). Used for manifest validation.
    pub bytes: u64,
}

/// **NUMERIC-POLICY-2** — on-disk format chosen for an expert tier tensor.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TierFmt {
    /// Always f32 (`numel*4`).
    F32,
    /// bf16 when losslessly representable, else f32 (MOE-PROD-6 default).
    Bf16Auto,
    /// Per-row symmetric int8 (lossy; `rows*4 + numel`). Experts only.
    QInt8,
}

/// **MOE-PROD-6** — if every value in `data` is **exactly** representable as
/// bf16 (its low 16 mantissa bits are zero — true for any f32 decoded from a
/// bf16 source via `(bits as u32) << 16`), return the truncated `Vec<u16>` bf16
/// bits; otherwise `None` (the tensor must stay f32 to preserve bit-exactness).
/// Truncating `bits >> 16` and later upcasting `<< 16` (`bf16_decode_bulk`) is
/// the identity for such values — so a bf16 tier is bit-identical to f32 here.
fn bf16_truncate_lossless(data: &[f32]) -> Option<Vec<u16>> {
    let mut out = Vec::with_capacity(data.len());
    for &v in data {
        let bits = v.to_bits();
        if bits & 0xFFFF != 0 {
            return None;
        }
        out.push((bits >> 16) as u16);
    }
    Some(out)
}

// ============================================================================
// **MOE-PERF-2** — bf16-resident expert cache + auto-sized capacity.
// ============================================================================

/// **MOE-PERF-2B** — whether cached experts are stored **bf16-resident** (default
/// **on**). bf16 storage is **lossless by construction** (a projection is kept
/// bf16 only when its f32 values are bf16-representable — low 16 bits zero — which
/// is exactly the bf16-tier case; otherwise it stays f32), so it **never changes
/// numerics**. The opt-out (`ATENIA_MOE_CACHE_BF16=0`) forces f32 storage (A/B /
/// debugging).
pub fn cache_bf16_enabled() -> bool {
    std::env::var("ATENIA_MOE_CACHE_BF16").as_deref() != Ok("0")
}

/// **MOE-PERF-2A** — the largest per-layer expert-cache capacity whose total
/// resident footprint (`n_layers × capacity × per_expert_bytes`) fits
/// `ram_budget_bytes`, clamped to `[1, num_experts]`. Pure + deterministic given
/// its inputs. The capacity affects only RAM / tier I-O, **never** the forward
/// numerics (`cached_forward_matches_uncached`), so model output is identical at
/// any capacity. Returns at least 1 (so the most-recent expert is always cached)
/// even under a tiny budget.
pub fn auto_expert_cache_capacity(
    ram_budget_bytes: u64,
    n_layers: usize,
    per_expert_bytes: usize,
    num_experts: usize,
) -> usize {
    let per_layer = (per_expert_bytes as u64).max(1);
    let denom = (n_layers as u64).max(1).saturating_mul(per_layer);
    let cap = (ram_budget_bytes / denom.max(1)) as usize;
    cap.clamp(1, num_experts.max(1))
}

/// **MOE-PERF-2B** — one cached projection weight: stored **bf16** when the f32
/// values are bf16-representable (the bf16-tier case), else verbatim **f32**.
/// Decoding back to f32 is **bit-exact** in both cases.
#[derive(Debug, Clone)]
enum CachedWeight {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}

impl CachedWeight {
    fn store(w: Vec<f32>, allow_bf16: bool) -> Self {
        if allow_bf16 {
            if let Some(bits) = bf16_truncate_lossless(&w) {
                return CachedWeight::Bf16(bits);
            }
        }
        CachedWeight::F32(w)
    }
    /// Decode to f32 — bit-exact (bf16 → `from_bits((b as u32) << 16)`, the inverse
    /// of [`bf16_truncate_lossless`]; f32 → clone).
    fn to_f32(&self) -> Vec<f32> {
        match self {
            CachedWeight::F32(v) => v.clone(),
            CachedWeight::Bf16(b) => {
                let mut out = vec![0.0_f32; b.len()];
                crate::simd_kernels::avx2::bf16_decode_bulk(b, &mut out);
                out
            }
        }
    }
    fn elems(&self) -> usize {
        match self {
            CachedWeight::F32(v) => v.len(),
            CachedWeight::Bf16(b) => b.len(),
        }
    }
    fn bytes(&self) -> usize {
        match self {
            CachedWeight::F32(v) => v.len() * 4,
            CachedWeight::Bf16(b) => b.len() * 2,
        }
    }
    fn is_bf16(&self) -> bool {
        matches!(self, CachedWeight::Bf16(_))
    }
}

/// **MOE-PERF-2B** — a cached expert (three projections), each bf16 or f32.
#[derive(Debug, Clone)]
struct CachedExpert {
    d_model: usize,
    d_ff: usize,
    gate: CachedWeight,
    up: CachedWeight,
    down: CachedWeight,
}

impl CachedExpert {
    fn store(e: &MoeDenseExpert, allow_bf16: bool) -> Self {
        Self {
            d_model: e.d_model,
            d_ff: e.d_ff,
            gate: CachedWeight::store(e.w_gate.clone(), allow_bf16),
            up: CachedWeight::store(e.w_up.clone(), allow_bf16),
            down: CachedWeight::store(e.w_down.clone(), allow_bf16),
        }
    }
    /// Materialise back to an f32 `MoeDenseExpert` (bit-exact to what was stored).
    fn to_dense(&self) -> MoeDenseExpert {
        MoeDenseExpert::new(self.d_model, self.d_ff, self.gate.to_f32(), self.up.to_f32(), self.down.to_f32())
            .expect("cached expert dims are valid by construction")
    }
    fn bytes(&self) -> usize {
        self.gate.bytes() + self.up.bytes() + self.down.bytes()
    }
    fn f32_bytes(&self) -> usize {
        (self.gate.elems() + self.up.elems() + self.down.elems()) * 4
    }
    fn is_bf16(&self) -> bool {
        self.gate.is_bf16()
    }
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
    fmt: TierFmt,
) -> io::Result<SharedParam> {
    let dtype = write_or_reuse_expert(ctx, key, data, &shape, fmt)?;
    let path = ctx.dir.join(format!("{key}.bin"));
    Ok(SharedParam::Disk { shape, handle: disk_tier::open_existing(&path, data.len(), dtype) })
}

/// **MOE-PROD-6 / NUMERIC-POLICY-2** — write (or reuse) an **expert** tier tensor
/// at `dir/<key>.bin` in the requested [`TierFmt`]: `QInt8` (per-row int8,
/// `rows*4 + numel` B — halves the bf16 tier), `Bf16Auto` (bf16 if lossless
/// else f32), or `F32`. Reuse skips the write when an existing file already
/// matches the chosen format's byte length. Records the manifest entry (dtype +
/// actual bytes) and returns the on-disk dtype.
pub fn write_or_reuse_expert(
    ctx: &mut TierContext,
    key: &str,
    data: &[f32],
    shape: &[usize],
    fmt: TierFmt,
) -> io::Result<DiskDtype> {
    let file = format!("{key}.bin");
    let path = ctx.dir.join(&file);
    let numel = data.len();
    // Quant axis = the tensor's first dim (rows). For non-quant fmts it's unused.
    let rows = shape.first().copied().filter(|&r| r > 0 && numel % r == 0).unwrap_or(numel.max(1));
    let bf16 = if matches!(fmt, TierFmt::Bf16Auto) { bf16_truncate_lossless(data) } else { None };

    let (dtype, disk_bytes) = match fmt {
        TierFmt::QInt8 => (DiskDtype::QInt8, disk_tier::qint8_disk_bytes(numel, rows) as u64),
        TierFmt::Bf16Auto if bf16.is_some() => (DiskDtype::BF16, (numel * 2) as u64),
        _ => (DiskDtype::F32, (numel * 4) as u64),
    };

    let valid = std::fs::metadata(&path).map(|m| m.len() == disk_bytes).unwrap_or(false);
    if valid {
        ctx.reused += 1;
    } else {
        match dtype {
            DiskDtype::QInt8 => {
                disk_tier::write_qint8_tensor_named(&path, data, rows, numel / rows)?;
            }
            DiskDtype::BF16 => {
                disk_tier::write_bf16_tensor_named(&path, bf16.as_ref().unwrap())?;
            }
            DiskDtype::F32 => {
                disk_tier::write_f32_tensor_named(&path, data)?;
            }
        }
        ctx.written += 1;
    }
    ctx.entries.push(TierEntry { key: key.to_string(), file, numel, dtype, bytes: disk_bytes });
    Ok(dtype)
}

/// **MOE-PROD-5 / MOE-PROD-7** — write `data` to `dir/<key>.bin` (reusing an
/// existing file with the expected byte length) and record the manifest entry.
/// Used for tier tensors read back to RAM at warm reconstruction (router /
/// shared-gate / attention / embed / lm_head). With `allow_bf16` the big
/// backend tensors are persisted as bf16 when losslessly representable (halving
/// the warm backend read), exactly like the experts; warm reads detect the
/// dtype by file size (`read_named_to_f32`).
pub fn write_or_reuse(
    ctx: &mut TierContext,
    key: &str,
    data: &[f32],
    allow_bf16: bool,
) -> io::Result<()> {
    // Backend tensors are never int8 (NUMERIC-POLICY-2 quantises experts only).
    let fmt = if allow_bf16 { TierFmt::Bf16Auto } else { TierFmt::F32 };
    write_or_reuse_expert(ctx, key, data, &[], fmt)?;
    Ok(())
}

/// **NUMERIC-POLICY-2/3** — whether to simulate the int8 expert tier numerically
/// (quantise→dequantise the resolved weights per-row). Source order:
/// **in-process override** (set by the certification runner, NUMERIC-POLICY-3) →
/// else the cached env `ATENIA_MOE_QUANT_SIM=int8`. The override lets the cert
/// runner toggle the int8 effect per case on one loaded (bf16) model.
const QSIM_UNSET: u8 = u8::MAX;
static QUANT_SIM_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(QSIM_UNSET);

/// **NUMERIC-POLICY-3** — force the int8 sim on/off in-process (`Some(true/false)`)
/// or clear the override (`None` → fall back to the env).
pub fn set_quant_sim_int8_override(v: Option<bool>) {
    use std::sync::atomic::Ordering;
    QUANT_SIM_OVERRIDE.store(
        match v {
            Some(true) => 1,
            Some(false) => 0,
            None => QSIM_UNSET,
        },
        Ordering::Relaxed,
    );
}

fn quant_sim_int8() -> bool {
    use std::sync::atomic::Ordering;
    use std::sync::OnceLock;
    match QUANT_SIM_OVERRIDE.load(Ordering::Relaxed) {
        1 => return true,
        0 => return false,
        _ => {}
    }
    static SIM: OnceLock<bool> = OnceLock::new();
    *SIM.get_or_init(|| std::env::var("ATENIA_MOE_QUANT_SIM").as_deref() == Ok("int8"))
}

/// Per-row int8 quantize→dequantize round-trip (the lossy effect of an int8 tier).
fn qdq_per_row_i8(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let (q, scales) = disk_tier::quantize_per_row_i8(data, rows, cols);
    disk_tier::dequantize_per_row_i8(&q, &scales, rows, cols)
}

/// Materialise a stored param to an owned `Vec<f32>` (reads NVMe for the
/// `Disk` tier; copies the Arc for the `Ram` tier).
///
/// **MOE-PROD-8** — for the `Disk` tier (the hot generation path) read the tier
/// file **directly** into one owned `Vec<f32>` instead of
/// `to_tensor().ensure_cpu().copy_to_cpu_vec()`, which allocated the f32 buffer
/// **and then cloned it** (a redundant full copy per projection, ×3 per expert
/// resolve). The bytes are identical (`read_f32_tensor` / `read_bf16_tensor` +
/// the same `bf16_decode_bulk` upcast `ensure_cpu` uses) — bit-exact, half the
/// allocation/memcpy traffic on every routed-expert miss.
fn materialize(p: &SharedParam) -> Vec<f32> {
    match p {
        SharedParam::Disk { handle, shape } => match handle.dtype() {
            DiskDtype::F32 => disk_tier::read_f32_tensor(handle)
                .expect("residency: read f32 tier file failed"),
            DiskDtype::BF16 => {
                let bits = disk_tier::read_bf16_tensor(handle)
                    .expect("residency: read bf16 tier file failed");
                let mut out = vec![0.0_f32; bits.len()];
                crate::simd_kernels::avx2::bf16_decode_bulk(&bits, &mut out);
                out
            }
            // **NUMERIC-POLICY-2** — per-row int8: dequantize to f32 (rows = the
            // tensor's first dim from the stored shape).
            DiskDtype::QInt8 => {
                let rows = shape.first().copied().unwrap_or(1);
                disk_tier::read_qint8_to_f32(handle, rows)
                    .expect("residency: read qint8 tier file failed")
            }
        },
        _ => {
            let mut t = p.to_tensor();
            t.ensure_cpu().expect("residency: ensure_cpu failed materialising expert weight");
            t.copy_to_cpu_vec()
        }
    }
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
        fmt: TierFmt,
    ) -> io::Result<Self> {
        Ok(Self {
            d_model: e.d_model,
            d_ff: e.d_ff,
            gate: place_at(ctx, &format!("{prefix}.gate"), vec![e.d_ff, e.d_model], &e.w_gate, fmt)?,
            up: place_at(ctx, &format!("{prefix}.up"), vec![e.d_ff, e.d_model], &e.w_up, fmt)?,
            down: place_at(ctx, &format!("{prefix}.down"), vec![e.d_model, e.d_ff], &e.w_down, fmt)?,
        })
    }

    /// **MOE-PROD-5** — wrap an expert's three projection files from the
    /// persistent tier (`dir/{prefix}.{gate,up,down}.bin`) as disk-backed
    /// `SharedParam`s, validating each file's presence + byte length so a
    /// missing/corrupt tier triggers the caller's fallback.
    fn from_tier(
        dir: &std::path::Path,
        prefix: &str,
        d_model: usize,
        d_ff: usize,
    ) -> io::Result<Self> {
        // **MOE-PROD-6** — detect the on-disk dtype from the file's byte length:
        // `numel*4` → F32, `numel*2` → BF16 (lossless upcast at materialise). A
        // length matching neither is corruption → error → caller falls back.
        let wrap = |role: &str, numel: usize, shape: Vec<usize>| -> io::Result<SharedParam> {
            let path = dir.join(format!("{prefix}.{role}.bin"));
            let md = std::fs::metadata(&path)?;
            let rows = shape.first().copied().unwrap_or(1);
            let dtype = if md.len() == (numel * 4) as u64 {
                DiskDtype::F32
            } else if md.len() == (numel * 2) as u64 {
                DiskDtype::BF16
            } else if md.len() == disk_tier::qint8_disk_bytes(numel, rows) as u64 {
                DiskDtype::QInt8
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "tier file {path:?}: {} bytes, expected {} (f32) / {} (bf16) / {} (qint8)",
                        md.len(),
                        numel * 4,
                        numel * 2,
                        disk_tier::qint8_disk_bytes(numel, rows),
                    ),
                ));
            };
            Ok(SharedParam::Disk { shape, handle: disk_tier::open_existing(&path, numel, dtype) })
        };
        Ok(Self {
            d_model,
            d_ff,
            gate: wrap("gate", d_ff * d_model, vec![d_ff, d_model])?,
            up: wrap("up", d_ff * d_model, vec![d_ff, d_model])?,
            down: wrap("down", d_model * d_ff, vec![d_model, d_ff])?,
        })
    }

    /// Materialise the three projections into a transient `MoeDenseExpert`.
    fn resolve(&self) -> Result<(MoeDenseExpert, usize), MoeDenseError> {
        let mut g = materialize(&self.gate);
        let mut u = materialize(&self.up);
        let mut d = materialize(&self.down);
        // **NUMERIC-POLICY-2** — int8 **simulation**: quantise+dequantise the
        // resolved weights per-row to reproduce the *numerical* effect of an
        // int8 expert tier, reusing the existing bf16 tier (no cold rebuild).
        // Gates the real int8 tier on a cheap certification run.
        if quant_sim_int8() {
            g = qdq_per_row_i8(&g, self.d_ff, self.d_model);
            u = qdq_per_row_i8(&u, self.d_ff, self.d_model);
            d = qdq_per_row_i8(&d, self.d_model, self.d_ff);
        }
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
        fmt: TierFmt,
    ) -> io::Result<Self> {
        // Experts (routed + shared) use the requested format (incl. QInt8);
        // the router + shared-gate stay bf16 (tiny, kept accurate).
        let experts = layer
            .routed
            .experts
            .iter()
            .enumerate()
            .map(|(i, e)| ResidentExpert::from_dense_at(ctx, &format!("L{layer_id}.e{i}"), e, fmt))
            .collect::<io::Result<Vec<_>>>()?;
        let shared = match &layer.shared {
            Some(se) => {
                Some(ResidentExpert::from_dense_at(ctx, &format!("L{layer_id}.shared"), se, fmt)?)
            }
            None => None,
        };
        // **MOE-PROD-5** — persist router + shared-gate too, so a warm load can
        // reconstruct the layer without re-reading the shards.
        write_or_reuse(ctx, &format!("L{layer_id}.router"), &layer.routed.w_router, true)?;
        if let Some(g) = &layer.shared_gate {
            write_or_reuse(ctx, &format!("L{layer_id}.shared_gate"), g, true)?;
        }
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

    /// **MOE-PROD-5** — reconstruct a disk-tier layer **directly from the
    /// persistent tier** (`dir`), without a `RealMoeLayer` and without reading
    /// the shards: experts wrap existing tier files (`open_existing_f32`), the
    /// router + optional shared-gate are read from their small tier files. The
    /// caller must have validated the manifest (all files present, correct
    /// sizes) — otherwise it falls back to the certified shard path. Output is
    /// bit-identical (the tier holds the exact f32 bytes a cold load wrote).
    #[allow(clippy::too_many_arguments)]
    pub fn from_tier(
        config: MoeLayerConfig,
        convention: MoeExecutionConvention,
        dir: &std::path::Path,
        layer_id: usize,
        has_shared: bool,
        shared_gate_present: bool,
        shared_d_ff: usize,
    ) -> io::Result<Self> {
        let d_model = config.d_model;
        let d_ff = config.d_ff;
        let n = config.num_experts;
        // **MOE-PROD-7** — router/shared-gate may be bf16 (dtype-detecting read).
        let w_router = disk_tier::read_named_to_f32(
            &dir.join(format!("L{layer_id}.router.bin")),
            n * d_model,
        )?;
        let experts = (0..n)
            .map(|i| ResidentExpert::from_tier(dir, &format!("L{layer_id}.e{i}"), d_model, d_ff))
            .collect::<io::Result<Vec<_>>>()?;
        // The shared expert may have a different FFN width
        // (`shared_expert_intermediate_size` ≠ `moe_intermediate_size` for
        // Qwen-MoE), so it uses `shared_d_ff`, not the routed `d_ff`.
        let shared = if has_shared {
            Some(ResidentExpert::from_tier(
                dir,
                &format!("L{layer_id}.shared"),
                d_model,
                shared_d_ff,
            )?)
        } else {
            None
        };
        let shared_gate = if shared_gate_present {
            Some(disk_tier::read_named_to_f32(
                &dir.join(format!("L{layer_id}.shared_gate.bin")),
                d_model,
            )?)
        } else {
            None
        };
        Ok(Self { config, convention, tier: ExpertTier::Disk, w_router, experts, shared, shared_gate })
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
    /// **MOE-PROD-6** — shared-expert lookups served from the pinned cache slot
    /// (the shared expert is identical every token; cached once per layer).
    pub shared_hits: usize,
    /// **MOE-PROD-6** — shared-expert lookups that required a tier read (the
    /// first token, or every token when the shared cache is disabled).
    pub shared_misses: usize,
    /// **MOE-PERF-3 (instrumentation)** — cumulative nanoseconds spent in the
    /// **shared expert's `forward`** (the matmul that VRAM residency would
    /// accelerate). Bounds the maximum possible benefit of GPU residency.
    pub shared_fwd_nanos: u128,
    /// **MOE-PERF-3 (instrumentation)** — cumulative nanoseconds spent in the
    /// **routed experts' `forward`** (matmul only, excluding the tier resolve).
    pub routed_fwd_nanos: u128,
    /// **MOE-IO (instrumentation)** — cumulative nanoseconds spent in the tier
    /// **`resolve()`** (NVMe read + bf16→f32 decode of expert weights, routed +
    /// shared). Isolates the disk-I/O cost from the matmul and the GraphBuilder.
    pub resolve_nanos: u128,
}

/// Bounded LRU cache of materialised experts, keyed by routed-expert index.
/// Capacity 0 disables caching (every lookup is a miss). Owned by the caller
/// and threaded across forwards so residency is amortised over a generation.
#[derive(Debug)]
pub struct ExpertCache {
    capacity: usize,
    clock: u64,
    /// **MOE-PERF-2B** — routed entries stored bf16-resident (lossless) or f32.
    entries: HashMap<usize, (CachedExpert, u64)>,
    /// **MOE-PERF-2B** — whether new entries are stored bf16 (lossless). Captured
    /// at construction from `cache_bf16_enabled()`.
    allow_bf16: bool,
    stats: CacheStats,
    /// **MOE-PROD-6** — pinned cache slot for the **shared** expert (read every
    /// token in HF-Qwen/Atenia conventions). The shared expert's weights never
    /// change during a generation, so it is materialised once and reused. This
    /// is the highest-leverage per-token I/O cut (the shared FFN is ~4× a routed
    /// expert on Qwen-MoE). Bounded by construction: one expert per layer.
    shared: Option<MoeDenseExpert>,
    /// Whether the pinned shared slot is enabled. When `false`, the shared
    /// expert is resolved from the tier every token (the certified MOE-PROD-5
    /// behaviour) — the safe fallback.
    shared_enabled: bool,
}

impl ExpertCache {
    /// New cache holding at most `capacity` routed experts (0 = no caching).
    /// The shared-expert pinned slot is **enabled** by default.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            clock: 0,
            entries: HashMap::new(),
            allow_bf16: cache_bf16_enabled(),
            stats: CacheStats::default(),
            shared: None,
            shared_enabled: true,
        }
    }

    /// **MOE-PERF-2B** — force the cache's bf16-resident storage on/off (defaults
    /// to `cache_bf16_enabled()`). Test/A-B hook; never changes numerics.
    pub fn set_bf16_resident(&mut self, on: bool) {
        self.allow_bf16 = on;
    }
    /// Whether new routed entries are stored bf16-resident.
    pub fn bf16_resident(&self) -> bool {
        self.allow_bf16
    }
    /// **MOE-PERF-2B** — number of routed entries currently stored as bf16.
    pub fn bf16_entries(&self) -> usize {
        self.entries.values().filter(|(e, _)| e.is_bf16()).count()
    }
    /// **MOE-PERF-2B** — what the resident routed entries WOULD cost as f32.
    pub fn resident_bytes_f32_equiv(&self) -> usize {
        self.entries.values().map(|(e, _)| e.f32_bytes()).sum()
    }
    /// **MOE-PERF-2B** — host-RAM bytes saved vs storing the same entries as f32.
    pub fn resident_bytes_saved(&self) -> usize {
        self.resident_bytes_f32_equiv().saturating_sub(self.routed_resident_bytes())
    }
    fn routed_resident_bytes(&self) -> usize {
        self.entries.values().map(|(e, _)| e.bytes()).sum()
    }

    /// **MOE-PROD-6** — toggle the pinned shared-expert slot. Disabling reverts
    /// to per-token resolution of the shared expert (safe fallback / A-B test).
    pub fn set_shared_enabled(&mut self, enabled: bool) {
        self.shared_enabled = enabled;
        if !enabled {
            self.shared = None;
        }
    }

    /// Whether the pinned shared-expert slot is currently populated.
    pub fn shared_cached(&self) -> bool {
        self.shared.is_some()
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
    /// Host-RAM bytes the cached experts occupy (routed entries bf16/f32 per
    /// **MOE-PERF-2B**, plus the pinned shared slot which stays f32).
    pub fn resident_bytes(&self) -> usize {
        self.routed_resident_bytes()
            + self
                .shared
                .as_ref()
                .map(|e| (e.w_gate.len() + e.w_up.len() + e.w_down.len()) * 4)
                .unwrap_or(0)
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

    /// Insert (or refresh) a materialised expert (stored bf16-resident when
    /// lossless, per **MOE-PERF-2B**). Used by `prefetch` and the miss path.
    /// Counts neither hit nor miss by itself.
    fn put(&mut self, idx: usize, expert: &MoeDenseExpert) {
        let cached = CachedExpert::store(expert, self.allow_bf16);
        let t = self.bump();
        self.entries.insert(idx, (cached, t));
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
        if let Some((cached, used)) = cache.entries.get_mut(&idx) {
            cache.clock += 1;
            *used = cache.clock;
            // **MOE-PERF-2B** — decode the cached (bf16/f32) expert back to f32;
            // bit-exact to the original resolve.
            let dense = cached.to_dense();
            cache.stats.hits += 1;
            return Ok(dense);
        }
        let t_res = std::time::Instant::now();
        let (expert, bytes) = self.experts[idx]
            .resolve()
            .map_err(|e| MoeLayerError::Binding(super::binding::MoeBindingError::Dense(e)))?;
        cache.stats.resolve_nanos += t_res.elapsed().as_nanos();
        cache.stats.misses += 1;
        cache.stats.tier_bytes_read += bytes;
        if cache.capacity > 0 {
            cache.put(idx, &expert);
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
            cache.put(idx, &expert);
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
            let t_fwd = std::time::Instant::now();
            let y = expert
                .forward(x)
                .map_err(|err| MoeLayerError::Binding(super::binding::MoeBindingError::Dense(err)))?;
            cache.stats.routed_fwd_nanos += t_fwd.elapsed().as_nanos();
            let w = selection.weights[slot];
            for d in 0..self.config.d_model {
                out[d] += w * y[d];
            }
        }

        // Shared expert (Mixtral has none). **MOE-PROD-6** — the shared expert
        // is identical every token, so it is materialised once into the cache's
        // pinned slot and reused (the biggest per-token I/O cut). With the slot
        // disabled it is resolved from the tier every token (MOE-PROD-5).
        if let Some(se) = &self.shared {
            // Populate the pinned slot on a miss (or every token if disabled).
            if cache.shared_enabled {
                if cache.shared.is_none() {
                    let t_res = std::time::Instant::now();
                    let (expert, bytes) = se.resolve().map_err(|err| {
                        MoeLayerError::Binding(super::binding::MoeBindingError::Dense(err))
                    })?;
                    cache.stats.resolve_nanos += t_res.elapsed().as_nanos();
                    cache.stats.shared_misses += 1;
                    cache.stats.tier_bytes_read += bytes;
                    materialized_bytes += bytes;
                    cache.shared = Some(expert);
                } else {
                    cache.stats.shared_hits += 1;
                }
            }
            let t_sfwd = std::time::Instant::now();
            let s = if cache.shared_enabled {
                // Borrow the pinned expert (no clone of the large weights).
                cache.shared.as_ref().unwrap().forward(x)
            } else {
                let (expert, bytes) = se.resolve().map_err(|err| {
                    MoeLayerError::Binding(super::binding::MoeBindingError::Dense(err))
                })?;
                cache.stats.shared_misses += 1;
                cache.stats.tier_bytes_read += bytes;
                materialized_bytes += bytes;
                expert.forward(x)
            }
            .map_err(|err| MoeLayerError::Binding(super::binding::MoeBindingError::Dense(err)))?;
            cache.stats.shared_fwd_nanos += t_sfwd.elapsed().as_nanos();
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

    /// **MOE-PROD-6** — Qwen-MoE-style real layer WITH a shared expert (its FFN
    /// width `d_ff_s` may differ from the routed `d_ff`) + a sigmoid shared gate.
    /// Exercises the pinned shared-expert cache and the HF-Qwen convention.
    fn build_real_with_shared(n: usize, d_model: usize, d_ff: usize, d_ff_s: usize) -> RealMoeLayer {
        let mut ns: Vec<(String, Vec<usize>)> = Vec::new();
        let mut store: HashMap<String, Vec<f32>> = HashMap::new();
        let router = "model.layers.0.mlp.gate.weight".to_string();
        ns.push((router.clone(), vec![n, d_model]));
        store.insert(router, seeded(1, n * d_model));
        for e in 0..n {
            let base = 100 + e as u64;
            let g = format!("model.layers.0.mlp.experts.{e}.gate_proj.weight");
            let u = format!("model.layers.0.mlp.experts.{e}.up_proj.weight");
            let d = format!("model.layers.0.mlp.experts.{e}.down_proj.weight");
            ns.push((g.clone(), vec![d_ff, d_model]));
            ns.push((u.clone(), vec![d_ff, d_model]));
            ns.push((d.clone(), vec![d_model, d_ff]));
            store.insert(g, seeded(base * 10 + 1, d_ff * d_model));
            store.insert(u, seeded(base * 10 + 2, d_ff * d_model));
            store.insert(d, seeded(base * 10 + 3, d_model * d_ff));
        }
        // Shared expert (different FFN width) + sigmoid gate.
        let sg = "model.layers.0.mlp.shared_expert.gate_proj.weight".to_string();
        let su = "model.layers.0.mlp.shared_expert.up_proj.weight".to_string();
        let sd = "model.layers.0.mlp.shared_expert.down_proj.weight".to_string();
        let sgate = "model.layers.0.mlp.shared_expert_gate.weight".to_string();
        ns.push((sg.clone(), vec![d_ff_s, d_model]));
        ns.push((su.clone(), vec![d_ff_s, d_model]));
        ns.push((sd.clone(), vec![d_model, d_ff_s]));
        ns.push((sgate.clone(), vec![1, d_model]));
        store.insert(sg, seeded(9001, d_ff_s * d_model));
        store.insert(su, seeded(9002, d_ff_s * d_model));
        store.insert(sd, seeded(9003, d_model * d_ff_s));
        store.insert(sgate, seeded(9004, d_model));
        let map = MoeWeightMap::from_tensors(ns.iter().map(|(n, s)| (n.as_str(), s.clone())));
        let resolve = move |name: &str| store.get(name).cloned();
        let cfg = MoeLayerConfig::new(n, 2, true, d_model, d_ff).unwrap();
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

    // ---- MOE-PROD-6: bf16 lossless detection + shared cache ----

    #[test]
    fn bf16_truncate_lossless_detects_representability() {
        // bf16-source values (low 16 bits zero) → Some, truncation is exact.
        let bf16_src: Vec<f32> =
            [0.0_f32, -2.5, 1.5, 256.0].iter().map(|f| f32::from_bits(f.to_bits() & 0xFFFF_0000)).collect();
        let bits = bf16_truncate_lossless(&bf16_src).expect("bf16-source must be representable");
        assert_eq!(bits.len(), bf16_src.len());
        // Upcast bf16 → f32 must recover the exact value.
        for (b, v) in bits.iter().zip(bf16_src.iter()) {
            assert_eq!(f32::from_bits((*b as u32) << 16).to_bits(), v.to_bits());
        }
        // A value with non-zero low 16 bits is NOT representable → None (stays f32).
        let arbitrary = vec![0.0_f32, f32::from_bits(0x3F80_0001)];
        assert!(bf16_truncate_lossless(&arbitrary).is_none());
    }

    #[test]
    fn shared_cache_pins_and_is_bit_exact() {
        // A layer WITH a shared expert: forward_cached must (a) read the shared
        // expert from the tier exactly once across many tokens (pinned slot),
        // and (b) produce output identical to the uncached forward.
        let real = build_real_with_shared(4, 8, 16, 12);
        let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();
        let mut cache = ExpertCache::new(4);
        for s in 0..5u64 {
            let x = seeded(700 + s, 8);
            let (cached, _) = res.forward_cached(&mut cache, &x).unwrap();
            let (plain, _) = res.forward(&x).unwrap();
            assert_eq!(cached, plain, "shared-cached forward must equal uncached (seed {s})");
        }
        let st = cache.stats();
        assert_eq!(st.shared_misses, 1, "shared expert read from tier exactly once");
        assert_eq!(st.shared_hits, 4, "subsequent tokens reuse the pinned shared slot");
        assert!(cache.shared_cached());
    }

    #[test]
    fn shared_cache_disabled_resolves_every_token() {
        let real = build_real_with_shared(4, 8, 16, 12);
        let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Disk).unwrap();
        let mut cache = ExpertCache::new(4);
        cache.set_shared_enabled(false);
        for s in 0..3u64 {
            let x = seeded(800 + s, 8);
            let (off, _) = res.forward_cached(&mut cache, &x).unwrap();
            let (plain, _) = res.forward(&x).unwrap();
            assert_eq!(off, plain, "disabled shared cache still bit-exact");
        }
        let st = cache.stats();
        assert_eq!(st.shared_misses, 3, "disabled slot resolves shared every token");
        assert_eq!(st.shared_hits, 0);
        assert!(!cache.shared_cached());
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

    // ---- MOE-PERF-2: auto-sized capacity + bf16-resident cache ----

    /// bf16-representable weights (low 16 mantissa bits zero) — what the bf16 tier
    /// produces. The cache stores these bf16-resident, losslessly.
    fn bf16_vec(seed: u64, n: usize) -> Vec<f32> {
        seeded(seed, n).into_iter().map(|v| f32::from_bits(v.to_bits() & 0xFFFF_0000)).collect()
    }

    /// Mixtral-style layer whose expert weights are bf16-representable.
    fn build_real_bf16(n: usize, d_model: usize, d_ff: usize) -> RealMoeLayer {
        let mut ns: Vec<(String, Vec<usize>)> = Vec::new();
        let mut store: HashMap<String, Vec<f32>> = HashMap::new();
        let router = "model.layers.0.block_sparse_moe.gate.weight".to_string();
        ns.push((router.clone(), vec![n, d_model]));
        store.insert(router, bf16_vec(1, n * d_model));
        for e in 0..n {
            let base = 100 + e as u64;
            let g = format!("model.layers.0.block_sparse_moe.experts.{e}.w1.weight");
            let u = format!("model.layers.0.block_sparse_moe.experts.{e}.w3.weight");
            let d = format!("model.layers.0.block_sparse_moe.experts.{e}.w2.weight");
            ns.push((g.clone(), vec![d_ff, d_model]));
            ns.push((u.clone(), vec![d_ff, d_model]));
            ns.push((d.clone(), vec![d_model, d_ff]));
            store.insert(g, bf16_vec(base * 10 + 1, d_ff * d_model));
            store.insert(u, bf16_vec(base * 10 + 2, d_ff * d_model));
            store.insert(d, bf16_vec(base * 10 + 3, d_model * d_ff));
        }
        let map = MoeWeightMap::from_tensors(ns.iter().map(|(n, s)| (n.as_str(), s.clone())));
        let resolve = move |name: &str| store.get(name).cloned();
        let cfg = MoeLayerConfig::new(n, 2, false, d_model, d_ff).unwrap();
        RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap()
    }

    #[test]
    fn perf2_auto_capacity_fits_budget() {
        let pe = 1usize << 20; // 1 MiB per expert
        // Plenty of budget → clamp to num_experts.
        assert_eq!(auto_expert_cache_capacity(u64::MAX, 1, pe, 4), 4);
        // 100 MiB / (10 layers × 1 MiB) = 10 → clamp to 8 experts.
        assert_eq!(auto_expert_cache_capacity(100 << 20, 10, pe, 8), 8);
        // 20 MiB / (10 layers × 1 MiB) = 2 per layer.
        assert_eq!(auto_expert_cache_capacity(20 << 20, 10, pe, 8), 2);
        // Tiny budget → at least 1 (never 0).
        assert_eq!(auto_expert_cache_capacity(1, 10, pe, 8), 1);
    }

    #[test]
    fn perf2_cached_weight_bf16_is_lossless() {
        let bf = bf16_vec(123, 64);
        let w = CachedWeight::store(bf.clone(), true);
        assert!(matches!(w, CachedWeight::Bf16(_)), "bf16-representable → bf16");
        assert_eq!(w.to_f32(), bf, "bf16 round-trip is bit-exact");
        assert_eq!(w.bytes(), bf.len() * 2, "bf16 is half the f32 bytes");
        // Non-representable values stay f32 (still exact).
        let arb = seeded(7, 64);
        let w2 = CachedWeight::store(arb.clone(), true);
        assert!(matches!(w2, CachedWeight::F32(_)), "non-bf16 → f32 fallback");
        assert_eq!(w2.to_f32(), arb);
        // Opt-out forces f32.
        assert!(matches!(CachedWeight::store(bf, false), CachedWeight::F32(_)));
    }

    #[test]
    fn perf2_bf16_cache_is_bit_exact_and_smaller() {
        let real = build_real_bf16(8, 8, 16);
        let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Ram).unwrap();
        let mut cache = ExpertCache::new(8);
        assert!(cache.bf16_resident(), "bf16-resident on by default");
        for s in 0..6u64 {
            let x = seeded(500 + s, 8);
            let (cached, _) = res.forward_cached(&mut cache, &x).unwrap();
            let (plain, _) = res.forward(&x).unwrap();
            assert_eq!(cached, plain, "bf16-resident cache must be bit-exact (seed {s})");
        }
        assert!(cache.len() > 0);
        assert_eq!(cache.bf16_entries(), cache.len(), "all entries bf16 (weights bf16-representable)");
        // No shared expert here → resident == routed; bf16 is half the f32-equiv.
        assert_eq!(cache.resident_bytes_f32_equiv(), 2 * cache.resident_bytes());
        assert!(cache.resident_bytes_saved() > 0, "bf16 residency must save bytes");
    }

    #[test]
    fn perf2_bf16_disabled_is_f32_and_still_bit_exact() {
        let real = build_real_bf16(6, 8, 16);
        let res = ResidentExpertLayer::from_real_layer(&real, ExpertTier::Ram).unwrap();
        let mut cache = ExpertCache::new(6);
        cache.set_bf16_resident(false);
        for s in 0..4u64 {
            let x = seeded(600 + s, 8);
            let (cached, _) = res.forward_cached(&mut cache, &x).unwrap();
            let (plain, _) = res.forward(&x).unwrap();
            assert_eq!(cached, plain, "f32 cache must be bit-exact");
        }
        assert_eq!(cache.bf16_entries(), 0, "bf16 disabled → all f32");
        assert_eq!(cache.resident_bytes_saved(), 0);
    }
}
