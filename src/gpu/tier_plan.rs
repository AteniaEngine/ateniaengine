//! M6 replan sub-fase 1 — pre-load tier planner.
//!
//! Pure function from `(tensor metadata, free RAM, free VRAM)` to a
//! per-tensor [`Tier`] assignment. Consumed by sub-fase 2's
//! `WeightMapper::load_into_with_residency_plan` to decide whether
//! each parameter lands directly in VRAM, in RAM, or on disk —
//! replacing the post-load `WeightStore::upload_layer_bf16_to_vram`
//! approach (M6 step 4c) which couldn't free RAM by construction
//! because everything was loaded to RAM first.
//!
//! # Inputs
//!
//! [`TierPlanInput`] carries the full safetensors header view
//! (one entry per tensor: name, shape, source dtype) plus a
//! free-memory snapshot (RAM and VRAM). Both numbers are intended
//! to come from `gpu::safety::resource_check::probe_free_*`. The
//! planner does not probe itself — it stays a pure function so
//! unit tests can exercise the policy without touching real
//! hardware.
//!
//! # Policy
//!
//! 1. **GPU-eligible filter**. A tensor is a VRAM candidate iff
//!    its name ends in `_proj.weight` and its shape has rank ≥ 2.
//!    This precisely captures Llama-family Q/K/V/O attention
//!    projections (`*.self_attn.q_proj.weight` etc.) and the FFN
//!    gate/up/down projections (`*.mlp.gate_proj.weight` etc.).
//!    Every other tensor (`*norm*`, `embed_tokens.weight`,
//!    `lm_head.weight`, biases, 1D weights) is auto-excluded
//!    because:
//!    - The Llama executor's RmsNorm / RoPE / SiLU / Softmax /
//!      BroadcastMul / BroadcastAdd / LogSoftmax / IndexSelect
//!      arms call `ensure_cpu` on the operand, downloading any
//!      `Cuda` storage every step — net regression on hot path.
//!    - `Linear` has `try_gpu_linear` hardcoded to `false`
//!      (`gpu/dispatch/hooks.rs`) so `lm_head.weight` would never
//!      be consumed on GPU even if uploaded.
//!
//! 2. **Greedy bin packing** in input order. For each tensor:
//!    - If GPU-eligible and `vram_remaining ≥ vram_cost(tensor)`
//!      → assign [`Tier::Vram`].
//!    - Else if `ram_remaining ≥ ram_cost(tensor)` → assign
//!      [`Tier::Ram`].
//!    - Else → assign [`Tier::Disk`].
//!
//! 3. **Headrooms** (subtracted up-front from the input numbers):
//!    - VRAM: 1 GiB reserved for working buffers (activation
//!      uploads, output downloads, KV-cache slice).
//!    - RAM: 8 GiB reserved as the safety floor that protects
//!      against the May 2 BSOD scenario (a 32 GiB system loading
//!      a 24 GiB model needs at least 8 GiB headroom for OS,
//!      file caches, and transient activations).
//!
//! # Cost calculations
//!
//! - VRAM cost is **always F32** (`numel * 4`). Even when the
//!   source is BF16 on disk, the upload kernel
//!   (`cuda::bf16_to_f32::bf16_to_f32_resident_in_vram`) writes
//!   F32 into the device buffer. M6 step 3a documented this is
//!   the only kernel ABI today; a future BF16-resident kernel
//!   would let this drop to `numel * 2`.
//! - RAM cost is the source dtype size (`numel *
//!   dtype.size_in_bytes()`).
//! - Disk cost is identical to RAM cost (same byte layout on the
//!   `disk_tier` files).

use std::collections::HashMap;

use crate::tensor::DType;

/// Tier assignment for a single parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tier {
    /// Resident in VRAM as F32. Consumed directly by
    /// `cuda_matmul_inplace` on the residency or mixed-residency
    /// path.
    Vram,
    /// Resident in host RAM. Loaded as the source dtype (F32 or
    /// BF16). Consumed by the CPU dispatcher via
    /// `try_gpu_matmul`'s pool / non-pooled paths.
    Ram,
    /// Resident on NVMe via `disk_tier`. Materialised to RAM
    /// on demand by `Tensor::ensure_cpu`'s `Disk` arm.
    Disk,
}

/// Per-tensor metadata for the planner. Mirrors the public
/// surface of `safetensors::TensorEntry` — name, shape, source
/// dtype.
#[derive(Clone, Debug)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl TensorMeta {
    fn numel(&self) -> u64 {
        self.shape.iter().product::<usize>() as u64
    }

    /// VRAM cost if this tensor is uploaded to the device
    /// buffer of the requested `kernel_dtype`.
    ///
    /// **M6 / M7 path** (`kernel_dtype = F32`): `numel × 4`.
    /// The device buffer holds F32 because the M6 upload kernel
    /// (`bf16_to_f32_resident_in_vram`) upcasts BF16 sources at
    /// load time and the matmul kernel
    /// (`matmul_f32_launch_device`) accepts F32 only.
    ///
    /// **M8 path** (`kernel_dtype = BF16`): `numel × 2`.
    /// The device buffer keeps BF16 (allocated via M8.1's
    /// `bf16_to_vram_no_upcast`) and the matmul kernel
    /// (`cuda_matmul_bf16_inplace`, M8.2) consumes BF16 directly.
    /// Halves the VRAM footprint per weight, doubling the
    /// effective capacity of the hardware.
    ///
    /// F16 / Int8 paths would slot in the same way; for now
    /// `BF16` is the only non-F32 supported value (`F16` and
    /// the int dtypes still map to `numel × 4` because no
    /// resident matmul exists for them in M8).
    fn vram_cost_bytes(&self, kernel_dtype: DType) -> u64 {
        match kernel_dtype {
            DType::BF16 => self.numel() * 2,
            _ => self.numel() * 4,
        }
    }

    /// Host RAM cost in the source dtype. Used both for the
    /// `Tier::Ram` and `Tier::Disk` budget calculations (Disk
    /// stores the same byte layout).
    fn ram_cost_bytes(&self) -> u64 {
        self.numel() * (self.dtype.size_in_bytes() as u64)
    }
}

/// Input to [`plan`]. The free-memory numbers are bytes, not
/// MiB — keep the precision; the planner subtracts the
/// headrooms internally before bin-packing.
///
/// **M7.2 — adaptive headroom**. `model_total_bytes` is the sum
/// of `numel × dtype_size` across all input tensors (the on-disk
/// BF16/F32 footprint of the model). `total_ram_bytes` is the
/// box's installed physical RAM (not free). Together they let the
/// planner detect "model dominates RAM" scenarios (e.g. 13B in a
/// 32 GiB box, where the BF16 weights are ~25 GiB and free RAM
/// is ~26 GiB) and inflate the RAM headroom proportionally so
/// some tensors overflow to Disk by construction — the M6 fixed
/// 8 GiB headroom would otherwise let everything fit in RAM and
/// the Disk tier would never trigger.
#[derive(Clone, Debug)]
pub struct TierPlanInput {
    pub tensors: Vec<TensorMeta>,
    pub free_vram_bytes: u64,
    pub free_ram_bytes: u64,
    /// Total raw byte size of the model in source dtype. Sum of
    /// `numel × dtype.size_in_bytes()` across `tensors`. Used by
    /// the adaptive-headroom rule.
    pub model_total_bytes: u64,
    /// Box's installed physical RAM. Probed via
    /// `gpu::safety::resource_check::probe_total_ram_bytes`.
    /// Logged for telemetry; the policy itself uses `free_ram_bytes`.
    pub total_ram_bytes: u64,
    /// **M8.3** — kernel dtype of the VRAM-resident matmul path.
    ///
    /// `F32` (default — backward-compatible with M6 / M7) costs
    /// `numel × 4` per VRAM-eligible weight. `BF16` (M8 path,
    /// gated by `ATENIA_M8_BF16_KERNEL=1` in `pipeline.rs`)
    /// costs `numel × 2`, doubling effective VRAM capacity.
    ///
    /// The planner does not validate that the `BF16` path is
    /// actually wired in the dispatcher — that contract is held
    /// by the caller (M8.4 wires it; before then the flag is a
    /// dry-run for plan inspection).
    pub kernel_dtype: DType,
}

/// Output of [`plan`]. Insertion order preserved for caller
/// inspection; `get` provides O(1) lookup by name.
#[derive(Clone, Debug, Default)]
pub struct TierPlan {
    /// `(name, tier)` pairs in input order.
    pub assignments: Vec<(String, Tier)>,
    /// Total bytes assigned per tier — useful for telemetry and
    /// for the caller to log a residency summary.
    pub vram_bytes_assigned: u64,
    pub ram_bytes_assigned: u64,
    pub disk_bytes_assigned: u64,
    /// **M7.2** — RAM headroom actually applied by the planner.
    /// Equals [`RAM_HEADROOM_BASE_BYTES`] in the small-model
    /// case; bumped above it when `model_total > 0.7 × free_ram`.
    /// Exposed so the caller can log the breakdown without
    /// re-running the policy.
    pub ram_headroom_bytes: u64,
    /// **M7.2** — overflow component of `ram_headroom_bytes`
    /// (i.e. `ram_headroom_bytes - RAM_HEADROOM_BASE_BYTES`).
    /// Zero when the adaptive rule did not trigger.
    pub ram_headroom_overflow_bytes: u64,
    /// Index for O(1) name → tier lookup. Populated alongside
    /// `assignments`.
    by_name: HashMap<String, Tier>,
}

impl TierPlan {
    pub fn get(&self, name: &str) -> Option<Tier> {
        self.by_name.get(name).copied()
    }

    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }

    /// Count of tensors assigned to a given tier. O(n) — intended
    /// for telemetry, not a hot-path lookup.
    pub fn count(&self, tier: Tier) -> usize {
        self.assignments.iter().filter(|(_, t)| *t == tier).count()
    }

    /// Convenience aliases for the per-tier counts. O(n).
    pub fn vram_count(&self) -> usize {
        self.count(Tier::Vram)
    }
    pub fn ram_count(&self) -> usize {
        self.count(Tier::Ram)
    }
    pub fn disk_count(&self) -> usize {
        self.count(Tier::Disk)
    }
}

const VRAM_HEADROOM_BYTES: u64 = 1024 * 1024 * 1024;
/// **M8.7 prerequisite** — staging buffers reserved at the top of the
/// VRAM budget when the plan would otherwise put any tensor on Disk.
///
/// The Disk → GPU JIT pipeline (M8.7) streams BF16 weights from NVMe
/// through two pinned-host slots into VRAM staging buffers, then
/// dispatches `cuda_matmul_bf16_inplace` against each slot. The
/// upper-bound payload is the 13B `mlp.down_proj.weight`
/// (`numel = 5120 × 13824 = 70 778 880`, BF16 = 141 557 760 bytes).
/// Two slots × ~135 MiB ≈ 283 MiB. We allocate 2 × 135 MiB
/// (`135 × 2²⁰`) exactly so the constant matches the documented
/// staging budget and lines up with the M8.0b two-buffer bench
/// (`examples/bench_disk_gpu_pipeline.rs`).
///
/// Reserved only when `disk_bytes_assigned > 0` in the dry-run pass —
/// 7B-class models that fit entirely in VRAM/RAM keep the M8 budget
/// unchanged, so this constant never penalises the fast-path
/// configuration.
pub const DISK_PIPELINE_STAGING_BYTES: u64 = 2 * 135 * 1024 * 1024;
/// **M7.2** — base RAM headroom. Always reserved regardless of
/// model size (matches the M6 fixed value). The adaptive rule
/// below can only *increase* the headroom, never reduce it.
pub const RAM_HEADROOM_BASE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
/// **M7.2** — adaptive trigger threshold. When `model_total
/// > 0.7 × free_ram` the planner inflates the headroom by the
/// excess so that some tensors are forced to overflow to Disk.
/// Encoded as numerator/denominator to keep the math integer.
const ADAPTIVE_TRIGGER_NUM: u64 = 7;
const ADAPTIVE_TRIGGER_DEN: u64 = 10;

/// **M7.2** — pure helper that computes the RAM headroom for a
/// given `(model_total_bytes, free_ram_bytes)` pair. Returns
/// `(ram_headroom_bytes, overflow_bytes)` where `overflow_bytes`
/// is the amount above [`RAM_HEADROOM_BASE_BYTES`] (zero when
/// the adaptive rule did not trigger).
///
/// Policy:
/// - Threshold = `0.7 × free_ram_bytes`.
/// - If `model_total_bytes > threshold`:
///   `extra = model_total_bytes - threshold` and the headroom is
///   `RAM_HEADROOM_BASE_BYTES + extra`.
/// - Otherwise the headroom stays at the base value.
pub fn adaptive_ram_headroom(model_total_bytes: u64, free_ram_bytes: u64) -> (u64, u64) {
    let threshold = free_ram_bytes
        .saturating_mul(ADAPTIVE_TRIGGER_NUM)
        / ADAPTIVE_TRIGGER_DEN;
    if model_total_bytes > threshold {
        let extra = model_total_bytes - threshold;
        (RAM_HEADROOM_BASE_BYTES + extra, extra)
    } else {
        (RAM_HEADROOM_BASE_BYTES, 0)
    }
}

/// **GPU-eligible** classifier. Returns `true` if and only if the
/// tensor is a Llama-family attention/FFN projection weight that
/// the M6 step 4d mixed-residency dispatch path can consume from
/// VRAM. See module docs for the full rationale.
fn is_gpu_eligible(meta: &TensorMeta) -> bool {
    meta.shape.len() >= 2 && meta.name.ends_with("_proj.weight")
}

/// Pure function — no I/O, no logging, no allocation beyond the
/// output `TierPlan`. Suitable for direct unit testing.
///
/// **M8.7 prerequisite — two-pass staging reservation**.
/// The Disk → GPU JIT pipeline (M8.7) needs ~283 MiB of VRAM for two
/// streaming staging buffers when at least one tensor lands on Disk.
/// Allocating those buffers without telling the planner would OOM the
/// device (the M8 baseline already fills VRAM up to a 1 GiB headroom).
///
/// Policy:
/// 1. Dry-run with the M8 budget (`free_vram - VRAM_HEADROOM_BYTES`).
/// 2. If `disk_bytes_assigned == 0` → return the dry-run plan
///    unchanged. 7B-class fully-resident configurations are never
///    penalised.
/// 3. Otherwise re-plan with `vram_budget -= DISK_PIPELINE_STAGING_BYTES`
///    so M8.7's staging slots have guaranteed VRAM at load time.
///
/// The two-pass cost is a single extra walk of `input.tensors`; the
/// planner stays a pure function (no I/O, no allocations beyond the
/// returned `TierPlan`).
pub fn plan(input: &TierPlanInput) -> TierPlan {
    let m8_budget = input.free_vram_bytes.saturating_sub(VRAM_HEADROOM_BYTES);
    let dry_run = plan_inner(input, m8_budget);
    if dry_run.disk_bytes_assigned == 0 {
        return dry_run;
    }
    let m8_7_budget = m8_budget.saturating_sub(DISK_PIPELINE_STAGING_BYTES);
    plan_inner(input, m8_7_budget)
}

/// Internal bin-packer. `vram_budget_bytes` is the post-headroom (and
/// post-staging-reservation, when applicable) VRAM budget — the
/// planner will not subtract `VRAM_HEADROOM_BYTES` again.
fn plan_inner(input: &TierPlanInput, vram_budget_bytes: u64) -> TierPlan {
    let (ram_headroom, overflow) =
        adaptive_ram_headroom(input.model_total_bytes, input.free_ram_bytes);

    let mut vram_remaining = vram_budget_bytes;
    let mut ram_remaining = input.free_ram_bytes.saturating_sub(ram_headroom);

    let mut out = TierPlan::default();
    out.assignments.reserve(input.tensors.len());
    out.by_name.reserve(input.tensors.len());
    out.ram_headroom_bytes = ram_headroom;
    out.ram_headroom_overflow_bytes = overflow;

    for meta in &input.tensors {
        let vram_cost = meta.vram_cost_bytes(input.kernel_dtype);
        let ram_cost = meta.ram_cost_bytes();

        let tier = if is_gpu_eligible(meta) && vram_remaining >= vram_cost {
            vram_remaining -= vram_cost;
            out.vram_bytes_assigned += vram_cost;
            Tier::Vram
        } else if ram_remaining >= ram_cost {
            ram_remaining -= ram_cost;
            out.ram_bytes_assigned += ram_cost;
            Tier::Ram
        } else {
            out.disk_bytes_assigned += ram_cost;
            Tier::Disk
        };

        out.assignments.push((meta.name.clone(), tier));
        out.by_name.insert(meta.name.clone(), tier);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_meta(name: &str, shape: Vec<usize>, dtype: DType) -> TensorMeta {
        TensorMeta {
            name: name.to_string(),
            shape,
            dtype,
        }
    }

    /// Helper — sum of source-dtype byte sizes across a tensor list.
    /// Used by tests to feed the M7.2 `model_total_bytes` field.
    fn sum_model_bytes(tensors: &[TensorMeta]) -> u64 {
        tensors.iter().map(|t| t.ram_cost_bytes()).sum()
    }

    /// Llama-family layer template — 4 attention proj + 3 FFN
    /// proj + 2 norms. Matches the HuggingFace naming convention
    /// used by Llama 2, Llama 3, Qwen 2.5, TinyLlama, SmolLM2.
    fn llama_layer(layer_idx: usize, hidden: usize, intermediate: usize, dtype: DType) -> Vec<TensorMeta> {
        let prefix = format!("model.layers.{}.", layer_idx);
        vec![
            make_meta(&format!("{}self_attn.q_proj.weight", prefix), vec![hidden, hidden], dtype),
            make_meta(&format!("{}self_attn.k_proj.weight", prefix), vec![hidden, hidden], dtype),
            make_meta(&format!("{}self_attn.v_proj.weight", prefix), vec![hidden, hidden], dtype),
            make_meta(&format!("{}self_attn.o_proj.weight", prefix), vec![hidden, hidden], dtype),
            make_meta(&format!("{}mlp.gate_proj.weight", prefix), vec![intermediate, hidden], dtype),
            make_meta(&format!("{}mlp.up_proj.weight", prefix), vec![intermediate, hidden], dtype),
            make_meta(&format!("{}mlp.down_proj.weight", prefix), vec![hidden, intermediate], dtype),
            make_meta(&format!("{}input_layernorm.weight", prefix), vec![hidden], dtype),
            make_meta(&format!("{}post_attention_layernorm.weight", prefix), vec![hidden], dtype),
        ]
    }

    /// 1. VRAM abundante → proj a VRAM, norms a RAM.
    #[test]
    fn abundant_vram_routes_proj_to_vram_norms_to_ram() {
        let tensors = llama_layer(0, 4096, 11008, DType::BF16);
        let model_total = sum_model_bytes(&tensors);
        // 16 GiB free VRAM, 32 GiB free RAM — enough for one layer's
        // ~1.5 GiB F32 projection footprint plus headrooms.
        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 16 * 1024 * 1024 * 1024,
            free_ram_bytes: 32 * 1024 * 1024 * 1024,
            model_total_bytes: model_total,
            total_ram_bytes: 32 * 1024 * 1024 * 1024,
            kernel_dtype: DType::F32,
        };
        let p = plan(&input);

        // 7 _proj.weight tensors → Vram. 2 layernorm.weight tensors
        // (1D shape) → Ram via the rank-≥-2 filter.
        assert_eq!(p.count(Tier::Vram), 7,
            "expected 7 proj weights on Vram, got {}", p.count(Tier::Vram));
        assert_eq!(p.count(Tier::Ram), 2,
            "expected 2 norm weights on Ram, got {}", p.count(Tier::Ram));
        assert_eq!(p.count(Tier::Disk), 0);

        // Spot-check by name.
        assert_eq!(p.get("model.layers.0.self_attn.q_proj.weight"), Some(Tier::Vram));
        assert_eq!(p.get("model.layers.0.mlp.down_proj.weight"), Some(Tier::Vram));
        assert_eq!(p.get("model.layers.0.input_layernorm.weight"), Some(Tier::Ram));
        assert_eq!(p.get("model.layers.0.post_attention_layernorm.weight"), Some(Tier::Ram));
    }

    /// 2. VRAM llena → overflow a RAM. Greedy bin packing keeps
    /// filling VRAM with smaller tensors after the first layer
    /// runs out — *not* a layer-atomic boundary. The expected
    /// distribution is therefore "layer 0 in full + layer 1
    /// partial" rather than "layer 0 only".
    #[test]
    fn vram_overflow_falls_through_to_ram() {
        // Two Llama 2 13B-class layers.
        // Layer 0 footprint (F32, GPU-eligible only):
        //   - Q,K,V,O proj: 4 × (5120² × 4)        = 400 MiB
        //   - gate proj  : 1 × (5120 × 13824 × 4) = 270 MiB
        //   - up proj    : 1 × (5120 × 13824 × 4) = 270 MiB
        //   - down proj  : 1 × (13824 × 5120 × 4) = 270 MiB
        //   Total = 1210 MiB = ~1.181 GiB.
        // VRAM = 2.5 GiB free, headroom = 1 GiB → 1.5 GiB usable.
        // After layer 0: ~326 MiB remain → fits 3 more 100-MiB
        // proj weights (Q, K, V of layer 1). Layer 1's O / gate /
        // up / down (all ≥100 MiB) overflow to RAM along with
        // every norm.
        let mut tensors = llama_layer(0, 5120, 13824, DType::BF16);
        tensors.extend(llama_layer(1, 5120, 13824, DType::BF16));
        let model_total = sum_model_bytes(&tensors);

        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 25 * 1024 * 1024 * 1024 / 10, // 2.5 GiB
            free_ram_bytes: 32 * 1024 * 1024 * 1024,
            model_total_bytes: model_total,
            total_ram_bytes: 32 * 1024 * 1024 * 1024,
            kernel_dtype: DType::F32,
        };
        let p = plan(&input);

        // 7 (layer 0) + 3 (layer 1 Q,K,V) = 10 Vram assignments.
        assert_eq!(
            p.count(Tier::Vram), 10,
            "greedy continues past layer boundary; expected 10 Vram, got {}",
            p.count(Tier::Vram)
        );
        // Layer 1: O, gate, up, down (4 proj overflows to Ram).
        // 4 norms (2 per layer × 2 layers) → Ram (1D, not GPU-eligible).
        assert_eq!(p.count(Tier::Ram), 4 + 4);
        assert_eq!(p.count(Tier::Disk), 0);

        // Spot-checks: layer 0 fully VRAM, layer 1 mixed.
        assert_eq!(p.get("model.layers.0.self_attn.q_proj.weight"), Some(Tier::Vram));
        assert_eq!(p.get("model.layers.0.mlp.down_proj.weight"), Some(Tier::Vram));
        assert_eq!(p.get("model.layers.1.self_attn.q_proj.weight"), Some(Tier::Vram));
        assert_eq!(p.get("model.layers.1.self_attn.v_proj.weight"), Some(Tier::Vram));
        // O is the 4th attn proj — overflowed because Q,K,V already
        // consumed the ~326 MiB residual.
        assert_eq!(p.get("model.layers.1.self_attn.o_proj.weight"), Some(Tier::Ram));
        assert_eq!(p.get("model.layers.1.mlp.gate_proj.weight"), Some(Tier::Ram));
    }

    /// 3. RAM también llena → overflow a Disk.
    #[test]
    fn ram_full_overflow_to_disk() {
        // One layer Llama 2 13B-class. ~1.21 GiB F32 GPU footprint,
        // ~620 MB BF16 RAM footprint. VRAM = 0 (forces RAM/Disk
        // for everything). RAM = 8.5 GiB free → 0.5 GiB usable
        // after the 8 GiB safety headroom → fits ~one tensor at
        // BF16, the rest go to Disk.
        let tensors = llama_layer(0, 5120, 13824, DType::BF16);
        let model_total = sum_model_bytes(&tensors);
        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 0,
            free_ram_bytes: (85 * 1024 * 1024 * 1024) / 10, // 8.5 GiB
            model_total_bytes: model_total,
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            kernel_dtype: DType::F32,
        };
        let p = plan(&input);

        assert_eq!(p.count(Tier::Vram), 0,
            "no VRAM available, expected 0 Vram assignments");
        assert!(p.count(Tier::Disk) > 0,
            "with 0.5 GiB RAM usable, some tensors must overflow to Disk");
        // Total tensors must equal the input length.
        assert_eq!(p.len(), 9);
        assert_eq!(
            p.count(Tier::Vram) + p.count(Tier::Ram) + p.count(Tier::Disk),
            9
        );
    }

    /// 4. Norms siempre a RAM — aunque haya VRAM libre.
    #[test]
    fn norms_and_embed_always_ram_even_with_abundant_vram() {
        // Mix the special-name tensors that should NEVER go to
        // VRAM regardless of available headroom: norms,
        // embed_tokens, lm_head, model.norm. With 100 GiB free
        // VRAM (way more than the test's input demands), only
        // _proj.weight tensors should land on Vram.
        let tensors = vec![
            make_meta("model.embed_tokens.weight", vec![32000, 4096], DType::BF16),
            make_meta("model.layers.0.input_layernorm.weight", vec![4096], DType::BF16),
            make_meta("model.layers.0.self_attn.q_proj.weight", vec![4096, 4096], DType::BF16),
            make_meta("model.layers.0.post_attention_layernorm.weight", vec![4096], DType::BF16),
            make_meta("model.norm.weight", vec![4096], DType::BF16),
            make_meta("lm_head.weight", vec![32000, 4096], DType::BF16),
        ];
        let model_total = sum_model_bytes(&tensors);
        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 100 * 1024 * 1024 * 1024,
            free_ram_bytes: 100 * 1024 * 1024 * 1024,
            model_total_bytes: model_total,
            total_ram_bytes: 128 * 1024 * 1024 * 1024,
            kernel_dtype: DType::F32,
        };
        let p = plan(&input);

        // Only the _proj.weight tensor on Vram.
        assert_eq!(p.get("model.layers.0.self_attn.q_proj.weight"), Some(Tier::Vram));
        assert_eq!(p.get("model.embed_tokens.weight"), Some(Tier::Ram));
        assert_eq!(p.get("lm_head.weight"), Some(Tier::Ram));
        assert_eq!(p.get("model.norm.weight"), Some(Tier::Ram));
        assert_eq!(p.get("model.layers.0.input_layernorm.weight"), Some(Tier::Ram));
        assert_eq!(p.get("model.layers.0.post_attention_layernorm.weight"), Some(Tier::Ram));

        assert_eq!(p.count(Tier::Vram), 1);
        assert_eq!(p.count(Tier::Ram), 5);
        assert_eq!(p.count(Tier::Disk), 0);
    }

    /// 5. Llama 2 13B real shape — 40 capas, 8 GiB VRAM libre,
    /// 32 GiB RAM. Verifica distribución:
    /// - ~5 capas de proj a VRAM (7 GiB usable / 1.21 GiB per layer).
    /// - El resto de proj weights + 80 norms + embed + lm_head a RAM.
    /// - Cero Disk (32 GiB - 8 GiB headroom = 24 GiB para ~13 GiB BF16 = sobra).
    #[test]
    fn llama2_13b_realistic_distribution() {
        let mut tensors = Vec::new();

        // Embed tokens (BF16 32000 × 5120 = 327 MB).
        tensors.push(make_meta("model.embed_tokens.weight", vec![32000, 5120], DType::BF16));

        // 40 transformer layers, Llama 2 13B shape.
        for i in 0..40 {
            tensors.extend(llama_layer(i, 5120, 13824, DType::BF16));
        }

        // Final norm + LM head.
        tensors.push(make_meta("model.norm.weight", vec![5120], DType::BF16));
        tensors.push(make_meta("lm_head.weight", vec![32000, 5120], DType::BF16));

        // 8 GiB VRAM free → 7 GiB usable after 1 GiB headroom.
        // Per-layer proj F32 cost = 4 × 5120² + 3 × 5120 × 13824
        //                         = 100 MiB + ~810 MiB = 1.21 GiB.
        // 7 / 1.21 ≈ 5.78 layers — but the planner iterates per-
        // tensor, so partial layers are possible. Verify at least
        // 5 layers' worth of proj weights landed on Vram.
        let model_total = sum_model_bytes(&tensors);
        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 8 * 1024 * 1024 * 1024,
            free_ram_bytes: 32 * 1024 * 1024 * 1024,
            model_total_bytes: model_total,
            total_ram_bytes: 32 * 1024 * 1024 * 1024,
            kernel_dtype: DType::F32,
        };
        let p = plan(&input);

        // Expected counts. 40 layers × 7 proj + 40 layers × 2
        // norms + embed + final_norm + lm_head = 280 + 80 + 3 = 363.
        assert_eq!(p.len(), 363);

        // Vram count: must be at least 5 layers' worth of proj
        // (5 × 7 = 35) and at most 6 layers' worth (6 × 7 = 42).
        let vram_count = p.count(Tier::Vram);
        assert!(
            vram_count >= 35 && vram_count <= 42,
            "expected 35..=42 proj weights on Vram (5-6 layers), got {}",
            vram_count
        );

        // Disk count: 0 (32 GiB RAM has plenty of room for the
        // remaining ~12 GiB of BF16 storage after the 8 GiB
        // headroom is reserved).
        assert_eq!(p.count(Tier::Disk), 0,
            "expected 0 Disk assignments with abundant RAM, got {}",
            p.count(Tier::Disk));

        // Ram count = total - vram.
        assert_eq!(p.count(Tier::Ram), 363 - vram_count);

        // Spot-check first uploaded layer.
        assert_eq!(
            p.get("model.layers.0.self_attn.q_proj.weight"),
            Some(Tier::Vram)
        );
        // Embed and lm_head must be RAM regardless.
        assert_eq!(p.get("model.embed_tokens.weight"), Some(Tier::Ram));
        assert_eq!(p.get("lm_head.weight"), Some(Tier::Ram));
        assert_eq!(p.get("model.norm.weight"), Some(Tier::Ram));

        // Spot-check that some late layer's proj overflowed to RAM.
        assert_eq!(
            p.get("model.layers.30.self_attn.q_proj.weight"),
            Some(Tier::Ram)
        );

        // Aggregate byte accounting sanity: vram_bytes_assigned
        // should be > 5 GiB (5 layers × 1.21 GiB) and < 7 GiB
        // (the usable budget).
        let vram_gib = p.vram_bytes_assigned as f64 / (1024.0_f64.powi(3));
        assert!(
            vram_gib >= 5.0 && vram_gib <= 7.0,
            "expected 5..=7 GiB on Vram, got {:.2} GiB",
            vram_gib
        );
    }

    // ----------------------------------------------------------------
    // M7.2 — adaptive RAM headroom tests.
    //
    // The fixed-headroom planner (M6) reserved exactly 8 GiB of RAM
    // regardless of model size, which let a 13B BF16 model (~24 GiB)
    // fit entirely in a 32 GiB box's RAM budget — meaning the Disk
    // tier never triggered. M7.2 inflates the headroom proportionally
    // when `model_total > 0.7 × free_ram` so genuine overflow scenarios
    // route to NVMe by construction.
    // ----------------------------------------------------------------

    /// Build a synthetic 7B-class Llama (32 layers) for the
    /// adaptive-headroom tests. Hidden=4096, intermediate=11008.
    fn synth_7b_model() -> Vec<TensorMeta> {
        let mut tensors = Vec::new();
        tensors.push(make_meta(
            "model.embed_tokens.weight",
            vec![32000, 4096],
            DType::BF16,
        ));
        for i in 0..32 {
            tensors.extend(llama_layer(i, 4096, 11008, DType::BF16));
        }
        tensors.push(make_meta("model.norm.weight", vec![4096], DType::BF16));
        tensors.push(make_meta("lm_head.weight", vec![32000, 4096], DType::BF16));
        tensors
    }

    /// Build a synthetic 13B-class Llama (40 layers). Hidden=5120,
    /// intermediate=13824. Same shape used in test 5 above.
    fn synth_13b_model() -> Vec<TensorMeta> {
        let mut tensors = Vec::new();
        tensors.push(make_meta(
            "model.embed_tokens.weight",
            vec![32000, 5120],
            DType::BF16,
        ));
        for i in 0..40 {
            tensors.extend(llama_layer(i, 5120, 13824, DType::BF16));
        }
        tensors.push(make_meta("model.norm.weight", vec![5120], DType::BF16));
        tensors.push(make_meta("lm_head.weight", vec![32000, 5120], DType::BF16));
        tensors
    }

    /// 6. **13B en 32 GiB** — adaptive trigger fires; headroom
    /// inflates above 8 GiB and forces Disk overflow.
    ///
    /// Model BF16 ≈ 24.24 GiB, free_ram = 26 GiB (realistic on a
    /// 32 GiB box with OS/drivers using ~6 GiB). Threshold =
    /// 0.7 × 26 = 18.2 GiB. Model exceeds threshold by ~6 GiB →
    /// headroom bumps to ~14 GiB. With free_vram = 0 and budget
    /// of ~12 GiB, the ~24 GiB BF16 model overflows ~12 GiB to Disk.
    #[test]
    fn m7_2_adaptive_13b_on_32gib_box_triggers_disk_overflow() {
        let tensors = synth_13b_model();
        let model_total = sum_model_bytes(&tensors);
        let free_ram = 26 * 1024 * 1024 * 1024_u64;

        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 0,
            free_ram_bytes: free_ram,
            model_total_bytes: model_total,
            total_ram_bytes: 32 * 1024 * 1024 * 1024,
            kernel_dtype: DType::F32,
        };
        let p = plan(&input);

        // Adaptive trigger must fire — overflow > 0.
        assert!(
            p.ram_headroom_overflow_bytes > 0,
            "expected adaptive headroom to trigger, got base headroom only"
        );
        assert!(
            p.ram_headroom_bytes > RAM_HEADROOM_BASE_BYTES,
            "expected ram_headroom > 8 GiB base, got {} bytes",
            p.ram_headroom_bytes
        );
        // Some tensors must overflow to Disk.
        assert!(
            p.count(Tier::Disk) > 0,
            "expected Disk overflow on 13B/32GiB box, got {} Disk assignments",
            p.count(Tier::Disk)
        );
        assert!(
            p.disk_bytes_assigned > 0,
            "expected nonzero Disk bytes assigned, got {}",
            p.disk_bytes_assigned
        );
    }

    /// 7. **7B en 32 GiB** — adaptive does NOT trigger; behaviour
    /// is identical to M6 (everything fits in RAM).
    ///
    /// Model BF16 ≈ 13 GiB, free_ram = 26 GiB. Threshold =
    /// 18.2 GiB. Model is well below → headroom stays at 8 GiB.
    /// 26 - 8 = 18 GiB budget > 13 GiB model → no Disk.
    #[test]
    fn m7_2_adaptive_7b_on_32gib_box_keeps_base_headroom() {
        let tensors = synth_7b_model();
        let model_total = sum_model_bytes(&tensors);

        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 0,
            free_ram_bytes: 26 * 1024 * 1024 * 1024,
            model_total_bytes: model_total,
            total_ram_bytes: 32 * 1024 * 1024 * 1024,
            kernel_dtype: DType::F32,
        };
        let p = plan(&input);

        assert_eq!(
            p.ram_headroom_bytes, RAM_HEADROOM_BASE_BYTES,
            "expected base 8 GiB headroom for 7B/32GiB, got {} bytes",
            p.ram_headroom_bytes
        );
        assert_eq!(p.ram_headroom_overflow_bytes, 0);
        assert_eq!(
            p.count(Tier::Disk), 0,
            "expected 0 Disk assignments for 7B/32GiB (M6 behaviour preserved)"
        );
    }

    /// 8. **7B en 16 GiB** — RAM-constrained box, adaptive forces
    /// Disk overflow.
    ///
    /// Model BF16 ≈ 13 GiB, free_ram = 12 GiB. Threshold =
    /// 0.7 × 12 = 8.4 GiB. Model exceeds threshold by ~4.6 GiB →
    /// headroom bumps to ~12.6 GiB. Budget = 12 - 12.6 saturates
    /// to 0 → every tensor overflows to Disk.
    #[test]
    fn m7_2_adaptive_7b_on_16gib_box_forces_disk() {
        let tensors = synth_7b_model();
        let model_total = sum_model_bytes(&tensors);
        let total_count = tensors.len();

        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 0,
            free_ram_bytes: 12 * 1024 * 1024 * 1024,
            model_total_bytes: model_total,
            total_ram_bytes: 16 * 1024 * 1024 * 1024,
            kernel_dtype: DType::F32,
        };
        let p = plan(&input);

        assert!(
            p.ram_headroom_overflow_bytes > 0,
            "expected adaptive trigger on 7B/16GiB box"
        );
        // RAM budget saturated — every tensor must land on Disk.
        assert_eq!(
            p.count(Tier::Disk), total_count,
            "expected all {} tensors on Disk (RAM budget saturated to 0), got {}",
            total_count,
            p.count(Tier::Disk)
        );
        assert_eq!(p.count(Tier::Ram), 0);
        assert_eq!(p.count(Tier::Vram), 0);
    }

    /// 9. **Modelo pequeño en hardware modesto** — model fits
    /// comfortably; adaptive must NOT trigger.
    ///
    /// 1B-class synthetic model (~2 GiB BF16), free_ram = 6 GiB.
    /// Threshold = 0.7 × 6 = 4.2 GiB. Model 2 < 4.2 → headroom
    /// stays at 8 GiB base. (RAM budget saturates to 0 since
    /// 6 - 8 < 0, but that's the M6 behaviour for any small box,
    /// not an M7.2 regression — the test asserts the *headroom*,
    /// not the placement.)
    #[test]
    fn m7_2_adaptive_small_model_keeps_base_headroom() {
        // ~2 GiB BF16 of synthetic _proj.weight tensors.
        // 16 × 64 MiB = 1024 MiB (one weight = 64 MiB BF16 →
        // shape [4096, 8192], numel 33.5M, ×2 = 67.1 MB ≈ 64 MiB).
        let mut tensors = Vec::new();
        for i in 0..16 {
            tensors.push(make_meta(
                &format!("model.layers.{}.self_attn.q_proj.weight", i),
                vec![4096, 8192],
                DType::BF16,
            ));
        }
        let model_total = sum_model_bytes(&tensors);
        // Sanity: ~1 GiB BF16, well below the 4.2 GiB threshold.
        assert!(model_total < 4 * 1024 * 1024 * 1024);

        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 0,
            free_ram_bytes: 6 * 1024 * 1024 * 1024,
            model_total_bytes: model_total,
            total_ram_bytes: 8 * 1024 * 1024 * 1024,
            kernel_dtype: DType::F32,
        };
        let p = plan(&input);

        assert_eq!(
            p.ram_headroom_bytes, RAM_HEADROOM_BASE_BYTES,
            "expected base headroom for small model, got {} bytes",
            p.ram_headroom_bytes
        );
        assert_eq!(p.ram_headroom_overflow_bytes, 0);
    }

    /// 10. Pure-helper sanity — direct unit test of
    /// `adaptive_ram_headroom` covering both branches and the
    /// boundary condition.
    #[test]
    fn m7_2_adaptive_ram_headroom_helper_branches() {
        let gib = 1024_u64 * 1024 * 1024;

        // Below threshold → base headroom, zero overflow.
        let (h, o) = adaptive_ram_headroom(10 * gib, 32 * gib);
        assert_eq!(h, RAM_HEADROOM_BASE_BYTES);
        assert_eq!(o, 0);

        // Exactly at threshold (0.7 × 32 = 22.4 GiB) — must NOT
        // trigger (rule is strict `>`).
        let threshold = 32 * gib * 7 / 10;
        let (h, o) = adaptive_ram_headroom(threshold, 32 * gib);
        assert_eq!(h, RAM_HEADROOM_BASE_BYTES);
        assert_eq!(o, 0);

        // Above threshold → headroom bumps by exactly the excess.
        let (h, o) = adaptive_ram_headroom(threshold + 5 * gib, 32 * gib);
        assert_eq!(o, 5 * gib);
        assert_eq!(h, RAM_HEADROOM_BASE_BYTES + 5 * gib);

        // Zero free RAM → threshold is 0, any positive model
        // triggers; whole model becomes the overflow.
        let (h, o) = adaptive_ram_headroom(3 * gib, 0);
        assert_eq!(o, 3 * gib);
        assert_eq!(h, RAM_HEADROOM_BASE_BYTES + 3 * gib);
    }

    // ----------------------------------------------------------------
    // M8.3 — kernel_dtype-aware VRAM cost.
    //
    // The M6 / M7 planner hardcoded `vram_cost_bytes = numel × 4`
    // because the only matmul kernel for VRAM-resident weights was
    // F32 (`matmul_f32_launch_device`). M8.2 introduced a BF16
    // matmul (`cuda_matmul_bf16_inplace` via `cublasGemmEx`) that
    // consumes BF16 directly. With `kernel_dtype = BF16`, the
    // planner counts each VRAM-eligible weight at `numel × 2`,
    // doubling effective VRAM capacity. These tests confirm the
    // policy without exercising the dispatcher (M8.4 wires that).
    // ----------------------------------------------------------------

    /// 11. **13B with BF16 kernel** — VRAM tensor count roughly
    /// doubles vs F32. With 8 GiB free VRAM (7 GiB usable), F32
    /// fits ~5.78 layers (~40 _proj.weight tensors); BF16 fits
    /// ~11.56 layers (~80 _proj.weight tensors). Same input,
    /// only `kernel_dtype` differs.
    #[test]
    fn m8_3_bf16_kernel_doubles_vram_tensor_count_on_13b() {
        let mut tensors = Vec::new();
        tensors.push(make_meta(
            "model.embed_tokens.weight",
            vec![32000, 5120],
            DType::BF16,
        ));
        for i in 0..40 {
            tensors.extend(llama_layer(i, 5120, 13824, DType::BF16));
        }
        tensors.push(make_meta("model.norm.weight", vec![5120], DType::BF16));
        tensors.push(make_meta("lm_head.weight", vec![32000, 5120], DType::BF16));

        let model_total = sum_model_bytes(&tensors);

        // F32 baseline (same input as M7.2 test 5).
        let input_f32 = TierPlanInput {
            tensors: tensors.clone(),
            free_vram_bytes: 8 * 1024 * 1024 * 1024,
            free_ram_bytes: 32 * 1024 * 1024 * 1024,
            model_total_bytes: model_total,
            total_ram_bytes: 32 * 1024 * 1024 * 1024,
            kernel_dtype: DType::F32,
        };
        let p_f32 = plan(&input_f32);

        let input_bf16 = TierPlanInput {
            tensors: tensors.clone(),
            free_vram_bytes: 8 * 1024 * 1024 * 1024,
            free_ram_bytes: 32 * 1024 * 1024 * 1024,
            model_total_bytes: model_total,
            total_ram_bytes: 32 * 1024 * 1024 * 1024,
            kernel_dtype: DType::BF16,
        };
        let p_bf16 = plan(&input_bf16);

        let vram_f32 = p_f32.vram_count();
        let vram_bf16 = p_bf16.vram_count();

        // F32 fits ~5.78 layers worth = roughly 35..=42 _proj
        // tensors. The M7.2 test already locks this range; we
        // re-confirm here as a baseline anchor.
        assert!(
            (35..=42).contains(&vram_f32),
            "F32 baseline expected 35..=42 VRAM tensors, got {}",
            vram_f32
        );
        // BF16 should fit ~11.56 layers = 78..=84 _proj tensors.
        // The exact count depends on greedy bin-packing residual
        // (last layer's down_proj may or may not fit).
        assert!(
            (78..=84).contains(&vram_bf16),
            "BF16 path expected 78..=84 VRAM tensors, got {}",
            vram_bf16
        );

        // Strong invariant: BF16 must fit at least 1.9× more
        // VRAM tensors than F32 (allowing some slack for the
        // last-layer residuals).
        let ratio = (vram_bf16 as f64) / (vram_f32 as f64);
        assert!(
            ratio >= 1.9,
            "BF16/F32 VRAM tensor ratio expected ≥ 1.9, got {:.2} \
             (F32={}, BF16={})",
            ratio,
            vram_f32,
            vram_bf16
        );

        // Aggregate byte accounting: BF16 vram_bytes_assigned
        // is per-tensor `numel × 2`. F32 was `numel × 4`.
        // Both should respect the 7 GiB usable budget.
        let usable_vram = (8 - 1) * 1024 * 1024 * 1024_u64;
        assert!(p_f32.vram_bytes_assigned <= usable_vram);
        assert!(p_bf16.vram_bytes_assigned <= usable_vram);
    }

    /// 12. **13B with F32 kernel default** — regression-zero
    /// gate. With `kernel_dtype = F32` the M7.2 baseline (test 5)
    /// must reproduce: ~35..=42 VRAM tensors.
    #[test]
    fn m8_3_f32_kernel_default_matches_m7_2_baseline() {
        let mut tensors = Vec::new();
        tensors.push(make_meta(
            "model.embed_tokens.weight",
            vec![32000, 5120],
            DType::BF16,
        ));
        for i in 0..40 {
            tensors.extend(llama_layer(i, 5120, 13824, DType::BF16));
        }
        tensors.push(make_meta("model.norm.weight", vec![5120], DType::BF16));
        tensors.push(make_meta("lm_head.weight", vec![32000, 5120], DType::BF16));

        let model_total = sum_model_bytes(&tensors);
        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 8 * 1024 * 1024 * 1024,
            free_ram_bytes: 32 * 1024 * 1024 * 1024,
            model_total_bytes: model_total,
            total_ram_bytes: 32 * 1024 * 1024 * 1024,
            kernel_dtype: DType::F32,
        };
        let p = plan(&input);

        // Same range as M7.2 test 5.
        let vram_count = p.vram_count();
        assert!(
            (35..=42).contains(&vram_count),
            "F32 default expected 35..=42 VRAM tensors (M7.2 baseline), got {}",
            vram_count
        );
        // The bytes should be in the F32 ballpark (vram_count × ~150 MiB).
        // Sanity: F32 VRAM bytes per layer = 1.21 GiB; 5 layers = 6.05 GiB.
        let vram_gib = p.vram_bytes_assigned as f64 / (1024.0_f64.powi(3));
        assert!(
            vram_gib >= 5.0 && vram_gib <= 7.0,
            "F32 VRAM bytes expected 5..=7 GiB, got {:.2}",
            vram_gib
        );
    }

    /// 13. **7B with BF16 kernel** — even on a model that fully
    /// fits at F32, BF16 fits more `_proj.weight` tensors.
    /// 7B has 32 layers × 7 _proj = 224 GPU-eligible tensors.
    /// At 8 GiB VRAM with F32, ~all of them fit (7B is small).
    /// With BF16, even more fit (still all in this case) plus
    /// the per-tensor cost halves the budget consumption — the
    /// test asserts both: vram_bytes_assigned halves with BF16.
    #[test]
    fn m8_3_bf16_kernel_halves_vram_bytes_on_7b() {
        let mut tensors = Vec::new();
        tensors.push(make_meta(
            "model.embed_tokens.weight",
            vec![32000, 4096],
            DType::BF16,
        ));
        for i in 0..32 {
            tensors.extend(llama_layer(i, 4096, 11008, DType::BF16));
        }
        tensors.push(make_meta("model.norm.weight", vec![4096], DType::BF16));
        tensors.push(make_meta("lm_head.weight", vec![32000, 4096], DType::BF16));

        let model_total = sum_model_bytes(&tensors);

        let mk_input = |kd| TierPlanInput {
            tensors: tensors.clone(),
            free_vram_bytes: 8 * 1024 * 1024 * 1024,
            free_ram_bytes: 32 * 1024 * 1024 * 1024,
            model_total_bytes: model_total,
            total_ram_bytes: 32 * 1024 * 1024 * 1024,
            kernel_dtype: kd,
        };

        let p_f32 = plan(&mk_input(DType::F32));
        let p_bf16 = plan(&mk_input(DType::BF16));

        // BF16 must place ≥ as many VRAM tensors as F32.
        assert!(
            p_bf16.vram_count() >= p_f32.vram_count(),
            "BF16 VRAM count ({}) must be ≥ F32 VRAM count ({})",
            p_bf16.vram_count(),
            p_f32.vram_count()
        );

        // For the tensors that BOTH plans placed on VRAM, the
        // BF16 byte cost should be roughly half. Use the
        // overlapping subset by walking the BF16 plan; any
        // tensor on Vram in BF16 plan that is also on Vram in
        // F32 plan contributes to the sum.
        let mut overlap_bytes_bf16 = 0_u64;
        let mut overlap_bytes_f32 = 0_u64;
        for meta in &tensors {
            if !is_gpu_eligible(meta) {
                continue;
            }
            let on_bf16 = p_bf16.get(&meta.name) == Some(Tier::Vram);
            let on_f32 = p_f32.get(&meta.name) == Some(Tier::Vram);
            if on_bf16 && on_f32 {
                overlap_bytes_bf16 += meta.vram_cost_bytes(DType::BF16);
                overlap_bytes_f32 += meta.vram_cost_bytes(DType::F32);
            }
        }
        assert!(overlap_bytes_f32 > 0, "expected non-empty overlap");
        let ratio = (overlap_bytes_bf16 as f64) / (overlap_bytes_f32 as f64);
        assert!(
            (ratio - 0.5).abs() < 0.001,
            "expected BF16/F32 overlap byte ratio = 0.5, got {:.4}",
            ratio
        );
    }

    /// 14. **`vram_cost_bytes` direct dtype check** — the helper
    /// produces the right per-element multiplier for both
    /// supported dtypes, and falls back to F32 for unsupported
    /// (defensive: if a future dtype is added without an
    /// explicit M8 entry, the planner won't silently route it
    /// to BF16 cost).
    #[test]
    fn m8_3_vram_cost_bytes_helper_matches_dtype() {
        let m = make_meta(
            "model.layers.0.self_attn.q_proj.weight",
            vec![5120, 5120],
            DType::BF16,
        );
        let numel = 5120 * 5120;
        assert_eq!(m.vram_cost_bytes(DType::F32), (numel * 4) as u64);
        assert_eq!(m.vram_cost_bytes(DType::BF16), (numel * 2) as u64);
        // F16 (and any future non-BF16/F32) defaults to F32 cost.
        assert_eq!(m.vram_cost_bytes(DType::F16), (numel * 4) as u64);
    }

    /// 15. **Headroom + adaptive interaction** — switching to
    /// BF16 must not change RAM headroom logic. The adaptive
    /// rule uses `model_total_bytes` and `free_ram_bytes`;
    /// neither depends on `kernel_dtype`. Verify the same
    /// `(model_total, free_ram)` pair produces the same
    /// `ram_headroom_bytes` regardless of `kernel_dtype`.
    #[test]
    fn m8_3_kernel_dtype_does_not_affect_ram_headroom() {
        let tensors = synth_13b_model();
        let model_total = sum_model_bytes(&tensors);

        let mk_input = |kd| TierPlanInput {
            tensors: tensors.clone(),
            free_vram_bytes: 0,
            free_ram_bytes: 26 * 1024 * 1024 * 1024,
            model_total_bytes: model_total,
            total_ram_bytes: 32 * 1024 * 1024 * 1024,
            kernel_dtype: kd,
        };

        let p_f32 = plan(&mk_input(DType::F32));
        let p_bf16 = plan(&mk_input(DType::BF16));

        assert_eq!(
            p_f32.ram_headroom_bytes, p_bf16.ram_headroom_bytes,
            "ram_headroom_bytes must be invariant under kernel_dtype \
             (F32: {}, BF16: {})",
            p_f32.ram_headroom_bytes, p_bf16.ram_headroom_bytes
        );
        assert_eq!(
            p_f32.ram_headroom_overflow_bytes, p_bf16.ram_headroom_overflow_bytes
        );
    }

    // ----------------------------------------------------------------
    // M8.7 prerequisite — Disk pipeline staging reservation.
    //
    // When the plan would otherwise put any tensor on Disk, the
    // planner must reserve `DISK_PIPELINE_STAGING_BYTES` (≈283 MiB)
    // at the top of the VRAM budget so the M8.7 streaming staging
    // slots have guaranteed VRAM at load time. Plans that do not
    // touch Disk (7B-class fully-resident configs) must remain
    // bit-identical to the M8 baseline — the staging cost is paid
    // only by configurations that actually need it.
    // ----------------------------------------------------------------

    /// 16. **Disk plan triggers staging reservation** — 13B on a
    /// 32 GiB box with adaptive overflow drops 2 _proj weights from
    /// VRAM (≈270 MiB BF16 freed for two streaming slots) compared
    /// to the same input run through `plan_inner` without the
    /// staging reservation.
    #[test]
    fn m8_7_prereq_disk_plan_reserves_staging_vram() {
        let tensors = synth_13b_model();
        let model_total = sum_model_bytes(&tensors);

        let input = TierPlanInput {
            tensors,
            // 8 GiB free VRAM → 7 GiB usable after the M8 headroom,
            // matching the production 13B-on-this-box configuration.
            free_vram_bytes: 8 * 1024 * 1024 * 1024,
            free_ram_bytes: 26 * 1024 * 1024 * 1024,
            model_total_bytes: model_total,
            total_ram_bytes: 32 * 1024 * 1024 * 1024,
            kernel_dtype: DType::BF16,
        };

        // Reference: M8 baseline budget (no staging reservation).
        let baseline_budget = input
            .free_vram_bytes
            .saturating_sub(VRAM_HEADROOM_BYTES);
        let p_baseline = plan_inner(&input, baseline_budget);

        // Sanity: the dry-run scenario actually overflows to Disk —
        // otherwise the staging reservation wouldn't apply.
        assert!(
            p_baseline.disk_bytes_assigned > 0,
            "test precondition: 13B/32GiB must overflow to Disk in baseline"
        );

        // The public planner should detect Disk overflow and re-plan
        // with the staging reservation.
        let p = plan(&input);

        // Disk overflow remains.
        assert!(p.disk_bytes_assigned > 0);

        // Strong invariant: VRAM count must drop by at least one
        // tensor relative to the no-staging baseline (the BF16 13B
        // FFN-down weight is ~135 MiB, so ~283 MiB of reservation
        // displaces ≥ 1 weight; usually ≥ 2 ≈ 2 × 50 MiB Q/K/V
        // weights or ≥ 1 FFN weight).
        assert!(
            p.vram_count() < p_baseline.vram_count(),
            "expected staging reservation to displace ≥1 VRAM tensor; \
             baseline={} vram, with-staging={} vram",
            p_baseline.vram_count(),
            p.vram_count()
        );

        // Strong invariant: the freed VRAM bytes must be ≥
        // DISK_PIPELINE_STAGING_BYTES − one-tensor-slack. We check
        // the simpler form: bytes_assigned drops by ≥ 0.5 ×
        // staging budget (the displaced tensors won't sum exactly
        // to 283 MiB; greedy bin packing leaves residuals).
        let half_staging = DISK_PIPELINE_STAGING_BYTES / 2;
        assert!(
            p_baseline
                .vram_bytes_assigned
                .saturating_sub(p.vram_bytes_assigned)
                >= half_staging,
            "expected ≥ {} bytes freed by staging reservation; \
             baseline={} vram bytes, with-staging={} vram bytes",
            half_staging,
            p_baseline.vram_bytes_assigned,
            p.vram_bytes_assigned
        );

        // RAM headroom invariant: must be untouched (staging only
        // touches the VRAM budget, not the RAM headroom rule).
        assert_eq!(p.ram_headroom_bytes, p_baseline.ram_headroom_bytes);
        assert_eq!(
            p.ram_headroom_overflow_bytes,
            p_baseline.ram_headroom_overflow_bytes
        );

        // Observability — `cargo test -- --nocapture` prints the
        // actual placement counts so the M8.7 follow-up can confirm
        // the staging reservation displaces the expected number of
        // tensors on the production 13B configuration.
        let baseline_vram_gib =
            p_baseline.vram_bytes_assigned as f64 / (1024.0_f64.powi(3));
        let staged_vram_gib =
            p.vram_bytes_assigned as f64 / (1024.0_f64.powi(3));
        eprintln!(
            "[M8.7-prereq 13B/26GiB-RAM/8GiB-VRAM] \
             baseline: vram={} ({:.2} GiB)  ram={}  disk={}",
            p_baseline.vram_count(),
            baseline_vram_gib,
            p_baseline.ram_count(),
            p_baseline.disk_count()
        );
        eprintln!(
            "[M8.7-prereq 13B/26GiB-RAM/8GiB-VRAM] \
             with-staging: vram={} ({:.2} GiB)  ram={}  disk={}  \
             staging-reserved={} MiB",
            p.vram_count(),
            staged_vram_gib,
            p.ram_count(),
            p.disk_count(),
            DISK_PIPELINE_STAGING_BYTES / (1024 * 1024)
        );
    }

    /// 17. **No-Disk plan keeps full VRAM budget** — 7B on a 32 GiB
    /// box fits entirely in VRAM/RAM with the BF16 kernel; the
    /// staging reservation must NOT activate. The output of
    /// `plan(input)` must equal the no-staging dry-run byte for byte.
    #[test]
    fn m8_7_prereq_no_disk_plan_keeps_m8_budget() {
        let tensors = synth_7b_model();
        let model_total = sum_model_bytes(&tensors);

        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 8 * 1024 * 1024 * 1024,
            free_ram_bytes: 26 * 1024 * 1024 * 1024,
            model_total_bytes: model_total,
            total_ram_bytes: 32 * 1024 * 1024 * 1024,
            kernel_dtype: DType::BF16,
        };

        let baseline_budget = input
            .free_vram_bytes
            .saturating_sub(VRAM_HEADROOM_BYTES);
        let p_baseline = plan_inner(&input, baseline_budget);

        // Sanity: no Disk overflow on this configuration.
        assert_eq!(
            p_baseline.disk_bytes_assigned, 0,
            "test precondition: 7B/32GiB BF16 must fit fully in VRAM/RAM"
        );

        // Public planner result must match the no-staging dry run.
        let p = plan(&input);
        assert_eq!(p.vram_count(), p_baseline.vram_count());
        assert_eq!(p.ram_count(), p_baseline.ram_count());
        assert_eq!(p.disk_count(), 0);
        assert_eq!(p.vram_bytes_assigned, p_baseline.vram_bytes_assigned);
        assert_eq!(p.ram_bytes_assigned, p_baseline.ram_bytes_assigned);
        assert_eq!(p.disk_bytes_assigned, 0);
    }
}
