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

    /// VRAM cost if this tensor is uploaded to the F32 device
    /// buffer. The size of the device-side allocation that the
    /// `bf16_to_f32_resident_in_vram` / direct-upload path
    /// materialises. Always F32.
    fn vram_cost_bytes(&self) -> u64 {
        self.numel() * 4
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
#[derive(Clone, Debug)]
pub struct TierPlanInput {
    pub tensors: Vec<TensorMeta>,
    pub free_vram_bytes: u64,
    pub free_ram_bytes: u64,
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
const RAM_HEADROOM_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// **GPU-eligible** classifier. Returns `true` if and only if the
/// tensor is a Llama-family attention/FFN projection weight that
/// the M6 step 4d mixed-residency dispatch path can consume from
/// VRAM. See module docs for the full rationale.
fn is_gpu_eligible(meta: &TensorMeta) -> bool {
    meta.shape.len() >= 2 && meta.name.ends_with("_proj.weight")
}

/// Pure function — no I/O, no logging, no allocation beyond the
/// output `TierPlan`. Suitable for direct unit testing.
pub fn plan(input: &TierPlanInput) -> TierPlan {
    let mut vram_remaining = input.free_vram_bytes.saturating_sub(VRAM_HEADROOM_BYTES);
    let mut ram_remaining = input.free_ram_bytes.saturating_sub(RAM_HEADROOM_BYTES);

    let mut out = TierPlan::default();
    out.assignments.reserve(input.tensors.len());
    out.by_name.reserve(input.tensors.len());

    for meta in &input.tensors {
        let vram_cost = meta.vram_cost_bytes();
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
        // 16 GiB free VRAM, 32 GiB free RAM — enough for one layer's
        // ~1.5 GiB F32 projection footprint plus headrooms.
        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 16 * 1024 * 1024 * 1024,
            free_ram_bytes: 32 * 1024 * 1024 * 1024,
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

        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 25 * 1024 * 1024 * 1024 / 10, // 2.5 GiB
            free_ram_bytes: 32 * 1024 * 1024 * 1024,
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
        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 0,
            free_ram_bytes: (85 * 1024 * 1024 * 1024) / 10, // 8.5 GiB
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
        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 100 * 1024 * 1024 * 1024,
            free_ram_bytes: 100 * 1024 * 1024 * 1024,
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
        let input = TierPlanInput {
            tensors,
            free_vram_bytes: 8 * 1024 * 1024 * 1024,
            free_ram_bytes: 32 * 1024 * 1024 * 1024,
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
}
