//! **MOE-FULL-10** — controlled productive Mixtral runtime (opt-in).
//!
//! The first **productive** MoE path in Atenia: load a real tiny Mixtral
//! checkpoint and generate to EOS, reusing the certified MoE components
//! (family recognition, adapter validation, GQA tiling, residency + expert
//! cache, the prefill/KV-cache/decode generation loop). It is gated behind an
//! explicit opt-in (`ATENIA_EXPERIMENTAL_MOE=1`):
//!
//! ```text
//!   ATENIA_EXPERIMENTAL_MOE=1  +  Mixtral checkpoint
//!        │
//!        ▼
//!   classify_family → MixtralAdapter::recognize → validate config
//!        │  per layer: RealMoeLayer::assemble + gqa::to_mha_kv
//!        ▼
//!   ResidentExpertLayer (RAM) + ExpertCache   (wired & self-validated)
//!        │
//!        ▼
//!   generate_greedy_tiny_eos  →  tokens … EOS
//! ```
//!
//! ## What this is / is NOT
//!
//! * It **is** a controlled, explicit, opt-in productive entry for **Mixtral
//!   only**. Without the opt-in it refuses (controlled fail-loud), exactly like
//!   before. The dense loader's fail-loud guard is **unchanged** — it never
//!   loads MoE; this is a *separate* runtime the opt-in selects.
//! * It does **not** declare general MoE support, enable Qwen-MoE / DeepSeek-MoE,
//!   touch the CLI, use a VRAM tier, quantise, batch, or optimise. Correctness
//!   first. The generation MoE block runs through the certified RAM path
//!   (bit-identical to the residency RAM tier, proven MOE-FULL-8/9); the
//!   residency+cache layers are constructed and self-validated here as the
//!   wired storage backend.

use std::path::Path;

use serde_json::Value;

use super::full_forward::{TinyDecoderWeights, TinyMixtralConfig, TinyMixtralWeights};
use super::generate::generate_greedy_tiny_eos;
use super::gqa::to_mha_kv;
use super::layer::{MoeLayerConfig, RealMoeLayer};
use super::mixtral_adapter::MixtralAdapter;
use super::residency::{ExpertCache, ExpertTier, ResidentExpertLayer};
use super::family::{classify_family, experimental_moe_enabled, validate_family_config, MoeFamily};
use crate::nn::llama::moe_config::MoeConfig;
use crate::v17::loader::safetensors_reader::SafetensorsReader;

/// Errors from the controlled Mixtral runtime.
#[derive(Debug)]
pub enum MixtralRuntimeError {
    /// The opt-in flag is not set; the experimental path refuses (fail-loud).
    OptInDisabled,
    /// The checkpoint is not a recognised Mixtral checkpoint.
    NotMixtral,
    /// `config.json` could not be read / parsed, or a required field is missing.
    Config(String),
    /// Safetensors read / tensor assembly error.
    Load(String),
    /// The declared config disagreed with the tensors.
    ConfigInconsistent(Vec<String>),
    /// The residency+cache wiring self-check failed (should never happen —
    /// residency is certified bit-identical in MOE-FULL-8/9).
    ResidencyMismatch { layer: usize, max_abs_diff: f32 },
}

impl std::fmt::Display for MixtralRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MixtralRuntimeError::OptInDisabled => write!(
                f,
                "moe-runtime: experimental MoE is opt-in; set ATENIA_EXPERIMENTAL_MOE=1 to enable \
                 the controlled Mixtral runtime (the dense loader still refuses MoE)"
            ),
            MixtralRuntimeError::NotMixtral => {
                write!(f, "moe-runtime: checkpoint is not a recognised Mixtral MoE checkpoint")
            }
            MixtralRuntimeError::Config(m) => write!(f, "moe-runtime: config error: {m}"),
            MixtralRuntimeError::Load(m) => write!(f, "moe-runtime: load error: {m}"),
            MixtralRuntimeError::ConfigInconsistent(notes) => {
                write!(f, "moe-runtime: config inconsistent with tensors: {}", notes.join("; "))
            }
            MixtralRuntimeError::ResidencyMismatch { layer, max_abs_diff } => write!(
                f,
                "moe-runtime: residency self-check failed at layer {layer} (max_abs_diff={max_abs_diff:.3e})"
            ),
        }
    }
}

impl std::error::Error for MixtralRuntimeError {}

fn cfg_u(v: &Value, key: &str) -> Result<usize, MixtralRuntimeError> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .ok_or_else(|| MixtralRuntimeError::Config(format!("missing/invalid '{key}'")))
}

/// Parse `eos_token_id` from a HF config: a single int or an array of ints.
fn parse_eos(v: &Value) -> Vec<u32> {
    match v.get("eos_token_id") {
        Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect(),
        _ => Vec::new(),
    }
}

/// The loaded, ready-to-generate controlled Mixtral runtime.
#[derive(Debug)]
pub struct MixtralRuntime {
    weights: TinyMixtralWeights,
    eos_token_ids: Vec<u32>,
    /// Per-layer residency-backed expert storage (RAM tier) — the wired,
    /// self-validated storage backend (bit-identical to the generation block).
    residency: Vec<ResidentExpertLayer>,
    /// Per-layer expert LRU cache (MOE-FULL-9), available to the runtime.
    caches: Vec<ExpertCache>,
}

impl MixtralRuntime {
    /// Load a controlled Mixtral runtime from a HF `config.json` and a
    /// safetensors weights file. **Refuses unless `ATENIA_EXPERIMENTAL_MOE=1`.**
    ///
    /// Reuses the certified pipeline: family recognition → adapter validation →
    /// config cross-check → per-layer `RealMoeLayer::assemble` + GQA K/V tiling
    /// → residency + expert cache (self-validated). No dense-loader change.
    pub fn load_from_files(
        config_path: &Path,
        weights_path: &Path,
    ) -> Result<Self, MixtralRuntimeError> {
        // 1. Controlled opt-in gate. Without it, fail loud exactly as before.
        if !experimental_moe_enabled() {
            return Err(MixtralRuntimeError::OptInDisabled);
        }

        // 2. Parse config.json.
        let text = std::fs::read_to_string(config_path)
            .map_err(|e| MixtralRuntimeError::Config(format!("read {config_path:?}: {e}")))?;
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| MixtralRuntimeError::Config(format!("parse json: {e}")))?;
        let vocab = cfg_u(&v, "vocab_size")?;
        let hidden = cfg_u(&v, "hidden_size")?;
        let n_layers = cfg_u(&v, "num_hidden_layers")?;
        let n_heads = cfg_u(&v, "num_attention_heads")?;
        let n_kv_heads = cfg_u(&v, "num_key_value_heads")?;
        let head_dim = v
            .get("head_dim")
            .and_then(Value::as_u64)
            .map(|n| n as usize)
            .unwrap_or(hidden / n_heads.max(1));
        let d_ff = cfg_u(&v, "intermediate_size")?;
        let n_experts = cfg_u(&v, "num_local_experts")?;
        let topk = cfg_u(&v, "num_experts_per_tok")?;
        let rope_theta = v.get("rope_theta").and_then(Value::as_f64).unwrap_or(10000.0) as u32;
        let rms_eps = v.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-5) as f32;
        let eos_token_ids = parse_eos(&v);

        // 3. Open the checkpoint and recognise the family.
        let reader = SafetensorsReader::open(weights_path)
            .map_err(|e| MixtralRuntimeError::Load(format!("open {weights_path:?}: {e:?}")))?;
        let names: Vec<String> = reader.iter().map(|e| e.name.to_string()).collect();
        if classify_family(names.iter().map(|s| s.as_str())) != Some(MoeFamily::Mixtral) {
            return Err(MixtralRuntimeError::NotMixtral);
        }

        // 4. Adapter validation (load-only metadata) + config cross-check.
        let moe_cfg = MoeConfig::from_json_str(&text)
            .map_err(|e| MixtralRuntimeError::Config(format!("moe config: {e}")))?;
        let tensors: Vec<(&str, Vec<usize>)> =
            reader.iter().map(|e| (e.name, e.shape.to_vec())).collect();
        MixtralAdapter::recognize(tensors.iter().map(|(n, s)| (*n, s.clone())), &moe_cfg)
            .map_err(|e| MixtralRuntimeError::Load(format!("mixtral adapter: {e}")))?;
        let validation = validate_family_config(names.iter().map(|s| s.as_str()), &moe_cfg);
        if !validation.consistent {
            return Err(MixtralRuntimeError::ConfigInconsistent(validation.notes));
        }

        // 5. Assemble per-layer weights (certified) + GQA K/V tiling.
        let map = super::data_plane::MoeWeightMap::from_tensors(
            reader.iter().map(|e| (e.name, e.shape.to_vec())),
        );
        let resolve = |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok());
        let get = |name: &str| -> Result<Vec<f32>, MixtralRuntimeError> {
            reader
                .get(name)
                .and_then(|e| e.to_vec_f32().ok())
                .ok_or_else(|| MixtralRuntimeError::Load(format!("missing tensor {name}")))
        };

        let mut layers: Vec<TinyDecoderWeights> = Vec::with_capacity(n_layers);
        let mut residency: Vec<ResidentExpertLayer> = Vec::with_capacity(n_layers);
        let mut caches: Vec<ExpertCache> = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            let p = format!("model.layers.{l}");
            let layer_cfg = MoeLayerConfig::new(n_experts, topk, false, hidden, d_ff)
                .map_err(|e| MixtralRuntimeError::Load(format!("layer cfg: {e}")))?;
            let moe = RealMoeLayer::assemble(&map, l, layer_cfg, &resolve)
                .map_err(|e| MixtralRuntimeError::Load(format!("assemble layer {l}: {e}")))?;

            // Residency + cache wiring (RAM tier), self-validated below.
            let resident = ResidentExpertLayer::from_real_layer(&moe, ExpertTier::Ram)
                .map_err(|e| MixtralRuntimeError::Load(format!("residency layer {l}: {e}")))?;
            Self::self_validate_residency(l, &moe, &resident)?;
            residency.push(resident);
            caches.push(ExpertCache::new(n_experts));

            // GQA: tile K/V to MHA shape so the certified MHA graph is reused.
            let w_k = to_mha_kv(&get(&format!("{p}.self_attn.k_proj.weight"))?, n_kv_heads, n_heads, head_dim, hidden)
                .map_err(|e| MixtralRuntimeError::Load(format!("k tile layer {l}: {e}")))?;
            let w_v = to_mha_kv(&get(&format!("{p}.self_attn.v_proj.weight"))?, n_kv_heads, n_heads, head_dim, hidden)
                .map_err(|e| MixtralRuntimeError::Load(format!("v tile layer {l}: {e}")))?;

            layers.push(TinyDecoderWeights {
                input_ln: get(&format!("{p}.input_layernorm.weight"))?,
                w_q: get(&format!("{p}.self_attn.q_proj.weight"))?,
                w_k,
                w_v,
                w_o: get(&format!("{p}.self_attn.o_proj.weight"))?,
                post_ln: get(&format!("{p}.post_attention_layernorm.weight"))?,
                moe,
            });
        }

        let weights = TinyMixtralWeights {
            config: TinyMixtralConfig {
                vocab_size: vocab,
                hidden_size: hidden,
                num_hidden_layers: n_layers,
                num_attention_heads: n_heads,
                head_dim,
                rope_theta,
            },
            embed_tokens: get("model.embed_tokens.weight")?,
            layers,
            final_norm: get("model.norm.weight")?,
            lm_head: get("lm_head.weight")?,
            rms_eps,
        };

        Ok(Self { weights, eos_token_ids, residency, caches })
    }

    /// Self-check that the residency+cache path reproduces the certified MoE
    /// block bit-exactly (MOE-FULL-8/9 invariant), proving the wiring is real.
    fn self_validate_residency(
        layer: usize,
        moe: &RealMoeLayer,
        resident: &ResidentExpertLayer,
    ) -> Result<(), MixtralRuntimeError> {
        let dm = moe.config.d_model;
        // Deterministic probe vector.
        let mut state = (layer as u64).wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let probe: Vec<f32> = (0..dm)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 11) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        let mut cache = ExpertCache::new(moe.num_experts());
        let (got, _) = resident
            .forward_cached(&mut cache, &probe)
            .map_err(|e| MixtralRuntimeError::Load(format!("residency probe: {e}")))?;
        let want = moe
            .forward_auto(&probe)
            .map_err(|e| MixtralRuntimeError::Load(format!("block probe: {e}")))?;
        let max_abs = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        if max_abs > 1e-5 {
            return Err(MixtralRuntimeError::ResidencyMismatch { layer, max_abs_diff: max_abs });
        }
        Ok(())
    }

    /// The EOS token ids parsed from the checkpoint config.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.weights.config.num_hidden_layers
    }

    /// Residency-backed expert storage (RAM tier) per layer — the wired,
    /// self-validated backend. Exposed for inspection/telemetry.
    pub fn residency(&self) -> &[ResidentExpertLayer] {
        &self.residency
    }

    /// Per-layer expert caches (MOE-FULL-9).
    pub fn caches(&self) -> &[ExpertCache] {
        &self.caches
    }

    /// **Generate** greedily from `prompt`, stopping at EOS or after
    /// `max_new_tokens`. Reuses the certified prefill + KV-cache + decode loop.
    /// Returns the generated token ids (the emitted EOS token, if any, is
    /// included). Deterministic.
    pub fn generate(&self, prompt: &[u32], max_new_tokens: usize) -> Vec<u32> {
        generate_greedy_tiny_eos(&self.weights, prompt, max_new_tokens, &self.eos_token_ids).tokens
    }
}

// ============================================================================
// Tests (the real-checkpoint end-to-end run is in
// tests/moe_mixtral_runtime_test.rs; here we cover the opt-in gate).
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Without the opt-in, loading must refuse (controlled fail-loud). This is
    /// safe regardless of env because we assert the *disabled* branch only when
    /// the flag is unset.
    #[test]
    fn opt_in_disabled_refuses() {
        if experimental_moe_enabled() {
            // The CI/dev shell has the flag set; skip (the enabled path is
            // covered by the integration test).
            return;
        }
        let err = MixtralRuntime::load_from_files(
            &PathBuf::from("/nonexistent/config.json"),
            &PathBuf::from("/nonexistent/model.safetensors"),
        )
        .unwrap_err();
        assert!(matches!(err, MixtralRuntimeError::OptInDisabled));
    }
}
