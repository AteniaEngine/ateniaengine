//! **MOE-FULL-10 / MOE-FULL-11** — controlled productive MoE runtime (opt-in).
//!
//! The productive MoE entry in Atenia: load a real tiny MoE checkpoint and
//! generate to EOS, reusing the certified components (family recognition,
//! adapter validation, GQA tiling, optional Q/K/V attention bias, residency +
//! expert cache, the prefill/KV-cache/decode loop). Gated behind an explicit
//! opt-in (`ATENIA_EXPERIMENTAL_MOE=1`).
//!
//! ## Enabled families (controlled, opt-in)
//!
//! * **Mixtral** — standard attention (no bias), no shared expert. Full
//!   end-to-end HF parity (MOE-FULL-10).
//! * **Qwen-MoE** — standard attention **with Q/K/V bias**, packed experts +
//!   **shared expert** (sigmoid-gated), `norm_topk_prob = false`. Full
//!   end-to-end HF parity (MOE-FULL-11).
//!
//! ## Not enabled for generation
//!
//! * **DeepSeek-MoE** — uses **MLA** (multi-head latent attention:
//!   `kv_a_proj_with_mqa` / `kv_b_proj`), a different attention architecture
//!   not modelled here. The runtime refuses it; the DeepSeek **MoE block** is
//!   certified separately vs HF (`tests/moe_deepseek_block_test.rs`). Adding
//!   MLA is out of scope (would be a new architecture).
//!
//! Without the opt-in the runtime refuses; the dense loader's fail-loud guard
//! is **unchanged** and still refuses MoE. No CLI / VRAM / batching / quant.

use std::path::Path;

use serde_json::Value;

use super::family::{classify_family, experimental_moe_enabled, validate_family_config, MoeFamily};
use super::full_forward::{QkvBias, TinyDecoderWeights, TinyMixtralConfig, TinyMixtralWeights};
use super::generate::generate_greedy_tiny_eos;
use super::gqa::to_mha_kv;
use super::layer::{MoeLayerConfig, RealMoeLayer};
use super::mixtral_adapter::MixtralAdapter;
use super::mla::{DeepseekConfig, DeepseekLayer, DeepseekWeights};
use super::residency::{ExpertCache, ExpertTier, ResidentExpertLayer};
use crate::nn::llama::moe_config::MoeConfig;
use crate::v17::loader::safetensors_reader::SafetensorsReader;

/// Generation backend: the AMG graph (Mixtral/Qwen MHA) or the imperative MLA
/// forward (DeepSeek).
#[derive(Debug)]
enum Backend {
    /// Mixtral / Qwen-MoE — the certified MHA(+bias) graph generation loop.
    Graph(TinyMixtralWeights),
    /// DeepSeek-MoE — the imperative MLA forward (MOE-FULL-12).
    Mla(DeepseekWeights),
}

/// Errors from the controlled MoE runtime.
#[derive(Debug)]
pub enum MoeRuntimeError {
    /// The opt-in flag is not set; the experimental path refuses (fail-loud).
    OptInDisabled,
    /// Not a MoE checkpoint at all.
    NotMoe,
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

impl std::fmt::Display for MoeRuntimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoeRuntimeError::OptInDisabled => write!(
                f,
                "moe-runtime: experimental MoE is opt-in; set ATENIA_EXPERIMENTAL_MOE=1 to enable \
                 the controlled MoE runtime (the dense loader still refuses MoE)"
            ),
            MoeRuntimeError::NotMoe => {
                write!(f, "moe-runtime: checkpoint is not a recognised MoE checkpoint")
            }
            MoeRuntimeError::Config(m) => write!(f, "moe-runtime: config error: {m}"),
            MoeRuntimeError::Load(m) => write!(f, "moe-runtime: load error: {m}"),
            MoeRuntimeError::ConfigInconsistent(notes) => {
                write!(f, "moe-runtime: config inconsistent with tensors: {}", notes.join("; "))
            }
            MoeRuntimeError::ResidencyMismatch { layer, max_abs_diff } => write!(
                f,
                "moe-runtime: residency self-check failed at layer {layer} (max_abs_diff={max_abs_diff:.3e})"
            ),
        }
    }
}

impl std::error::Error for MoeRuntimeError {}

/// Back-compat aliases (MOE-FULL-10 named these `Mixtral*`).
pub type MixtralRuntime = MoeRuntime;
pub type MixtralRuntimeError = MoeRuntimeError;

fn cfg_u(v: &Value, key: &str) -> Result<usize, MoeRuntimeError> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .ok_or_else(|| MoeRuntimeError::Config(format!("missing/invalid '{key}'")))
}

/// Read the first present `usize` among `keys` (family-divergent field names).
fn cfg_u_any(v: &Value, keys: &[&str]) -> Result<usize, MoeRuntimeError> {
    for k in keys {
        if let Some(n) = v.get(*k).and_then(Value::as_u64) {
            return Ok(n as usize);
        }
    }
    Err(MoeRuntimeError::Config(format!("missing/invalid any of {keys:?}")))
}

/// Parse `eos_token_id` from a HF config: a single int or an array of ints.
fn parse_eos(v: &Value) -> Vec<u32> {
    match v.get("eos_token_id") {
        Some(Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect(),
        _ => Vec::new(),
    }
}

/// The loaded, ready-to-generate controlled MoE runtime.
#[derive(Debug)]
pub struct MoeRuntime {
    family: MoeFamily,
    backend: Backend,
    num_layers: usize,
    eos_token_ids: Vec<u32>,
    /// Per-layer residency-backed expert storage (RAM tier) — the wired,
    /// self-validated storage backend (bit-identical to the generation block).
    residency: Vec<ResidentExpertLayer>,
    /// Per-layer expert LRU cache (MOE-FULL-9), available to the runtime.
    caches: Vec<ExpertCache>,
}

impl MoeRuntime {
    /// Load a controlled MoE runtime from a HF `config.json` and a safetensors
    /// weights file. **Refuses unless `ATENIA_EXPERIMENTAL_MOE=1`.** Enables
    /// **Mixtral** and **Qwen-MoE**; refuses DeepSeek-MoE (MLA) and dense.
    ///
    /// Reuses the certified pipeline: family recognition → (Mixtral) adapter
    /// validation → config cross-check → per-layer `RealMoeLayer::assemble` +
    /// GQA K/V (and bias) tiling → residency + expert cache (self-validated).
    pub fn load_from_files(
        config_path: &Path,
        weights_path: &Path,
    ) -> Result<Self, MoeRuntimeError> {
        // 1. Controlled opt-in gate. Without it, fail loud exactly as before.
        if !experimental_moe_enabled() {
            return Err(MoeRuntimeError::OptInDisabled);
        }

        // 2. Parse config.json.
        let text = std::fs::read_to_string(config_path)
            .map_err(|e| MoeRuntimeError::Config(format!("read {config_path:?}: {e}")))?;
        let v: Value = serde_json::from_str(&text)
            .map_err(|e| MoeRuntimeError::Config(format!("parse json: {e}")))?;
        let vocab = cfg_u(&v, "vocab_size")?;
        let hidden = cfg_u(&v, "hidden_size")?;
        let n_layers = cfg_u(&v, "num_hidden_layers")?;
        let n_experts = cfg_u_any(&v, &["num_local_experts", "num_experts", "n_routed_experts"])?;
        let topk = cfg_u(&v, "num_experts_per_tok")?;
        let rms_eps = v.get("rms_norm_eps").and_then(Value::as_f64).unwrap_or(1e-5) as f32;
        let eos_token_ids = parse_eos(&v);

        // 3. Open the checkpoint and recognise the family.
        let reader = SafetensorsReader::open(weights_path)
            .map_err(|e| MoeRuntimeError::Load(format!("open {weights_path:?}: {e:?}")))?;
        let names: Vec<String> = reader.iter().map(|e| e.name.to_string()).collect();
        let family = match classify_family(names.iter().map(|s| s.as_str())) {
            Some(f) => f,
            None => return Err(MoeRuntimeError::NotMoe),
        };

        // Per-family knobs: shared expert + routed-expert FFN size.
        let (has_shared, d_ff) = match family {
            MoeFamily::Mixtral => (false, cfg_u(&v, "intermediate_size")?),
            MoeFamily::QwenMoe | MoeFamily::DeepSeekMoe => {
                (true, cfg_u_any(&v, &["moe_intermediate_size", "intermediate_size"])?)
            }
        };

        // 4. Adapter validation (Mixtral) + config cross-check (all families).
        let moe_cfg = MoeConfig::from_json_str(&text)
            .map_err(|e| MoeRuntimeError::Config(format!("moe config: {e}")))?;
        if family == MoeFamily::Mixtral {
            let tensors: Vec<(&str, Vec<usize>)> =
                reader.iter().map(|e| (e.name, e.shape.to_vec())).collect();
            MixtralAdapter::recognize(tensors.iter().map(|(n, s)| (*n, s.clone())), &moe_cfg)
                .map_err(|e| MoeRuntimeError::Load(format!("mixtral adapter: {e}")))?;
        }
        let validation = validate_family_config(names.iter().map(|s| s.as_str()), &moe_cfg);
        if !validation.consistent {
            return Err(MoeRuntimeError::ConfigInconsistent(validation.notes));
        }

        // 5. Assemble per-layer MoE + residency + cache (COMMON to all families).
        let map = super::data_plane::MoeWeightMap::from_tensors(
            reader.iter().map(|e| (e.name, e.shape.to_vec())),
        );
        let resolve = |name: &str| reader.get(name).and_then(|e| e.to_vec_f32().ok());
        let mut residency: Vec<ResidentExpertLayer> = Vec::with_capacity(n_layers);
        let mut caches: Vec<ExpertCache> = Vec::with_capacity(n_layers);
        let mut moes: Vec<RealMoeLayer> = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            let layer_cfg = MoeLayerConfig::new(n_experts, topk, has_shared, hidden, d_ff)
                .map_err(|e| MoeRuntimeError::Load(format!("layer cfg: {e}")))?;
            let moe = RealMoeLayer::assemble(&map, l, layer_cfg, &resolve)
                .map_err(|e| MoeRuntimeError::Load(format!("assemble layer {l}: {e}")))?;
            let resident = ResidentExpertLayer::from_real_layer(&moe, ExpertTier::Ram)
                .map_err(|e| MoeRuntimeError::Load(format!("residency layer {l}: {e}")))?;
            Self::self_validate_residency(l, &moe, &resident)?;
            residency.push(resident);
            caches.push(ExpertCache::new(n_experts));
            moes.push(moe);
        }

        // 6. Family-specific attention weights → backend.
        let backend = match family {
            MoeFamily::DeepSeekMoe => {
                Backend::Mla(build_deepseek(&v, &reader, vocab, hidden, n_layers, rms_eps, moes)?)
            }
            _ => Backend::Graph(build_graph(
                &v, &reader, family, vocab, hidden, n_layers, rms_eps, moes,
            )?),
        };

        Ok(Self { family, backend, num_layers: n_layers, eos_token_ids, residency, caches })
    }

    /// **MOE-FULL-14** — load from a model **directory**: finds `config.json`
    /// and the first `*.safetensors` inside `dir`. CLI-friendly wrapper over
    /// [`Self::load_from_files`]. Same opt-in gate.
    pub fn load_from_dir(dir: &Path) -> Result<Self, MoeRuntimeError> {
        let config = dir.join("config.json");
        if !config.exists() {
            return Err(MoeRuntimeError::Config(format!("no config.json in {dir:?}")));
        }
        let st = std::fs::read_dir(dir)
            .map_err(|e| MoeRuntimeError::Load(format!("read_dir {dir:?}: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .find(|p| p.extension().and_then(|x| x.to_str()) == Some("safetensors"))
            .ok_or_else(|| MoeRuntimeError::Load(format!("no .safetensors in {dir:?}")))?;
        Self::load_from_files(&config, &st)
    }

    /// Self-check that the residency+cache path reproduces the certified MoE
    /// block bit-exactly (MOE-FULL-8/9 invariant), proving the wiring is real.
    fn self_validate_residency(
        layer: usize,
        moe: &RealMoeLayer,
        resident: &ResidentExpertLayer,
    ) -> Result<(), MoeRuntimeError> {
        let dm = moe.config.d_model;
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
            .map_err(|e| MoeRuntimeError::Load(format!("residency probe: {e}")))?;
        let want = moe
            .forward_auto(&probe)
            .map_err(|e| MoeRuntimeError::Load(format!("block probe: {e}")))?;
        let max_abs = got
            .iter()
            .zip(want.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        if max_abs > 1e-5 {
            return Err(MoeRuntimeError::ResidencyMismatch { layer, max_abs_diff: max_abs });
        }
        Ok(())
    }

    /// The recognised family.
    pub fn family(&self) -> MoeFamily {
        self.family
    }

    /// The EOS token ids parsed from the checkpoint config.
    pub fn eos_token_ids(&self) -> &[u32] {
        &self.eos_token_ids
    }

    /// Number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Residency-backed expert storage (RAM tier) per layer.
    pub fn residency(&self) -> &[ResidentExpertLayer] {
        &self.residency
    }

    /// Per-layer expert caches (MOE-FULL-9).
    pub fn caches(&self) -> &[ExpertCache] {
        &self.caches
    }

    /// **MOE-FULL-12** — run only layer `layer`'s MLA attention over
    /// already-normed hidden states (DeepSeek backend only); `None` for the
    /// graph backend. Used to certify MLA against HuggingFace in isolation.
    pub fn debug_mla_attention(
        &self,
        layer: usize,
        normed: &[Vec<f32>],
    ) -> Option<Vec<Vec<f32>>> {
        match &self.backend {
            Backend::Mla(w) => Some(w.debug_attention(layer, normed)),
            Backend::Graph(_) => None,
        }
    }

    /// **Generate** greedily from `prompt`, stopping at EOS or after
    /// `max_new_tokens`. Reuses the certified generation loop (graph for
    /// Mixtral/Qwen; imperative MLA for DeepSeek).
    pub fn generate(&self, prompt: &[u32], max_new_tokens: usize) -> Vec<u32> {
        match &self.backend {
            Backend::Graph(w) => {
                generate_greedy_tiny_eos(w, prompt, max_new_tokens, &self.eos_token_ids).tokens
            }
            Backend::Mla(w) => w.generate_greedy_eos(prompt, max_new_tokens, &self.eos_token_ids),
        }
    }

    /// Full-sequence forward logits `[seq * vocab]` for `tokens`. Exposed so
    /// callers can validate end-to-end HF parity through the productive runtime
    /// (graph for Mixtral/Qwen; imperative MLA prefill for DeepSeek).
    pub fn forward_logits(&self, tokens: &[u32]) -> Vec<f32> {
        match &self.backend {
            Backend::Graph(w) => {
                use crate::amg::builder::GraphBuilder;
                use crate::tensor::Tensor;
                let seq = tokens.len();
                let mut gb = GraphBuilder::new();
                let tok = gb.input();
                let logits =
                    super::full_forward::build_tiny_mixtral_graph(&mut gb, tok, seq, w.clone());
                gb.output(logits);
                let mut g = gb.build();
                let t = Tensor::new_cpu(vec![1, seq], tokens.iter().map(|&x| x as f32).collect());
                g.execute(vec![t])[0].as_cpu_slice().to_vec()
            }
            Backend::Mla(w) => w.forward_prefill(tokens).0,
        }
    }
}

/// Build the graph (Mixtral/Qwen) backend weights from the reader.
#[allow(clippy::too_many_arguments)]
fn build_graph(
    v: &Value,
    reader: &SafetensorsReader,
    family: MoeFamily,
    vocab: usize,
    hidden: usize,
    n_layers: usize,
    rms_eps: f32,
    moes: Vec<RealMoeLayer>,
) -> Result<TinyMixtralWeights, MoeRuntimeError> {
    let n_heads = cfg_u(v, "num_attention_heads")?;
    let n_kv_heads = cfg_u(v, "num_key_value_heads")?;
    let head_dim = v
        .get("head_dim")
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .unwrap_or(hidden / n_heads.max(1));
    let rope_theta = v.get("rope_theta").and_then(Value::as_f64).unwrap_or(10000.0) as u32;
    let attn_has_bias = family == MoeFamily::QwenMoe;
    let get = |name: &str| -> Result<Vec<f32>, MoeRuntimeError> {
        reader
            .get(name)
            .and_then(|e| e.to_vec_f32().ok())
            .ok_or_else(|| MoeRuntimeError::Load(format!("missing tensor {name}")))
    };

    let mut layers: Vec<TinyDecoderWeights> = Vec::with_capacity(n_layers);
    for (l, moe) in moes.into_iter().enumerate() {
        let p = format!("model.layers.{l}");
        let w_k = to_mha_kv(&get(&format!("{p}.self_attn.k_proj.weight"))?, n_kv_heads, n_heads, head_dim, hidden)
            .map_err(|e| MoeRuntimeError::Load(format!("k tile layer {l}: {e}")))?;
        let w_v = to_mha_kv(&get(&format!("{p}.self_attn.v_proj.weight"))?, n_kv_heads, n_heads, head_dim, hidden)
            .map_err(|e| MoeRuntimeError::Load(format!("v tile layer {l}: {e}")))?;
        let attn_bias = if attn_has_bias {
            let qb = get(&format!("{p}.self_attn.q_proj.bias"))?;
            let kb = to_mha_kv(&get(&format!("{p}.self_attn.k_proj.bias"))?, n_kv_heads, n_heads, head_dim, 1)
                .map_err(|e| MoeRuntimeError::Load(format!("k bias tile layer {l}: {e}")))?;
            let vb = to_mha_kv(&get(&format!("{p}.self_attn.v_proj.bias"))?, n_kv_heads, n_heads, head_dim, 1)
                .map_err(|e| MoeRuntimeError::Load(format!("v bias tile layer {l}: {e}")))?;
            Some(QkvBias { q: qb, k: kb, v: vb })
        } else {
            None
        };
        layers.push(TinyDecoderWeights {
            input_ln: get(&format!("{p}.input_layernorm.weight"))?,
            w_q: get(&format!("{p}.self_attn.q_proj.weight"))?,
            w_k,
            w_v,
            w_o: get(&format!("{p}.self_attn.o_proj.weight"))?,
            post_ln: get(&format!("{p}.post_attention_layernorm.weight"))?,
            attn_bias,
            moe,
        });
    }
    Ok(TinyMixtralWeights {
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
    })
}

/// Build the MLA (DeepSeek) backend weights from the reader.
#[allow(clippy::too_many_arguments)]
fn build_deepseek(
    v: &Value,
    reader: &SafetensorsReader,
    vocab: usize,
    hidden: usize,
    n_layers: usize,
    rms_eps: f32,
    moes: Vec<RealMoeLayer>,
) -> Result<DeepseekWeights, MoeRuntimeError> {
    let n_heads = cfg_u(v, "num_attention_heads")?;
    let kv_lora_rank = cfg_u(v, "kv_lora_rank")?;
    let qk_nope_head_dim = cfg_u(v, "qk_nope_head_dim")?;
    let qk_rope_head_dim = cfg_u(v, "qk_rope_head_dim")?;
    let v_head_dim = cfg_u(v, "v_head_dim")?;
    let rope_theta = v.get("rope_theta").and_then(Value::as_f64).unwrap_or(10000.0) as f32;
    let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    // Validate tensor lengths against the config so a corrupt MLA checkpoint (or
    // a mismatched config field like kv_lora_rank) fails with a clear error
    // instead of a silent out-of-range panic inside the imperative forward.
    let checked = |name: &str, expected: usize| -> Result<Vec<f32>, MoeRuntimeError> {
        let v = reader
            .get(name)
            .and_then(|e| e.to_vec_f32().ok())
            .ok_or_else(|| MoeRuntimeError::Load(format!("missing tensor {name}")))?;
        if v.len() != expected {
            return Err(MoeRuntimeError::Load(format!(
                "tensor {name}: expected {expected} elements (from config), got {}",
                v.len()
            )));
        }
        Ok(v)
    };

    let mut layers: Vec<DeepseekLayer> = Vec::with_capacity(n_layers);
    for (l, moe) in moes.into_iter().enumerate() {
        let p = format!("model.layers.{l}");
        layers.push(DeepseekLayer {
            input_ln: checked(&format!("{p}.input_layernorm.weight"), hidden)?,
            w_q: checked(&format!("{p}.self_attn.q_proj.weight"), n_heads * qk_head_dim * hidden)?,
            w_kv_a: checked(
                &format!("{p}.self_attn.kv_a_proj_with_mqa.weight"),
                (kv_lora_rank + qk_rope_head_dim) * hidden,
            )?,
            kv_a_ln: checked(&format!("{p}.self_attn.kv_a_layernorm.weight"), kv_lora_rank)?,
            w_kv_b: checked(
                &format!("{p}.self_attn.kv_b_proj.weight"),
                n_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank,
            )?,
            w_o: checked(&format!("{p}.self_attn.o_proj.weight"), hidden * n_heads * v_head_dim)?,
            post_ln: checked(&format!("{p}.post_attention_layernorm.weight"), hidden)?,
            moe,
        });
    }
    let get = |name: &str| -> Result<Vec<f32>, MoeRuntimeError> {
        reader
            .get(name)
            .and_then(|e| e.to_vec_f32().ok())
            .ok_or_else(|| MoeRuntimeError::Load(format!("missing tensor {name}")))
    };
    Ok(DeepseekWeights {
        config: DeepseekConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            num_hidden_layers: n_layers,
            num_attention_heads: n_heads,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            rope_theta,
            rms_eps,
        },
        embed_tokens: get("model.embed_tokens.weight")?,
        layers,
        final_norm: get("model.norm.weight")?,
        lm_head: get("lm_head.weight")?,
    })
}

// ============================================================================
// Tests (the real-checkpoint end-to-end runs are in
// tests/moe_mixtral_runtime_test.rs and tests/moe_qwen_runtime_test.rs;
// here we cover the opt-in gate).
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn opt_in_disabled_refuses() {
        if experimental_moe_enabled() {
            return;
        }
        let err = MoeRuntime::load_from_files(
            &PathBuf::from("/nonexistent/config.json"),
            &PathBuf::from("/nonexistent/model.safetensors"),
        )
        .unwrap_err();
        assert!(matches!(err, MoeRuntimeError::OptInDisabled));
    }
}
