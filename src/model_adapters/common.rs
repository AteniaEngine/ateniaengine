use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::KvCacheBuildSpec;
use crate::amg::weight_store::WeightStore;
use crate::nn::llama::builder::{LlamaRuntime, build_llama};
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::{ConfigError, LlamaConfig, RopeScaling};

use super::ScratchGraphBuild;

/// **Phase 14** — shared `rope_scaling.type = "llama3"` parser for
/// the Llama-family adapters (Llama / Qwen2 / Mistral).
///
/// This is the Llama 3 piecewise inverse-frequency parser relocated
/// out of `config.rs::get_rope_scaling`, parallel to how Phi-3
/// LongRope moved to `Phi3Adapter::parse_rope_scaling` at Phase
/// 12.4. `config.rs` no longer knows the `"llama3"` schema; the
/// Llama-family adapter owns it.
///
/// Mirrors the [`crate::model_adapters::ConfigPolicy::parse_rope_scaling`]
/// contract:
/// - `Ok(None)` — not a `"llama3"` block (caller falls back to the
///   `config.rs` residual, which fails fast on a recognised type
///   declared under the wrong family).
/// - `Ok(Some(_))` — recognised and parsed.
/// - `Err(_)` — recognised as `"llama3"` but a required field is
///   missing or malformed; fail fast.
///
/// Honours both discriminator conventions: `"rope_type"`
/// (transformers >= 4.43) and the legacy `"type"` alias
/// (pre-4.43). The non-object-`rope_scaling` hard error stays in
/// `config.rs` (it guards before the adapter is consulted), so a
/// non-object block here returns `Ok(None)` — symmetric with
/// `Phi3Adapter::parse_rope_scaling`. Error message strings are
/// byte-identical to the pre-Phase-14 `config.rs` text.
pub(super) fn parse_llama3_rope_scaling(
    outer: &serde_json::Value,
) -> Result<Option<RopeScaling>, ConfigError> {
    let block = match outer.get("rope_scaling") {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(other) => other,
    };
    if !block.is_object() {
        return Ok(None);
    }
    let rope_type = block
        .get("rope_type")
        .or_else(|| block.get("type"))
        .and_then(|x| x.as_str());
    if rope_type != Some("llama3") {
        return Ok(None);
    }
    let factor = block
        .get("factor")
        .and_then(|x| x.as_f64())
        .ok_or_else(|| {
            ConfigError::Parse("rope_scaling.llama3 requires numeric `factor`".into())
        })? as f32;
    let low_freq_factor = block
        .get("low_freq_factor")
        .and_then(|x| x.as_f64())
        .ok_or_else(|| {
            ConfigError::Parse("rope_scaling.llama3 requires numeric `low_freq_factor`".into())
        })? as f32;
    let high_freq_factor = block
        .get("high_freq_factor")
        .and_then(|x| x.as_f64())
        .ok_or_else(|| {
            ConfigError::Parse("rope_scaling.llama3 requires numeric `high_freq_factor`".into())
        })? as f32;
    let original_max_position_embeddings = block
        .get("original_max_position_embeddings")
        .and_then(|x| x.as_u64())
        .ok_or_else(|| {
            ConfigError::Parse(
                "rope_scaling.llama3 requires integer `original_max_position_embeddings`".into(),
            )
        })? as u32;
    Ok(Some(RopeScaling::Llama3 {
        factor,
        low_freq_factor,
        high_freq_factor,
        original_max_position_embeddings,
    }))
}

/// **Phase 13** — shared family-config validation for the
/// Llama-family adapters (Llama / Qwen2 / Mistral).
///
/// This is the V2 check relocated out of
/// `LlamaConfig::validate()`. It encodes the *derived-head_dim*
/// assumption (`head_dim = hidden_size / num_attention_heads`)
/// that holds for the classic Llama family but **not** for
/// explicit-`head_dim` architectures (Phi-3 / Gemma 2), which is
/// why it lives in the adapter layer rather than in the
/// structural core validator.
///
/// The `ConfigError::Validation` message is byte-identical to
/// the pre-Phase-13 `LlamaConfig::validate()` text so existing
/// string-coupled tests stay green.
pub(super) fn validate_llama_family_config(config: &LlamaConfig) -> Result<(), ConfigError> {
    if config.hidden_size % config.num_attention_heads != 0 {
        return Err(ConfigError::Validation(format!(
            "hidden_size ({}) must be divisible by num_attention_heads ({})",
            config.hidden_size, config.num_attention_heads
        )));
    }
    Ok(())
}

pub(super) fn build_llama_scratch(
    gb: &mut GraphBuilder,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    token_input_id: usize,
) -> ScratchGraphBuild {
    let h = build_llama(gb, config, runtime, token_input_id);
    ScratchGraphBuild {
        logits_id: h.logits_id,
        param_ids: h.param_ids,
        param_names: h.param_names,
    }
}

pub(super) fn build_llama_store_graph(
    gb: &mut GraphBuilder,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    token_input_id: usize,
    store: &WeightStore,
    kv_cache: Option<&KvCacheBuildSpec>,
) -> Result<LlamaHandlesShared, BuildError> {
    crate::nn::llama::builder_shared::build_llama_with_store(
        gb,
        config,
        runtime,
        token_input_id,
        store,
        kv_cache,
    )
}
