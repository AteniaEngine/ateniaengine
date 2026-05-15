use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::KvCacheBuildSpec;
use crate::amg::weight_store::WeightStore;
use crate::nn::llama::builder::{LlamaRuntime, build_llama};
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::{ConfigError, LlamaConfig};

use super::ScratchGraphBuild;

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
