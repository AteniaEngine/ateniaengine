use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::KvCacheBuildSpec;
use crate::amg::weight_store::WeightStore;
use crate::nn::llama::builder::LlamaRuntime;
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::LlamaConfig;
use crate::nn::llama::gguf_weight_loading::llama_gguf_weight_mapper;
use crate::nn::llama::qwen3::{build_qwen3, build_qwen3_with_store, qwen3_weight_mapper};
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::WeightMapper;

use super::{
    AdapterCapabilities, ConfigPolicy, GgufNameMapper, GgufWeightMapper, HfWeightMapper,
    ModelAdapter, ModelFamily, ModelMetadata, ResidencyHints, ResidencyPolicyHints,
    ScratchGraphBuild, StoreBackedGraphBuilder, llama_like_residency_hints,
};

/// **Phase Q (Qwen3 family support).**
///
/// Qwen3 is a Llama-topology family with two attention-block deltas vs
/// Llama / Qwen2 / Mistral:
///
/// 1. **Per-head QK-Norm** (`*.self_attn.q_norm.weight` /
///    `*.self_attn.k_norm.weight`, both shape `[head_dim]`): an extra
///    RMSNorm applied to the q / k projections AFTER reshape-to-heads
///    and BEFORE RoPE. γ is broadcast over the head dimension, shared
///    across heads. Topologically expressible with the existing AMG
///    `RmsNorm` op (no new node types).
/// 2. **No QKV biases** (`attention_bias=false`) — opposite of
///    Qwen2's family default.
///
/// MLP is standard SwiGLU (separate gate_proj / up_proj / down_proj —
/// not fused). No sliding window, no rope_scaling, no softcap, no
/// fused QKV / gate_up. GQA via explicit `head_dim` (1024/16=64 ≠
/// 128 in the 0.6B checkpoint, so head_dim must come from the config
/// — already handled by `LlamaConfig`'s explicit `head_dim` field).
///
/// **Q-1 scaffold:** this adapter resolves Qwen3 and routes through
/// the *Llama* graph builder / weight mapper temporarily. Q-2 will
/// introduce `QWEN3_SPEC` (`q_norm` / `k_norm` transforms); Q-3 will
/// introduce `build_qwen3` and `qwen3_weight_mapper` and update the
/// delegations below.
pub(super) struct Qwen3Adapter;

impl ModelAdapter for Qwen3Adapter {
    fn id(&self) -> &'static str {
        "qwen3"
    }

    fn family(&self) -> ModelFamily {
        ModelFamily::Qwen3
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities::llama_like()
    }

    fn supported_architectures(&self) -> &'static [&'static str] {
        &["Qwen3ForCausalLM"]
    }

    fn supported_model_types(&self) -> &'static [&'static str] {
        &["qwen3"]
    }

    fn supports(&self, metadata: &ModelMetadata<'_>) -> bool {
        metadata.architecture == "Qwen3ForCausalLM" || metadata.model_type == Some("qwen3")
    }

    fn log_selection(&self) {
        eprintln!(
            "[ATENIA] Architecture: Qwen3ForCausalLM - routing to qwen3 adapter \
             (per-head QK-Norm pre-RoPE, GQA, separate SwiGLU, no QKV biases)."
        );
    }

    fn build_scratch_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
    ) -> ScratchGraphBuild {
        let h = build_qwen3(gb, config, runtime, token_input_id);
        ScratchGraphBuild {
            logits_id: h.logits_id,
            param_ids: h.param_ids,
            param_names: h.param_names,
        }
    }
}

impl HfWeightMapper for Qwen3Adapter {
    fn map_hf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        qwen3_weight_mapper(config, param_names, param_ids)
    }
}

impl GgufWeightMapper for Qwen3Adapter {
    fn map_gguf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        // GGUF Qwen3 is out of scope for Phase Q (no checkpoint
        // validated yet). The trait impl reuses the Llama GGUF mapper
        // as a defensive default; not a tested path.
        llama_gguf_weight_mapper(config, param_names, param_ids)
    }
}

// Qwen3 GGUF, if/when supported, uses the same Llama-layout common
// names as Qwen2 (the official Qwen3 GGUF conversion follows
// llama.cpp's `LLM_ARCH_QWEN3` which mirrors the Llama tensor
// naming). No family-specific name extras.
impl GgufNameMapper for Qwen3Adapter {}

impl StoreBackedGraphBuilder for Qwen3Adapter {
    fn build_store_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
        store: &WeightStore,
        kv_cache: Option<&KvCacheBuildSpec>,
    ) -> Result<LlamaHandlesShared, BuildError> {
        build_qwen3_with_store(gb, config, runtime, token_input_id, store, kv_cache)
    }
}

impl ResidencyHints for Qwen3Adapter {
    fn residency_hints(&self, _config: &LlamaConfig) -> ResidencyPolicyHints {
        llama_like_residency_hints()
    }
}

impl ConfigPolicy for Qwen3Adapter {
    /// Qwen3 has **no** QKV biases (`attention_bias=false`) — opposite
    /// of Qwen2's family default. Defensive: the canonical Qwen3
    /// config.json sets `attention_bias` explicitly, so this only
    /// fires for malformed / partial configs.
    fn default_attention_bias(&self) -> Option<bool> {
        Some(false)
    }
}
