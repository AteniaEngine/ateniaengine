use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::KvCacheBuildSpec;
use crate::amg::weight_store::WeightStore;
use crate::nn::llama::builder::LlamaRuntime;
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::LlamaConfig;
use crate::nn::llama::gguf_weight_loading::gemma2_gguf_weight_mapper;
use crate::v17::loader::gguf_to_hf_naming::{gemma2_gguf_extra, gguf_to_hf_name_common};
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::WeightMapper;

use super::{
    AdapterCapabilities, ConfigPolicy, GgufNameMapper, GgufWeightMapper, HfWeightMapper,
    ModelAdapter, ModelFamily, ModelMetadata, ResidencyHints, ScratchGraphBuild,
    StoreBackedGraphBuilder,
};

pub(super) struct Gemma2Adapter;

impl ModelAdapter for Gemma2Adapter {
    fn id(&self) -> &'static str {
        "gemma2"
    }

    fn family(&self) -> ModelFamily {
        ModelFamily::Gemma2
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            gemma2_softcaps: true,
            ..AdapterCapabilities::llama_like()
        }
    }

    fn supported_architectures(&self) -> &'static [&'static str] {
        &["Gemma2ForCausalLM"]
    }

    fn supported_model_types(&self) -> &'static [&'static str] {
        &["gemma2"]
    }

    fn supports(&self, metadata: &ModelMetadata<'_>) -> bool {
        metadata.architecture == "Gemma2ForCausalLM" || metadata.model_type == Some("gemma2")
    }

    fn log_selection(&self) {
        eprintln!(
            "[ATENIA] Architecture: Gemma2ForCausalLM - routing to gemma2 adapter \
             (dual-norm, GeGLU, SoftCap@50/30, embedding scale; sliding-window \
             deferred - full causal attention for context < 4096)."
        );
    }

    fn build_scratch_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
    ) -> ScratchGraphBuild {
        let h = crate::nn::llama::gemma2::build_gemma2(gb, config, runtime, token_input_id);
        ScratchGraphBuild {
            logits_id: h.logits_id,
            param_ids: h.param_ids,
            param_names: h.param_names,
        }
    }
}

impl HfWeightMapper for Gemma2Adapter {
    fn map_hf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        crate::nn::llama::gemma2::gemma2_weight_mapper(config, param_names, param_ids)
    }
}

impl GgufWeightMapper for Gemma2Adapter {
    fn map_gguf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        gemma2_gguf_weight_mapper(config, param_names, param_ids)
    }
}

// **Phase 16.2 / G-1b2 (GAP-N2)** — Gemma 2 has four per-layer
// norms. The Gemma 2 extras are tried **first**: the common
// Llama-layout table also matches `ffn_norm.weight` (→ the 2-norm
// `post_attention_layernorm`), which would shadow Gemma 2's
// `ffn_norm` → `pre_feedforward_layernorm` override (and leave
// `pre_feedforward_layernorm` with no source). Extra-first mirrors
// `Phi3Adapter` (the same composition-order class as Phi-3 #5a).
// Names with no Gemma 2 override fall through to
// `gguf_to_hf_name_common` unchanged.
impl GgufNameMapper for Gemma2Adapter {
    fn gguf_to_hf_name(&self, gguf_name: &str) -> Option<String> {
        gemma2_gguf_extra(gguf_name).or_else(|| gguf_to_hf_name_common(gguf_name))
    }
}

impl StoreBackedGraphBuilder for Gemma2Adapter {
    fn build_store_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
        store: &WeightStore,
        kv_cache: Option<&KvCacheBuildSpec>,
    ) -> Result<LlamaHandlesShared, BuildError> {
        crate::nn::llama::gemma2::build_gemma2_with_store(
            gb,
            config,
            runtime,
            token_input_id,
            store,
            kv_cache,
        )
    }
}

impl ResidencyHints for Gemma2Adapter {}

impl ConfigPolicy for Gemma2Adapter {
    /// Gemma 2's official HuggingFace config omits
    /// `tie_word_embeddings` entirely; upstream `Gemma2Config`
    /// defaults it to `True`. Phase 12 surfaces that default
    /// via the adapter trait instead of a `match model_type`
    /// branch inside `LlamaConfig::from_json_str`. The same
    /// default applies to any future `gemma1` checkpoint that
    /// resolves through this adapter family.
    fn default_tie_word_embeddings(&self) -> Option<bool> {
        Some(true)
    }

    /// **Phase 15** — Gemma 2 family config defaults, relocated
    /// from the `if arch == "gemma2"` block in
    /// `llama_config_from_gguf`. Each setter only fires when the
    /// field is absent, so an explicit `config.json` value (HF)
    /// or GGUF metadata value always wins; behaviour for real
    /// Gemma 2 checkpoints (which ship the caps explicitly) is
    /// unchanged. `head_dim` is made explicit when absent so the
    /// graph/validator see the value the kernel uses
    /// (`effective_head_dim()` = `hidden_size / num_attention_heads`
    /// when the field is `None`).
    fn apply_config_defaults(&self, config: &mut LlamaConfig) {
        if config.head_dim.is_none() {
            config.head_dim = Some(config.effective_head_dim());
        }
        if config.attn_logit_softcapping.is_none() {
            config.attn_logit_softcapping = Some(50.0);
        }
        if config.final_logit_softcapping.is_none() {
            config.final_logit_softcapping = Some(30.0);
        }
        if config.query_pre_attn_scalar.is_none() {
            config.query_pre_attn_scalar = Some(config.effective_head_dim() as f32);
        }
    }
}
