use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::KvCacheBuildSpec;
use crate::amg::weight_store::WeightStore;
use crate::nn::llama::builder::LlamaRuntime;
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::LlamaConfig;
use crate::nn::llama::gguf_weight_loading::gemma2_gguf_weight_mapper;
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::WeightMapper;

use super::{
    AdapterCapabilities, ConfigPolicy, GgufWeightMapper, HfWeightMapper, ModelAdapter,
    ModelFamily, ModelMetadata, ResidencyHints, ScratchGraphBuild, StoreBackedGraphBuilder,
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
}
