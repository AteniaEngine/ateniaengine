use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::KvCacheBuildSpec;
use crate::amg::weight_store::WeightStore;
use crate::nn::llama::builder::LlamaRuntime;
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::LlamaConfig;
use crate::nn::llama::gguf_weight_loading::llama_gguf_weight_mapper;
use crate::nn::llama::weight_loading::llama_weight_mapper;
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::WeightMapper;

use super::common::{build_llama_scratch, build_llama_store_graph, validate_llama_family_config};
use super::{
    AdapterCapabilities, ConfigPolicy, GgufWeightMapper, HfWeightMapper, ModelAdapter,
    ModelFamily, ModelMetadata, ResidencyHints, ResidencyPolicyHints, ScratchGraphBuild,
    StoreBackedGraphBuilder, llama_like_residency_hints,
};

pub(super) struct LlamaFamilyAdapter;
pub(super) struct Qwen2Adapter;
pub(super) struct MistralAdapter;

impl ModelAdapter for LlamaFamilyAdapter {
    fn id(&self) -> &'static str {
        "llama"
    }

    fn family(&self) -> ModelFamily {
        ModelFamily::Llama
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities::llama_like()
    }

    fn supported_architectures(&self) -> &'static [&'static str] {
        &["LlamaForCausalLM"]
    }

    fn supported_model_types(&self) -> &'static [&'static str] {
        &["llama"]
    }

    fn supports(&self, metadata: &ModelMetadata<'_>) -> bool {
        metadata.architecture == "LlamaForCausalLM"
    }

    fn build_scratch_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
    ) -> ScratchGraphBuild {
        build_llama_scratch(gb, config, runtime, token_input_id)
    }
}

impl HfWeightMapper for LlamaFamilyAdapter {
    fn map_hf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        llama_weight_mapper(config, param_names, param_ids)
    }
}

impl GgufWeightMapper for LlamaFamilyAdapter {
    fn map_gguf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        llama_gguf_weight_mapper(config, param_names, param_ids)
    }
}

impl StoreBackedGraphBuilder for LlamaFamilyAdapter {
    fn build_store_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
        store: &WeightStore,
        kv_cache: Option<&KvCacheBuildSpec>,
    ) -> Result<LlamaHandlesShared, BuildError> {
        build_llama_store_graph(gb, config, runtime, token_input_id, store, kv_cache)
    }
}

impl ResidencyHints for LlamaFamilyAdapter {
    fn residency_hints(&self, _config: &LlamaConfig) -> ResidencyPolicyHints {
        llama_like_residency_hints()
    }
}

impl ConfigPolicy for LlamaFamilyAdapter {
    fn validate_config(
        &self,
        config: &LlamaConfig,
    ) -> Result<(), crate::nn::llama::config::ConfigError> {
        validate_llama_family_config(config)
    }
}

impl ModelAdapter for Qwen2Adapter {
    fn id(&self) -> &'static str {
        "qwen2"
    }

    fn family(&self) -> ModelFamily {
        ModelFamily::Qwen2
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities::llama_like()
    }

    fn supported_architectures(&self) -> &'static [&'static str] {
        &["Qwen2ForCausalLM"]
    }

    fn supported_model_types(&self) -> &'static [&'static str] {
        &["qwen2"]
    }

    fn supports(&self, metadata: &ModelMetadata<'_>) -> bool {
        metadata.architecture == "Qwen2ForCausalLM" || metadata.model_type == Some("qwen2")
    }

    fn build_scratch_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
    ) -> ScratchGraphBuild {
        build_llama_scratch(gb, config, runtime, token_input_id)
    }
}

impl HfWeightMapper for Qwen2Adapter {
    fn map_hf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        llama_weight_mapper(config, param_names, param_ids)
    }
}

impl GgufWeightMapper for Qwen2Adapter {
    fn map_gguf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        llama_gguf_weight_mapper(config, param_names, param_ids)
    }
}

impl StoreBackedGraphBuilder for Qwen2Adapter {
    fn build_store_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
        store: &WeightStore,
        kv_cache: Option<&KvCacheBuildSpec>,
    ) -> Result<LlamaHandlesShared, BuildError> {
        build_llama_store_graph(gb, config, runtime, token_input_id, store, kv_cache)
    }
}

impl ResidencyHints for Qwen2Adapter {
    fn residency_hints(&self, _config: &LlamaConfig) -> ResidencyPolicyHints {
        llama_like_residency_hints()
    }
}

impl ConfigPolicy for Qwen2Adapter {
    /// Qwen2 omits `attention_bias` from its official config
    /// and hard-codes QKV biases on inside `Qwen2Attention`.
    /// Phase 12 surfaces that default here so
    /// `LlamaConfig::effective_attention_bias()` no longer
    /// needs to hard-code `matches!(model_type, Some("qwen2"))`.
    fn default_attention_bias(&self) -> Option<bool> {
        Some(true)
    }

    fn validate_config(
        &self,
        config: &LlamaConfig,
    ) -> Result<(), crate::nn::llama::config::ConfigError> {
        validate_llama_family_config(config)
    }
}

impl ModelAdapter for MistralAdapter {
    fn id(&self) -> &'static str {
        "mistral"
    }

    fn family(&self) -> ModelFamily {
        ModelFamily::Mistral
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities::llama_like()
    }

    fn supported_architectures(&self) -> &'static [&'static str] {
        &["MistralForCausalLM"]
    }

    fn supported_model_types(&self) -> &'static [&'static str] {
        &["mistral"]
    }

    fn supports(&self, metadata: &ModelMetadata<'_>) -> bool {
        metadata.architecture == "MistralForCausalLM" || metadata.model_type == Some("mistral")
    }

    fn build_scratch_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
    ) -> ScratchGraphBuild {
        build_llama_scratch(gb, config, runtime, token_input_id)
    }
}

impl HfWeightMapper for MistralAdapter {
    fn map_hf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        llama_weight_mapper(config, param_names, param_ids)
    }
}

impl GgufWeightMapper for MistralAdapter {
    fn map_gguf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        llama_gguf_weight_mapper(config, param_names, param_ids)
    }
}

impl StoreBackedGraphBuilder for MistralAdapter {
    fn build_store_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
        store: &WeightStore,
        kv_cache: Option<&KvCacheBuildSpec>,
    ) -> Result<LlamaHandlesShared, BuildError> {
        build_llama_store_graph(gb, config, runtime, token_input_id, store, kv_cache)
    }
}

impl ResidencyHints for MistralAdapter {
    fn residency_hints(&self, _config: &LlamaConfig) -> ResidencyPolicyHints {
        llama_like_residency_hints()
    }
}

impl ConfigPolicy for MistralAdapter {
    fn validate_config(
        &self,
        config: &LlamaConfig,
    ) -> Result<(), crate::nn::llama::config::ConfigError> {
        validate_llama_family_config(config)
    }
}
