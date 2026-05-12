//! Internal model adapter layer.
//!
//! This is deliberately not a public SDK yet. The first contract is small and
//! internal: adapters describe model-family differences at load/build time,
//! while Atenia Core keeps owning execution, memory planning, kernels, and
//! numeric policy.

use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::KvCacheBuildSpec;
use crate::amg::weight_store::WeightStore;
use crate::nn::llama::builder::{LlamaRuntime, build_llama};
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::LlamaConfig;
use crate::nn::llama::gguf_weight_loading::{
    gemma2_gguf_weight_mapper, llama_gguf_weight_mapper, phi3_gguf_weight_mapper,
};
use crate::nn::llama::weight_loading::llama_weight_mapper;
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::WeightMapper;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFormat {
    HfSafetensors,
    Gguf,
}

#[derive(Clone, Copy, Debug)]
pub struct ModelMetadata<'a> {
    pub architecture: &'a str,
    pub model_type: Option<&'a str>,
    pub format: ModelFormat,
}

#[derive(Debug)]
pub struct ScratchGraphBuild {
    pub logits_id: usize,
    pub param_ids: Vec<usize>,
    pub param_names: Vec<String>,
}

pub trait ModelAdapter: Sync {
    fn id(&self) -> &'static str;
    fn supports(&self, metadata: &ModelMetadata<'_>) -> bool;
    fn log_selection(&self) {}

    fn build_scratch_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
    ) -> ScratchGraphBuild;
}

pub trait HfWeightMapper {
    fn map_hf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError>;
}

pub trait GgufWeightMapper {
    fn map_gguf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError>;
}

pub trait StoreBackedGraphBuilder {
    fn build_store_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
        store: &WeightStore,
        kv_cache: Option<&KvCacheBuildSpec>,
    ) -> Result<LlamaHandlesShared, BuildError>;
}

pub trait AteniaModelAdapter:
    ModelAdapter + HfWeightMapper + GgufWeightMapper + StoreBackedGraphBuilder
{
}

impl<T> AteniaModelAdapter for T where
    T: ModelAdapter + HfWeightMapper + GgufWeightMapper + StoreBackedGraphBuilder
{
}

struct LlamaFamilyAdapter;
struct Qwen2Adapter;
struct MistralAdapter;
struct Phi3Adapter;
struct Gemma2Adapter;

fn build_llama_scratch(
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

fn build_llama_store_graph(
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

impl ModelAdapter for LlamaFamilyAdapter {
    fn id(&self) -> &'static str {
        "llama"
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

impl ModelAdapter for Qwen2Adapter {
    fn id(&self) -> &'static str {
        "qwen2"
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

impl ModelAdapter for MistralAdapter {
    fn id(&self) -> &'static str {
        "mistral"
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

impl ModelAdapter for Phi3Adapter {
    fn id(&self) -> &'static str {
        "phi3"
    }

    fn supports(&self, metadata: &ModelMetadata<'_>) -> bool {
        metadata.architecture == "Phi3ForCausalLM" || metadata.model_type == Some("phi3")
    }

    fn log_selection(&self) {
        eprintln!(
            "[ATENIA] Architecture: Phi3ForCausalLM - routing to phi3 adapter \
             (LongRope + fused QKV / gate_up split via SliceLastDim)."
        );
    }

    fn build_scratch_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
    ) -> ScratchGraphBuild {
        let h = crate::nn::llama::phi3::build_phi3(gb, config, runtime, token_input_id);
        ScratchGraphBuild {
            logits_id: h.logits_id,
            param_ids: h.param_ids,
            param_names: h.param_names,
        }
    }
}

impl HfWeightMapper for Phi3Adapter {
    fn map_hf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        crate::nn::llama::phi3::phi3_weight_mapper(config, param_names, param_ids)
    }
}

impl GgufWeightMapper for Phi3Adapter {
    fn map_gguf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        phi3_gguf_weight_mapper(config, param_names, param_ids)
    }
}

impl StoreBackedGraphBuilder for Phi3Adapter {
    fn build_store_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
        store: &WeightStore,
        kv_cache: Option<&KvCacheBuildSpec>,
    ) -> Result<LlamaHandlesShared, BuildError> {
        crate::nn::llama::phi3::build_phi3_with_store(
            gb,
            config,
            runtime,
            token_input_id,
            store,
            kv_cache,
        )
    }
}

impl ModelAdapter for Gemma2Adapter {
    fn id(&self) -> &'static str {
        "gemma2"
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

static LLAMA_FAMILY_ADAPTER: LlamaFamilyAdapter = LlamaFamilyAdapter;
static QWEN2_ADAPTER: Qwen2Adapter = Qwen2Adapter;
static MISTRAL_ADAPTER: MistralAdapter = MistralAdapter;
static PHI3_ADAPTER: Phi3Adapter = Phi3Adapter;
static GEMMA2_ADAPTER: Gemma2Adapter = Gemma2Adapter;

static ADAPTERS: [&'static dyn AteniaModelAdapter; 5] = [
    &PHI3_ADAPTER,
    &GEMMA2_ADAPTER,
    &QWEN2_ADAPTER,
    &MISTRAL_ADAPTER,
    &LLAMA_FAMILY_ADAPTER,
];

pub fn resolve_adapter(metadata: &ModelMetadata<'_>) -> Option<&'static dyn AteniaModelAdapter> {
    ADAPTERS
        .iter()
        .copied()
        .find(|adapter| adapter.supports(metadata))
}

pub fn resolve_adapter_for_config(config: &LlamaConfig) -> &'static dyn AteniaModelAdapter {
    let architecture = match config.model_type.as_deref() {
        Some("phi3") => "Phi3ForCausalLM",
        Some("gemma2") => "Gemma2ForCausalLM",
        Some("qwen2") => "Qwen2ForCausalLM",
        Some("mistral") => "MistralForCausalLM",
        _ => "LlamaForCausalLM",
    };
    let metadata = ModelMetadata {
        architecture,
        model_type: config.model_type.as_deref(),
        format: ModelFormat::HfSafetensors,
    };
    resolve_adapter(&metadata).unwrap_or(&LLAMA_FAMILY_ADAPTER)
}

pub fn supported_architectures_message() -> &'static str {
    "Atenia today supports LlamaForCausalLM, Qwen2ForCausalLM, \
     MistralForCausalLM, Phi3ForCausalLM, and Gemma2ForCausalLM \
     through the internal model adapter layer."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_resolves_core_architectures() {
        let cases = [
            ("LlamaForCausalLM", None, "llama"),
            ("Qwen2ForCausalLM", Some("qwen2"), "qwen2"),
            ("MistralForCausalLM", Some("mistral"), "mistral"),
            ("Phi3ForCausalLM", Some("phi3"), "phi3"),
            ("Gemma2ForCausalLM", Some("gemma2"), "gemma2"),
        ];
        for (architecture, model_type, expected_id) in cases {
            let metadata = ModelMetadata {
                architecture,
                model_type,
                format: ModelFormat::HfSafetensors,
            };
            let adapter = resolve_adapter(&metadata).expect("adapter must resolve");
            assert_eq!(adapter.id(), expected_id);
        }
    }

    #[test]
    fn registry_rejects_unknown_architecture() {
        let metadata = ModelMetadata {
            architecture: "UnknownForCausalLM",
            model_type: None,
            format: ModelFormat::HfSafetensors,
        };
        assert!(resolve_adapter(&metadata).is_none());
    }
}
