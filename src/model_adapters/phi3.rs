use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::KvCacheBuildSpec;
use crate::amg::weight_store::WeightStore;
use crate::nn::llama::builder::LlamaRuntime;
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::LlamaConfig;
use crate::nn::llama::gguf_weight_loading::phi3_gguf_weight_mapper;
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::WeightMapper;

use super::{
    AdapterCapabilities, GgufWeightMapper, HfWeightMapper, ModelAdapter, ModelFamily,
    ModelMetadata, ResidencyHints, ScratchGraphBuild, StoreBackedGraphBuilder,
};

pub(super) struct Phi3Adapter;

impl ModelAdapter for Phi3Adapter {
    fn id(&self) -> &'static str {
        "phi3"
    }

    fn family(&self) -> ModelFamily {
        ModelFamily::Phi3
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            fused_qkv_weight_mapping: true,
            fused_gate_up_weight_mapping: true,
            ..AdapterCapabilities::llama_like()
        }
    }

    fn supported_architectures(&self) -> &'static [&'static str] {
        &["Phi3ForCausalLM"]
    }

    fn supported_model_types(&self) -> &'static [&'static str] {
        &["phi3"]
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

impl ResidencyHints for Phi3Adapter {}
