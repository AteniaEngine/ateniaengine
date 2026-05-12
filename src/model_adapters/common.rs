use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::KvCacheBuildSpec;
use crate::amg::weight_store::WeightStore;
use crate::nn::llama::builder::{LlamaRuntime, build_llama};
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::LlamaConfig;

use super::ScratchGraphBuild;

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
