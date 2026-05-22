use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::KvCacheBuildSpec;
use crate::amg::weight_store::WeightStore;
use crate::nn::llama::builder::LlamaRuntime;
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::LlamaConfig;
use crate::nn::llama::gemma3::{
    build_gemma3, build_gemma3_with_store, gemma3_gguf_weight_mapper, gemma3_weight_mapper,
};
use crate::v17::loader::gguf_to_hf_naming::{gemma3_gguf_extra, gguf_to_hf_name_common};
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::WeightMapper;

use super::{
    AdapterCapabilities, ConfigPolicy, GgufNameMapper, GgufWeightMapper, HfWeightMapper,
    ModelAdapter, ModelFamily, ModelMetadata, ResidencyHints, ScratchGraphBuild,
    StoreBackedGraphBuilder,
};

/// **Gemma 3 (text) family adapter.**
///
/// Gemma 3 `Gemma3ForCausalLM` / `model_type = gemma3_text` is the
/// Gemma 2 topology (dual-norm, GeGLU, embedding scale, tied LM
/// head) with three deltas, all expressible with the existing AMG
/// op set — see `crate::nn::llama::gemma3`:
///
/// 1. per-head QK-Norm on q / k (RMSNorm γ `[head_dim]`), with the
///    attention scale folded into the k_norm γ;
/// 2. no soft-cap (`attn`/`final_logit_softcapping` are `None`);
/// 3. dual RoPE base frequency — local (sliding-window) layers use
///    `rope_local_base_freq`, global layers use `rope_theta`,
///    selected per layer by `sliding_window_pattern`.
///
/// **Scope:** text-only. The multimodal Gemma 3 variants
/// (`Gemma3ForConditionalGeneration`, 4B / 12B / 27B with a vision
/// tower) are out of scope; only the text path resolves here.
pub(super) struct Gemma3Adapter;

impl ModelAdapter for Gemma3Adapter {
    fn id(&self) -> &'static str {
        "gemma3"
    }

    fn family(&self) -> ModelFamily {
        ModelFamily::Gemma3
    }

    fn capabilities(&self) -> AdapterCapabilities {
        // Gemma 3 dropped Gemma 2's soft-caps — `gemma2_softcaps`
        // stays `false` (the default).
        AdapterCapabilities::llama_like()
    }

    fn supported_architectures(&self) -> &'static [&'static str] {
        &["Gemma3ForCausalLM"]
    }

    fn supported_model_types(&self) -> &'static [&'static str] {
        &["gemma3_text"]
    }

    fn supports(&self, metadata: &ModelMetadata<'_>) -> bool {
        metadata.architecture == "Gemma3ForCausalLM"
            || metadata.model_type == Some("gemma3_text")
            || metadata.model_type == Some("gemma3")
    }

    fn log_selection(&self) {
        eprintln!(
            "[ATENIA] Architecture: Gemma3ForCausalLM - routing to gemma3 adapter \
             (text-only: dual-norm, GeGLU, per-head QK-Norm, dual-RoPE local/global, \
             no soft-cap; sliding-window deferred - full causal attention for \
             context < sliding_window)."
        );
    }

    fn build_scratch_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
    ) -> ScratchGraphBuild {
        let h = build_gemma3(gb, config, runtime, token_input_id);
        ScratchGraphBuild {
            logits_id: h.logits_id,
            param_ids: h.param_ids,
            param_names: h.param_names,
        }
    }
}

impl HfWeightMapper for Gemma3Adapter {
    fn map_hf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        gemma3_weight_mapper(config, param_names, param_ids)
    }
}

impl GgufWeightMapper for Gemma3Adapter {
    fn map_gguf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        gemma3_gguf_weight_mapper(config, param_names, param_ids)
    }
}

// Gemma 3 GGUF: four per-layer norms plus the QK-Norm γ tensors.
// Extra-first composition so the `ffn_norm` →
// `pre_feedforward_layernorm` override wins over the common
// Llama-layout table (mirrors `Gemma2Adapter`).
impl GgufNameMapper for Gemma3Adapter {
    fn gguf_to_hf_name(&self, gguf_name: &str) -> Option<String> {
        gemma3_gguf_extra(gguf_name).or_else(|| gguf_to_hf_name_common(gguf_name))
    }
}

impl StoreBackedGraphBuilder for Gemma3Adapter {
    fn build_store_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
        store: &WeightStore,
        kv_cache: Option<&KvCacheBuildSpec>,
    ) -> Result<LlamaHandlesShared, BuildError> {
        build_gemma3_with_store(gb, config, runtime, token_input_id, store, kv_cache)
    }
}

impl ResidencyHints for Gemma3Adapter {}

impl ConfigPolicy for Gemma3Adapter {
    /// Gemma 3's HF config omits `tie_word_embeddings`; upstream
    /// defaults it to `True` (the small Gemma 3 text checkpoints
    /// are genuinely tied — no physical `lm_head.weight`).
    fn default_tie_word_embeddings(&self) -> Option<bool> {
        Some(true)
    }

    /// Gemma 3 family config defaults. `head_dim` is made explicit
    /// when absent (the graph/validator must see the value the
    /// kernel uses). The dual-RoPE constants are defaulted only
    /// when the GGUF metadata omits them — the canonical Gemma 3
    /// HF `config.json` always ships them, so this only fires for
    /// a GGUF that lacks the keys. Each setter is `if is_none()`
    /// so an explicit value always wins.
    fn apply_config_defaults(&self, config: &mut LlamaConfig) {
        if config.head_dim.is_none() {
            config.head_dim = Some(config.effective_head_dim());
        }
        if config.sliding_window_pattern.is_none() {
            config.sliding_window_pattern = Some(6);
        }
        if config.rope_local_base_freq.is_none() {
            config.rope_local_base_freq = Some(10_000);
        }
    }
}
