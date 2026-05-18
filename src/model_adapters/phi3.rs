use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::KvCacheBuildSpec;
use crate::amg::weight_store::WeightStore;
use crate::nn::llama::builder::LlamaRuntime;
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::LlamaConfig;
use crate::nn::llama::gguf_weight_loading::phi3_gguf_weight_mapper;
use crate::v17::loader::gguf_to_hf_naming::{gguf_to_hf_name_common, phi3_gguf_extra};
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::WeightMapper;

use super::{
    AdapterCapabilities, ConfigPolicy, GgufNameMapper, GgufWeightMapper, HfWeightMapper,
    ModelAdapter, ModelFamily, ModelMetadata, ResidencyHints, ScratchGraphBuild,
    StoreBackedGraphBuilder,
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

// **Phase 16** — Phi-3 fuses QKV (`attn_qkv`→`qkv_proj`) and MLP
// gate/up (`ffn_up`→`gate_up_proj`). The Phi-3 overrides are tried
// **first**: the common Llama-layout table also matches
// `ffn_up.weight` (→ the separate `up_proj`), which would shadow
// the fused `gate_up_proj` Phi-3's graph expects. Names with no
// Phi-3 override fall through to `gguf_to_hf_name_common`
// unchanged.
impl GgufNameMapper for Phi3Adapter {
    fn gguf_to_hf_name(&self, gguf_name: &str) -> Option<String> {
        phi3_gguf_extra(gguf_name).or_else(|| gguf_to_hf_name_common(gguf_name))
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

impl ConfigPolicy for Phi3Adapter {
    /// **Phase 12.4** — decode Phi-3 / Phi-3.5 LongRope
    /// scaling from the outer config JSON. Pre-Phase-12.4 this
    /// branch lived inside `LlamaConfig::from_json_str`'s
    /// `get_rope_scaling` helper. Moved here so the Phi-3
    /// family owns its own rope_scaling schema; the shared
    /// `Llama 3 piecewise` parser remains in config.rs as a
    /// fallback for the family that doesn't need adapter
    /// dispatch.
    ///
    /// The function reads:
    /// - `rope_scaling.type` (or `rope_scaling.rope_type`)
    ///   to confirm the shape is `"longrope"`.
    /// - `rope_scaling.short_factor`, `rope_scaling.long_factor`
    ///   per-dimension factor arrays.
    /// - **Top-level** `original_max_position_embeddings` and
    ///   `max_position_embeddings` (Phi-3 stores these outside
    ///   the `rope_scaling` block — the doc-comment on
    ///   `RopeScaling::LongRope` documents the choice).
    ///
    /// Returns `Ok(None)` when:
    /// - `rope_scaling` is absent.
    /// - `rope_scaling.type` is not `"longrope"`.
    ///
    /// Returns `Err(_)` when the shape is recognised as
    /// longrope but a required field is missing or malformed.
    fn parse_rope_scaling(
        &self,
        outer: &serde_json::Value,
    ) -> Result<
        Option<crate::nn::llama::config::RopeScaling>,
        crate::nn::llama::config::ConfigError,
    > {
        use crate::nn::llama::config::{parse_f32_array, ConfigError, RopeScaling};
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
        if rope_type != Some("longrope") {
            return Ok(None);
        }
        let short_factor = parse_f32_array(block, "short_factor", "rope_scaling.longrope")?;
        let long_factor = parse_f32_array(block, "long_factor", "rope_scaling.longrope")?;
        let original_max_position_embeddings = outer
            .get("original_max_position_embeddings")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| {
                ConfigError::Parse(
                    "rope_scaling.longrope requires top-level integer \
                     `original_max_position_embeddings`"
                        .into(),
                )
            })? as u32;
        let max_position_embeddings = outer
            .get("max_position_embeddings")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| {
                ConfigError::Parse(
                    "rope_scaling.longrope requires top-level integer \
                     `max_position_embeddings`"
                        .into(),
                )
            })? as u32;
        Ok(Some(RopeScaling::LongRope {
            short_factor,
            long_factor,
            original_max_position_embeddings,
            max_position_embeddings,
        }))
    }

    /// **Phase 15** — Phi-3 makes `head_dim` explicit when the
    /// input did not. Relocated from the `if arch == "phi3"`
    /// block in `llama_config_from_gguf`. Behaviour-equivalent:
    /// GGUF set `head_dim = explicit_head_dim.unwrap_or(inferred)`
    /// for Phi-3, which equals `effective_head_dim()` once the
    /// generic GGUF `head_size` extraction has run. An explicit
    /// value (HF `config.json` or GGUF metadata) is preserved
    /// because the setter only fires when `head_dim.is_none()`.
    fn apply_config_defaults(&self, config: &mut LlamaConfig) {
        if config.head_dim.is_none() {
            config.head_dim = Some(config.effective_head_dim());
        }
    }
}
