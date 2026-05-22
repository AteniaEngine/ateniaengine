//! **Adapter Toolkit v2 — Part 3: the adapter generator.**
//!
//! [`GeneratedAdapter`] is a v2 adapter built from a
//! [`ResolvedAdapterSpec`]. It is **not** generated Rust code and it
//! does **not** reimplement any graph builder or weight mapper: it
//! holds a `&'static dyn AteniaModelAdapter` — the v1 hand-written
//! adapter for the resolved family — and implements the v1 adapter
//! supertrait by **pure delegation** to it.
//!
//! The DSL therefore *parameterises* an existing family; it never
//! defines a new architecture. Graph topology, weight mapping, and
//! GGUF naming are all v1's, untouched. What the DSL adds on top —
//! the multi-EOS set, turn terminators, per-checkpoint overrides —
//! is generation-time metadata, exposed via [`GeneratedAdapter`]'s
//! own accessors ([`GeneratedAdapter::spec`]) rather than smuggled
//! into the graph build. This keeps the v2 adapter a strict,
//! behaviour-preserving superset of the v1 adapter it wraps.

use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::KvCacheBuildSpec;
use crate::amg::weight_store::WeightStore;
use crate::model_adapters::{
    resolve_adapter, AdapterCapabilities, AteniaModelAdapter, ConfigPolicy, GgufNameMapper,
    GgufWeightMapper, HfWeightMapper, ModelAdapter, ModelFamily, ModelFormat, ModelMetadata,
    ResidencyHints, ResidencyPolicyHints, ScratchGraphBuild, StoreBackedGraphBuilder,
};
use crate::nn::llama::builder::LlamaRuntime;
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::{ConfigError, LlamaConfig, RopeScaling};
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::WeightMapper;

use super::spec::ResolvedAdapterSpec;
use super::ToolkitError;

/// A v2 adapter: a [`ResolvedAdapterSpec`] bound to the v1 base
/// adapter it delegates to.
pub struct GeneratedAdapter {
    /// The v1 hand-written adapter for the resolved family. Every
    /// supertrait method delegates here.
    base: &'static dyn AteniaModelAdapter,
    /// The resolved DSL spec — the source of the v2-only metadata.
    spec: ResolvedAdapterSpec,
}

impl GeneratedAdapter {
    /// Build a [`GeneratedAdapter`] from a resolved spec by binding
    /// it to the v1 base adapter for the spec's family. Fails if v1
    /// has no adapter for the resolved architecture/model_type — the
    /// toolkit never invents a builder.
    pub fn from_spec(spec: ResolvedAdapterSpec) -> Result<Self, ToolkitError> {
        let metadata = ModelMetadata {
            architecture: spec.architecture,
            model_type: Some(spec.model_type),
            // Format does not affect v1 adapter selection
            // (`supports()` keys off architecture / model_type
            // only); HfSafetensors is the neutral choice.
            format: ModelFormat::HfSafetensors,
        };
        let base = resolve_adapter(&metadata).ok_or_else(|| {
            ToolkitError::Resolution(format!(
                "no Atenia v1 adapter resolves architecture `{}` / model_type `{}` — \
                 Adapter Toolkit v2 parameterises existing families, it does not \
                 add new architectures",
                spec.architecture, spec.model_type
            ))
        })?;
        // Defensive: the v1 adapter's family must match the family
        // the spec resolved to. A mismatch means the DSL's explicit
        // architecture/model_type disagree — fail loud rather than
        // silently delegating to the wrong builder.
        if base.family() != spec.family {
            return Err(ToolkitError::Resolution(format!(
                "DSL family {:?} resolved v1 adapter `{}` of family {:?} — \
                 architecture/model_type are inconsistent with `family`",
                spec.family,
                base.id(),
                base.family()
            )));
        }
        Ok(Self { base, spec })
    }

    /// The resolved spec backing this adapter.
    pub fn spec(&self) -> &ResolvedAdapterSpec {
        &self.spec
    }

    /// The v1 base adapter id this v2 adapter delegates to
    /// (`llama`, `qwen2`, `phi3`, …).
    pub fn base_id(&self) -> &'static str {
        self.base.id()
    }
}

// ----------------------------------------------------------------
// Supertrait delegation. Every method forwards to `self.base`; the
// v2 adapter's behaviour on the v1 surface is byte-identical to the
// hand-written adapter it wraps.
// ----------------------------------------------------------------

impl ModelAdapter for GeneratedAdapter {
    fn id(&self) -> &'static str {
        self.base.id()
    }

    fn family(&self) -> ModelFamily {
        self.base.family()
    }

    fn capabilities(&self) -> AdapterCapabilities {
        self.base.capabilities()
    }

    fn supported_architectures(&self) -> &'static [&'static str] {
        self.base.supported_architectures()
    }

    fn supported_model_types(&self) -> &'static [&'static str] {
        self.base.supported_model_types()
    }

    fn supports(&self, metadata: &ModelMetadata<'_>) -> bool {
        self.base.supports(metadata)
    }

    fn log_selection(&self) {
        self.base.log_selection();
    }

    fn build_scratch_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
    ) -> ScratchGraphBuild {
        self.base
            .build_scratch_graph(gb, config, runtime, token_input_id)
    }
}

impl HfWeightMapper for GeneratedAdapter {
    fn map_hf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        self.base.map_hf_weights(config, param_names, param_ids)
    }
}

impl GgufWeightMapper for GeneratedAdapter {
    fn map_gguf_weights(
        &self,
        config: &LlamaConfig,
        param_names: &[String],
        param_ids: &[usize],
    ) -> Result<WeightMapper, LoaderError> {
        self.base.map_gguf_weights(config, param_names, param_ids)
    }
}

impl GgufNameMapper for GeneratedAdapter {
    fn gguf_to_hf_name(&self, gguf_name: &str) -> Option<String> {
        self.base.gguf_to_hf_name(gguf_name)
    }
}

impl StoreBackedGraphBuilder for GeneratedAdapter {
    fn build_store_graph(
        &self,
        gb: &mut GraphBuilder,
        config: &LlamaConfig,
        runtime: &LlamaRuntime,
        token_input_id: usize,
        store: &WeightStore,
        kv_cache: Option<&KvCacheBuildSpec>,
    ) -> Result<LlamaHandlesShared, BuildError> {
        self.base
            .build_store_graph(gb, config, runtime, token_input_id, store, kv_cache)
    }
}

impl ResidencyHints for GeneratedAdapter {
    fn residency_hints(&self, config: &LlamaConfig) -> ResidencyPolicyHints {
        self.base.residency_hints(config)
    }
}

impl ConfigPolicy for GeneratedAdapter {
    fn default_tie_word_embeddings(&self) -> Option<bool> {
        self.base.default_tie_word_embeddings()
    }

    fn default_attention_bias(&self) -> Option<bool> {
        self.base.default_attention_bias()
    }

    fn parse_rope_scaling(
        &self,
        outer: &serde_json::Value,
    ) -> Result<Option<RopeScaling>, ConfigError> {
        self.base.parse_rope_scaling(outer)
    }

    fn validate_config(&self, config: &LlamaConfig) -> Result<(), ConfigError> {
        self.base.validate_config(config)
    }

    fn apply_config_defaults(&self, config: &mut LlamaConfig) {
        self.base.apply_config_defaults(config);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter_toolkit::dsl::AdapterDsl;

    fn generated(text: &str) -> GeneratedAdapter {
        let dsl = AdapterDsl::from_str(text, true).expect("dsl parses");
        let spec = ResolvedAdapterSpec::resolve(&dsl).expect("spec resolves");
        GeneratedAdapter::from_spec(spec).expect("adapter generates")
    }

    #[test]
    fn generated_llama_adapter_delegates_to_v1_llama() {
        let adapter = generated("family: llama\n");
        assert_eq!(adapter.base_id(), "llama");
        assert_eq!(adapter.family(), ModelFamily::Llama);
        assert_eq!(adapter.id(), "llama");
    }

    #[test]
    fn generated_phi_adapter_keeps_phi_capabilities() {
        let adapter = generated("family: phi\n");
        let caps = adapter.capabilities();
        // Delegation must preserve Phi-3's fused-weight capability
        // flags exactly — proof the v2 adapter is a faithful
        // superset of the v1 adapter.
        assert!(caps.fused_qkv_weight_mapping);
        assert!(caps.fused_gate_up_weight_mapping);
    }

    #[test]
    fn generated_qwen3_adapter_resolves_distinctly() {
        let adapter = generated("family: qwen3\n");
        assert_eq!(adapter.base_id(), "qwen3");
        assert_eq!(adapter.family(), ModelFamily::Qwen3);
    }

    #[test]
    fn generated_adapter_gguf_naming_delegates() {
        // Phi-3 fused-QKV GGUF name mapping must survive delegation.
        let adapter = generated("family: phi\n");
        assert_eq!(
            adapter.gguf_to_hf_name("blk.3.attn_qkv.weight").as_deref(),
            Some("model.layers.3.self_attn.qkv_proj.weight")
        );
        // A llama-family v2 adapter inherits the common table only.
        let llama = generated("family: llama\n");
        assert_eq!(llama.gguf_to_hf_name("blk.3.attn_qkv.weight"), None);
    }

    #[test]
    fn generated_adapter_exposes_dsl_overrides() {
        let adapter = generated(
            "family: qwen\n\
             overrides:\n  deepseek-distill:\n    tokenizer:\n      eos_tokens: [1, 106]\n",
        );
        let ov = adapter
            .spec()
            .override_for("deepseek-distill")
            .expect("override present");
        assert_eq!(ov.tokenizer.eos_tokens, Some(vec![1, 106]));
    }

    #[test]
    fn generated_adapter_is_usable_as_atenia_model_adapter() {
        // The blanket impl must hold: a GeneratedAdapter is an
        // AteniaModelAdapter and can be used behind the v1 trait
        // object, drop-in.
        let adapter = generated("family: mistral\n");
        let obj: &dyn AteniaModelAdapter = &adapter;
        assert_eq!(obj.id(), "mistral");
        assert_eq!(obj.family(), ModelFamily::Mistral);
    }
}
