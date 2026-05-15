//! Internal model adapter layer.
//!
//! This is deliberately not a public SDK yet. The first contract is small and
//! internal: adapters describe model-family differences at load/build time,
//! while Atenia Core keeps owning execution, memory planning, kernels, and
//! numeric policy.

mod common;
mod gemma2;
mod llama_family;
mod phi3;

use crate::amg::builder::GraphBuilder;
use crate::amg::kv_cache::KvCacheBuildSpec;
use crate::amg::weight_store::WeightStore;
use crate::nn::llama::builder::LlamaRuntime;
use crate::nn::llama::builder_shared::{BuildError, LlamaHandlesShared};
use crate::nn::llama::config::LlamaConfig;
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::weight_mapper::WeightMapper;

use gemma2::Gemma2Adapter;
use llama_family::{LlamaFamilyAdapter, MistralAdapter, Qwen2Adapter};
use phi3::Phi3Adapter;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelFamily {
    Llama,
    Qwen2,
    Mistral,
    Phi3,
    Gemma2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdapterCapabilities {
    pub hf_safetensors: bool,
    pub gguf: bool,
    pub store_backed_generation: bool,
    pub fused_qkv_weight_mapping: bool,
    pub fused_gate_up_weight_mapping: bool,
    pub gemma2_softcaps: bool,
}

impl AdapterCapabilities {
    pub(super) const fn llama_like() -> Self {
        Self {
            hf_safetensors: true,
            gguf: true,
            store_backed_generation: true,
            fused_qkv_weight_mapping: false,
            fused_gate_up_weight_mapping: false,
            gemma2_softcaps: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdapterContract {
    pub id: &'static str,
    pub family: ModelFamily,
    pub supported_architectures: &'static [&'static str],
    pub supported_model_types: &'static [&'static str],
    pub capabilities: AdapterCapabilities,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LmHeadResidency {
    UntiedOnly,
    KeepCpu,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidencyPolicyHints {
    pub projection_weights_vram_eligible: bool,
    pub lm_head: LmHeadResidency,
    pub embeddings_cpu_only: bool,
    pub norms_cpu_only: bool,
    pub prefer_layer_local_projection_packing: bool,
}

impl Default for ResidencyPolicyHints {
    fn default() -> Self {
        Self {
            projection_weights_vram_eligible: true,
            lm_head: LmHeadResidency::UntiedOnly,
            embeddings_cpu_only: true,
            norms_cpu_only: true,
            prefer_layer_local_projection_packing: true,
        }
    }
}

pub(in crate::model_adapters) fn llama_like_residency_hints() -> ResidencyPolicyHints {
    llama_like_residency_hints_from_flag(
        std::env::var("ATENIA_ADAPTER_LM_HEAD_CPU").as_deref() == Ok("1"),
    )
}

pub(in crate::model_adapters) fn llama_like_residency_hints_from_flag(
    keep_lm_head_cpu: bool,
) -> ResidencyPolicyHints {
    let mut hints = ResidencyPolicyHints::default();
    if keep_lm_head_cpu {
        hints.lm_head = LmHeadResidency::KeepCpu;
    }
    hints
}

pub trait ModelAdapter: Sync {
    fn id(&self) -> &'static str;
    fn family(&self) -> ModelFamily;
    fn capabilities(&self) -> AdapterCapabilities;
    fn supported_architectures(&self) -> &'static [&'static str] {
        &[]
    }
    fn supported_model_types(&self) -> &'static [&'static str] {
        &[]
    }
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

pub trait ResidencyHints {
    fn residency_hints(&self, _config: &LlamaConfig) -> ResidencyPolicyHints {
        ResidencyPolicyHints::default()
    }
}

/// **Phase 12** — adapter-supplied defaults for HuggingFace
/// `config.json` fields that some checkpoints omit. Used by
/// [`crate::nn::llama::config::LlamaConfig::from_json_str`] to
/// resolve missing fields at parse time without hard-coding
/// `match model_type` branches inside the config parser.
///
/// Default impl returns `None` for every method, meaning "no
/// adapter-supplied default — the field must be present in
/// the JSON or the parser hard-errors". Adapters opt in by
/// overriding individual methods.
///
/// Today's per-adapter overrides:
/// - `Gemma2Adapter::default_tie_word_embeddings = Some(true)`
///   — Gemma 2's official config omits the field and upstream
///   `Gemma2Config` defaults it to `True`.
/// - `Qwen2Adapter::default_attention_bias = Some(true)` —
///   Qwen2 omits the field entirely and hard-codes QKV biases
///   on inside `Qwen2Attention`.
///
/// Every other adapter (Llama, Mistral, Phi-3) inherits the
/// default `None` for both methods, preserving the existing
/// fail-loud behaviour for unexpected config shapes.
pub trait ConfigPolicy {
    /// Default value for `tie_word_embeddings` when the field
    /// is absent from `config.json`. `None` means "no
    /// adapter-supplied default; require the field". `Some(b)`
    /// means "treat as `b` if absent".
    fn default_tie_word_embeddings(&self) -> Option<bool> {
        None
    }

    /// Default value for `attention_bias` when the field is
    /// absent from `config.json`. `None` means "no
    /// adapter-supplied default; treat as `false`". `Some(b)`
    /// means "treat as `b` if absent". The semantic difference
    /// from `default_tie_word_embeddings` is that
    /// `attention_bias` is consumed via
    /// `LlamaConfig::effective_attention_bias()`, which
    /// already had a `false` fallback for unknown families
    /// before Phase 12. Adapters that need a different default
    /// return `Some(true)` (Qwen2 today).
    fn default_attention_bias(&self) -> Option<bool> {
        None
    }

    /// **Phase 12.4** — family-specific parser for the
    /// `rope_scaling` JSON block. The full **outer** config
    /// JSON is passed (not just the `rope_scaling` sub-object)
    /// because some shapes — Phi-3 LongRope — read top-level
    /// fields like `original_max_position_embeddings` that
    /// don't live inside the `rope_scaling` block.
    ///
    /// Return value:
    /// - `Ok(None)` — adapter does not recognise the
    ///   `rope_scaling` shape; the caller falls back to the
    ///   shared parser (currently Llama 3 piecewise) or
    ///   silently downgrades to no-scaling.
    /// - `Ok(Some(scaling))` — adapter recognised and
    ///   successfully parsed the block.
    /// - `Err(_)` — adapter recognised the shape but found a
    ///   malformed field; the parse fails fast.
    ///
    /// Default impl returns `Ok(None)`. Today the only override
    /// is `Phi3Adapter::parse_rope_scaling`, which decodes
    /// `type = "longrope"`.
    fn parse_rope_scaling(
        &self,
        _outer: &serde_json::Value,
    ) -> Result<Option<crate::nn::llama::config::RopeScaling>, crate::nn::llama::config::ConfigError>
    {
        Ok(None)
    }
}

pub trait AteniaModelAdapter:
    ModelAdapter
    + HfWeightMapper
    + GgufWeightMapper
    + StoreBackedGraphBuilder
    + ResidencyHints
    + ConfigPolicy
{
}

impl<T> AteniaModelAdapter for T where
    T: ModelAdapter
        + HfWeightMapper
        + GgufWeightMapper
        + StoreBackedGraphBuilder
        + ResidencyHints
        + ConfigPolicy
{
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

pub fn registered_adapter_contracts() -> Vec<AdapterContract> {
    ADAPTERS
        .iter()
        .map(|adapter| AdapterContract {
            id: adapter.id(),
            family: adapter.family(),
            supported_architectures: adapter.supported_architectures(),
            supported_model_types: adapter.supported_model_types(),
            capabilities: adapter.capabilities(),
        })
        .collect()
}

pub fn model_metadata_for_config(config: &LlamaConfig) -> ModelMetadata<'_> {
    model_metadata_from_parts(
        None,
        config.model_type.as_deref(),
        ModelFormat::HfSafetensors,
    )
}

pub fn model_metadata_from_parts<'a>(
    architecture: Option<&'a str>,
    model_type: Option<&'a str>,
    format: ModelFormat,
) -> ModelMetadata<'a> {
    let architecture = architecture
        .or_else(|| model_type.and_then(default_architecture_for_model_type))
        .unwrap_or("LlamaForCausalLM");
    ModelMetadata {
        architecture,
        model_type,
        format,
    }
}

fn default_architecture_for_model_type(model_type: &str) -> Option<&'static str> {
    ADAPTERS.iter().find_map(|adapter| {
        if adapter.supported_model_types().contains(&model_type) {
            adapter.supported_architectures().first().copied()
        } else {
            None
        }
    })
}

pub fn resolve_adapter_for_config(config: &LlamaConfig) -> &'static dyn AteniaModelAdapter {
    let metadata = model_metadata_for_config(config);
    resolve_adapter(&metadata).unwrap_or(&LLAMA_FAMILY_ADAPTER)
}

pub fn supported_architectures_message() -> String {
    let entries = registered_adapter_contracts()
        .into_iter()
        .map(|contract| {
            let architectures = contract.supported_architectures.join("/");
            if contract.supported_model_types.is_empty() {
                format!("{} [{}]", contract.id, architectures)
            } else {
                format!(
                    "{} [{}; model_type={}]",
                    contract.id,
                    architectures,
                    contract.supported_model_types.join("/")
                )
            }
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "Atenia today supports these internal adapters: {entries}. \
         Adapters are selected from architecture/model_type metadata."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_resolves_core_architectures() {
        let cases = [
            ("LlamaForCausalLM", None, "llama", ModelFamily::Llama),
            (
                "Qwen2ForCausalLM",
                Some("qwen2"),
                "qwen2",
                ModelFamily::Qwen2,
            ),
            (
                "MistralForCausalLM",
                Some("mistral"),
                "mistral",
                ModelFamily::Mistral,
            ),
            ("Phi3ForCausalLM", Some("phi3"), "phi3", ModelFamily::Phi3),
            (
                "Gemma2ForCausalLM",
                Some("gemma2"),
                "gemma2",
                ModelFamily::Gemma2,
            ),
        ];
        for (architecture, model_type, expected_id, expected_family) in cases {
            let metadata = ModelMetadata {
                architecture,
                model_type,
                format: ModelFormat::HfSafetensors,
            };
            let adapter = resolve_adapter(&metadata).expect("adapter must resolve");
            assert_eq!(adapter.id(), expected_id);
            assert_eq!(adapter.family(), expected_family);
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

    #[test]
    fn falcon3_llama_compatible_config_resolves_to_llama_adapter() {
        let metadata = ModelMetadata {
            architecture: "LlamaForCausalLM",
            model_type: Some("llama"),
            format: ModelFormat::HfSafetensors,
        };
        let adapter = resolve_adapter(&metadata).expect("falcon3 config shape resolves");
        assert_eq!(adapter.id(), "llama");
        assert_eq!(adapter.family(), ModelFamily::Llama);
    }

    #[test]
    fn adapter_capabilities_document_family_specific_features() {
        let phi3 = resolve_adapter(&ModelMetadata {
            architecture: "Phi3ForCausalLM",
            model_type: Some("phi3"),
            format: ModelFormat::HfSafetensors,
        })
        .expect("phi3 adapter");
        let phi3_caps = phi3.capabilities();
        assert!(phi3_caps.fused_qkv_weight_mapping);
        assert!(phi3_caps.fused_gate_up_weight_mapping);
        assert!(!phi3_caps.gemma2_softcaps);

        let gemma2 = resolve_adapter(&ModelMetadata {
            architecture: "Gemma2ForCausalLM",
            model_type: Some("gemma2"),
            format: ModelFormat::HfSafetensors,
        })
        .expect("gemma2 adapter");
        let gemma2_caps = gemma2.capabilities();
        assert!(gemma2_caps.gemma2_softcaps);
        assert!(!gemma2_caps.fused_qkv_weight_mapping);
    }

    #[test]
    fn default_residency_hints_match_current_tier_policy() {
        let adapter = resolve_adapter(&ModelMetadata {
            architecture: "MistralForCausalLM",
            model_type: Some("mistral"),
            format: ModelFormat::HfSafetensors,
        })
        .expect("mistral adapter");
        let config = LlamaConfig::from_json_str(
            r#"{
                "vocab_size": 32000,
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "intermediate_size": 128,
                "max_position_embeddings": 128,
                "rope_theta": 10000,
                "rms_norm_eps": 0.000001,
                "tie_word_embeddings": true,
                "model_type": "mistral",
                "bos_token_id": 1,
                "eos_token_id": 2
            }"#,
        )
        .expect("test config parses");
        let hints = adapter.residency_hints(&config);
        assert!(hints.projection_weights_vram_eligible);
        assert_eq!(hints.lm_head, LmHeadResidency::UntiedOnly);
        assert!(hints.embeddings_cpu_only);
        assert!(hints.norms_cpu_only);
        assert!(hints.prefer_layer_local_projection_packing);
    }

    #[test]
    fn llama_like_residency_flag_can_keep_lm_head_on_cpu() {
        let hints = llama_like_residency_hints_from_flag(true);
        assert!(hints.projection_weights_vram_eligible);
        assert_eq!(hints.lm_head, LmHeadResidency::KeepCpu);
        assert!(hints.embeddings_cpu_only);
        assert!(hints.norms_cpu_only);
        assert!(hints.prefer_layer_local_projection_packing);
    }

    #[test]
    fn llama_like_residency_default_keeps_current_lm_head_policy() {
        let hints = llama_like_residency_hints_from_flag(false);
        assert_eq!(hints, ResidencyPolicyHints::default());
    }

    #[test]
    fn registered_adapter_contracts_are_stable_and_unique() {
        let contracts = registered_adapter_contracts();
        let ids: Vec<_> = contracts.iter().map(|contract| contract.id).collect();
        assert_eq!(ids, ["phi3", "gemma2", "qwen2", "mistral", "llama"]);

        let mut unique_ids = std::collections::BTreeSet::new();
        for contract in &contracts {
            assert!(unique_ids.insert(contract.id));
            assert!(!contract.supported_architectures.is_empty());
            assert!(contract.capabilities.hf_safetensors);
            assert!(contract.capabilities.gguf);
            assert!(contract.capabilities.store_backed_generation);
        }
    }

    #[test]
    fn registered_adapter_contracts_capture_family_specialization() {
        let contracts = registered_adapter_contracts();
        let phi3 = contracts
            .iter()
            .find(|contract| contract.id == "phi3")
            .expect("phi3 contract");
        assert_eq!(phi3.family, ModelFamily::Phi3);
        assert!(phi3.capabilities.fused_qkv_weight_mapping);
        assert!(phi3.capabilities.fused_gate_up_weight_mapping);
        assert!(!phi3.capabilities.gemma2_softcaps);

        let gemma2 = contracts
            .iter()
            .find(|contract| contract.id == "gemma2")
            .expect("gemma2 contract");
        assert_eq!(gemma2.family, ModelFamily::Gemma2);
        assert!(gemma2.capabilities.gemma2_softcaps);
        assert!(!gemma2.capabilities.fused_qkv_weight_mapping);
    }

    #[test]
    fn supported_architectures_message_is_registry_backed() {
        let message = supported_architectures_message();
        for expected in [
            "phi3",
            "gemma2",
            "qwen2",
            "mistral",
            "llama",
            "Phi3ForCausalLM",
            "Gemma2ForCausalLM",
            "Qwen2ForCausalLM",
            "MistralForCausalLM",
            "LlamaForCausalLM",
        ] {
            assert!(
                message.contains(expected),
                "{expected} missing from {message}"
            );
        }
    }

    #[test]
    fn config_metadata_uses_registered_model_type_contracts() {
        let cases = [
            ("phi3", "Phi3ForCausalLM"),
            ("gemma2", "Gemma2ForCausalLM"),
            ("qwen2", "Qwen2ForCausalLM"),
            ("mistral", "MistralForCausalLM"),
            ("llama", "LlamaForCausalLM"),
        ];
        for (model_type, expected_architecture) in cases {
            assert_eq!(
                default_architecture_for_model_type(model_type),
                Some(expected_architecture)
            );
        }
    }

    #[test]
    fn config_metadata_falls_back_to_llama_for_unknown_or_absent_model_type() {
        let config = LlamaConfig::from_json_str(
            r#"{
                "vocab_size": 32000,
                "hidden_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "intermediate_size": 128,
                "max_position_embeddings": 128,
                "rope_theta": 10000,
                "rms_norm_eps": 0.000001,
                "tie_word_embeddings": true,
                "model_type": "custom_llama_like",
                "bos_token_id": 1,
                "eos_token_id": 2
            }"#,
        )
        .expect("test config parses");
        let metadata = model_metadata_for_config(&config);
        assert_eq!(metadata.architecture, "LlamaForCausalLM");
        assert_eq!(metadata.model_type, Some("custom_llama_like"));
    }

    #[test]
    fn metadata_from_parts_preserves_explicit_architecture_and_format() {
        let metadata =
            model_metadata_from_parts(Some("Phi3ForCausalLM"), Some("phi3"), ModelFormat::Gguf);
        assert_eq!(metadata.architecture, "Phi3ForCausalLM");
        assert_eq!(metadata.model_type, Some("phi3"));
        assert_eq!(metadata.format, ModelFormat::Gguf);
    }

    #[test]
    fn config_policy_defaults_are_family_specific() {
        // Phase 12 — only Gemma2Adapter opts into the
        // tie_word_embeddings default. Llama / Mistral / Phi-3 /
        // Qwen2 must return None so config.rs hard-errors when
        // the field is missing on those families.
        let gemma2 = resolve_adapter(&ModelMetadata {
            architecture: "Gemma2ForCausalLM",
            model_type: Some("gemma2"),
            format: ModelFormat::HfSafetensors,
        })
        .expect("gemma2 adapter");
        assert_eq!(gemma2.default_tie_word_embeddings(), Some(true));
        assert_eq!(gemma2.default_attention_bias(), None);

        // Phase 12 — only Qwen2Adapter opts into the
        // attention_bias default. Everyone else returns None
        // (which `effective_attention_bias` translates to
        // `false` if `attention_bias` field is absent).
        let qwen2 = resolve_adapter(&ModelMetadata {
            architecture: "Qwen2ForCausalLM",
            model_type: Some("qwen2"),
            format: ModelFormat::HfSafetensors,
        })
        .expect("qwen2 adapter");
        assert_eq!(qwen2.default_attention_bias(), Some(true));
        assert_eq!(qwen2.default_tie_word_embeddings(), None);

        // Llama / Mistral / Phi-3: no adapter-supplied default
        // for either field.
        for (arch, model_type) in [
            ("LlamaForCausalLM", Some("llama")),
            ("MistralForCausalLM", Some("mistral")),
            ("Phi3ForCausalLM", Some("phi3")),
        ] {
            let adapter = resolve_adapter(&ModelMetadata {
                architecture: arch,
                model_type,
                format: ModelFormat::HfSafetensors,
            })
            .expect("adapter resolves");
            assert_eq!(
                adapter.default_tie_word_embeddings(),
                None,
                "{arch} must not supply a tie_word_embeddings default"
            );
            assert_eq!(
                adapter.default_attention_bias(),
                None,
                "{arch} must not supply an attention_bias default"
            );
        }
    }

    #[test]
    fn metadata_from_parts_infers_architecture_from_model_type() {
        let metadata = model_metadata_from_parts(None, Some("gemma2"), ModelFormat::HfSafetensors);
        assert_eq!(metadata.architecture, "Gemma2ForCausalLM");
        assert_eq!(metadata.model_type, Some("gemma2"));
        assert_eq!(metadata.format, ModelFormat::HfSafetensors);
    }
}
