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

pub trait ModelAdapter: Sync {
    fn id(&self) -> &'static str;
    fn family(&self) -> ModelFamily;
    fn capabilities(&self) -> AdapterCapabilities;
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

pub trait AteniaModelAdapter:
    ModelAdapter + HfWeightMapper + GgufWeightMapper + StoreBackedGraphBuilder + ResidencyHints
{
}

impl<T> AteniaModelAdapter for T where
    T: ModelAdapter + HfWeightMapper + GgufWeightMapper + StoreBackedGraphBuilder + ResidencyHints
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
}
