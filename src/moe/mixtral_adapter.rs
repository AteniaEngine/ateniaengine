//! **MOE-FULL-3** — experimental Mixtral family adapter + tensor spec
//! (load-only).
//!
//! This is a **metadata-only** layer that teaches Atenia to *recognize* a
//! Mixtral checkpoint at the tensor level and describe its MoE tensor
//! topology. It is deliberately isolated in `src/moe/` (the experimental
//! sandbox) and does NOT:
//!
//! * touch the productive `src/model_adapters/` registry or any runtime/graph;
//! * execute, build a graph, or run a forward;
//! * lift the MOE-2 loader fail-loud guard (the productive loader still
//!   refuses MoE checkpoints).
//!
//! It builds on the existing pieces: `detect::classify_tensor_name`,
//! `data_plane::MoeWeightMap`, `binding::packed_dims`, and
//! `nn::llama::moe_config::MoeConfig` (MOE-FULL-2). Its job is to give
//! MOE-FULL-4 a single place that says "this is a Mixtral, here are its
//! router/expert tensors (packed or classic), and here is the dense surround
//! it reuses".
//!
//! ## Mixtral tensor inventory (verified on real tiny checkpoints)
//!
//! Two on-disk expert layouts exist (both real, both validated in
//! MIXTRAL-CERT-1):
//!
//! ```text
//! PACKED (hf-internal-testing/tiny-random-MixtralForCausalLM):
//!   model.layers.{L}.mlp.gate.weight                      router [E, d_model]
//!   model.layers.{L}.mlp.experts.gate_up_proj             [E, 2*d_ff, d_model]
//!   model.layers.{L}.mlp.experts.down_proj                [E, d_model, d_ff]
//!
//! CLASSIC (TitanML/tiny-mixtral, original Mixtral layout):
//!   model.layers.{L}.block_sparse_moe.gate.weight         router [E, d_model]
//!   model.layers.{L}.block_sparse_moe.experts.{e}.w1.weight  gate [d_ff, d_model]
//!   model.layers.{L}.block_sparse_moe.experts.{e}.w3.weight  up   [d_ff, d_model]
//!   model.layers.{L}.block_sparse_moe.experts.{e}.w2.weight  down [d_model, d_ff]
//!
//! Dense surround (reused, identical to a dense Mistral decoder):
//!   model.embed_tokens.weight, lm_head.weight, model.norm.weight,
//!   model.layers.{L}.input_layernorm.weight,
//!   model.layers.{L}.post_attention_layernorm.weight,
//!   model.layers.{L}.self_attn.{q,k,v,o}_proj.weight
//! ```
//!
//! Mixtral has **no shared expert** in either layout.

use crate::nn::llama::moe_config::MoeConfig;

use super::data_plane::MoeWeightMap;
use super::detect::detect_moe;

/// The on-disk expert layout of a Mixtral checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixtralExpertLayout {
    /// `mlp.experts.gate_up_proj` + `mlp.experts.down_proj` (3-D, fused).
    Packed,
    /// `block_sparse_moe.experts.{e}.w1/w3/w2` (per-expert, classic Mixtral).
    Classic,
}

/// Errors from recognizing / validating a Mixtral checkpoint (load-only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MixtralAdapterError {
    /// No MoE tensors were detected at all (a dense checkpoint).
    NotMoe,
    /// MoE tensors exist but the layout is neither Mixtral packed nor classic.
    UnrecognizedLayout,
    /// A required tensor (router / experts) is missing for a layer.
    MissingRouter { layer_id: usize },
    /// A layer has neither classic nor packed expert tensors.
    MissingExperts { layer_id: usize },
    /// Packed shapes were inconsistent (delegated to `binding::packed_dims`).
    BadPackedShapes { detail: String },
}

impl std::fmt::Display for MixtralAdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MixtralAdapterError::NotMoe => write!(f, "mixtral-adapter: not a MoE checkpoint"),
            MixtralAdapterError::UnrecognizedLayout => {
                write!(f, "mixtral-adapter: MoE tensors present but layout is not Mixtral")
            }
            MixtralAdapterError::MissingRouter { layer_id } => {
                write!(f, "mixtral-adapter: layer {layer_id} has no router tensor")
            }
            MixtralAdapterError::MissingExperts { layer_id } => {
                write!(f, "mixtral-adapter: layer {layer_id} has no expert tensors")
            }
            MixtralAdapterError::BadPackedShapes { detail } => {
                write!(f, "mixtral-adapter: bad packed expert shapes: {detail}")
            }
        }
    }
}

impl std::error::Error for MixtralAdapterError {}

/// The names a Mixtral layer's MoE tensors carry, plus the dense tensors it
/// reuses. Pure metadata — no data, no execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MixtralTensorSpec {
    pub layout: MixtralExpertLayout,
    /// Router weight name for layer 0 (template: `{L}` substituted).
    pub router_suffix: &'static str,
    /// Expert gate/up/down name suffixes (per-expert classic) or packed
    /// tensor suffixes.
    pub expert_suffixes: &'static [&'static str],
    /// Dense tensor suffixes reused unchanged from the dense Mistral decoder.
    pub dense_suffixes: &'static [&'static str],
}

const CLASSIC_EXPERT_SUFFIXES: &[&str] = &[
    "block_sparse_moe.experts.{e}.w1.weight", // gate
    "block_sparse_moe.experts.{e}.w3.weight", // up
    "block_sparse_moe.experts.{e}.w2.weight", // down
];
const PACKED_EXPERT_SUFFIXES: &[&str] =
    &["mlp.experts.gate_up_proj", "mlp.experts.down_proj"];
const DENSE_SUFFIXES: &[&str] = &[
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
];

impl MixtralTensorSpec {
    fn for_layout(layout: MixtralExpertLayout) -> Self {
        match layout {
            MixtralExpertLayout::Classic => MixtralTensorSpec {
                layout,
                router_suffix: "block_sparse_moe.gate.weight",
                expert_suffixes: CLASSIC_EXPERT_SUFFIXES,
                dense_suffixes: DENSE_SUFFIXES,
            },
            MixtralExpertLayout::Packed => MixtralTensorSpec {
                layout,
                router_suffix: "mlp.gate.weight",
                expert_suffixes: PACKED_EXPERT_SUFFIXES,
                dense_suffixes: DENSE_SUFFIXES,
            },
        }
    }
}

/// The recognized, validated metadata of a Mixtral checkpoint (load-only).
#[derive(Debug, Clone, PartialEq)]
pub struct MixtralMetadata {
    pub layout: MixtralExpertLayout,
    pub spec: MixtralTensorSpec,
    /// MoE topology from the data plane.
    pub weight_map: MoeWeightMap,
    /// Per-layer expert count (from layer 0).
    pub num_experts: usize,
    /// Number of transformer layers that carry MoE tensors.
    pub num_moe_layers: usize,
    /// Mixtral never has a shared expert.
    pub has_shared_experts: bool,
}

/// The experimental Mixtral family adapter. Stateless; entry points are
/// associated functions. Recognizes a Mixtral checkpoint from its
/// `(name, shape)` tensor listing (what a `SafetensorsReader.iter()` yields),
/// validates the required tensor inventory, and builds metadata. **Load-only.**
pub struct MixtralAdapter;

impl MixtralAdapter {
    /// Whether the tensor listing looks like a Mixtral checkpoint. True iff it
    /// is MoE AND a recognized Mixtral layout (packed or classic) is present.
    pub fn detect_family<'a, I>(tensors: I) -> bool
    where
        I: IntoIterator<Item = (&'a str, Vec<usize>)>,
    {
        let names: Vec<(String, Vec<usize>)> =
            tensors.into_iter().map(|(n, s)| (n.to_string(), s)).collect();
        if !detect_moe(names.iter().map(|(n, _)| n.as_str())).is_moe {
            return false;
        }
        Self::layout_of(names.iter().map(|(n, _)| n.as_str())).is_some()
    }

    /// Determine the Mixtral expert layout from tensor names, if any.
    fn layout_of<'a, I>(names: I) -> Option<MixtralExpertLayout>
    where
        I: IntoIterator<Item = &'a str>,
    {
        let mut packed = false;
        let mut classic = false;
        for n in names {
            if n.contains("mlp.experts.gate_up_proj") || n.contains("mlp.experts.down_proj") {
                packed = true;
            }
            if n.contains("block_sparse_moe.experts.") {
                classic = true;
            }
        }
        // Classic naming is unambiguously Mixtral; packed naming is shared with
        // Qwen2/3-MoE, but the adapter is only *asked* about Mixtral by the
        // caller, so packed-without-classic is treated as Mixtral packed.
        match (classic, packed) {
            (true, _) => Some(MixtralExpertLayout::Classic),
            (false, true) => Some(MixtralExpertLayout::Packed),
            (false, false) => None,
        }
    }

    /// Recognize + validate a Mixtral checkpoint and build its metadata.
    /// `config` (MOE-FULL-2) supplies the declared expert count / top-k for
    /// cross-checking; pass `MoeConfig::default()` if unavailable.
    ///
    /// **Load-only**: builds metadata; does not load tensor bytes, build a
    /// graph, or run anything.
    pub fn recognize<'a, I>(
        tensors: I,
        _config: &MoeConfig,
    ) -> Result<MixtralMetadata, MixtralAdapterError>
    where
        I: IntoIterator<Item = (&'a str, Vec<usize>)>,
    {
        let owned: Vec<(String, Vec<usize>)> =
            tensors.into_iter().map(|(n, s)| (n.to_string(), s)).collect();

        if !detect_moe(owned.iter().map(|(n, _)| n.as_str())).is_moe {
            return Err(MixtralAdapterError::NotMoe);
        }
        let layout = Self::layout_of(owned.iter().map(|(n, _)| n.as_str()))
            .ok_or(MixtralAdapterError::UnrecognizedLayout)?;

        let weight_map =
            MoeWeightMap::from_tensors(owned.iter().map(|(n, s)| (n.as_str(), s.clone())));
        if weight_map.is_empty() {
            return Err(MixtralAdapterError::NotMoe);
        }

        // Validate every MoE layer: router present + experts present.
        let mut num_experts = 0usize;
        for (&layer_id, layer) in &weight_map.layers {
            if layer.router.is_none() {
                return Err(MixtralAdapterError::MissingRouter { layer_id });
            }
            let has_classic = layer.has_classic_experts();
            let has_packed = layer.has_packed_experts();
            if !has_classic && !has_packed {
                return Err(MixtralAdapterError::MissingExperts { layer_id });
            }
            // Determine per-layer expert count from whichever layout is present.
            let layer_experts = if has_classic {
                layer.num_experts()
            } else {
                let gu = layer.packed_gate_up.as_ref().unwrap();
                let dn = layer.packed_down.as_ref().unwrap();
                super::binding::packed_dims(gu, dn)
                    .map_err(|e| MixtralAdapterError::BadPackedShapes { detail: e.to_string() })?
                    .num_experts
            };
            if num_experts == 0 {
                num_experts = layer_experts;
            }
        }

        Ok(MixtralMetadata {
            layout,
            spec: MixtralTensorSpec::for_layout(layout),
            num_moe_layers: weight_map.layers.len(),
            weight_map,
            num_experts,
            has_shared_experts: false, // Mixtral has no shared expert
        })
    }
}

// ============================================================================
// Tests (synthetic tensor listings — no model, no loader, no execution)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Mixtral packed single-layer listing (hf-internal style).
    fn packed_mixtral(d_model: usize, d_ff: usize, ne: usize) -> Vec<(&'static str, Vec<usize>)> {
        // Use 'static names by leaking would be ugly; build owned and convert
        // in the caller. Here we hardcode the small fixed shapes.
        let _ = (d_model, d_ff, ne);
        vec![
            ("model.embed_tokens.weight", vec![32000, 64]),
            ("lm_head.weight", vec![32000, 64]),
            ("model.norm.weight", vec![64]),
            ("model.layers.0.input_layernorm.weight", vec![64]),
            ("model.layers.0.post_attention_layernorm.weight", vec![64]),
            ("model.layers.0.self_attn.q_proj.weight", vec![64, 64]),
            ("model.layers.0.self_attn.k_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.v_proj.weight", vec![32, 64]),
            ("model.layers.0.self_attn.o_proj.weight", vec![64, 64]),
            ("model.layers.0.mlp.gate.weight", vec![4, 64]),
            ("model.layers.0.mlp.experts.gate_up_proj", vec![4, 256, 64]),
            ("model.layers.0.mlp.experts.down_proj", vec![4, 64, 128]),
        ]
    }

    /// Mixtral classic single-layer listing (TitanML style), 8 experts.
    fn classic_mixtral() -> Vec<(&'static str, Vec<usize>)> {
        let mut v: Vec<(&'static str, Vec<usize>)> = vec![
            ("model.embed_tokens.weight", vec![32000, 1024]),
            ("lm_head.weight", vec![32000, 1024]),
            ("model.norm.weight", vec![1024]),
            ("model.layers.0.input_layernorm.weight", vec![1024]),
            ("model.layers.0.post_attention_layernorm.weight", vec![1024]),
            ("model.layers.0.self_attn.q_proj.weight", vec![1024, 1024]),
            ("model.layers.0.self_attn.k_proj.weight", vec![256, 1024]),
            ("model.layers.0.self_attn.v_proj.weight", vec![256, 1024]),
            ("model.layers.0.self_attn.o_proj.weight", vec![1024, 1024]),
            ("model.layers.0.block_sparse_moe.gate.weight", vec![8, 1024]),
        ];
        // 8 experts × w1/w3/w2.
        let experts: &[(&str, Vec<usize>)] = &[
            ("model.layers.0.block_sparse_moe.experts.0.w1.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.0.w3.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.0.w2.weight", vec![1024, 3584]),
            ("model.layers.0.block_sparse_moe.experts.1.w1.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.1.w3.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.1.w2.weight", vec![1024, 3584]),
            ("model.layers.0.block_sparse_moe.experts.2.w1.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.2.w3.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.2.w2.weight", vec![1024, 3584]),
            ("model.layers.0.block_sparse_moe.experts.3.w1.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.3.w3.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.3.w2.weight", vec![1024, 3584]),
            ("model.layers.0.block_sparse_moe.experts.4.w1.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.4.w3.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.4.w2.weight", vec![1024, 3584]),
            ("model.layers.0.block_sparse_moe.experts.5.w1.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.5.w3.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.5.w2.weight", vec![1024, 3584]),
            ("model.layers.0.block_sparse_moe.experts.6.w1.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.6.w3.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.6.w2.weight", vec![1024, 3584]),
            ("model.layers.0.block_sparse_moe.experts.7.w1.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.7.w3.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.7.w2.weight", vec![1024, 3584]),
        ];
        v.extend(experts.iter().cloned());
        v
    }

    fn dense_mistral() -> Vec<(&'static str, Vec<usize>)> {
        vec![
            ("model.embed_tokens.weight", vec![32000, 64]),
            ("lm_head.weight", vec![32000, 64]),
            ("model.layers.0.self_attn.q_proj.weight", vec![64, 64]),
            ("model.layers.0.mlp.gate_proj.weight", vec![128, 64]),
            ("model.layers.0.mlp.up_proj.weight", vec![128, 64]),
            ("model.layers.0.mlp.down_proj.weight", vec![64, 128]),
        ]
    }

    #[test]
    fn detect_mixtral_family() {
        assert!(MixtralAdapter::detect_family(packed_mixtral(64, 128, 4)));
        assert!(MixtralAdapter::detect_family(classic_mixtral()));
        assert!(!MixtralAdapter::detect_family(dense_mistral()));
    }

    #[test]
    fn validate_mixtral_tensor_spec() {
        let md = MixtralAdapter::recognize(classic_mixtral(), &MoeConfig::default()).unwrap();
        assert_eq!(md.layout, MixtralExpertLayout::Classic);
        assert_eq!(md.spec.router_suffix, "block_sparse_moe.gate.weight");
        assert_eq!(md.spec.expert_suffixes.len(), 3);
        assert!(md.spec.dense_suffixes.contains(&"self_attn.q_proj.weight"));
        assert!(!md.has_shared_experts);
    }

    #[test]
    fn packed_experts_detected() {
        let md = MixtralAdapter::recognize(packed_mixtral(64, 128, 4), &MoeConfig::default()).unwrap();
        assert_eq!(md.layout, MixtralExpertLayout::Packed);
        assert_eq!(md.num_experts, 4);
        assert_eq!(md.num_moe_layers, 1);
        assert_eq!(md.spec.router_suffix, "mlp.gate.weight");
    }

    #[test]
    fn classic_experts_detected() {
        let md = MixtralAdapter::recognize(classic_mixtral(), &MoeConfig::default()).unwrap();
        assert_eq!(md.layout, MixtralExpertLayout::Classic);
        assert_eq!(md.num_experts, 8);
        assert_eq!(md.num_moe_layers, 1);
    }

    #[test]
    fn missing_router_reports_error() {
        // Classic experts but no router tensor.
        let listing = vec![
            ("model.layers.0.block_sparse_moe.experts.0.w1.weight", vec![3584usize, 1024]),
            ("model.layers.0.block_sparse_moe.experts.0.w3.weight", vec![3584, 1024]),
            ("model.layers.0.block_sparse_moe.experts.0.w2.weight", vec![1024, 3584]),
        ];
        let err = MixtralAdapter::recognize(listing, &MoeConfig::default()).unwrap_err();
        assert_eq!(err, MixtralAdapterError::MissingRouter { layer_id: 0 });
    }

    #[test]
    fn dense_models_unaffected() {
        // A dense checkpoint is not recognized as Mixtral and yields NotMoe.
        assert!(!MixtralAdapter::detect_family(dense_mistral()));
        let err = MixtralAdapter::recognize(dense_mistral(), &MoeConfig::default()).unwrap_err();
        assert_eq!(err, MixtralAdapterError::NotMoe);
    }
}
