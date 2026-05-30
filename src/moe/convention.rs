//! **MOE-18** — automatic MoE execution-convention selection.
//!
//! MOE-17 added two execution conventions (`Atenia`, `HuggingFaceQwen`) but
//! required the caller to pick one. MOE-18 infers the right convention from
//! **metadata that already exists** — no `config.json` parsing, no new
//! metadata, no change to the MoE math.
//!
//! ## Signal
//!
//! The discriminator is the presence of a **`shared_expert_gate`** tensor:
//!
//! * Qwen-MoE ships `…mlp.shared_expert_gate.weight` (the sigmoid gate that
//!   scales the shared expert). Its presence means the checkpoint follows the
//!   HF Qwen convention (`norm_topk_prob = false` + sigmoid-gated shared
//!   expert) → [`MoeExecutionConvention::HuggingFaceQwen`].
//! * Mixtral (and any checkpoint without a shared-expert gate) has no such
//!   tensor → [`MoeExecutionConvention::Atenia`] (renormalise, ungated), which
//!   already matches the HF Mixtral block exactly.
//!
//! This signal is exactly what the `HuggingFaceQwen` forward needs (the gate
//! weight), so detection and execution use the same fact. The selection
//! changes **which** correct forward runs; it does not change any forward's
//! math (MOE-17 numerics are preserved bit-for-bit per convention).

use super::data_plane::MoeWeightMap;
use super::layer::MoeExecutionConvention;

/// Resolves the [`MoeExecutionConvention`] for a checkpoint from its MoE
/// metadata. Stateless.
pub struct MoeConventionResolver;

impl MoeConventionResolver {
    /// Infer the convention from a [`MoeWeightMap`]: `HuggingFaceQwen` if any
    /// layer carries a `shared_expert_gate` tensor, else `Atenia`.
    pub fn from_weight_map(map: &MoeWeightMap) -> MoeExecutionConvention {
        let has_shared_gate = map
            .layers
            .values()
            .any(|l| l.shared.iter().any(|e| e.name.contains("shared_expert_gate")));
        Self::from_shared_gate_present(has_shared_gate)
    }

    /// Infer the convention from whether a shared-expert sigmoid gate is
    /// present (the assembly-level signal `RealMoeLayer.shared_gate`).
    pub fn from_shared_gate_present(present: bool) -> MoeExecutionConvention {
        if present {
            MoeExecutionConvention::HuggingFaceQwen
        } else {
            MoeExecutionConvention::Atenia
        }
    }
}

// ============================================================================
// Tests (synthetic metadata — no model, no fixtures)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn map_of(names_shapes: &[(&str, Vec<usize>)]) -> MoeWeightMap {
        MoeWeightMap::from_tensors(names_shapes.iter().map(|(n, s)| (*n, s.clone())))
    }

    #[test]
    fn qwen_with_shared_gate_resolves_huggingface() {
        let map = map_of(&[
            ("model.layers.0.mlp.gate.weight", vec![4, 8]),
            ("model.layers.0.mlp.experts.0.gate_proj.weight", vec![16, 8]),
            ("model.layers.0.mlp.experts.0.up_proj.weight", vec![16, 8]),
            ("model.layers.0.mlp.experts.0.down_proj.weight", vec![8, 16]),
            ("model.layers.0.mlp.shared_expert.gate_proj.weight", vec![16, 8]),
            ("model.layers.0.mlp.shared_expert_gate.weight", vec![1, 8]),
        ]);
        assert_eq!(
            MoeConventionResolver::from_weight_map(&map),
            MoeExecutionConvention::HuggingFaceQwen
        );
    }

    #[test]
    fn mixtral_without_shared_gate_resolves_atenia() {
        let map = map_of(&[
            ("model.layers.0.block_sparse_moe.gate.weight", vec![8, 16]),
            ("model.layers.0.block_sparse_moe.experts.0.w1.weight", vec![32, 16]),
            ("model.layers.0.block_sparse_moe.experts.0.w3.weight", vec![32, 16]),
            ("model.layers.0.block_sparse_moe.experts.0.w2.weight", vec![16, 32]),
        ]);
        assert_eq!(
            MoeConventionResolver::from_weight_map(&map),
            MoeExecutionConvention::Atenia
        );
    }

    #[test]
    fn shared_expert_without_gate_resolves_atenia() {
        // A shared expert but NO gate tensor → Atenia (ungated).
        let map = map_of(&[
            ("model.layers.0.mlp.gate.weight", vec![4, 8]),
            ("model.layers.0.mlp.experts.0.gate_proj.weight", vec![16, 8]),
            ("model.layers.0.mlp.experts.0.up_proj.weight", vec![16, 8]),
            ("model.layers.0.mlp.experts.0.down_proj.weight", vec![8, 16]),
            ("model.layers.0.mlp.shared_expert.gate_proj.weight", vec![16, 8]),
        ]);
        assert_eq!(
            MoeConventionResolver::from_weight_map(&map),
            MoeExecutionConvention::Atenia
        );
    }

    #[test]
    fn shared_gate_present_helper() {
        assert_eq!(
            MoeConventionResolver::from_shared_gate_present(true),
            MoeExecutionConvention::HuggingFaceQwen
        );
        assert_eq!(
            MoeConventionResolver::from_shared_gate_present(false),
            MoeExecutionConvention::Atenia
        );
    }
}
