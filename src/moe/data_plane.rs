//! **MOE-9** — real MoE data-plane metadata + expert registry.
//!
//! Turns a checkpoint's `(tensor_name, shape)` listing into a structured
//! map of MoE weights: router per layer, expert (gate/up/down) per
//! `(layer, expert)`, and shared experts. It builds on the MOE-2 name
//! classifier (`detect::classify_tensor_name`) so detection and mapping
//! share one source of truth.
//!
//! ## Scope (MOE-9)
//!
//! This is the **metadata / lookup** layer only. It does NOT:
//! * load tensor *data* (only names + shapes are consumed);
//! * touch the productive loader load path — the MOE-2 fail-loud guard is
//!   unchanged, so a real MoE checkpoint still refuses to load as a model;
//! * touch the Adapter Toolkit or claim real Mixtral / Qwen-MoE support.
//!
//! It is the substrate a future milestone will feed with real tensor data
//! to actually run a MoE model. For now it lets the graph layer *see* the
//! structure of expert weights (which experts exist, their shapes, their
//! names) without bespoke per-call parsing.
//!
//! Experimental, CPU-only, deterministic (BTreeMap-backed).

use std::collections::BTreeMap;

use super::detect::{classify_tensor_name, TensorRole};

/// One MoE tensor's structured metadata (no data — name + shape + role).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeTensorEntry {
    pub name: String,
    pub layer_id: Option<usize>,
    pub expert_id: Option<usize>,
    pub role: TensorRole,
    pub shape: Vec<usize>,
}

/// The three projections of one expert (Mixtral `w1/w3/w2`,
/// Qwen/DeepSeek `gate_proj/up_proj/down_proj`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExpertTensors {
    pub gate: Option<MoeTensorEntry>,
    pub up: Option<MoeTensorEntry>,
    pub down: Option<MoeTensorEntry>,
}

impl ExpertTensors {
    /// Whether all three projections are present.
    pub fn is_complete(&self) -> bool {
        self.gate.is_some() && self.up.is_some() && self.down.is_some()
    }
}

/// All MoE weights for one transformer layer.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MoeLayerMap {
    pub router: Option<MoeTensorEntry>,
    /// Expert id → its three projections (sorted by expert id). Classic
    /// per-expert format.
    pub experts: BTreeMap<usize, ExpertTensors>,
    /// Shared-expert tensors (Qwen-MoE / DeepSeek), if any.
    pub shared: Vec<MoeTensorEntry>,
    /// **MOE-15** — packed/fused gate+up tensor (3-D, all experts stacked),
    /// if the checkpoint uses the packed format.
    pub packed_gate_up: Option<MoeTensorEntry>,
    /// **MOE-15** — packed/fused down tensor (3-D, all experts stacked).
    pub packed_down: Option<MoeTensorEntry>,
}

impl MoeLayerMap {
    /// Number of *classic* per-expert experts mapped for this layer.
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Whether this layer has classic per-expert tensors.
    pub fn has_classic_experts(&self) -> bool {
        !self.experts.is_empty()
    }

    /// **MOE-15** — whether this layer has both packed expert tensors
    /// (`gate_up_proj` + `down_proj`).
    pub fn has_packed_experts(&self) -> bool {
        self.packed_gate_up.is_some() && self.packed_down.is_some()
    }
}

/// The structured MoE weight map for a whole checkpoint, plus expert
/// lookup. Acts as the **expert registry** the graph layer queries.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MoeWeightMap {
    /// Layer id → that layer's MoE weights (sorted by layer id).
    pub layers: BTreeMap<usize, MoeLayerMap>,
}

impl MoeWeightMap {
    /// Build the map from an iterator of `(tensor_name, shape)`. Non-MoE
    /// tensors (attention, dense MLP, embeddings, norms) are ignored.
    /// Pure; consumes only names + shapes (what a reader's `iter()` yields).
    pub fn from_tensors<'a, I>(tensors: I) -> Self
    where
        I: IntoIterator<Item = (&'a str, Vec<usize>)>,
    {
        let mut map = MoeWeightMap::default();
        for (name, shape) in tensors {
            let info = classify_tensor_name(name);
            if !info.role.is_moe() {
                continue;
            }
            let Some(layer_id) = info.layer_id else {
                // A MoE tensor without a parseable layer id is unusual;
                // skip it rather than guess.
                continue;
            };
            let entry = MoeTensorEntry {
                name: name.to_string(),
                layer_id: info.layer_id,
                expert_id: info.expert_id,
                role: info.role,
                shape,
            };
            let layer = map.layers.entry(layer_id).or_default();
            match info.role {
                TensorRole::MoeRouter => layer.router = Some(entry),
                TensorRole::MoeSharedExpert => layer.shared.push(entry),
                TensorRole::MoePackedGateUp => layer.packed_gate_up = Some(entry),
                TensorRole::MoePackedDown => layer.packed_down = Some(entry),
                TensorRole::MoeExpertGate
                | TensorRole::MoeExpertUp
                | TensorRole::MoeExpertDown => {
                    if let Some(eid) = info.expert_id {
                        let ex = layer.experts.entry(eid).or_default();
                        match info.role {
                            TensorRole::MoeExpertGate => ex.gate = Some(entry),
                            TensorRole::MoeExpertUp => ex.up = Some(entry),
                            TensorRole::MoeExpertDown => ex.down = Some(entry),
                            _ => unreachable!(),
                        }
                    }
                }
                _ => {}
            }
        }
        map
    }

    /// `true` if no MoE tensors were found (a dense checkpoint).
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Lookup one layer's MoE map.
    pub fn layer(&self, layer_id: usize) -> Option<&MoeLayerMap> {
        self.layers.get(&layer_id)
    }

    /// Lookup the router tensor for a layer.
    pub fn router_weight(&self, layer_id: usize) -> Option<&MoeTensorEntry> {
        self.layers.get(&layer_id).and_then(|l| l.router.as_ref())
    }

    /// Lookup one expert's three projections.
    pub fn expert(&self, layer_id: usize, expert_id: usize) -> Option<&ExpertTensors> {
        self.layers.get(&layer_id).and_then(|l| l.experts.get(&expert_id))
    }

    /// Number of experts mapped for a layer.
    pub fn num_experts(&self, layer_id: usize) -> usize {
        self.layers.get(&layer_id).map_or(0, |l| l.num_experts())
    }

    /// Total expert *projection* tensors across all layers (gate+up+down).
    pub fn total_expert_tensors(&self) -> usize {
        self.layers
            .values()
            .flat_map(|l| l.experts.values())
            .map(|e| {
                e.gate.is_some() as usize + e.up.is_some() as usize + e.down.is_some() as usize
            })
            .sum()
    }

    /// Whether every mapped expert has all three projections.
    pub fn all_experts_complete(&self) -> bool {
        self.layers
            .values()
            .flat_map(|l| l.experts.values())
            .all(|e| e.is_complete())
    }
}

// ============================================================================
// Tests (synthetic names + shapes — no model, no loader, no data)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Mixtral-style 2-expert single-layer listing.
    fn mixtral_tensors() -> Vec<(&'static str, Vec<usize>)> {
        vec![
            ("model.layers.0.self_attn.q_proj.weight", vec![4096, 4096]),
            ("model.layers.0.block_sparse_moe.gate.weight", vec![8, 4096]),
            ("model.layers.0.block_sparse_moe.experts.0.w1.weight", vec![14336, 4096]),
            ("model.layers.0.block_sparse_moe.experts.0.w3.weight", vec![14336, 4096]),
            ("model.layers.0.block_sparse_moe.experts.0.w2.weight", vec![4096, 14336]),
            ("model.layers.0.block_sparse_moe.experts.1.w1.weight", vec![14336, 4096]),
            ("model.layers.0.block_sparse_moe.experts.1.w3.weight", vec![14336, 4096]),
            ("model.layers.0.block_sparse_moe.experts.1.w2.weight", vec![4096, 14336]),
        ]
    }

    /// Qwen-MoE-style listing with a shared expert.
    fn qwen_moe_tensors() -> Vec<(&'static str, Vec<usize>)> {
        vec![
            ("model.layers.0.mlp.gate.weight", vec![60, 2048]),
            ("model.layers.0.mlp.experts.0.gate_proj.weight", vec![1408, 2048]),
            ("model.layers.0.mlp.experts.0.up_proj.weight", vec![1408, 2048]),
            ("model.layers.0.mlp.experts.0.down_proj.weight", vec![2048, 1408]),
            ("model.layers.0.mlp.shared_expert.gate_proj.weight", vec![5632, 2048]),
        ]
    }

    #[test]
    fn mixtral_tensor_mapping() {
        let map = MoeWeightMap::from_tensors(mixtral_tensors());
        assert!(!map.is_empty());
        assert_eq!(map.num_experts(0), 2);
        // Router present with [8, 4096].
        let router = map.router_weight(0).unwrap();
        assert_eq!(router.role, TensorRole::MoeRouter);
        assert_eq!(router.shape, vec![8, 4096]);
        // Expert 0 complete (w1=gate, w3=up, w2=down).
        let e0 = map.expert(0, 0).unwrap();
        assert!(e0.is_complete());
        assert_eq!(e0.gate.as_ref().unwrap().shape, vec![14336, 4096]);
        assert_eq!(e0.down.as_ref().unwrap().shape, vec![4096, 14336]);
        assert!(map.all_experts_complete());
        assert_eq!(map.total_expert_tensors(), 6);
    }

    #[test]
    fn qwen_moe_tensor_mapping() {
        let map = MoeWeightMap::from_tensors(qwen_moe_tensors());
        assert_eq!(map.num_experts(0), 1);
        assert!(map.router_weight(0).is_some());
        let e0 = map.expert(0, 0).unwrap();
        assert!(e0.is_complete());
        // Shared expert captured.
        let layer = map.layer(0).unwrap();
        assert_eq!(layer.shared.len(), 1);
        assert_eq!(layer.shared[0].role, TensorRole::MoeSharedExpert);
    }

    #[test]
    fn expert_registry_lookup() {
        let map = MoeWeightMap::from_tensors(mixtral_tensors());
        assert!(map.expert(0, 0).is_some());
        assert!(map.expert(0, 1).is_some());
        // Out-of-range expert / layer return None.
        assert!(map.expert(0, 99).is_none());
        assert!(map.expert(5, 0).is_none());
    }

    #[test]
    fn router_weight_lookup() {
        let map = MoeWeightMap::from_tensors(mixtral_tensors());
        assert_eq!(map.router_weight(0).unwrap().name, "model.layers.0.block_sparse_moe.gate.weight");
        assert!(map.router_weight(7).is_none());
    }

    #[test]
    fn expert_metadata_roundtrip() {
        let map = MoeWeightMap::from_tensors(mixtral_tensors());
        let e1 = map.expert(0, 1).unwrap();
        let gate = e1.gate.as_ref().unwrap();
        assert_eq!(gate.layer_id, Some(0));
        assert_eq!(gate.expert_id, Some(1));
        assert_eq!(gate.role, TensorRole::MoeExpertGate);
        assert_eq!(gate.name, "model.layers.0.block_sparse_moe.experts.1.w1.weight");
    }

    #[test]
    fn dense_models_produce_empty_map() {
        // Dense Llama / DeepSeek-distill names → no MoE tensors.
        let dense = vec![
            ("model.embed_tokens.weight", vec![32000, 4096]),
            ("model.layers.0.self_attn.q_proj.weight", vec![4096, 4096]),
            ("model.layers.0.mlp.gate_proj.weight", vec![11008, 4096]),
            ("model.layers.0.mlp.up_proj.weight", vec![11008, 4096]),
            ("model.layers.0.mlp.down_proj.weight", vec![4096, 11008]),
            ("lm_head.weight", vec![32000, 4096]),
        ];
        let map = MoeWeightMap::from_tensors(dense);
        assert!(map.is_empty(), "dense checkpoint must produce no MoE map");
        assert_eq!(map.total_expert_tensors(), 0);
    }

    #[test]
    fn multi_layer_mapping_is_sorted_and_complete() {
        let mut tensors = mixtral_tensors();
        // Add a second layer's experts.
        tensors.push(("model.layers.1.block_sparse_moe.gate.weight", vec![8, 4096]));
        tensors.push(("model.layers.1.block_sparse_moe.experts.0.w1.weight", vec![14336, 4096]));
        tensors.push(("model.layers.1.block_sparse_moe.experts.0.w3.weight", vec![14336, 4096]));
        tensors.push(("model.layers.1.block_sparse_moe.experts.0.w2.weight", vec![4096, 14336]));
        let map = MoeWeightMap::from_tensors(tensors);
        let layer_ids: Vec<usize> = map.layers.keys().copied().collect();
        assert_eq!(layer_ids, vec![0, 1]); // BTreeMap → sorted
        assert_eq!(map.num_experts(1), 1);
    }
}
