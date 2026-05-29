//! **MOE-11** — Real MoE layer assembly (experimental, CPU-only).
//!
//! MOE-10 bound real checkpoint bytes to individual experts and ran the
//! sparse forward over them. MOE-11 assembles a **complete** MoE layer:
//!
//! ```text
//!   MoeLayerConfig (fixture)
//!     + MoeWeightMap (MOE-9 metadata)  + byte resolver (MOE-10)
//!     → router + routed experts  (build_real_layer, MOE-10)
//!     → optional shared expert   (rebuilt from the shared tensor set)
//!     → RealMoeLayer
//!     → forward: route → top-k → dispatch → combine → (+ shared) → output
//! ```
//!
//! ## What this is and is NOT
//!
//! * It **does** assemble a runnable single MoE layer (router + N routed
//!   experts + an optional shared expert) from real tensor data, and run a
//!   correctness-first CPU forward over it.
//! * It does **not** load a full model, parse `config.json`, build a graph /
//!   transformer, or lift the MOE-2 loader fail-loud guard. `MoeLayerConfig`
//!   is a **fixture** struct supplied by the caller, not parsed from a
//!   checkpoint. A real MoE checkpoint still refuses to load as a model.
//! * No Mixtral / Qwen-MoE end-to-end support is claimed: this validates the
//!   *layer-assembly mechanism* on small synthetic checkpoints that use real
//!   Mixtral / Qwen-MoE tensor names.

use super::binding::{build_real_layer, MoeBindingError, RealExpertTensorBinding};
use super::data_plane::MoeWeightMap;
use super::dense::{MoeDenseExpert, MoeDenseLayer};
use super::sparse::MoeSparseError;

/// Experimental MoE-layer configuration (a **fixture**, not parsed from a
/// real `config.json`). Describes the topology a caller expects so assembly
/// can cross-check it against the resolved tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeLayerConfig {
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub has_shared_expert: bool,
    pub d_model: usize,
    pub d_ff: usize,
}

/// Errors from validating a [`MoeLayerConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoeConfigError {
    ZeroExperts,
    ZeroExpertsPerToken,
    ExpertsPerTokenExceedsCount { per_token: usize, total: usize },
    ZeroDModel,
    ZeroDFf,
}

impl std::fmt::Display for MoeConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoeConfigError::ZeroExperts => write!(f, "moe-config: num_experts is 0"),
            MoeConfigError::ZeroExpertsPerToken => {
                write!(f, "moe-config: experts_per_token is 0")
            }
            MoeConfigError::ExpertsPerTokenExceedsCount { per_token, total } => write!(
                f,
                "moe-config: experts_per_token ({per_token}) exceeds num_experts ({total})"
            ),
            MoeConfigError::ZeroDModel => write!(f, "moe-config: d_model is 0"),
            MoeConfigError::ZeroDFf => write!(f, "moe-config: d_ff is 0"),
        }
    }
}

impl std::error::Error for MoeConfigError {}

impl MoeLayerConfig {
    /// Construct + validate a fixture config.
    pub fn new(
        num_experts: usize,
        experts_per_token: usize,
        has_shared_expert: bool,
        d_model: usize,
        d_ff: usize,
    ) -> Result<Self, MoeConfigError> {
        let cfg = Self {
            num_experts,
            experts_per_token,
            has_shared_expert,
            d_model,
            d_ff,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Structural validation. Pure; no I/O.
    pub fn validate(&self) -> Result<(), MoeConfigError> {
        if self.num_experts == 0 {
            return Err(MoeConfigError::ZeroExperts);
        }
        if self.experts_per_token == 0 {
            return Err(MoeConfigError::ZeroExpertsPerToken);
        }
        if self.experts_per_token > self.num_experts {
            return Err(MoeConfigError::ExpertsPerTokenExceedsCount {
                per_token: self.experts_per_token,
                total: self.num_experts,
            });
        }
        if self.d_model == 0 {
            return Err(MoeConfigError::ZeroDModel);
        }
        if self.d_ff == 0 {
            return Err(MoeConfigError::ZeroDFf);
        }
        Ok(())
    }
}

/// Errors from assembling / running a [`RealMoeLayer`].
#[derive(Debug, Clone, PartialEq)]
pub enum MoeLayerError {
    /// Invalid fixture config.
    Config(MoeConfigError),
    /// Error resolving / binding the routed experts or router.
    Binding(MoeBindingError),
    /// Error from the sparse forward (routing / dispatch).
    Sparse(MoeSparseError),
    /// The resolved topology disagreed with the supplied config.
    ConfigMismatch { detail: String },
    /// `has_shared_expert` was set but no shared-expert tensors were found.
    SharedExpertMissing { layer_id: usize },
    /// The shared expert was missing one of gate/up/down (by name).
    SharedExpertIncomplete { layer_id: usize },
    /// **MOE-15** — a layer has BOTH classic per-expert and packed/fused
    /// expert tensors; the format is ambiguous so assembly refuses.
    MixedExpertFormat { layer_id: usize },
}

impl std::fmt::Display for MoeLayerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoeLayerError::Config(e) => write!(f, "moe-layer: {e}"),
            MoeLayerError::Binding(e) => write!(f, "moe-layer: {e}"),
            MoeLayerError::Sparse(e) => write!(f, "moe-layer: {e}"),
            MoeLayerError::ConfigMismatch { detail } => {
                write!(f, "moe-layer: config mismatch: {detail}")
            }
            MoeLayerError::SharedExpertMissing { layer_id } => write!(
                f,
                "moe-layer: layer {layer_id} config has_shared_expert=true but no shared tensors found"
            ),
            MoeLayerError::SharedExpertIncomplete { layer_id } => write!(
                f,
                "moe-layer: layer {layer_id} shared expert is missing a gate/up/down projection"
            ),
            MoeLayerError::MixedExpertFormat { layer_id } => write!(
                f,
                "moe-layer: layer {layer_id} has both classic and packed expert tensors (ambiguous)"
            ),
        }
    }
}

impl std::error::Error for MoeLayerError {}

impl From<MoeConfigError> for MoeLayerError {
    fn from(e: MoeConfigError) -> Self {
        MoeLayerError::Config(e)
    }
}
impl From<MoeBindingError> for MoeLayerError {
    fn from(e: MoeBindingError) -> Self {
        MoeLayerError::Binding(e)
    }
}
impl From<MoeSparseError> for MoeLayerError {
    fn from(e: MoeSparseError) -> Self {
        MoeLayerError::Sparse(e)
    }
}

/// A fully assembled, runnable real MoE layer: router + routed experts +
/// an optional shared expert + the fixture config.
#[derive(Debug, Clone)]
pub struct RealMoeLayer {
    pub config: MoeLayerConfig,
    /// Router + routed experts (MOE-3/4 executor; `forward_sparse` lives
    /// here). The conceptual top-k stored equals `config.experts_per_token`.
    pub routed: MoeDenseLayer,
    /// Optional shared expert that runs on **every** token and is added to
    /// the routed combine (Qwen-MoE / DeepSeek-MoE convention).
    pub shared: Option<MoeDenseExpert>,
}

/// Build the shared expert (if present) from a layer's shared tensor set.
///
/// The MOE-2 classifier collapses every `shared_expert*` tensor to the role
/// `MoeSharedExpert`, so we recover gate/up/down by **name substring**
/// (`gate_proj` / `up_proj` / `down_proj`). Non-projection shared tensors
/// (e.g. a `shared_expert_gate` router) are ignored — only the three SwiGLU
/// projections are assembled.
fn build_shared_expert<F>(
    map: &MoeWeightMap,
    layer_id: usize,
    resolve: &F,
) -> Result<MoeDenseExpert, MoeLayerError>
where
    F: Fn(&str) -> Option<Vec<f32>>,
{
    let layer = map
        .layer(layer_id)
        .ok_or(MoeLayerError::SharedExpertMissing { layer_id })?;
    if layer.shared.is_empty() {
        return Err(MoeLayerError::SharedExpertMissing { layer_id });
    }

    let find = |needle: &str| layer.shared.iter().find(|e| e.name.contains(needle));
    let (gate, up, down) = match (
        find(".gate_proj."),
        find(".up_proj."),
        find(".down_proj."),
    ) {
        (Some(g), Some(u), Some(d)) => (g, u, d),
        _ => return Err(MoeLayerError::SharedExpertIncomplete { layer_id }),
    };

    // Reuse the MOE-10 expert resolver by wrapping the three entries in an
    // ExpertTensors and binding it. This keeps shape inference + byte
    // resolution + validation in one place.
    let tensors = super::data_plane::ExpertTensors {
        gate: Some(gate.clone()),
        up: Some(up.clone()),
        down: Some(down.clone()),
    };
    let binding = RealExpertTensorBinding::resolve(layer_id, usize::MAX, &tensors, resolve)?;
    Ok(binding.expert)
}

impl RealMoeLayer {
    /// Assemble a complete real MoE layer for `layer_id` from the metadata
    /// map + byte resolver, cross-checking against the fixture `config`.
    ///
    /// `config.experts_per_token` is carried into the routed layer's
    /// conceptual top-k and used as the default `k` by [`Self::forward`].
    pub fn assemble<F>(
        map: &MoeWeightMap,
        layer_id: usize,
        config: MoeLayerConfig,
        resolve: &F,
    ) -> Result<Self, MoeLayerError>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        config.validate()?;

        // MOE-15: select the expert format. Classic per-expert is preferred;
        // packed/fused is used when classic is absent; if BOTH are present
        // the checkpoint is ambiguous and we refuse rather than guess.
        let layer_meta = map
            .layer(layer_id)
            .ok_or(MoeLayerError::Binding(MoeBindingError::LayerNotFound { layer_id }))?;
        let has_classic = layer_meta.has_classic_experts();
        let has_packed = layer_meta.has_packed_experts();

        // Router + routed experts via the appropriate binding.
        let routed = if has_classic && has_packed {
            return Err(MoeLayerError::MixedExpertFormat { layer_id });
        } else if has_classic {
            build_real_layer(map, layer_id, config.experts_per_token, resolve)?
        } else if has_packed {
            super::binding::build_packed_layer(map, layer_id, config.experts_per_token, resolve)?
        } else {
            // No experts in either format → surface the classic NoExperts error.
            build_real_layer(map, layer_id, config.experts_per_token, resolve)?
        };

        // Cross-check resolved topology against the fixture config.
        if routed.num_experts() != config.num_experts {
            return Err(MoeLayerError::ConfigMismatch {
                detail: format!(
                    "resolved {} experts but config.num_experts = {}",
                    routed.num_experts(),
                    config.num_experts
                ),
            });
        }
        if routed.d_model != config.d_model || routed.d_ff != config.d_ff {
            return Err(MoeLayerError::ConfigMismatch {
                detail: format!(
                    "resolved (d_model={}, d_ff={}) but config (d_model={}, d_ff={})",
                    routed.d_model, routed.d_ff, config.d_model, config.d_ff
                ),
            });
        }

        let shared = if config.has_shared_expert {
            let se = build_shared_expert(map, layer_id, resolve)?;
            if se.d_model != config.d_model {
                return Err(MoeLayerError::ConfigMismatch {
                    detail: format!(
                        "shared expert d_model={} != config d_model={}",
                        se.d_model, config.d_model
                    ),
                });
            }
            Some(se)
        } else {
            None
        };

        Ok(Self { config, routed, shared })
    }

    /// Number of routed experts.
    pub fn num_experts(&self) -> usize {
        self.routed.num_experts()
    }

    /// Whether a shared expert is assembled.
    pub fn has_shared_expert(&self) -> bool {
        self.shared.is_some()
    }

    /// **Routed-only** forward: route → top-k(`experts_per_token`) →
    /// dispatch selected experts → renormalised combine. This is exactly the
    /// MOE-4 sparse reference path; used as the numeric oracle.
    pub fn forward_routed(&self, x: &[f32]) -> Result<Vec<f32>, MoeLayerError> {
        let out = self.routed.forward_sparse(x, self.config.experts_per_token)?;
        Ok(out.output)
    }

    /// **Full layer forward**: routed sparse combine, plus the shared expert
    /// (if present) added on top. CPU-only, single token, single layer.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, MoeLayerError> {
        let mut out = self.forward_routed(x)?;
        if let Some(se) = &self.shared {
            let s = se
                .forward(x)
                .map_err(|e| MoeLayerError::Binding(MoeBindingError::Dense(e)))?;
            for d in 0..self.config.d_model {
                out[d] += s[d];
            }
        }
        Ok(out)
    }
}

// ============================================================================
// Tests (synthetic in-memory tensors — no model, no loader, no graph)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn seeded(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as u32;
            out.push((u as f32 / u32::MAX as f32) * 2.0 - 1.0);
        }
        out
    }

    fn resolver(store: &HashMap<String, Vec<f32>>) -> impl Fn(&str) -> Option<Vec<f32>> + '_ {
        move |name: &str| store.get(name).cloned()
    }

    /// Mixtral-style (no shared expert): `n` experts, dims `(d_model, d_ff)`.
    fn mixtral(
        n: usize,
        d_model: usize,
        d_ff: usize,
    ) -> (MoeWeightMap, HashMap<String, Vec<f32>>) {
        let mut ns: Vec<(String, Vec<usize>)> = Vec::new();
        let mut store: HashMap<String, Vec<f32>> = HashMap::new();
        let router = "model.layers.0.block_sparse_moe.gate.weight".to_string();
        ns.push((router.clone(), vec![n, d_model]));
        store.insert(router, seeded(1, n * d_model));
        for e in 0..n {
            let base = 100 + e as u64;
            let g = format!("model.layers.0.block_sparse_moe.experts.{e}.w1.weight");
            let u = format!("model.layers.0.block_sparse_moe.experts.{e}.w3.weight");
            let d = format!("model.layers.0.block_sparse_moe.experts.{e}.w2.weight");
            ns.push((g.clone(), vec![d_ff, d_model]));
            ns.push((u.clone(), vec![d_ff, d_model]));
            ns.push((d.clone(), vec![d_model, d_ff]));
            store.insert(g, seeded(base * 10 + 1, d_ff * d_model));
            store.insert(u, seeded(base * 10 + 2, d_ff * d_model));
            store.insert(d, seeded(base * 10 + 3, d_model * d_ff));
        }
        let map = MoeWeightMap::from_tensors(ns.iter().map(|(n, s)| (n.as_str(), s.clone())));
        (map, store)
    }

    /// Qwen-MoE-style with a shared expert (shared d_ff may differ).
    fn qwen_moe(
        n: usize,
        d_model: usize,
        d_ff: usize,
        shared_ff: usize,
    ) -> (MoeWeightMap, HashMap<String, Vec<f32>>) {
        let mut ns: Vec<(String, Vec<usize>)> = Vec::new();
        let mut store: HashMap<String, Vec<f32>> = HashMap::new();
        let router = "model.layers.0.mlp.gate.weight".to_string();
        ns.push((router.clone(), vec![n, d_model]));
        store.insert(router, seeded(1, n * d_model));
        for e in 0..n {
            let base = 200 + e as u64;
            let g = format!("model.layers.0.mlp.experts.{e}.gate_proj.weight");
            let u = format!("model.layers.0.mlp.experts.{e}.up_proj.weight");
            let d = format!("model.layers.0.mlp.experts.{e}.down_proj.weight");
            ns.push((g.clone(), vec![d_ff, d_model]));
            ns.push((u.clone(), vec![d_ff, d_model]));
            ns.push((d.clone(), vec![d_model, d_ff]));
            store.insert(g, seeded(base * 10 + 1, d_ff * d_model));
            store.insert(u, seeded(base * 10 + 2, d_ff * d_model));
            store.insert(d, seeded(base * 10 + 3, d_model * d_ff));
        }
        // Shared expert projections.
        let sg = "model.layers.0.mlp.shared_expert.gate_proj.weight".to_string();
        let su = "model.layers.0.mlp.shared_expert.up_proj.weight".to_string();
        let sd = "model.layers.0.mlp.shared_expert.down_proj.weight".to_string();
        ns.push((sg.clone(), vec![shared_ff, d_model]));
        ns.push((su.clone(), vec![shared_ff, d_model]));
        ns.push((sd.clone(), vec![d_model, shared_ff]));
        store.insert(sg, seeded(9001, shared_ff * d_model));
        store.insert(su, seeded(9002, shared_ff * d_model));
        store.insert(sd, seeded(9003, d_model * shared_ff));
        let map = MoeWeightMap::from_tensors(ns.iter().map(|(n, s)| (n.as_str(), s.clone())));
        (map, store)
    }

    #[test]
    fn config_validates() {
        assert!(MoeLayerConfig::new(4, 2, false, 8, 16).is_ok());
        assert_eq!(
            MoeLayerConfig::new(0, 2, false, 8, 16).unwrap_err(),
            MoeConfigError::ZeroExperts
        );
        assert!(matches!(
            MoeLayerConfig::new(2, 4, false, 8, 16).unwrap_err(),
            MoeConfigError::ExpertsPerTokenExceedsCount { .. }
        ));
    }

    #[test]
    fn assembles_mixtral_layer_no_shared() {
        let (map, store) = mixtral(4, 8, 16);
        let resolve = resolver(&store);
        let cfg = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
        let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
        assert_eq!(layer.num_experts(), 4);
        assert!(!layer.has_shared_expert());
    }

    #[test]
    fn assembled_forward_matches_sparse_reference_no_shared() {
        let (map, store) = mixtral(4, 8, 16);
        let resolve = resolver(&store);
        let cfg = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
        let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
        let x = seeded(7, 8);
        // With no shared expert, full forward == routed sparse reference.
        let full = layer.forward(&x).unwrap();
        let reference = layer.routed.forward_sparse(&x, 2).unwrap().output;
        for d in 0..8 {
            assert!((full[d] - reference[d]).abs() < 1e-5);
        }
    }

    #[test]
    fn assembles_qwen_layer_with_shared_expert() {
        let (map, store) = qwen_moe(3, 4, 6, 10);
        let resolve = resolver(&store);
        let cfg = MoeLayerConfig::new(3, 2, true, 4, 6).unwrap();
        let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
        assert!(layer.has_shared_expert());
        let se = layer.shared.as_ref().unwrap();
        assert_eq!(se.d_model, 4);
        assert_eq!(se.d_ff, 10); // shared expert has its own (larger) d_ff
    }

    #[test]
    fn shared_expert_adds_on_top_of_routed() {
        let (map, store) = qwen_moe(3, 4, 6, 10);
        let resolve = resolver(&store);
        let cfg = MoeLayerConfig::new(3, 2, true, 4, 6).unwrap();
        let layer = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap();
        let x = seeded(13, 4);
        let full = layer.forward(&x).unwrap();
        let routed = layer.forward_routed(&x).unwrap();
        let shared = layer.shared.as_ref().unwrap().forward(&x).unwrap();
        for d in 0..4 {
            assert!((full[d] - (routed[d] + shared[d])).abs() < 1e-5);
        }
        // The shared contribution is non-trivial, so full != routed.
        assert!((0..4).any(|d| (full[d] - routed[d]).abs() > 1e-6));
    }

    #[test]
    fn config_mismatch_is_rejected() {
        let (map, store) = mixtral(4, 8, 16);
        let resolve = resolver(&store);
        // Claim 8 experts but only 4 exist.
        let cfg = MoeLayerConfig::new(8, 2, false, 8, 16).unwrap();
        let err = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap_err();
        assert!(matches!(err, MoeLayerError::ConfigMismatch { .. }));
    }

    #[test]
    fn missing_shared_expert_is_reported() {
        // Mixtral layer has no shared expert, but config requests one.
        let (map, store) = mixtral(4, 8, 16);
        let resolve = resolver(&store);
        let cfg = MoeLayerConfig::new(4, 2, true, 8, 16).unwrap();
        let err = RealMoeLayer::assemble(&map, 0, cfg, &resolve).unwrap_err();
        assert!(matches!(err, MoeLayerError::SharedExpertMissing { .. }));
    }
}
