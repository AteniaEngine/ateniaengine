//! **MOE-12** — Multi-layer real MoE stack (experimental, CPU-only).
//!
//! MOE-11 assembled a single complete MoE layer (router + routed experts +
//! optional shared expert). MOE-12 composes several such layers into a
//! sequential stack and runs them one after another:
//!
//! ```text
//!   MoeStackConfig + per-layer (layer_id, MoeLayerConfig)
//!     + MoeWeightMap (MOE-9)  + byte resolver (MOE-10)
//!     → RealMoeLayer per layer   (MOE-11 assembly)
//!     → RealMoeStack { layers }
//!     → forward: x → layer0 → layer1 → ... → output
//! ```
//!
//! ## What this is and is NOT
//!
//! * It **does** assemble N real MoE layers and run them in sequence,
//!   feeding each layer's output to the next, correctness-first, CPU-only.
//! * It does **not** add residual connections, attention, norms, embeddings,
//!   or any transformer-block structure — this is a *pure sequential
//!   composition of MoE layers*, the minimal multi-layer step. It is **not**
//!   a transformer and not a full model.
//! * It does **not** parse `config.json`, build a graph, touch the loader /
//!   Adapter Toolkit, or lift the MOE-2 fail-loud guard. A real MoE
//!   checkpoint still refuses to load as a model.
//! * No Mixtral / Qwen-MoE end-to-end support is claimed: this validates the
//!   *multi-layer composition mechanism* on small synthetic checkpoints with
//!   real Mixtral / Qwen-MoE tensor names.
//!
//! Because every MoE layer maps `d_model → d_model` (SwiGLU experts preserve
//! the model dimension), sequential composition is dimensionally valid with
//! no projection between layers.

use super::data_plane::MoeWeightMap;
use super::layer::{MoeLayerConfig, MoeLayerError, RealMoeLayer};

/// Experimental MoE-stack configuration (fixture, not parsed from a real
/// `config.json`). Minimal by design: just the layer count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeStackConfig {
    pub num_layers: usize,
}

/// Errors from validating a [`MoeStackConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoeStackConfigError {
    ZeroLayers,
}

impl std::fmt::Display for MoeStackConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoeStackConfigError::ZeroLayers => write!(f, "moe-stack-config: num_layers is 0"),
        }
    }
}

impl std::error::Error for MoeStackConfigError {}

impl MoeStackConfig {
    /// Construct + validate.
    pub fn new(num_layers: usize) -> Result<Self, MoeStackConfigError> {
        if num_layers == 0 {
            return Err(MoeStackConfigError::ZeroLayers);
        }
        Ok(Self { num_layers })
    }
}

/// Errors from assembling / running a [`RealMoeStack`].
#[derive(Debug, Clone, PartialEq)]
pub enum MoeStackError {
    /// Invalid stack config.
    Config(MoeStackConfigError),
    /// A layer failed to assemble or run.
    Layer(MoeLayerError),
    /// No layers supplied.
    EmptyStack,
    /// `config.num_layers` disagreed with the number of layers built.
    LayerCountMismatch { config: usize, actual: usize },
    /// Two layers disagreed on `d_model`, so they cannot be chained.
    DModelMismatch {
        layer: usize,
        expected: usize,
        actual: usize,
    },
    /// The forward input length did not match the stack `d_model`.
    InputDimMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for MoeStackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoeStackError::Config(e) => write!(f, "moe-stack: {e}"),
            MoeStackError::Layer(e) => write!(f, "moe-stack: {e}"),
            MoeStackError::EmptyStack => write!(f, "moe-stack: no layers supplied"),
            MoeStackError::LayerCountMismatch { config, actual } => write!(
                f,
                "moe-stack: config.num_layers ({config}) != assembled layers ({actual})"
            ),
            MoeStackError::DModelMismatch { layer, expected, actual } => write!(
                f,
                "moe-stack: layer {layer} d_model {actual} != stack d_model {expected}"
            ),
            MoeStackError::InputDimMismatch { expected, actual } => write!(
                f,
                "moe-stack: input length {actual} != stack d_model {expected}"
            ),
        }
    }
}

impl std::error::Error for MoeStackError {}

impl From<MoeStackConfigError> for MoeStackError {
    fn from(e: MoeStackConfigError) -> Self {
        MoeStackError::Config(e)
    }
}
impl From<MoeLayerError> for MoeStackError {
    fn from(e: MoeLayerError) -> Self {
        MoeStackError::Layer(e)
    }
}

/// A sequential stack of assembled real MoE layers. Runs each layer's
/// `forward` in order, feeding the output of layer `i` into layer `i+1`.
#[derive(Debug, Clone)]
pub struct RealMoeStack {
    pub config: MoeStackConfig,
    pub layers: Vec<RealMoeLayer>,
}

impl RealMoeStack {
    /// Build a stack from already-assembled layers, validating that the
    /// config layer count matches and every layer shares one `d_model`.
    pub fn new(config: MoeStackConfig, layers: Vec<RealMoeLayer>) -> Result<Self, MoeStackError> {
        if layers.is_empty() {
            return Err(MoeStackError::EmptyStack);
        }
        if layers.len() != config.num_layers {
            return Err(MoeStackError::LayerCountMismatch {
                config: config.num_layers,
                actual: layers.len(),
            });
        }
        let d_model = layers[0].config.d_model;
        for (i, l) in layers.iter().enumerate() {
            if l.config.d_model != d_model {
                return Err(MoeStackError::DModelMismatch {
                    layer: i,
                    expected: d_model,
                    actual: l.config.d_model,
                });
            }
        }
        Ok(Self { config, layers })
    }

    /// Assemble a stack directly from a weight map: build one
    /// [`RealMoeLayer`] per `(layer_id, MoeLayerConfig)` entry (in the given
    /// order), then validate consistency. `config.num_layers` must equal the
    /// number of entries.
    pub fn assemble<F>(
        map: &MoeWeightMap,
        config: MoeStackConfig,
        layer_specs: &[(usize, MoeLayerConfig)],
        resolve: &F,
    ) -> Result<Self, MoeStackError>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        if layer_specs.is_empty() {
            return Err(MoeStackError::EmptyStack);
        }
        let mut layers = Vec::with_capacity(layer_specs.len());
        for &(layer_id, cfg) in layer_specs {
            let layer = RealMoeLayer::assemble(map, layer_id, cfg, resolve)?;
            layers.push(layer);
        }
        Self::new(config, layers)
    }

    /// Number of layers in the stack.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// The model dimension carried through the stack.
    pub fn d_model(&self) -> usize {
        self.layers[0].config.d_model
    }

    /// **Stack forward.** Run each layer's full forward in sequence, feeding
    /// the output of one layer into the next. No residuals, no norms — pure
    /// sequential composition. Input/output length is `d_model`.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, MoeStackError> {
        let d_model = self.d_model();
        if input.len() != d_model {
            return Err(MoeStackError::InputDimMismatch {
                expected: d_model,
                actual: input.len(),
            });
        }
        let mut x = input.to_vec();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
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

    /// Build a multi-layer Mixtral-style map (no shared experts). Each layer
    /// has `n` experts of dims `(d_model, d_ff)`. Seeds vary per layer so the
    /// layers are genuinely different.
    fn mixtral_stack(
        num_layers: usize,
        n: usize,
        d_model: usize,
        d_ff: usize,
    ) -> (MoeWeightMap, HashMap<String, Vec<f32>>) {
        let mut ns: Vec<(String, Vec<usize>)> = Vec::new();
        let mut store: HashMap<String, Vec<f32>> = HashMap::new();
        for l in 0..num_layers {
            let lseed = (l as u64 + 1) * 1000;
            let router = format!("model.layers.{l}.block_sparse_moe.gate.weight");
            ns.push((router.clone(), vec![n, d_model]));
            store.insert(router, seeded(lseed + 1, n * d_model));
            for e in 0..n {
                let base = lseed + 100 + e as u64;
                let g = format!("model.layers.{l}.block_sparse_moe.experts.{e}.w1.weight");
                let u = format!("model.layers.{l}.block_sparse_moe.experts.{e}.w3.weight");
                let d = format!("model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight");
                ns.push((g.clone(), vec![d_ff, d_model]));
                ns.push((u.clone(), vec![d_ff, d_model]));
                ns.push((d.clone(), vec![d_model, d_ff]));
                store.insert(g, seeded(base * 10 + 1, d_ff * d_model));
                store.insert(u, seeded(base * 10 + 2, d_ff * d_model));
                store.insert(d, seeded(base * 10 + 3, d_model * d_ff));
            }
        }
        let map = MoeWeightMap::from_tensors(ns.iter().map(|(n, s)| (n.as_str(), s.clone())));
        (map, store)
    }

    /// Build a single Qwen-MoE-style layer (with shared expert) at layer id.
    fn qwen_layer_into(
        ns: &mut Vec<(String, Vec<usize>)>,
        store: &mut HashMap<String, Vec<f32>>,
        l: usize,
        n: usize,
        d_model: usize,
        d_ff: usize,
        shared_ff: usize,
    ) {
        let lseed = (l as u64 + 1) * 2000;
        let router = format!("model.layers.{l}.mlp.gate.weight");
        ns.push((router.clone(), vec![n, d_model]));
        store.insert(router, seeded(lseed + 1, n * d_model));
        for e in 0..n {
            let base = lseed + 100 + e as u64;
            let g = format!("model.layers.{l}.mlp.experts.{e}.gate_proj.weight");
            let u = format!("model.layers.{l}.mlp.experts.{e}.up_proj.weight");
            let d = format!("model.layers.{l}.mlp.experts.{e}.down_proj.weight");
            ns.push((g.clone(), vec![d_ff, d_model]));
            ns.push((u.clone(), vec![d_ff, d_model]));
            ns.push((d.clone(), vec![d_model, d_ff]));
            store.insert(g, seeded(base * 10 + 1, d_ff * d_model));
            store.insert(u, seeded(base * 10 + 2, d_ff * d_model));
            store.insert(d, seeded(base * 10 + 3, d_model * d_ff));
        }
        let sg = format!("model.layers.{l}.mlp.shared_expert.gate_proj.weight");
        let su = format!("model.layers.{l}.mlp.shared_expert.up_proj.weight");
        let sd = format!("model.layers.{l}.mlp.shared_expert.down_proj.weight");
        ns.push((sg.clone(), vec![shared_ff, d_model]));
        ns.push((su.clone(), vec![shared_ff, d_model]));
        ns.push((sd.clone(), vec![d_model, shared_ff]));
        store.insert(sg, seeded(lseed + 9001, shared_ff * d_model));
        store.insert(su, seeded(lseed + 9002, shared_ff * d_model));
        store.insert(sd, seeded(lseed + 9003, d_model * shared_ff));
    }

    #[test]
    fn stack_config_validates() {
        assert!(MoeStackConfig::new(2).is_ok());
        assert_eq!(MoeStackConfig::new(0).unwrap_err(), MoeStackConfigError::ZeroLayers);
    }

    #[test]
    fn real_moe_stack_assembly() {
        let (map, store) = mixtral_stack(2, 4, 8, 16);
        let resolve = resolver(&store);
        let cfg = MoeStackConfig::new(2).unwrap();
        let lc = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
        let stack = RealMoeStack::assemble(&map, cfg, &[(0, lc), (1, lc)], &resolve).unwrap();
        assert_eq!(stack.num_layers(), 2);
        assert_eq!(stack.d_model(), 8);
    }

    #[test]
    fn real_moe_stack_forward() {
        let (map, store) = mixtral_stack(2, 4, 8, 16);
        let resolve = resolver(&store);
        let cfg = MoeStackConfig::new(2).unwrap();
        let lc = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
        let stack = RealMoeStack::assemble(&map, cfg, &[(0, lc), (1, lc)], &resolve).unwrap();
        let x = seeded(7, 8);
        let out = stack.forward(&x).unwrap();
        assert_eq!(out.len(), 8);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn two_layer_stack_matches_manual_execution() {
        let (map, store) = mixtral_stack(2, 4, 8, 16);
        let resolve = resolver(&store);
        let cfg = MoeStackConfig::new(2).unwrap();
        let lc = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
        let stack = RealMoeStack::assemble(&map, cfg, &[(0, lc), (1, lc)], &resolve).unwrap();
        let x = seeded(11, 8);
        // Manual chaining.
        let h0 = stack.layers[0].forward(&x).unwrap();
        let manual = stack.layers[1].forward(&h0).unwrap();
        let got = stack.forward(&x).unwrap();
        for d in 0..8 {
            assert!((got[d] - manual[d]).abs() < 1e-5);
        }
    }

    #[test]
    fn three_layer_stack_matches_manual_execution() {
        let (map, store) = mixtral_stack(3, 4, 8, 16);
        let resolve = resolver(&store);
        let cfg = MoeStackConfig::new(3).unwrap();
        let lc = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
        let stack =
            RealMoeStack::assemble(&map, cfg, &[(0, lc), (1, lc), (2, lc)], &resolve).unwrap();
        let x = seeded(13, 8);
        let h0 = stack.layers[0].forward(&x).unwrap();
        let h1 = stack.layers[1].forward(&h0).unwrap();
        let manual = stack.layers[2].forward(&h1).unwrap();
        let got = stack.forward(&x).unwrap();
        for d in 0..8 {
            assert!((got[d] - manual[d]).abs() < 1e-5);
        }
    }

    #[test]
    fn stack_validates_d_model_consistency() {
        // Two layers with different d_model cannot be chained.
        let (m0, s0) = mixtral_stack(1, 4, 8, 16);
        let (m1, s1) = mixtral_stack(1, 4, 4, 16); // d_model 4
        let r0 = resolver(&s0);
        let r1 = resolver(&s1);
        let l0 = RealMoeLayer::assemble(&m0, 0, MoeLayerConfig::new(4, 2, false, 8, 16).unwrap(), &r0)
            .unwrap();
        let l1 = RealMoeLayer::assemble(&m1, 0, MoeLayerConfig::new(4, 2, false, 4, 16).unwrap(), &r1)
            .unwrap();
        let cfg = MoeStackConfig::new(2).unwrap();
        let err = RealMoeStack::new(cfg, vec![l0, l1]).unwrap_err();
        assert!(matches!(err, MoeStackError::DModelMismatch { layer: 1, .. }));
    }

    #[test]
    fn stack_with_shared_experts() {
        // Two Qwen-MoE layers, each with a shared expert.
        let mut ns: Vec<(String, Vec<usize>)> = Vec::new();
        let mut store: HashMap<String, Vec<f32>> = HashMap::new();
        qwen_layer_into(&mut ns, &mut store, 0, 3, 4, 6, 10);
        qwen_layer_into(&mut ns, &mut store, 1, 3, 4, 6, 10);
        let map = MoeWeightMap::from_tensors(ns.iter().map(|(n, s)| (n.as_str(), s.clone())));
        let resolve = resolver(&store);
        let cfg = MoeStackConfig::new(2).unwrap();
        let lc = MoeLayerConfig::new(3, 2, true, 4, 6).unwrap();
        let stack = RealMoeStack::assemble(&map, cfg, &[(0, lc), (1, lc)], &resolve).unwrap();
        assert!(stack.layers.iter().all(|l| l.has_shared_expert()));
        let x = seeded(21, 4);
        let h0 = stack.layers[0].forward(&x).unwrap();
        let manual = stack.layers[1].forward(&h0).unwrap();
        let got = stack.forward(&x).unwrap();
        for d in 0..4 {
            assert!((got[d] - manual[d]).abs() < 1e-5);
        }
    }

    #[test]
    fn layer_count_mismatch_is_rejected() {
        let (map, store) = mixtral_stack(2, 4, 8, 16);
        let resolve = resolver(&store);
        // Config says 3 layers but only 2 specs.
        let cfg = MoeStackConfig::new(3).unwrap();
        let lc = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
        let err = RealMoeStack::assemble(&map, cfg, &[(0, lc), (1, lc)], &resolve).unwrap_err();
        assert!(matches!(err, MoeStackError::LayerCountMismatch { config: 3, actual: 2 }));
    }

    #[test]
    fn forward_rejects_wrong_input_dim() {
        let (map, store) = mixtral_stack(1, 4, 8, 16);
        let resolve = resolver(&store);
        let cfg = MoeStackConfig::new(1).unwrap();
        let lc = MoeLayerConfig::new(4, 2, false, 8, 16).unwrap();
        let stack = RealMoeStack::assemble(&map, cfg, &[(0, lc)], &resolve).unwrap();
        let err = stack.forward(&[0.0; 4]).unwrap_err();
        assert!(matches!(err, MoeStackError::InputDimMismatch { expected: 8, actual: 4 }));
    }
}
