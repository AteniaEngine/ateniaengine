//! **MOE-13** — Real MoE checkpoint validation harness (experimental,
//! CPU-only, opt-in).
//!
//! MOE-9..12 built the pieces: metadata plane, real tensor binding, layer
//! assembly, multi-layer stack. MOE-13 ties them into a single **validation
//! path** that takes a checkpoint's `(name, shape)` listing + a byte resolver
//! and:
//!
//! ```text
//!   MoeWeightMap (MOE-9)
//!     → derive a per-layer MoeLayerConfig from metadata SHAPES (no config.json)
//!     → assemble a RealMoeStack            (MOE-12)
//!     → run one minimal opt-in forward     (MOE-11/12)
//!     → produce a ValidationReport
//! ```
//!
//! ## Critical scope notes
//!
//! * This is a **validation-only, opt-in** path. It is NOT the productive
//!   loader, runtime, or inference path. It never loads a model "as a model";
//!   it consumes the same `(name, shape)` listing a reader already exposes,
//!   exactly as MOE-9..12 do. The MOE-2 loader **fail-loud guard is
//!   untouched** — a real MoE checkpoint still refuses to load normally.
//! * No `config.json` parsing: the per-layer topology (num experts, shared
//!   expert presence, `d_model`, `d_ff`) is derived from tensor **shapes**.
//!   `experts_per_token` is the one value not present in the weights, so it is
//!   supplied by the caller (default for the probe).
//! * **No real Mixtral / Qwen-MoE support is claimed.** A real Qwen-MoE
//!   (~14B params) or Mixtral (~47B) does not fit CI and is not downloaded.
//!   The harness is *target-agnostic*: it validates any safetensors MoE
//!   checkpoint supplied as a resolver. In CI it is exercised against a tiny
//!   synthetic checkpoint that uses **real Qwen-MoE / Mixtral tensor naming**.
//!   A `forward_pass_ok = true` means "metadata → stack → forward executed and
//!   produced finite numbers", NOT "this model is correct / supported".

use super::data_plane::MoeWeightMap;
use super::layer::MoeLayerConfig;
use super::stack::{MoeStackConfig, RealMoeStack};

/// Structured result of validating a (real-format) MoE checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    /// Number of distinct MoE layers found in the metadata.
    pub layers_detected: usize,
    /// Total routed experts across all layers.
    pub experts_detected: usize,
    /// Number of layers that carry a shared expert.
    pub shared_experts: usize,
    /// `d_model` derived from the metadata (None if nothing could be derived).
    pub d_model: Option<usize>,
    /// Whether the minimal opt-in forward executed and produced finite output.
    pub forward_pass_ok: bool,
    /// Human-readable errors encountered (empty on full success).
    pub errors: Vec<String>,
}

impl ValidationReport {
    /// An empty report (no MoE detected).
    fn empty() -> Self {
        Self {
            layers_detected: 0,
            experts_detected: 0,
            shared_experts: 0,
            d_model: None,
            forward_pass_ok: false,
            errors: Vec::new(),
        }
    }

    /// `true` if at least one MoE layer was detected.
    pub fn is_moe(&self) -> bool {
        self.layers_detected > 0
    }

    /// One-line human summary.
    pub fn summary(&self) -> String {
        format!(
            "MoE validation: layers={}, experts={}, shared_expert_layers={}, d_model={}, forward_ok={}, errors={}",
            self.layers_detected,
            self.experts_detected,
            self.shared_experts,
            self.d_model.map(|d| d.to_string()).unwrap_or_else(|| "?".to_string()),
            self.forward_pass_ok,
            self.errors.len()
        )
    }
}

/// Deterministic xorshift probe input (no external dependency, no
/// `Math.random`). Used only to feed the validation forward.
fn probe_input(d_model: usize) -> Vec<f32> {
    let mut state: u64 = 0xA7E_1A_13u64.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut out = Vec::with_capacity(d_model);
    for _ in 0..d_model {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u = (state >> 11) as u32;
        out.push((u as f32 / u32::MAX as f32) * 2.0 - 1.0);
    }
    out
}

/// The MOE-13 validation harness. Stateless; all entry points are
/// associated functions.
pub struct RealMoeCheckpointValidation;

impl RealMoeCheckpointValidation {
    /// Derive a [`MoeLayerConfig`] for one layer **purely from metadata
    /// shapes** — no `config.json`. `experts_per_token` is supplied by the
    /// caller (clamped to `[1, num_experts]`) because it is not encoded in
    /// the weights.
    ///
    /// `d_model` / `d_ff` come from the first complete expert's gate shape
    /// (`[d_ff, d_model]`); shared-expert presence from the metadata.
    pub fn derive_layer_config(
        map: &MoeWeightMap,
        layer_id: usize,
        experts_per_token: usize,
    ) -> Result<MoeLayerConfig, String> {
        let layer = map
            .layer(layer_id)
            .ok_or_else(|| format!("layer {layer_id}: not present in metadata"))?;
        let num_experts = layer.num_experts();
        if num_experts == 0 {
            return Err(format!("layer {layer_id}: no experts"));
        }
        // First expert with a gate tensor gives the dimensions.
        let gate = layer
            .experts
            .values()
            .find_map(|e| e.gate.as_ref())
            .ok_or_else(|| format!("layer {layer_id}: no expert gate tensor to derive dims"))?;
        if gate.shape.len() != 2 {
            return Err(format!(
                "layer {layer_id}: gate shape {:?} is not 2-D",
                gate.shape
            ));
        }
        let d_ff = gate.shape[0];
        let d_model = gate.shape[1];
        let has_shared = !layer.shared.is_empty();
        let k = experts_per_token.clamp(1, num_experts);
        MoeLayerConfig::new(num_experts, k, has_shared, d_model, d_ff)
            .map_err(|e| format!("layer {layer_id}: {e}"))
    }

    /// Validate a checkpoint end-to-end (metadata → stack → minimal forward)
    /// and return a [`ValidationReport`]. Never panics; failures are recorded
    /// in `report.errors` and reflected in `forward_pass_ok`.
    ///
    /// `experts_per_token` is the routing top-k used for the probe forward
    /// (clamped per layer). This is an **opt-in validation path only**.
    pub fn validate<F>(
        map: &MoeWeightMap,
        experts_per_token: usize,
        resolve: &F,
    ) -> ValidationReport
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        let mut report = ValidationReport::empty();

        // --- Pure metadata inspection (always safe). ---
        report.layers_detected = map.layers.len();
        report.experts_detected = map.layers.values().map(|l| l.num_experts()).sum();
        report.shared_experts = map
            .layers
            .values()
            .filter(|l| !l.shared.is_empty())
            .count();

        if report.layers_detected == 0 {
            report.errors.push("no MoE layers detected".to_string());
            return report;
        }

        // --- Derive per-layer configs from shapes. ---
        let layer_ids: Vec<usize> = map.layers.keys().copied().collect();
        let mut specs: Vec<(usize, MoeLayerConfig)> = Vec::with_capacity(layer_ids.len());
        for &lid in &layer_ids {
            match Self::derive_layer_config(map, lid, experts_per_token) {
                Ok(cfg) => {
                    if report.d_model.is_none() {
                        report.d_model = Some(cfg.d_model);
                    }
                    specs.push((lid, cfg));
                }
                Err(e) => report.errors.push(e),
            }
        }
        if specs.len() != layer_ids.len() {
            // Some layer failed to derive; do not attempt the forward.
            return report;
        }

        // --- Assemble the stack. ---
        let stack_cfg = match MoeStackConfig::new(specs.len()) {
            Ok(c) => c,
            Err(e) => {
                report.errors.push(e.to_string());
                return report;
            }
        };
        let stack = match RealMoeStack::assemble(map, stack_cfg, &specs, resolve) {
            Ok(s) => s,
            Err(e) => {
                report.errors.push(format!("stack assembly failed: {e}"));
                return report;
            }
        };

        // --- Minimal opt-in forward. ---
        let x = probe_input(stack.d_model());
        match stack.forward(&x) {
            Ok(out) => {
                if out.len() == stack.d_model() && out.iter().all(|v| v.is_finite()) {
                    report.forward_pass_ok = true;
                } else {
                    report
                        .errors
                        .push("forward produced wrong length or non-finite values".to_string());
                }
            }
            Err(e) => report.errors.push(format!("forward failed: {e}")),
        }

        report
    }
}

// ============================================================================
// Tests (synthetic real-format checkpoints — no model download, no loader)
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

    /// Build a multi-layer Qwen-MoE-style checkpoint (with shared experts).
    fn qwen_checkpoint(
        num_layers: usize,
        n: usize,
        d_model: usize,
        d_ff: usize,
        shared_ff: usize,
    ) -> (MoeWeightMap, HashMap<String, Vec<f32>>) {
        let mut ns: Vec<(String, Vec<usize>)> = Vec::new();
        let mut store: HashMap<String, Vec<f32>> = HashMap::new();
        for l in 0..num_layers {
            let lseed = (l as u64 + 1) * 3000;
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
        let map = MoeWeightMap::from_tensors(ns.iter().map(|(n, s)| (n.as_str(), s.clone())));
        (map, store)
    }

    #[test]
    fn validation_report_builds() {
        let (map, store) = qwen_checkpoint(2, 4, 8, 16, 24);
        let resolve = resolver(&store);
        let report = RealMoeCheckpointValidation::validate(&map, 2, &resolve);
        assert!(report.is_moe());
        assert!(report.errors.is_empty(), "errors: {:?}", report.errors);
        assert!(!report.summary().is_empty());
    }

    #[test]
    fn validation_detects_experts() {
        let (map, store) = qwen_checkpoint(2, 4, 8, 16, 24);
        let resolve = resolver(&store);
        let report = RealMoeCheckpointValidation::validate(&map, 2, &resolve);
        assert_eq!(report.layers_detected, 2);
        assert_eq!(report.experts_detected, 8); // 4 per layer * 2
        assert_eq!(report.shared_experts, 2);
        assert_eq!(report.d_model, Some(8));
    }

    #[test]
    fn validation_builds_stack() {
        // Config derivation alone must succeed per layer.
        let (map, _store) = qwen_checkpoint(3, 4, 8, 16, 24);
        for lid in 0..3 {
            let cfg = RealMoeCheckpointValidation::derive_layer_config(&map, lid, 2).unwrap();
            assert_eq!(cfg.num_experts, 4);
            assert_eq!(cfg.experts_per_token, 2);
            assert!(cfg.has_shared_expert);
            assert_eq!(cfg.d_model, 8);
            assert_eq!(cfg.d_ff, 16);
        }
    }

    #[test]
    fn validation_runs_forward() {
        let (map, store) = qwen_checkpoint(2, 4, 8, 16, 24);
        let resolve = resolver(&store);
        let report = RealMoeCheckpointValidation::validate(&map, 2, &resolve);
        assert!(report.forward_pass_ok, "forward must run; errors: {:?}", report.errors);
    }

    #[test]
    fn validation_reports_missing_tensor_data() {
        // Metadata lists everything, but a tensor's bytes are missing → the
        // forward cannot run, recorded as an error, no panic.
        let (map, mut store) = qwen_checkpoint(1, 4, 8, 16, 24);
        store.remove("model.layers.0.mlp.experts.0.gate_proj.weight");
        let resolve = resolver(&store);
        let report = RealMoeCheckpointValidation::validate(&map, 2, &resolve);
        assert!(report.is_moe());
        assert!(!report.forward_pass_ok);
        assert!(!report.errors.is_empty());
    }

    #[test]
    fn validation_clamps_experts_per_token() {
        // Asking for more experts_per_token than exist clamps to num_experts.
        let (map, _store) = qwen_checkpoint(1, 4, 8, 16, 24);
        let cfg = RealMoeCheckpointValidation::derive_layer_config(&map, 0, 99).unwrap();
        assert_eq!(cfg.experts_per_token, 4);
    }

    #[test]
    fn dense_checkpoint_produces_empty_report() {
        let dense = vec![
            ("model.embed_tokens.weight", vec![16, 4]),
            ("model.layers.0.mlp.gate_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.up_proj.weight", vec![8, 4]),
            ("model.layers.0.mlp.down_proj.weight", vec![4, 8]),
        ];
        let map = MoeWeightMap::from_tensors(dense);
        let store: HashMap<String, Vec<f32>> = HashMap::new();
        let resolve = resolver(&store);
        let report = RealMoeCheckpointValidation::validate(&map, 2, &resolve);
        assert!(!report.is_moe());
        assert_eq!(report.layers_detected, 0);
        assert!(!report.forward_pass_ok);
    }
}
