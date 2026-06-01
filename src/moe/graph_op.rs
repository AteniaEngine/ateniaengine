//! **MOE-5** — experimental graph bridge for the sparse MoE reference.
//!
//! This is the **minimal** integration of MoE into the execution graph
//! (MOE-0's "principal blocker"). It uses the **fused-op** route (Option
//! A): a single experimental `NodeType::MoeSparseReference` node whose
//! forward simply calls the already-certified
//! [`MoeDenseLayer::forward_sparse`] (MOE-4) on a synthetic fixture layer.
//! There is no dynamic scheduler, no per-expert graph nodes, no real
//! checkpoint loading.
//!
//! ## Why a process-global registry
//!
//! `NodeType` derives `Eq`, so a node variant cannot embed the
//! `MoeDenseLayer` (its `f32` weights are not `Eq`). The node therefore
//! carries only a `layer_id: u32`, and the actual layer lives in this
//! small experimental registry. This keeps the graph change to a single
//! `Eq`-safe variant + one executor arm — no scheduler/graph redesign.
//!
//! **Experimental · CPU-only · synthetic fixture only · NOT production
//! MoE.** Real MoE checkpoints still fail loud (MOE-2). This registry is
//! only populated by tests / explicit experimental callers.

use std::sync::{Arc, Mutex, OnceLock};

use super::dense::MoeDenseLayer;
use super::layer::{MoeExecutionConvention, MoeLayerError, RealMoeLayer};
use super::residency::ResidentExpertLayer;
use super::sparse::MoeSparseError;

/// Process-global registry of synthetic MoE layers, indexed by the
/// `layer_id` carried in `NodeType::MoeSparseReference`.
fn registry() -> &'static Mutex<Vec<Arc<MoeDenseLayer>>> {
    static REG: OnceLock<Mutex<Vec<Arc<MoeDenseLayer>>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(Vec::new()))
}

/// Register a synthetic MoE layer and return its `layer_id`. Used by tests
/// / experimental callers to wire a `MoeSparseReference` node. The
/// registry only grows; ids are stable for the process lifetime.
pub fn register_layer(layer: MoeDenseLayer) -> u32 {
    let mut reg = registry().lock().expect("moe registry poisoned");
    reg.push(Arc::new(layer));
    (reg.len() - 1) as u32
}

/// Fetch a registered layer by id (clones the `Arc`).
pub fn get_layer(layer_id: u32) -> Option<Arc<MoeDenseLayer>> {
    registry()
        .lock()
        .expect("moe registry poisoned")
        .get(layer_id as usize)
        .cloned()
}

/// Execute the sparse MoE reference for a registered layer on `input`,
/// selecting top-`k` experts. This is exactly the MOE-4 contract — the
/// graph op is a thin wrapper so the runtime output equals
/// [`MoeDenseLayer::forward_sparse`].
///
/// Returns the output vector (length `d_model`). Errors if the id is
/// unknown or the sparse forward fails.
pub fn execute_sparse_reference(
    layer_id: u32,
    input: &[f32],
    k: usize,
) -> Result<Vec<f32>, MoeSparseError> {
    let layer = get_layer(layer_id).ok_or(MoeSparseError::Dense(
        super::dense::MoeDenseError::NoExperts,
    ))?;
    let out = layer.forward_sparse(input, k)?;
    Ok(out.output)
}

/// Output of [`execute_dynamic_dispatch`] — the combined vector plus the
/// experts that were **actually executed** (proving conditional dispatch:
/// only the selected experts' `forward` is called).
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicDispatchOutput {
    pub output: Vec<f32>,
    pub executed_experts: Vec<usize>,
}

/// **MOE-7** — dynamic expert dispatch: given a registered layer, the
/// model `input`, and a `selection` tensor `[idx0, w0, idx1, w1, …]`
/// (from `MoeTopK`), run **only** the selected experts and combine their
/// outputs by the selected weights.
///
/// Unlike MOE-5's `forward_sparse` (which internally selects), this takes
/// the selection as data and executes exactly the listed experts — the
/// experts not in the selection are never `forward`-ed. Returns the
/// combined output and the list of executed expert ids.
pub fn execute_dynamic_dispatch(
    layer_id: u32,
    input: &[f32],
    selection: &[f32],
) -> Result<DynamicDispatchOutput, MoeSparseError> {
    let layer = get_layer(layer_id).ok_or(MoeSparseError::Dense(
        super::dense::MoeDenseError::NoExperts,
    ))?;
    if selection.len() % 2 != 0 {
        return Err(MoeSparseError::Dense(super::dense::MoeDenseError::DimMismatch {
            what: "dispatch selection (must be [idx, w, ...])",
            expected: 0,
            actual: selection.len() % 2,
        }));
    }
    let pairs = selection.len() / 2;
    let d_model = layer.d_model;
    let num_experts = layer.num_experts();
    let mut output = vec![0.0_f32; d_model];
    let mut executed = Vec::with_capacity(pairs);
    for p in 0..pairs {
        let idx_f = selection[p * 2];
        let weight = selection[p * 2 + 1];
        // Decode + validate the expert index.
        if !idx_f.is_finite() || idx_f < 0.0 {
            return Err(MoeSparseError::NegativeWeight { index: p * 2 });
        }
        let idx = idx_f as usize;
        if idx >= num_experts {
            return Err(MoeSparseError::KExceedsExperts {
                k: idx + 1,
                num_experts,
            });
        }
        // Conditional execution: only the selected expert runs.
        let y = layer.experts[idx].forward(input)?;
        for d in 0..d_model {
            output[d] += weight * y[d];
        }
        executed.push(idx);
    }
    Ok(DynamicDispatchOutput {
        output,
        executed_experts: executed,
    })
}

/// Look up the routing weight assigned to `expert_id` in a flat
/// `[idx0, w0, idx1, w1, …]` selection tensor. Returns `Some(weight)` if
/// the expert is selected, `None` otherwise.
pub fn expert_weight_in_selection(selection: &[f32], expert_id: u32) -> Option<f32> {
    let pairs = selection.len() / 2;
    for p in 0..pairs {
        if selection[p * 2] as u32 == expert_id {
            return Some(selection[p * 2 + 1]);
        }
    }
    None
}

/// **MOE-8** — conditional execution of a single expert. If `expert_id`
/// appears in `selection`, the expert's `forward` runs and the result is
/// scaled by its routing weight (`executed = true`). If it is **not**
/// selected, the expert's `forward` is **never called** — a zero vector
/// of length `d_model` is returned (`executed = false`). This is the
/// gating primitive the scheduler drives, one node per expert.
///
/// Returns `(contribution[d_model], executed)`.
pub fn execute_conditional_expert(
    layer_id: u32,
    expert_id: u32,
    input: &[f32],
    selection: &[f32],
) -> Result<(Vec<f32>, bool), MoeSparseError> {
    let layer = get_layer(layer_id).ok_or(MoeSparseError::Dense(
        super::dense::MoeDenseError::NoExperts,
    ))?;
    let d_model = layer.d_model;
    if (expert_id as usize) >= layer.num_experts() {
        return Err(MoeSparseError::KExceedsExperts {
            k: expert_id as usize + 1,
            num_experts: layer.num_experts(),
        });
    }
    match expert_weight_in_selection(selection, expert_id) {
        Some(weight) => {
            // Selected → actually run the expert.
            let y = layer.experts[expert_id as usize].forward(input)?;
            let scaled: Vec<f32> = y.into_iter().map(|v| v * weight).collect();
            Ok((scaled, true))
        }
        None => {
            // Not selected → skip the forward entirely; contribute zeros.
            Ok((vec![0.0_f32; d_model], false))
        }
    }
}

// ============================================================================
// **MOE-FULL-4** — RealMoeLayer registry + graph-op executor.
//
// Separate from the synthetic `MoeDenseLayer` registry above: a `RealMoeLayer`
// (router + routed experts + optional shared expert, MOE-11) carries f32
// weights so it is not `Eq` and cannot live in a `NodeType` variant. The
// graph node `NodeType::MoeRealLayerReference { layer_id }` therefore carries
// only an `Eq`-safe `u32`, and the actual layer lives in this process-global
// registry. Experimental, CPU-only, test/opt-in callers only — NOT a
// production path; real MoE checkpoints still fail loud (MOE-2).
// ============================================================================

/// A registered MoE layer behind the `MoeRealLayerReference` node. Either the
/// RAM-f32 [`RealMoeLayer`] (default, fixtures) or — **MOE-PROD-2** — a
/// tier-able [`ResidentExpertLayer`] whose experts live in bf16-RAM or on NVMe.
/// `ResidentExpertLayer::forward` is **certified bit-identical** to
/// `RealMoeLayer::forward_auto` (MOE-FULL-8), so the node output is unchanged.
enum RegisteredMoe {
    Real(Arc<RealMoeLayer>),
    Resident(Arc<ResidentExpertLayer>),
}

/// Process-global registry of MoE layers, indexed by the `layer_id`
/// carried in `NodeType::MoeRealLayerReference`. A single registry keeps ids
/// unique across the `Real`/`Resident` variants.
fn real_registry() -> &'static Mutex<Vec<RegisteredMoe>> {
    static REG: OnceLock<Mutex<Vec<RegisteredMoe>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(Vec::new()))
}

/// Register a real (RAM-f32) MoE layer and return its `layer_id`. Registry only
/// grows; ids stable per process.
pub fn register_real_moe_layer(layer: RealMoeLayer) -> u32 {
    let mut reg = real_registry().lock().expect("real moe registry poisoned");
    reg.push(RegisteredMoe::Real(Arc::new(layer)));
    (reg.len() - 1) as u32
}

/// **MOE-PROD-2** — register a tier-able resident MoE layer (experts in
/// bf16-RAM or on NVMe) and return its `layer_id`. The node forward delegates
/// to the certified `ResidentExpertLayer::forward`.
pub fn register_resident_moe_layer(layer: Arc<ResidentExpertLayer>) -> u32 {
    let mut reg = real_registry().lock().expect("real moe registry poisoned");
    reg.push(RegisteredMoe::Resident(layer));
    (reg.len() - 1) as u32
}

/// Fetch a registered **real** (RAM-f32) MoE layer by id (clones the `Arc`).
/// Returns `None` for resident-tier entries (use [`execute_real_moe_layer`]).
pub fn get_real_moe_layer(layer_id: u32) -> Option<Arc<RealMoeLayer>> {
    match real_registry().lock().expect("real moe registry poisoned").get(layer_id as usize) {
        Some(RegisteredMoe::Real(l)) => Some(Arc::clone(l)),
        _ => None,
    }
}

/// Run one `[d_model]` row through a registered layer, dispatching on its tier.
/// `Real` → `forward_auto`; `Resident` → certified-equal `forward` (drops the
/// residency telemetry). Errors if the id is unknown.
fn execute_moe_row(layer_id: u32, row: &[f32]) -> Result<Vec<f32>, MoeLayerError> {
    let reg = real_registry().lock().expect("real moe registry poisoned");
    match reg.get(layer_id as usize) {
        Some(RegisteredMoe::Real(l)) => l.forward_auto(row),
        Some(RegisteredMoe::Resident(l)) => l.forward(row).map(|(out, _info)| out),
        None => Err(MoeLayerError::Binding(super::binding::MoeBindingError::LayerNotFound {
            layer_id: layer_id as usize,
        })),
    }
}

/// Execute a registered real MoE layer on `input` using its **auto-resolved
/// convention** (MOE-18) — exactly the certified `RealMoeLayer::forward_auto`
/// contract, so the graph op output equals the reference.
///
/// `input` may be a single token (`[d_model]` → `[d_model]`) or, as of
/// **MOE-FULL-6**, a flat multi-token batch (`[n * d_model]` → `[n * d_model]`)
/// in which case the layer is applied **independently per `d_model`-sized
/// row** (the MoE block is position-wise). The single-token case is bit-
/// identical to before. Errors if the id is unknown or a row forward fails.
pub fn execute_real_moe_layer(layer_id: u32, input: &[f32]) -> Result<Vec<f32>, MoeLayerError> {
    let d_model = {
        let reg = real_registry().lock().expect("real moe registry poisoned");
        match reg.get(layer_id as usize) {
            Some(RegisteredMoe::Real(l)) => l.config.d_model,
            Some(RegisteredMoe::Resident(l)) => l.config.d_model,
            None => {
                return Err(MoeLayerError::Binding(
                    super::binding::MoeBindingError::LayerNotFound { layer_id: layer_id as usize },
                ));
            }
        }
    };
    // Single token, or a shape the forward will reject — delegate directly.
    if input.len() == d_model || d_model == 0 || input.len() % d_model != 0 {
        return execute_moe_row(layer_id, input);
    }
    // MOE-FULL-6: multi-token — apply the position-wise MoE per row.
    let mut out = Vec::with_capacity(input.len());
    for chunk in input.chunks(d_model) {
        out.extend(execute_moe_row(layer_id, chunk)?);
    }
    Ok(out)
}

/// Like [`execute_real_moe_layer`] but with an explicit convention override
/// (MOE-17). Used by tests that pin a convention.
pub fn execute_real_moe_layer_with(
    layer_id: u32,
    input: &[f32],
    convention: MoeExecutionConvention,
) -> Result<Vec<f32>, MoeLayerError> {
    let layer = get_real_moe_layer(layer_id)
        .ok_or(MoeLayerError::Binding(super::binding::MoeBindingError::LayerNotFound {
            layer_id: layer_id as usize,
        }))?;
    layer.forward_with(input, convention)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::dense::build_fixture_layer;

    #[test]
    fn register_and_execute_roundtrip() {
        let layer = build_fixture_layer();
        let id = register_layer(layer.clone());
        assert!(get_layer(id).is_some());
        // Deterministic input.
        let x: Vec<f32> = (0..layer.d_model).map(|i| (i as f32) * 0.1 - 0.3).collect();
        let via_registry = execute_sparse_reference(id, &x, 2).unwrap();
        let direct = layer.forward_sparse(&x, 2).unwrap().output;
        assert_eq!(via_registry, direct);
    }

    #[test]
    fn unknown_layer_id_errors() {
        let err = execute_sparse_reference(u32::MAX, &[0.0; 8], 2).unwrap_err();
        assert!(matches!(err, MoeSparseError::Dense(_)));
    }

    // ---- MOE-FULL-4: real MoE layer registry ----

    /// Build a small real MoE layer from the synthetic fixture (no shared
    /// expert) for registry tests.
    fn fixture_real_layer() -> RealMoeLayer {
        let routed = build_fixture_layer(); // 4 experts, top-k 2, d_model 8, d_ff 16
        let config = crate::moe::MoeLayerConfig::new(
            routed.num_experts(),
            2,
            false,
            routed.d_model,
            routed.d_ff,
        )
        .unwrap();
        RealMoeLayer { config, routed, shared: None, shared_gate: None }
    }

    #[test]
    fn real_register_and_execute_roundtrip() {
        let layer = fixture_real_layer();
        let id = register_real_moe_layer(layer.clone());
        assert!(get_real_moe_layer(id).is_some());
        let x: Vec<f32> = (0..layer.config.d_model).map(|i| (i as f32) * 0.1 - 0.3).collect();
        let via_registry = execute_real_moe_layer(id, &x).unwrap();
        let direct = layer.forward_auto(&x).unwrap();
        assert_eq!(via_registry, direct);
    }

    #[test]
    fn real_unknown_layer_id_errors() {
        let err = execute_real_moe_layer(u32::MAX, &[0.0; 8]).unwrap_err();
        assert!(matches!(
            err,
            crate::moe::MoeLayerError::Binding(
                crate::moe::MoeBindingError::LayerNotFound { .. }
            )
        ));
    }
}
