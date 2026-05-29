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
}
