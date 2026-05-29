//! **MOE-4** — Sparse MoE reference execution path (correctness-first).
//!
//! Adds real **top-k** expert selection on top of the MOE-3 dense
//! reference: route → softmax → select the k highest-weight experts →
//! **renormalise** the selected weights to sum 1 → execute **only** the
//! selected experts → weighted combine.
//!
//! ```text
//!   router_logits = W_router · x
//!   weights       = softmax(router_logits)            (all experts)
//!   (idx, w)      = top_k(weights, k)                  (k highest, ties → lower index)
//!   w'            = w / Σ w                            (renormalise selected)
//!   output        = Σ_{e ∈ idx} w'[e] · expert_e(x)    (selected experts ONLY)
//! ```
//!
//! ## Why renormalise
//!
//! After top-k we keep only k of the softmax weights, whose sum is < 1.
//! Renormalising them to sum 1 makes the sparse output directly comparable
//! to a **dense reference restricted to the same experts** (same weights,
//! same renormalisation), which is the oracle this module validates
//! against. It is also the standard Mixtral/Qwen-MoE convention
//! (softmax-then-top-k-then-renormalise).
//!
//! ## Scope (MOE-4)
//!
//! Still **isolated in `src/moe/`** — no graph, no runtime, no loader, no
//! CUDA, no adapter toolkit, no CLI. Graph/runtime integration is MOE-5.
//! This validates the sparse *math* against the dense oracle in isolation.

use super::dense::{MoeDenseError, MoeDenseLayer};

/// Errors from the sparse MoE path.
#[derive(Debug, Clone, PartialEq)]
pub enum MoeSparseError {
    /// `k == 0`.
    ZeroK,
    /// `k > num_experts`.
    KExceedsExperts { k: usize, num_experts: usize },
    /// A routing weight is NaN or infinite.
    NonFiniteWeight { index: usize },
    /// A routing weight is negative (softmax weights must be ≥ 0).
    NegativeWeight { index: usize },
    /// The selected top-k weights sum to ≤ 0, so they cannot be
    /// renormalised.
    SelectedSumNonPositive,
    /// An error bubbled up from the underlying dense layer (e.g. a
    /// dimension mismatch in an expert forward).
    Dense(MoeDenseError),
}

impl std::fmt::Display for MoeSparseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoeSparseError::ZeroK => write!(f, "moe-sparse: k must be > 0"),
            MoeSparseError::KExceedsExperts { k, num_experts } => {
                write!(f, "moe-sparse: k ({k}) exceeds num_experts ({num_experts})")
            }
            MoeSparseError::NonFiniteWeight { index } => {
                write!(f, "moe-sparse: routing weight at {index} is non-finite")
            }
            MoeSparseError::NegativeWeight { index } => {
                write!(f, "moe-sparse: routing weight at {index} is negative")
            }
            MoeSparseError::SelectedSumNonPositive => {
                write!(f, "moe-sparse: selected top-k weights sum to <= 0")
            }
            MoeSparseError::Dense(e) => write!(f, "moe-sparse: {e}"),
        }
    }
}

impl std::error::Error for MoeSparseError {}

impl From<MoeDenseError> for MoeSparseError {
    fn from(e: MoeDenseError) -> Self {
        MoeSparseError::Dense(e)
    }
}

/// Result of a top-k selection over routing weights. `indices` are sorted
/// ascending; `weights[i]` is the **renormalised** weight for
/// `indices[i]` (the selected weights sum to 1).
#[derive(Debug, Clone, PartialEq)]
pub struct TopKSelection {
    pub indices: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Select the `k` experts with the highest routing weights, breaking ties
/// by **lower expert index**, and renormalise their weights to sum 1.
///
/// Deterministic. Requires `0 < k <= weights.len()`, all weights finite
/// and non-negative, and a positive selected sum.
pub fn top_k_routing(weights: &[f32], k: usize) -> Result<TopKSelection, MoeSparseError> {
    if k == 0 {
        return Err(MoeSparseError::ZeroK);
    }
    if k > weights.len() {
        return Err(MoeSparseError::KExceedsExperts {
            k,
            num_experts: weights.len(),
        });
    }
    for (i, &w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return Err(MoeSparseError::NonFiniteWeight { index: i });
        }
        if w < 0.0 {
            return Err(MoeSparseError::NegativeWeight { index: i });
        }
    }

    // Rank by (weight desc, index asc) for deterministic tie-breaking.
    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_by(|&a, &b| {
        weights[b]
            .partial_cmp(&weights[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut selected: Vec<usize> = order.into_iter().take(k).collect();
    // Stable output: indices ascending.
    selected.sort_unstable();

    let sum: f64 = selected.iter().map(|&i| weights[i] as f64).sum();
    if sum <= 0.0 {
        return Err(MoeSparseError::SelectedSumNonPositive);
    }
    let inv = 1.0 / sum;
    let renorm: Vec<f32> = selected
        .iter()
        .map(|&i| ((weights[i] as f64) * inv) as f32)
        .collect();

    Ok(TopKSelection {
        indices: selected,
        weights: renorm,
    })
}

/// Output of [`MoeDenseLayer::forward_sparse`] — the combined vector plus
/// the experts that were actually executed (for instrumentation /
/// validation that only the selected experts ran).
#[derive(Debug, Clone, PartialEq)]
pub struct MoeSparseForwardOutput {
    pub output: Vec<f32>,
    pub selected_experts: Vec<usize>,
}

impl MoeDenseLayer {
    /// **Sparse MoE forward.** Route, select top-k experts, renormalise,
    /// and combine **only the selected experts**. Returns the output plus
    /// the selected expert indices.
    pub fn forward_sparse(
        &self,
        input: &[f32],
        k: usize,
    ) -> Result<MoeSparseForwardOutput, MoeSparseError> {
        let routing = self.route(input)?;
        let selection = top_k_routing(&routing.weights, k)?;
        let mut output = vec![0.0_f32; self.d_model];
        for (slot, &e) in selection.indices.iter().enumerate() {
            let w = selection.weights[slot];
            let y_e = self.experts[e].forward(input)?;
            for d in 0..self.d_model {
                output[d] += w * y_e[d];
            }
        }
        Ok(MoeSparseForwardOutput {
            output,
            selected_experts: selection.indices,
        })
    }

    /// **Dense reference restricted to a given expert subset** — the
    /// oracle the sparse path is validated against. Computes the full
    /// softmax, keeps the weights of `selected_indices`, renormalises them
    /// to sum 1, and combines those experts. With `selected_indices` equal
    /// to the top-k set, this must equal [`Self::forward_sparse`].
    pub fn forward_dense_restricted(
        &self,
        input: &[f32],
        selected_indices: &[usize],
    ) -> Result<Vec<f32>, MoeSparseError> {
        let routing = self.route(input)?;
        let sum: f64 = selected_indices
            .iter()
            .map(|&i| routing.weights[i] as f64)
            .sum();
        if sum <= 0.0 {
            return Err(MoeSparseError::SelectedSumNonPositive);
        }
        let inv = 1.0 / sum;
        let mut output = vec![0.0_f32; self.d_model];
        for &e in selected_indices {
            let w = ((routing.weights[e] as f64) * inv) as f32;
            let y_e = self.experts[e].forward(input)?;
            for d in 0..self.d_model {
                output[d] += w * y_e[d];
            }
        }
        Ok(output)
    }
}

// ============================================================================
// Tests (synthetic — no model, no loader, no graph)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::dense::build_fixture_layer;

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

    #[test]
    fn top_k_selects_largest_weights() {
        let sel = top_k_routing(&[0.1, 0.5, 0.2, 0.4], 2).unwrap();
        assert_eq!(sel.indices, vec![1, 3]); // 0.5 and 0.4, sorted asc
    }

    #[test]
    fn top_k_tie_breaks_by_lower_index() {
        // Three equal weights; k=2 must pick the two lowest indices.
        let sel = top_k_routing(&[0.25, 0.25, 0.25, 0.25], 2).unwrap();
        assert_eq!(sel.indices, vec![0, 1]);
    }

    #[test]
    fn top_k_rejects_zero_k() {
        assert_eq!(top_k_routing(&[0.5, 0.5], 0).unwrap_err(), MoeSparseError::ZeroK);
    }

    #[test]
    fn top_k_rejects_k_larger_than_experts() {
        assert_eq!(
            top_k_routing(&[0.5, 0.5], 3).unwrap_err(),
            MoeSparseError::KExceedsExperts { k: 3, num_experts: 2 }
        );
    }

    #[test]
    fn top_k_weights_are_renormalized() {
        let sel = top_k_routing(&[0.1, 0.6, 0.3], 2).unwrap();
        // selected 0.6 (idx1) and 0.3 (idx2); renorm -> 0.6/0.9, 0.3/0.9.
        let s: f32 = sel.weights.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "renormalised weights must sum to 1");
        assert!((sel.weights[0] - (0.6 / 0.9)).abs() < 1e-5);
        assert!((sel.weights[1] - (0.3 / 0.9)).abs() < 1e-5);
    }

    #[test]
    fn top_k_rejects_non_finite_weights() {
        assert!(matches!(
            top_k_routing(&[0.5, f32::NAN, 0.5], 2).unwrap_err(),
            MoeSparseError::NonFiniteWeight { index: 1 }
        ));
        assert!(matches!(
            top_k_routing(&[0.5, -0.1], 1).unwrap_err(),
            MoeSparseError::NegativeWeight { index: 1 }
        ));
    }

    #[test]
    fn sparse_forward_matches_dense_restricted_oracle() {
        let layer = build_fixture_layer();
        let x = seeded(7, layer.d_model);
        let sparse = layer.forward_sparse(&x, 2).unwrap();
        let oracle = layer
            .forward_dense_restricted(&x, &sparse.selected_experts)
            .unwrap();
        for d in 0..layer.d_model {
            assert!(
                (sparse.output[d] - oracle[d]).abs() < 1e-5,
                "sparse forward must equal dense restricted oracle"
            );
        }
    }

    #[test]
    fn sparse_forward_executes_only_selected_experts() {
        let layer = build_fixture_layer();
        let x = seeded(9, layer.d_model);
        let out = layer.forward_sparse(&x, 2).unwrap();
        assert_eq!(out.selected_experts.len(), 2);
        // Selected experts are the top-2 of the full softmax.
        let routing = layer.route(&x).unwrap();
        let expected = top_k_routing(&routing.weights, 2).unwrap().indices;
        assert_eq!(out.selected_experts, expected);
        // And they are a subset of all experts.
        assert!(out.selected_experts.iter().all(|&e| e < layer.num_experts()));
    }

    #[test]
    fn sparse_forward_k_equals_all_matches_dense_forward() {
        let layer = build_fixture_layer();
        let x = seeded(11, layer.d_model);
        // k = all experts: top-k selects everyone, renorm over full softmax
        // (sums to 1 already) = identity, so sparse == dense forward.
        let sparse = layer.forward_sparse(&x, layer.num_experts()).unwrap();
        let dense = layer.forward(&x).unwrap();
        for d in 0..layer.d_model {
            assert!(
                (sparse.output[d] - dense[d]).abs() < 1e-5,
                "sparse with k=all must equal dense forward"
            );
        }
        assert_eq!(sparse.selected_experts.len(), layer.num_experts());
    }

    #[test]
    fn sparse_forward_is_deterministic() {
        let layer = build_fixture_layer();
        let x = seeded(13, layer.d_model);
        let a = layer.forward_sparse(&x, 2).unwrap();
        let b = layer.forward_sparse(&x, 2).unwrap();
        assert_eq!(a, b);
        assert!(a.output.iter().all(|v| v.is_finite()));
    }
}
