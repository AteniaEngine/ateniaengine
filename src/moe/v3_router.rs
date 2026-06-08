//! **MOE-V3-ROUTE-1** — DeepSeek-V3-like modern routing primitives (mechanism only).
//!
//! This is a self-contained, correctness-first **reference router** for the
//! DeepSeek-V3 family routing math, isolated in `src/moe/` exactly like
//! [`super::dense`] / [`super::sparse`]: **no runtime, no loader, no graph, no
//! CUDA, no Adapter Toolkit, no productive wiring**. It only computes a routing
//! decision (selected experts + combine weights) from router logits + the
//! per-expert bias, so the L0 *mechanism* can be certified on a reduced-dim
//! fixture vs a HuggingFace `DeepseekV3MoE` float64 reference
//! (`fixtures/moe/generate_v3_route_reference.py`,
//! `tests/moe_v3_route_scale_cert_test.rs`).
//!
//! It is **not** real-weight certified and **not** L1/L2/L3; it does not claim
//! the dense ADR-004 `CERTIFIED`. L4 remains reserved/unreachable.
//!
//! ## Algorithm (matches `transformers`'s `DeepseekV3MoE.route_tokens_to_experts`)
//!
//! ```text
//!   scores              = sigmoid(router_logits)                 (per expert, NOT softmax)
//!   scores_for_choice   = scores + e_score_correction_bias       (bias used for SELECTION ONLY)
//!   group_score[g]      = sum of the top-2 scores_for_choice in group g
//!   selected_groups     = top-`topk_group` groups by group_score
//!   masked_choice       = scores_for_choice, with experts outside selected groups → -inf
//!   selected_experts    = top-`top_k` experts by masked_choice
//!   combine_weight[i]   = scores[i]                              (ORIGINAL score, NO bias)
//!   if norm_topk_prob:  combine_weight /= (Σ combine_weight + 1e-20)
//!   combine_weight     *= routed_scaling_factor
//! ```
//!
//! Source (verbatim algorithm): `transformers` (v5.6.2)
//! `models/deepseek_v3/modeling_deepseek_v3.py::DeepseekV3MoE.route_tokens_to_experts`
//! + DeepSeek-V3 public config (`scoring_func="sigmoid"`, `topk_method="noaux_tc"`,
//! `n_group`, `topk_group`, `routed_scaling_factor`, `norm_topk_prob`,
//! per-layer `e_score_correction_bias`). All internals are f64 (reference-grade),
//! matching the discipline that Atenia's router stays f64 on every policy.

/// Expert scoring function. DeepSeek-V3 uses `sigmoid`; only `sigmoid` is
/// implemented here. Any other value is a hard error (fail-loud) — this module
/// never silently falls back to softmax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoringFunc {
    Sigmoid,
}

impl ScoringFunc {
    /// Parse a config string fail-loud. `"sigmoid"` → [`ScoringFunc::Sigmoid`];
    /// anything else (incl. `"softmax"`) is rejected — softmax routing has its
    /// own certified path ([`super::dense`]/[`super::sparse`]); this module is
    /// the DeepSeek-V3 sigmoid mechanism only.
    pub fn parse(s: &str) -> Result<Self, V3RouterError> {
        match s {
            "sigmoid" => Ok(ScoringFunc::Sigmoid),
            other => Err(V3RouterError::UnsupportedScoringFunc { got: other.to_string() }),
        }
    }
}

/// DeepSeek-V3-like router configuration (a fixture/struct, not parsed from a
/// real checkpoint here).
#[derive(Debug, Clone, PartialEq)]
pub struct V3RouterConfig {
    pub n_routed_experts: usize,
    /// `num_experts_per_tok` — experts selected per token.
    pub top_k: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
    pub norm_topk_prob: bool,
    pub scoring_func: ScoringFunc,
}

/// Errors from the V3 router — all are fail-loud (no silent fallback).
#[derive(Debug, Clone, PartialEq)]
pub enum V3RouterError {
    UnsupportedScoringFunc { got: String },
    /// `n_routed_experts` is not divisible by `n_group`.
    ExpertsNotDivisibleByGroups { n_routed: usize, n_group: usize },
    /// A group has fewer than 2 experts (HF takes the top-2 per group).
    GroupTooSmall { experts_per_group: usize },
    /// `topk_group > n_group`.
    TopkGroupExceedsGroups { topk_group: usize, n_group: usize },
    /// `top_k` exceeds the experts available inside the selected groups.
    TopKExceedsSelectable { top_k: usize, selectable: usize },
    /// `top_k == 0` or `topk_group == 0` or `n_group == 0`.
    ZeroParameter { what: &'static str },
    /// A length mismatch between logits/bias and `n_routed_experts`.
    LengthMismatch { what: &'static str, expected: usize, actual: usize },
    /// A router logit or bias value is NaN/inf.
    NonFinite { what: &'static str, index: usize },
}

impl std::fmt::Display for V3RouterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            V3RouterError::UnsupportedScoringFunc { got } => {
                write!(f, "v3-router: unsupported scoring_func '{got}' (only 'sigmoid')")
            }
            V3RouterError::ExpertsNotDivisibleByGroups { n_routed, n_group } => write!(
                f,
                "v3-router: n_routed_experts ({n_routed}) not divisible by n_group ({n_group})"
            ),
            V3RouterError::GroupTooSmall { experts_per_group } => write!(
                f,
                "v3-router: experts_per_group ({experts_per_group}) < 2 (HF takes top-2 per group)"
            ),
            V3RouterError::TopkGroupExceedsGroups { topk_group, n_group } => {
                write!(f, "v3-router: topk_group ({topk_group}) > n_group ({n_group})")
            }
            V3RouterError::TopKExceedsSelectable { top_k, selectable } => write!(
                f,
                "v3-router: top_k ({top_k}) exceeds experts in selected groups ({selectable})"
            ),
            V3RouterError::ZeroParameter { what } => write!(f, "v3-router: {what} must be > 0"),
            V3RouterError::LengthMismatch { what, expected, actual } => write!(
                f,
                "v3-router: {what} length {actual} != n_routed_experts {expected}"
            ),
            V3RouterError::NonFinite { what, index } => {
                write!(f, "v3-router: {what} at {index} is non-finite")
            }
        }
    }
}

impl std::error::Error for V3RouterError {}

/// Result of a DeepSeek-V3 routing decision for one token.
#[derive(Debug, Clone, PartialEq)]
pub struct V3Routing {
    /// Selected expert indices, **ascending** (stable order).
    pub indices: Vec<usize>,
    /// Combine weight for `indices[i]` (original sigmoid score, normalised if
    /// `norm_topk_prob`, then × `routed_scaling_factor`).
    pub weights: Vec<f32>,
    /// Dense `[n_routed_experts]` combine-weight vector: `weights` scattered at
    /// `indices`, `0.0` elsewhere. Makes index-aligned comparison trivial and
    /// precision-robust.
    pub dense_weights: Vec<f32>,
    /// Selection margin: `min(scores_for_choice over selected)` −
    /// `max(scores_for_choice over experts in selected groups but NOT selected)`.
    /// `+inf` when every expert in the selected groups was selected. A small
    /// margin flags a near-tie selection (diagnostic, like C2's routing margin).
    pub selection_margin: f64,
}

#[inline]
fn sigmoid_f64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl V3RouterConfig {
    fn validate(&self) -> Result<(), V3RouterError> {
        if self.top_k == 0 {
            return Err(V3RouterError::ZeroParameter { what: "top_k" });
        }
        if self.n_group == 0 {
            return Err(V3RouterError::ZeroParameter { what: "n_group" });
        }
        if self.topk_group == 0 {
            return Err(V3RouterError::ZeroParameter { what: "topk_group" });
        }
        if self.n_routed_experts % self.n_group != 0 {
            return Err(V3RouterError::ExpertsNotDivisibleByGroups {
                n_routed: self.n_routed_experts,
                n_group: self.n_group,
            });
        }
        let experts_per_group = self.n_routed_experts / self.n_group;
        if experts_per_group < 2 {
            return Err(V3RouterError::GroupTooSmall { experts_per_group });
        }
        if self.topk_group > self.n_group {
            return Err(V3RouterError::TopkGroupExceedsGroups {
                topk_group: self.topk_group,
                n_group: self.n_group,
            });
        }
        let selectable = self.topk_group * experts_per_group;
        if self.top_k > selectable {
            return Err(V3RouterError::TopKExceedsSelectable { top_k: self.top_k, selectable });
        }
        Ok(())
    }
}

/// Pick the `k` highest values of `scored` (a list of `(orig_index, score)`),
/// breaking ties by **lower original index**. Returns the chosen original
/// indices. `-inf` scores are eligible only if nothing else remains (they sort
/// last). Deterministic.
fn top_k_by_score(scored: &[(usize, f64)], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scored.len()).collect();
    order.sort_by(|&a, &b| {
        scored[b]
            .1
            .partial_cmp(&scored[a].1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(scored[a].0.cmp(&scored[b].0))
    });
    order.into_iter().take(k).map(|p| scored[p].0).collect()
}

/// Compute the DeepSeek-V3 routing decision for one token from its raw router
/// logits (`W_router · x`, length `n_routed_experts`) and the per-expert
/// `e_score_correction_bias`. f64 internals. Fail-loud on any invalid config /
/// input. See the module docs for the exact algorithm + source.
pub fn v3_route(
    router_logits: &[f32],
    e_score_correction_bias: &[f32],
    cfg: &V3RouterConfig,
) -> Result<V3Routing, V3RouterError> {
    cfg.validate()?;
    let n = cfg.n_routed_experts;
    if router_logits.len() != n {
        return Err(V3RouterError::LengthMismatch {
            what: "router_logits",
            expected: n,
            actual: router_logits.len(),
        });
    }
    if e_score_correction_bias.len() != n {
        return Err(V3RouterError::LengthMismatch {
            what: "e_score_correction_bias",
            expected: n,
            actual: e_score_correction_bias.len(),
        });
    }
    for (i, &v) in router_logits.iter().enumerate() {
        if !v.is_finite() {
            return Err(V3RouterError::NonFinite { what: "router_logits", index: i });
        }
    }
    for (i, &v) in e_score_correction_bias.iter().enumerate() {
        if !v.is_finite() {
            return Err(V3RouterError::NonFinite { what: "e_score_correction_bias", index: i });
        }
    }

    // scores = sigmoid(logits); scores_for_choice = scores + bias.
    let scores: Vec<f64> = router_logits.iter().map(|&v| sigmoid_f64(v as f64)).collect();
    let choice: Vec<f64> =
        scores.iter().zip(e_score_correction_bias).map(|(&s, &b)| s + b as f64).collect();

    let experts_per_group = n / cfg.n_group;

    // group_score[g] = sum of the top-2 `choice` values in group g.
    let mut group_scores: Vec<(usize, f64)> = Vec::with_capacity(cfg.n_group);
    for g in 0..cfg.n_group {
        let base = g * experts_per_group;
        let mut grp: Vec<f64> = choice[base..base + experts_per_group].to_vec();
        // descending; take top 2.
        grp.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top2 = grp[0] + grp[1];
        group_scores.push((g, top2));
    }
    let selected_groups: Vec<usize> = top_k_by_score(&group_scores, cfg.topk_group);
    let mut group_selected = vec![false; cfg.n_group];
    for &g in &selected_groups {
        group_selected[g] = true;
    }

    // masked_choice: experts outside selected groups → -inf.
    let masked: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let g = i / experts_per_group;
            if group_selected[g] {
                (i, choice[i])
            } else {
                (i, f64::NEG_INFINITY)
            }
        })
        .collect();

    let mut selected: Vec<usize> = top_k_by_score(&masked, cfg.top_k);
    selected.sort_unstable();

    // Selection margin (diagnostic): lowest selected choice − highest
    // not-selected choice among experts inside the selected groups.
    let selected_set: std::collections::HashSet<usize> = selected.iter().copied().collect();
    let min_selected = selected.iter().map(|&i| choice[i]).fold(f64::INFINITY, f64::min);
    let max_unselected_in_groups = (0..n)
        .filter(|i| group_selected[i / experts_per_group] && !selected_set.contains(i))
        .map(|i| choice[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let selection_margin = if max_unselected_in_groups.is_finite() {
        min_selected - max_unselected_in_groups
    } else {
        f64::INFINITY
    };

    // Combine weights use the ORIGINAL scores (no bias), normalised + scaled.
    let mut w: Vec<f64> = selected.iter().map(|&i| scores[i]).collect();
    if cfg.norm_topk_prob {
        let denom: f64 = w.iter().sum::<f64>() + 1e-20;
        for v in &mut w {
            *v /= denom;
        }
    }
    for v in &mut w {
        *v *= cfg.routed_scaling_factor;
    }
    let weights: Vec<f32> = w.iter().map(|&v| v as f32).collect();

    let mut dense_weights = vec![0.0_f32; n];
    for (slot, &e) in selected.iter().enumerate() {
        dense_weights[e] = weights[slot];
    }

    Ok(V3Routing { indices: selected, weights, dense_weights, selection_margin })
}

// ============================================================================
// Tests (synthetic — no model, no loader, no graph)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(n: usize, top_k: usize, n_group: usize, topk_group: usize, scale: f64, norm: bool) -> V3RouterConfig {
        V3RouterConfig {
            n_routed_experts: n,
            top_k,
            n_group,
            topk_group,
            routed_scaling_factor: scale,
            norm_topk_prob: norm,
            scoring_func: ScoringFunc::Sigmoid,
        }
    }

    #[test]
    fn scoring_func_parse_fail_loud() {
        assert_eq!(ScoringFunc::parse("sigmoid"), Ok(ScoringFunc::Sigmoid));
        assert!(matches!(
            ScoringFunc::parse("softmax"),
            Err(V3RouterError::UnsupportedScoringFunc { .. })
        ));
    }

    #[test]
    fn selects_topk_within_topk_groups() {
        // 8 experts, 4 groups of 2, pick 2 groups, then top-2 experts.
        let c = cfg(8, 2, 4, 2, 1.0, true);
        // Logits engineered: group 1 (idx 2,3) and group 3 (idx 6,7) have the
        // highest pair-sums; within them experts 3 and 7 are largest.
        let logits = [0.0_f32, 0.1, 3.0, 3.1, 0.2, 0.3, 2.5, 2.6];
        let bias = [0.0_f32; 8];
        let r = v3_route(&logits, &bias, &c).unwrap();
        // Selected groups are g1 (experts 2,3) and g3 (experts 6,7). The global
        // top-2 by score among those four are experts 2,3 (g1's pair is the
        // strongest), so group-limiting admits {2,3,6,7} and top-k picks {2,3}.
        assert_eq!(r.indices, vec![2, 3], "global top-2 experts within the selected groups");
        // dense weights nonzero exactly at the selected indices.
        for i in 0..8 {
            assert_eq!(r.dense_weights[i] != 0.0, r.indices.contains(&i));
        }
    }

    #[test]
    fn bias_changes_selection_not_combine_weight() {
        let c = cfg(8, 2, 4, 2, 1.0, true);
        let logits = [0.0_f32, 0.0, 1.0, 1.05, 0.0, 0.0, 1.0, 0.9];
        // Without bias: group(2,3) sum ~ sigmoid(1)+sigmoid(1.05); group(6,7) ~
        // sigmoid(1)+sigmoid(0.9). Bias lifts expert 7 so its group + itself win.
        let no_bias = v3_route(&logits, &[0.0_f32; 8], &c).unwrap();
        let mut bias = [0.0_f32; 8];
        bias[7] = 0.5; // pushes expert 7 (and group 3) up for SELECTION only
        let with_bias = v3_route(&logits, &bias, &c).unwrap();
        assert_ne!(no_bias.indices, with_bias.indices, "bias must influence selection");
        // The combine weight for a selected expert is the ORIGINAL sigmoid score
        // (no bias). Expert 7's raw score = sigmoid(0.9); check the renormalised
        // weight equals raw/(sum of raw selected), independent of the +0.5 bias.
        if with_bias.indices.contains(&7) {
            let raw7 = super::sigmoid_f64(0.9);
            let sum_raw: f64 = with_bias.indices.iter().map(|&i| super::sigmoid_f64(logits[i] as f64)).sum();
            let slot = with_bias.indices.iter().position(|&i| i == 7).unwrap();
            assert!((with_bias.weights[slot] as f64 - raw7 / sum_raw).abs() < 1e-6);
        }
    }

    #[test]
    fn routed_scaling_factor_multiplies_weights() {
        let logits = [0.2_f32, 0.4, 2.0, 2.1, 0.1, 0.0, 1.8, 1.9];
        let bias = [0.0_f32; 8];
        let a = v3_route(&logits, &bias, &cfg(8, 2, 4, 2, 1.0, true)).unwrap();
        let b = v3_route(&logits, &bias, &cfg(8, 2, 4, 2, 2.5, true)).unwrap();
        assert_eq!(a.indices, b.indices);
        for (wa, wb) in a.weights.iter().zip(&b.weights) {
            assert!((wb - wa * 2.5).abs() < 1e-5, "scaling multiplies the combine weights");
        }
    }

    #[test]
    fn norm_topk_prob_normalises_then_scales() {
        let logits = [0.0_f32, 0.0, 2.0, 2.2, 0.0, 0.0, 1.5, 1.7];
        let bias = [0.0_f32; 8];
        let r = v3_route(&logits, &bias, &cfg(8, 2, 4, 2, 1.0, true)).unwrap();
        let sum: f32 = r.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "with scale 1.0, normalised weights sum to 1");
        // Without norm, weights are the raw sigmoid scores (× scale 1.0).
        let raw = v3_route(&logits, &bias, &cfg(8, 2, 4, 2, 1.0, false)).unwrap();
        let slot = raw.indices.iter().position(|&i| i == 3).unwrap();
        assert!((raw.weights[slot] as f64 - super::sigmoid_f64(2.2)).abs() < 1e-6);
    }

    #[test]
    fn group_limiting_excludes_experts_outside_selected_groups() {
        // Make group 0 (idx 0,1) individually have the single largest expert,
        // but group sum lower than two other groups → expert 0 must be excluded.
        let c = cfg(8, 2, 4, 2, 1.0, true);
        let logits = [5.0_f32, -5.0, 1.0, 1.0, 1.0, 1.0, -5.0, -5.0];
        // group0 sum = sig(5)+sig(-5)≈1.0; group1=sig1+sig1≈1.46; group2≈1.46;
        // group3≈0.013 → groups 1,2 selected; expert 0 (largest single) excluded.
        let bias = [0.0_f32; 8];
        let r = v3_route(&logits, &bias, &c).unwrap();
        assert!(!r.indices.contains(&0), "expert 0 excluded by group limiting");
        assert!(r.indices.iter().all(|&i| (2..6).contains(&i)));
    }

    #[test]
    fn deterministic() {
        let logits = [0.3_f32, 0.4, 1.1, 1.2, 0.9, 0.8, 1.5, 1.6];
        let bias = [0.05_f32, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0];
        let c = cfg(8, 3, 4, 2, 2.5, true);
        let a = v3_route(&logits, &bias, &c).unwrap();
        let b = v3_route(&logits, &bias, &c).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn fail_loud_on_bad_config_and_inputs() {
        // not divisible
        assert!(matches!(
            v3_route(&[0.0; 7], &[0.0; 7], &cfg(7, 2, 4, 2, 1.0, true)),
            Err(V3RouterError::ExpertsNotDivisibleByGroups { .. })
        ));
        // group too small (1 expert per group)
        assert!(matches!(
            v3_route(&[0.0; 4], &[0.0; 4], &cfg(4, 1, 4, 2, 1.0, true)),
            Err(V3RouterError::GroupTooSmall { .. })
        ));
        // topk_group > n_group
        assert!(matches!(
            v3_route(&[0.0; 8], &[0.0; 8], &cfg(8, 2, 4, 5, 1.0, true)),
            Err(V3RouterError::TopkGroupExceedsGroups { .. })
        ));
        // top_k exceeds selectable (2 groups × 2 = 4 selectable, ask 5)
        assert!(matches!(
            v3_route(&[0.0; 8], &[0.0; 8], &cfg(8, 5, 4, 2, 1.0, true)),
            Err(V3RouterError::TopKExceedsSelectable { .. })
        ));
        // length mismatch
        assert!(matches!(
            v3_route(&[0.0; 6], &[0.0; 8], &cfg(8, 2, 4, 2, 1.0, true)),
            Err(V3RouterError::LengthMismatch { .. })
        ));
        // non-finite
        assert!(matches!(
            v3_route(&[f32::NAN; 8], &[0.0; 8], &cfg(8, 2, 4, 2, 1.0, true)),
            Err(V3RouterError::NonFinite { .. })
        ));
    }
}
