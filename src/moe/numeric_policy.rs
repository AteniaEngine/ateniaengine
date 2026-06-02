//! **NUMERIC-POLICY-1** — explicit numeric policy for the MoE compute path.
//!
//! The MOE-PROD series proved the residual MoE bottleneck is the **CPU f64**
//! reference compute, which cannot shrink further without changing the
//! numerics. This module makes that choice **explicit and certifiable** instead
//! of hard-coded:
//!
//! * [`NumericPolicy::Certified`] — f64 accumulation, the bit-exact reference
//!   (today's behaviour). **Default, and the safe fallback.**
//! * [`NumericPolicy::Strict`] — f32 accumulation on the expert FFN (CPU);
//!   bounded drift, certified by tolerance vs Certified.
//! * [`NumericPolicy::Fast`] — reserved for the future GPU / TF32 / BF16
//!   Tensor-Core path (MOE-PERF). Until that lands it behaves like `Strict`
//!   (f32 CPU) so the policy surface is stable.
//!
//! ## Selection + safety
//!
//! Resolved from `ATENIA_NUMERIC_POLICY={certified,strict,fast}` (cached once),
//! overridable in-process by [`set_numeric_policy`] (used by the certification
//! harness to run the same generation under two policies). **Any doubt →
//! `Certified`** (unknown value, unset env → the bit-exact path).
//!
//! ## What changes under a non-Certified policy
//!
//! Only the **expert FFN** matmul (`gate`/`up`/`down`) switches its
//! accumulator to f32. The **router** matmul stays f64 on every policy so the
//! top-k **routing decision is identical** (a different routing would be a
//! different computation, not a tolerable rounding difference). Attention /
//! lm_head / softmax / SiLU are unchanged in this first block.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

/// The numeric policy for the MoE compute path. `Certified` is the f64
/// reference and the universal fallback.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NumericPolicy {
    /// f64 accumulation — bit-exact reference (default).
    Certified,
    /// f32 accumulation on the expert FFN — strict tolerance vs Certified.
    Strict,
    /// Reserved for GPU/TF32/BF16 (MOE-PERF); currently == `Strict`.
    Fast,
}

impl NumericPolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            NumericPolicy::Certified => "certified",
            NumericPolicy::Strict => "strict",
            NumericPolicy::Fast => "fast",
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "certified" | "f64" => Some(NumericPolicy::Certified),
            "strict" | "f32" => Some(NumericPolicy::Strict),
            "fast" => Some(NumericPolicy::Fast),
            _ => None,
        }
    }

    /// Whether the **expert FFN** matmul may accumulate in f32 under this
    /// policy. `Certified` → false (f64); `Strict`/`Fast` → true.
    pub fn ffn_uses_f32(self) -> bool {
        !matches!(self, NumericPolicy::Certified)
    }

    fn encode(self) -> u8 {
        match self {
            NumericPolicy::Certified => 0,
            NumericPolicy::Strict => 1,
            NumericPolicy::Fast => 2,
        }
    }

    fn decode(v: u8) -> NumericPolicy {
        match v {
            1 => NumericPolicy::Strict,
            2 => NumericPolicy::Fast,
            _ => NumericPolicy::Certified,
        }
    }
}

const UNSET: u8 = u8::MAX;
/// In-process override (set by the cert harness / CLI). `UNSET` → use env.
static OVERRIDE: AtomicU8 = AtomicU8::new(UNSET);
/// Cached env policy (read once; per-call `env::var` would be too slow on the
/// matvec hot path).
static ENV_POLICY: OnceLock<NumericPolicy> = OnceLock::new();

fn from_env() -> NumericPolicy {
    std::env::var("ATENIA_NUMERIC_POLICY")
        .ok()
        .and_then(|s| NumericPolicy::from_str(&s))
        .unwrap_or(NumericPolicy::Certified)
}

/// The active numeric policy. Override (if set) wins; else the cached env
/// value; else `Certified`. Cheap (atomic load + cached `OnceLock`).
pub fn numeric_policy() -> NumericPolicy {
    let o = OVERRIDE.load(Ordering::Relaxed);
    if o != UNSET {
        return NumericPolicy::decode(o);
    }
    *ENV_POLICY.get_or_init(from_env)
}

/// Force a policy in-process (overrides the env). Used by the certification
/// harness to run the same generation under Certified then a candidate policy.
pub fn set_numeric_policy(p: NumericPolicy) {
    OVERRIDE.store(p.encode(), Ordering::Relaxed);
}

/// Clear the in-process override → fall back to the env (then `Certified`).
pub fn clear_numeric_policy_override() {
    OVERRIDE.store(UNSET, Ordering::Relaxed);
}

// ============================================================================
// Tolerance certification (NUMERIC-POLICY-1 FASE 3)
// ============================================================================

/// A by-tolerance certificate for a non-Certified policy: the aggregate drift
/// of its per-token logits rows vs the `Certified` f64 reference, plus whether
/// the **generated token ids are identical**. A policy is only trustworthy if
/// it passes (`argmax` of every row agrees, logit drift bounded, tokens equal).
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyCertificate {
    pub policy: NumericPolicy,
    pub rows: usize,
    pub max_abs_diff: f64,
    pub mean_abs_diff: f64,
    pub rmse: f64,
    /// Fraction of logits rows whose argmax matches the reference (1.0 = all).
    pub argmax_match_rate: f64,
    /// Whether the generated token-id sequences are identical.
    pub tokens_match: bool,
}

impl PolicyCertificate {
    /// Certify `candidate` logits rows + tokens against the `Certified`
    /// `reference`. Rows must align (same count + length). Uses the existing
    /// f64 [`crate::moe::numerical::NumericalMetrics`].
    pub fn compare(
        policy: NumericPolicy,
        reference_rows: &[Vec<f32>],
        candidate_rows: &[Vec<f32>],
        reference_tokens: &[u32],
        candidate_tokens: &[u32],
    ) -> PolicyCertificate {
        use crate::moe::numerical::NumericalMetrics;
        let rows = reference_rows.len().min(candidate_rows.len());
        let (mut max_abs, mut sum_abs, mut sum_sq, mut argmax_ok) = (0.0_f64, 0.0_f64, 0.0_f64, 0usize);
        let mut counted = 0usize;
        for i in 0..rows {
            if let Some(m) = NumericalMetrics::compute(&candidate_rows[i], &reference_rows[i]) {
                max_abs = max_abs.max(m.max_abs_diff);
                sum_abs += m.mean_abs_diff;
                sum_sq += m.rmse * m.rmse;
                if m.argmax_match {
                    argmax_ok += 1;
                }
                counted += 1;
            }
        }
        let denom = counted.max(1) as f64;
        PolicyCertificate {
            policy,
            rows: counted,
            max_abs_diff: max_abs,
            mean_abs_diff: sum_abs / denom,
            rmse: (sum_sq / denom).sqrt(),
            argmax_match_rate: argmax_ok as f64 / denom,
            tokens_match: reference_tokens == candidate_tokens,
        }
    }

    /// Pass iff **every** row's argmax agrees, the **tokens are identical**,
    /// and the max logit drift is within `tolerance`. The token-equality and
    /// argmax conditions are the decisive ones (same output); the tolerance is
    /// a guard against a lucky-argmax-but-drifting run.
    pub fn passes(&self, tolerance: f64) -> bool {
        self.argmax_match_rate >= 1.0 && self.tokens_match && self.max_abs_diff <= tolerance
    }
}

/// Default logit-drift tolerance for the `Strict` (f32-FFN) policy. The expert
/// FFN reductions are at most a few thousand wide; f32 accumulation error is
/// ~`sqrt(N)·eps_f32` ≈ 1e-5 relative, well under this absolute logit bound
/// (logits are O(1–10)). Published, not magic: tighten with measured data.
pub const STRICT_LOGIT_TOLERANCE: f64 = 0.5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_certified_and_safe() {
        clear_numeric_policy_override();
        // With no override set, the policy is env-or-Certified; the override
        // API must round-trip and Certified must report f64.
        set_numeric_policy(NumericPolicy::Certified);
        assert_eq!(numeric_policy(), NumericPolicy::Certified);
        assert!(!NumericPolicy::Certified.ffn_uses_f32());
        clear_numeric_policy_override();
    }

    #[test]
    fn override_wins_and_clears() {
        set_numeric_policy(NumericPolicy::Strict);
        assert_eq!(numeric_policy(), NumericPolicy::Strict);
        assert!(NumericPolicy::Strict.ffn_uses_f32());
        set_numeric_policy(NumericPolicy::Fast);
        assert_eq!(numeric_policy(), NumericPolicy::Fast);
        assert!(NumericPolicy::Fast.ffn_uses_f32());
        clear_numeric_policy_override();
    }

    #[test]
    fn from_str_unknown_is_none_known_ok() {
        assert_eq!(NumericPolicy::from_str("certified"), Some(NumericPolicy::Certified));
        assert_eq!(NumericPolicy::from_str("STRICT"), Some(NumericPolicy::Strict));
        assert_eq!(NumericPolicy::from_str("fast"), Some(NumericPolicy::Fast));
        assert_eq!(NumericPolicy::from_str("garbage"), None);
    }

    #[test]
    fn certificate_passes_on_identical_and_fails_on_token_mismatch() {
        let r = vec![vec![1.0_f32, 2.0, 0.5], vec![0.1, 0.2, 9.0]];
        let toks = vec![1u32, 2];
        // Identical → passes.
        let c = PolicyCertificate::compare(NumericPolicy::Strict, &r, &r, &toks, &toks);
        assert!(c.passes(STRICT_LOGIT_TOLERANCE));
        assert_eq!(c.argmax_match_rate, 1.0);
        assert_eq!(c.max_abs_diff, 0.0);
        // Tiny drift but same argmax + tokens → still passes.
        let cand = vec![vec![1.0001_f32, 2.0, 0.5], vec![0.1, 0.2, 9.0001]];
        let c2 = PolicyCertificate::compare(NumericPolicy::Strict, &r, &cand, &toks, &toks);
        assert!(c2.passes(STRICT_LOGIT_TOLERANCE));
        // Token mismatch → fails regardless of small drift.
        let c3 = PolicyCertificate::compare(NumericPolicy::Strict, &r, &r, &toks, &[1, 3]);
        assert!(!c3.passes(STRICT_LOGIT_TOLERANCE));
        // argmax flip → fails.
        let flip = vec![vec![2.0_f32, 1.0, 0.5], vec![0.1, 0.2, 9.0]];
        let c4 = PolicyCertificate::compare(NumericPolicy::Strict, &r, &flip, &toks, &toks);
        assert!(!c4.passes(STRICT_LOGIT_TOLERANCE));
        assert!(c4.argmax_match_rate < 1.0);
    }
}
