//! **AQS-6** — certification report, deterministic policy ranking, and
//! experimental manifest draft.
//!
//! This layer turns the raw [`EndToEndEvalResult`]s produced by the
//! AQS-4 / AQS-5 harness into an **auditable surface**: each candidate is
//! classified, the set is ranked deterministically, the best certified
//! and best useful-lossy policies are selected, and a human report +
//! manifest *draft* are rendered.
//!
//! ## Why this exists (the AQS pivot)
//!
//! AQS-1..5 falsified five distinct weight-only quantisation mechanisms
//! against ADR-004 strict (`max_abs_diff < 0.5`) on TinyLlama: plain
//! INT8, β outlier, β-pivot AWQ, β-pivot hybrid, and real GPTQ. None
//! crossed the gate. Rather than chase a sixth technique, AQS-6 delivers
//! the audit's actual differentiator — **automatically classifying and
//! certifying which quantisation is safe for a given model against F64**.
//! The value is the report, not a magic algorithm.
//!
//! ## Scope contract (AQS-6)
//!
//! * **No new quantisation techniques.** Consumes existing results only.
//! * **No CLI, no runtime manifest integration, no `numcert` changes.**
//!   The manifest here is an explicitly-labelled `*-draft` string.
//! * **No default-on policies.** Experimental, CPU-only, isolated.
//! * AWQ is **never** labelled certified while `max_abs_diff ≥ 0.5`.
//!
//! See `docs/HANDOFF_AQS_6.md`.

use crate::quant::end_to_end::EndToEndEvalResult;

/// ADR-004 strict logit gate.
pub const ADR_004_GATE: f32 = 0.5;

// ============================================================================
// Certification status
// ============================================================================

/// Classification of one policy's end-to-end result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CertificationStatus {
    /// `max_abs_diff < 0.5` AND `argmax_match`. Safe to ship under
    /// ADR-004 strict.
    Adr004Certified,
    /// `argmax_match` holds but `max_abs_diff ≥ 0.5`, and the policy
    /// actually compresses (`compression_ratio > 1.0`). Not certified,
    /// but a useful trade-off worth surfacing.
    UsefulLossy,
    /// Argmax broke, metrics are non-finite, or the policy does not
    /// compress while being lossy. Not recommendable.
    Failed,
}

impl CertificationStatus {
    /// Stable lowercase identifier for manifests / reports.
    pub fn as_str(&self) -> &'static str {
        match self {
            CertificationStatus::Adr004Certified => "adr004_certified",
            CertificationStatus::UsefulLossy => "useful_lossy",
            CertificationStatus::Failed => "failed",
        }
    }

    /// Ranking rank: lower is better (certified < useful_lossy < failed).
    fn group_rank(&self) -> u8 {
        match self {
            CertificationStatus::Adr004Certified => 0,
            CertificationStatus::UsefulLossy => 1,
            CertificationStatus::Failed => 2,
        }
    }
}

// ============================================================================
// Per-policy report
// ============================================================================

/// One classified policy result.
#[derive(Debug, Clone, PartialEq)]
pub struct AqsPolicyReport {
    pub candidate_name: String,
    pub status: CertificationStatus,
    pub max_abs_diff: f32,
    pub mean_abs_diff: f32,
    pub rmse: f32,
    pub argmax_match: bool,
    pub memory_bytes: u64,
    /// `baseline_memory_bytes / memory_bytes`. `None` when either is 0
    /// (e.g. the certified baseline row carries `memory_bytes = 0` by
    /// harness convention — we do not invent a ratio).
    pub compression_ratio: Option<f32>,
}

impl AqsPolicyReport {
    /// Classify a single [`EndToEndEvalResult`] against the ADR-004 gate.
    ///
    /// `baseline_memory_bytes` is used only to derive `compression_ratio`.
    fn classify(
        result: &EndToEndEvalResult,
        baseline_memory_bytes: Option<u64>,
    ) -> Self {
        let finite = result.max_abs_diff.is_finite()
            && result.mean_abs_diff.is_finite()
            && result.rmse.is_finite();

        let compression_ratio = match (baseline_memory_bytes, result.memory_bytes) {
            (Some(base), mem) if base > 0 && mem > 0 => Some(base as f32 / mem as f32),
            _ => None,
        };

        let status = if !finite || !result.argmax_match {
            CertificationStatus::Failed
        } else if (result.max_abs_diff as f64) < ADR_004_GATE as f64 {
            CertificationStatus::Adr004Certified
        } else {
            // argmax holds, drift >= gate → useful only if it compresses.
            match compression_ratio {
                Some(r) if r > 1.0 => CertificationStatus::UsefulLossy,
                _ => CertificationStatus::Failed,
            }
        };

        Self {
            candidate_name: result.candidate_name.clone(),
            status,
            max_abs_diff: result.max_abs_diff,
            mean_abs_diff: result.mean_abs_diff,
            rmse: result.rmse,
            argmax_match: result.argmax_match,
            memory_bytes: result.memory_bytes,
            compression_ratio,
        }
    }

    /// Deterministic ordering key within a status group:
    /// (lower max_abs_diff, higher compression, lower memory, name asc).
    fn rank_key(&self) -> (u8, OrderedF32, OrderedF32, u64, &str) {
        (
            self.status.group_rank(),
            OrderedF32(self.max_abs_diff),
            // Negate compression so that "higher is better" sorts ascending.
            OrderedF32(-self.compression_ratio.unwrap_or(0.0)),
            self.memory_bytes,
            self.candidate_name.as_str(),
        )
    }
}

/// Total-order wrapper over f32 for deterministic sorting. NaN sorts
/// last (it should never appear in a finite-checked report, but we stay
/// defensive rather than panic in `partial_cmp`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF32(f32);
impl Eq for OrderedF32 {}
impl PartialOrd for OrderedF32 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for OrderedF32 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.0.is_nan(), other.0.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => self.0.partial_cmp(&other.0).unwrap(),
        }
    }
}

// ============================================================================
// Full certification report
// ============================================================================

/// The full auditable report over a candidate set, for one model.
#[derive(Debug, Clone, PartialEq)]
pub struct AqsCertificationReport {
    pub model_name: String,
    pub adr004_gate: f32,
    pub baseline_memory_bytes: Option<u64>,
    /// Policies, **sorted by deterministic rank** (best first).
    pub policies: Vec<AqsPolicyReport>,
}

impl AqsCertificationReport {
    /// Build a report from raw end-to-end results.
    ///
    /// `baseline_memory_bytes` is the reference memory used to derive
    /// compression ratios (typically the BF16 / certified footprint). If
    /// `None`, ratios are `None` everywhere.
    pub fn build(
        model_name: impl Into<String>,
        results: &[EndToEndEvalResult],
        baseline_memory_bytes: Option<u64>,
    ) -> Self {
        let mut policies: Vec<AqsPolicyReport> = results
            .iter()
            .map(|r| AqsPolicyReport::classify(r, baseline_memory_bytes))
            .collect();
        policies.sort_by(|a, b| a.rank_key().cmp(&b.rank_key()));
        Self {
            model_name: model_name.into(),
            adr004_gate: ADR_004_GATE,
            baseline_memory_bytes,
            policies,
        }
    }

    /// Best ADR-004-certified policy name (already rank-sorted, so the
    /// first certified entry is the best).
    pub fn best_certified(&self) -> Option<&str> {
        self.policies
            .iter()
            .find(|p| p.status == CertificationStatus::Adr004Certified)
            .map(|p| p.candidate_name.as_str())
    }

    /// Best useful-lossy policy name.
    pub fn best_useful_lossy(&self) -> Option<&str> {
        self.policies
            .iter()
            .find(|p| p.status == CertificationStatus::UsefulLossy)
            .map(|p| p.candidate_name.as_str())
    }

    /// The recommended policy: prefer the best certified one. A lossy
    /// candidate is **never** auto-substituted for a certified one — it
    /// is surfaced separately via [`Self::best_useful_lossy`]. Returns
    /// `None` only if there is no certified policy at all (in which case
    /// the caller should consult `best_useful_lossy` explicitly).
    pub fn recommended_policy(&self) -> Option<&str> {
        self.best_certified()
    }

    /// Find a policy report by candidate name.
    pub fn policy(&self, name: &str) -> Option<&AqsPolicyReport> {
        self.policies.iter().find(|p| p.candidate_name == name)
    }

    // ------------------------------------------------------------------
    // Renderers
    // ------------------------------------------------------------------

    /// Render a deterministic human-readable table.
    pub fn render_human_report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "AQS certification report — model: {} (ADR-004 gate = {})\n",
            self.model_name, self.adr004_gate
        ));
        out.push_str(&format!(
            "{:<18} {:<18} {:>12} {:>8} {:>13}\n",
            "candidate", "status", "max_diff", "argmax", "compression"
        ));
        out.push_str(&"-".repeat(73));
        out.push('\n');
        for p in &self.policies {
            let comp = match p.compression_ratio {
                Some(r) => format!("{r:.2}x"),
                None => "-".to_string(),
            };
            out.push_str(&format!(
                "{:<18} {:<18} {:>12.6} {:>8} {:>13}\n",
                p.candidate_name,
                p.status.as_str(),
                p.max_abs_diff,
                if p.argmax_match { "true" } else { "false" },
                comp,
            ));
        }
        out.push_str(&format!(
            "best certified     : {}\n",
            self.best_certified().unwrap_or("(none)")
        ));
        out.push_str(&format!(
            "best useful lossy  : {}\n",
            self.best_useful_lossy().unwrap_or("(none)")
        ));
        out
    }

    /// Render an **experimental, draft-only** manifest. This is NOT the
    /// productive `numcert` manifest and is never consumed by the
    /// runtime — note the `-draft` schema suffix. Hand-rendered YAML-ish
    /// string (no YAML crate dependency).
    pub fn render_manifest_draft(&self) -> String {
        let mut out = String::new();
        out.push_str("schema_version: \"3.0.0-draft\"\n");
        out.push_str(&format!("model: {}\n", self.model_name));
        out.push_str("adr004:\n");
        out.push_str(&format!("  gate: {}\n", self.adr004_gate));
        out.push_str(&format!(
            "  certified_policy: {}\n",
            self.best_certified().unwrap_or("null")
        ));
        match self.best_useful_lossy() {
            Some(name) => {
                out.push_str("lossy_recommendation:\n");
                out.push_str(&format!("  policy: {name}\n"));
                out.push_str(
                    "  reason: \"argmax stable, best useful lossy compression\"\n",
                );
            }
            None => {
                out.push_str("lossy_recommendation: null\n");
            }
        }
        out.push_str("policies:\n");
        for p in &self.policies {
            out.push_str(&format!("  - name: {}\n", p.candidate_name));
            out.push_str(&format!("    status: {}\n", p.status.as_str()));
            out.push_str(&format!("    max_abs_diff: {:.6}\n", p.max_abs_diff));
            out.push_str(&format!("    mean_abs_diff: {:.6}\n", p.mean_abs_diff));
            out.push_str(&format!("    rmse: {:.6}\n", p.rmse));
            out.push_str(&format!("    argmax_match: {}\n", p.argmax_match));
            out.push_str(&format!("    memory_bytes: {}\n", p.memory_bytes));
            if let Some(r) = p.compression_ratio {
                out.push_str(&format!("    compression_ratio: {r:.4}\n"));
            }
        }
        out
    }
}

// ============================================================================
// Tests (synthetic results — no model, no I/O)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn res(
        name: &str,
        max: f32,
        argmax: bool,
        mem: u64,
    ) -> EndToEndEvalResult {
        EndToEndEvalResult {
            candidate_name: name.to_string(),
            max_abs_diff: max,
            mean_abs_diff: max * 0.1,
            rmse: max * 0.15,
            argmax_match: argmax,
            memory_bytes: mem,
        }
    }

    /// The real consolidated TinyLlama results from AQS-5.
    fn tinyllama_results() -> Vec<EndToEndEvalResult> {
        vec![
            res("bf16", 0.000063, true, 2_260_729_856),
            res("plain_int8", 1.260771, false, 1_165_688_832),
            res("awq", 0.889217, true, 1_165_688_832),
            res("hybrid", 0.831786, false, 1_266_614_272),
            res("gptq", 1.405399, false, 1_167_265_792),
        ]
    }

    const BASE: Option<u64> = Some(2_260_729_856);

    #[test]
    fn classifies_bf16_as_certified() {
        let r = AqsCertificationReport::build("tinyllama", &tinyllama_results(), BASE);
        assert_eq!(
            r.policy("bf16").unwrap().status,
            CertificationStatus::Adr004Certified
        );
    }

    #[test]
    fn classifies_awq_as_useful_lossy_not_certified() {
        let r = AqsCertificationReport::build("tinyllama", &tinyllama_results(), BASE);
        let awq = r.policy("awq").unwrap();
        assert_eq!(awq.status, CertificationStatus::UsefulLossy);
        assert!(awq.max_abs_diff >= ADR_004_GATE, "AWQ must not be certified");
        assert!(awq.argmax_match);
        assert!(awq.compression_ratio.unwrap() > 1.0);
    }

    #[test]
    fn classifies_argmax_false_as_failed() {
        let r = AqsCertificationReport::build("tinyllama", &tinyllama_results(), BASE);
        // plain_int8, hybrid, gptq all have argmax false.
        for name in ["plain_int8", "hybrid", "gptq"] {
            assert_eq!(
                r.policy(name).unwrap().status,
                CertificationStatus::Failed,
                "{name} must be failed"
            );
        }
    }

    #[test]
    fn ranking_prefers_certified_over_lossy() {
        let r = AqsCertificationReport::build("tinyllama", &tinyllama_results(), BASE);
        // First entry is certified, last entries are failed.
        assert_eq!(r.policies[0].status, CertificationStatus::Adr004Certified);
        assert_eq!(r.policies[0].candidate_name, "bf16");
        // AWQ (useful_lossy) ranks above any failed.
        let awq_pos = r.policies.iter().position(|p| p.candidate_name == "awq").unwrap();
        let int8_pos = r.policies.iter().position(|p| p.candidate_name == "plain_int8").unwrap();
        assert!(awq_pos < int8_pos);
    }

    #[test]
    fn ranking_prefers_lower_drift_within_status() {
        // Two certified policies; lower drift ranks first.
        let results = vec![
            res("a_hi", 0.4, true, 100),
            res("b_lo", 0.1, true, 100),
        ];
        let r = AqsCertificationReport::build("m", &results, Some(200));
        assert_eq!(r.policies[0].candidate_name, "b_lo");
    }

    #[test]
    fn ranking_uses_compression_as_tiebreaker() {
        // Same drift + status; higher compression (smaller memory) wins.
        let results = vec![
            res("big", 0.1, true, 200),   // ratio 2.0
            res("small", 0.1, true, 100), // ratio 4.0
        ];
        let r = AqsCertificationReport::build("m", &results, Some(400));
        assert_eq!(r.policies[0].candidate_name, "small");
    }

    #[test]
    fn selects_best_certified_and_best_lossy() {
        let r = AqsCertificationReport::build("tinyllama", &tinyllama_results(), BASE);
        assert_eq!(r.best_certified(), Some("bf16"));
        assert_eq!(r.best_useful_lossy(), Some("awq"));
    }

    #[test]
    fn manifest_draft_contains_gate_and_policy_status() {
        let r = AqsCertificationReport::build("tinyllama-1.1b", &tinyllama_results(), BASE);
        let m = r.render_manifest_draft();
        assert!(m.contains("schema_version: \"3.0.0-draft\""));
        assert!(m.contains("model: tinyllama-1.1b"));
        assert!(m.contains("gate: 0.5"));
        assert!(m.contains("certified_policy: bf16"));
        assert!(m.contains("policy: awq"));
        assert!(m.contains("status: adr004_certified"));
        assert!(m.contains("status: useful_lossy"));
        assert!(m.contains("status: failed"));
    }

    #[test]
    fn human_report_renderer_is_deterministic() {
        let r = AqsCertificationReport::build("tinyllama", &tinyllama_results(), BASE);
        let a = r.render_human_report();
        let b = r.render_human_report();
        assert_eq!(a, b);
        assert!(a.contains("best certified     : bf16"));
        assert!(a.contains("best useful lossy  : awq"));
    }

    #[test]
    fn recommended_policy_does_not_replace_certified_with_lossy() {
        let r = AqsCertificationReport::build("tinyllama", &tinyllama_results(), BASE);
        // Even though AWQ compresses better, the recommendation stays the
        // certified bf16 — lossy is surfaced separately, never substituted.
        assert_eq!(r.recommended_policy(), Some("bf16"));
        assert_ne!(r.recommended_policy(), r.best_useful_lossy());
    }

    #[test]
    fn no_certified_yields_none_recommendation_but_keeps_lossy() {
        // All-lossy set: no certified, but a useful-lossy recommendation.
        let results = vec![res("awq", 0.8, true, 100)];
        let r = AqsCertificationReport::build("m", &results, Some(200));
        assert_eq!(r.recommended_policy(), None);
        assert_eq!(r.best_useful_lossy(), Some("awq"));
    }

    #[test]
    fn lossy_without_compression_is_failed() {
        // argmax true, drift over gate, but no compression (mem == baseline)
        // → not useful → failed.
        let results = vec![res("noop_lossy", 0.8, true, 200)];
        let r = AqsCertificationReport::build("m", &results, Some(200));
        assert_eq!(
            r.policy("noop_lossy").unwrap().status,
            CertificationStatus::Failed
        );
    }
}
