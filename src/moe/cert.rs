//! **NUMERIC-POLICY-3** — certification governance for the MoE numeric modes.
//!
//! NUMERIC-POLICY-1/2 built the fast modes (`Strict` f32, `Fast` GPU, bf16 /
//! int8 tier) and a `PolicyCertificate` *metric*. The audit
//! ([docs/NUMERIC_POLICY_AUDIT.md]) found the gap: certification was **manual,
//! offline, and not persisted**, so a non-`Certified` mode could only be trusted
//! by an operator running an ad-hoc check. This module closes that gap:
//!
//! - a **persisted [`NumericCertificate`]** artifact (JSON) per (model, policy,
//!   tier dtype), produced by a **validation prompt set**;
//! - a **runner** that generates the set under `Certified` (reference) vs the
//!   candidate mode on **one loaded model** (policy toggle + the int8 *sim*),
//!   so no extra tier rebuild / cold load is needed;
//! - a **loader + validator** (model id / policy / tier dtype / manifest version
//!   / prompt-set id / pass-fail must all match);
//! - a **runtime guard**: under `ATENIA_NUMERIC_REQUIRE_CERT=1`, a non-`Certified`
//!   **compute policy** without a valid passing certificate **falls back to
//!   `Certified`** (the tier dtype is a build-time choice — see the guard docs).
//!
//! The `Certified` default is never changed; without `REQUIRE_CERT` the existing
//! opt-in/experimental behaviour is untouched.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::numeric_policy::{
    clear_numeric_policy_override, set_numeric_policy, NumericPolicy, PolicyCertificate,
    STRICT_LOGIT_TOLERANCE,
};

/// Version of the validation prompt set. Bump on any change to [`validation_set`]
/// so stale certificates are rejected.
pub const VALIDATION_SET_ID: &str = "moe-greedy-v1";

/// Version of the certificate format itself.
pub const CERTIFICATE_VERSION: u32 = 1;

/// One validation case: a deterministic greedy generation from fixed prompt ids.
#[derive(Clone, Copy, Debug)]
pub struct ValidationCase {
    pub prompt_ids: &'static [u32],
    pub max_new: usize,
}

/// A small, real, offline validation set: several short prompts of differing
/// lengths. Greedy + deterministic → reproducible token ids. No network, no
/// external model, nothing downloaded. (Token ids are generic small integers;
/// the *reference* output is recomputed per model by the runner, never
/// hard-coded — so the set is model-agnostic.)
pub fn validation_set() -> &'static [ValidationCase] {
    // Token ids are kept in `0..=9` so the set is valid for any real vocab and
    // for the tiny test fixtures (vocab as small as ~16). Greedy + deterministic.
    &[
        ValidationCase { prompt_ids: &[1], max_new: 4 },
        ValidationCase { prompt_ids: &[2, 5, 9], max_new: 4 },
        ValidationCase { prompt_ids: &[3, 7], max_new: 6 },
        ValidationCase { prompt_ids: &[0, 4, 8, 6], max_new: 4 },
        ValidationCase { prompt_ids: &[2, 2, 2], max_new: 8 },
        ValidationCase { prompt_ids: &[5], max_new: 8 },
    ]
}

/// Per-case certification result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CertCaseResult {
    pub prompt_ids: Vec<u32>,
    pub max_new: usize,
    pub expected_tokens: Vec<u32>,
    pub observed_tokens: Vec<u32>,
    pub tokens_match: bool,
    pub argmax_match_rate: f64,
    pub max_abs_diff: f64,
    pub mean_abs_diff: f64,
    pub rmse: f64,
}

/// The persisted certificate: "running `model_id` with (`policy`, `tier_dtype`)
/// reproduces the `Certified` reference on `validation_set_id`, within tolerance".
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NumericCertificate {
    pub certificate_version: u32,
    pub created_at_unix: u64,
    pub code_version: String,
    pub model_id: String,
    pub manifest_version: u32,
    /// Certified / Strict / Fast (the compute policy).
    pub numeric_policy: String,
    /// f32 / bf16 / qint8 (the expert tier dtype being certified).
    pub tier_dtype: String,
    pub validation_set_id: String,
    pub tolerance: f64,
    pub cases: Vec<CertCaseResult>,
    /// Overall: every case's argmax agrees, tokens identical, drift within tol.
    pub pass: bool,
    /// Worst-case max_abs_diff across the set (informative).
    pub worst_max_abs_diff: f64,
}

impl NumericCertificate {
    /// Did the certification pass? (every case argmax + tokens match, drift ≤ tol).
    pub fn passed(&self) -> bool {
        self.pass
    }

    /// Is this certificate valid **for** the requested run: same model, policy,
    /// tier dtype, manifest version, prompt-set id, certificate version — and it
    /// passed. Any mismatch → invalid (→ the caller must fall back / re-certify).
    #[allow(clippy::too_many_arguments)]
    pub fn is_valid_for(
        &self,
        model_id: &str,
        policy: NumericPolicy,
        tier_dtype: &str,
        manifest_version: u32,
        validation_set_id: &str,
    ) -> bool {
        self.certificate_version == CERTIFICATE_VERSION
            && self.model_id == model_id
            && self.numeric_policy == policy.as_str()
            && self.tier_dtype == tier_dtype
            && self.manifest_version == manifest_version
            && self.validation_set_id == validation_set_id
            && self.pass
    }
}

/// Deterministic certificate file path for a (model, policy, tier dtype) combo.
pub fn certificate_path(tier_dir: &Path, policy: NumericPolicy, tier_dtype: &str) -> PathBuf {
    tier_dir.join(format!("numeric_cert_{}_{}.json", policy.as_str(), tier_dtype))
}

/// Load + parse a certificate, or `None` (missing / unparseable → invalid).
pub fn load_certificate(path: &Path) -> Option<NumericCertificate> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// **NUMERIC-POLICY-3** — run the validation set under `Certified` (reference) vs
/// the candidate (`policy` compute + `tier_dtype` storage, the latter modelled
/// by the int8 sim when `qint8`) on **one already-loaded model** (loaded from a
/// lossless bf16 tier), and build a [`NumericCertificate`].
///
/// The model must be loaded with a **lossless** tier (f32/bf16) so the reference
/// is the true Certified output and the candidate's int8 effect is applied via
/// the sim — this avoids a second cold load / tier rebuild. `manifest_version`
/// is the loaded tier's manifest version (recorded for validity matching).
pub fn run_certification(
    rt: &super::runtime::MoeRuntime,
    model_id: &str,
    policy: NumericPolicy,
    tier_dtype: &str,
    manifest_version: u32,
) -> NumericCertificate {
    let sim_int8 = tier_dtype == "qint8";
    let mut cases = Vec::new();
    let mut all_pass = true;
    let mut worst = 0.0f64;

    for case in validation_set() {
        // Reference: Certified compute, no quant sim.
        set_numeric_policy(NumericPolicy::Certified);
        super::residency::set_quant_sim_int8_override(Some(false));
        let reference = rt.generate_full(case.prompt_ids, case.max_new);

        // Candidate: requested compute policy + (int8 sim if qint8 tier).
        set_numeric_policy(policy);
        super::residency::set_quant_sim_int8_override(Some(sim_int8));
        let candidate = rt.generate_full(case.prompt_ids, case.max_new);

        let pc = PolicyCertificate::compare(
            policy,
            &reference.step_logits,
            &candidate.step_logits,
            &reference.tokens,
            &candidate.tokens,
        );
        let case_pass = pc.passes(STRICT_LOGIT_TOLERANCE);
        all_pass &= case_pass;
        worst = worst.max(pc.max_abs_diff);
        cases.push(CertCaseResult {
            prompt_ids: case.prompt_ids.to_vec(),
            max_new: case.max_new,
            expected_tokens: reference.tokens,
            observed_tokens: candidate.tokens,
            tokens_match: pc.tokens_match,
            argmax_match_rate: pc.argmax_match_rate,
            max_abs_diff: pc.max_abs_diff,
            mean_abs_diff: pc.mean_abs_diff,
            rmse: pc.rmse,
        });
    }
    // Clear the in-process overrides so policy/sim resolution reverts to the
    // caller's configuration (env) — never leave a stale override behind.
    clear_numeric_policy_override();
    super::residency::set_quant_sim_int8_override(None);

    NumericCertificate {
        certificate_version: CERTIFICATE_VERSION,
        created_at_unix: now_unix(),
        code_version: env!("CARGO_PKG_VERSION").to_string(),
        model_id: model_id.to_string(),
        manifest_version,
        numeric_policy: policy.as_str().to_string(),
        tier_dtype: tier_dtype.to_string(),
        validation_set_id: VALIDATION_SET_ID.to_string(),
        tolerance: STRICT_LOGIT_TOLERANCE,
        cases,
        pass: all_pass,
        worst_max_abs_diff: worst,
    }
}

/// Persist a certificate to `path` (pretty JSON). Best-effort; returns the path.
pub fn save_certificate(cert: &NumericCertificate, path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_vec_pretty(cert)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

// ============================================================================
// Runtime guard + numeric-mode descriptor
// ============================================================================

/// Whether trusted-mode governance is on (`ATENIA_NUMERIC_REQUIRE_CERT=1`).
pub fn require_cert() -> bool {
    std::env::var("ATENIA_NUMERIC_REQUIRE_CERT").as_deref() == Ok("1")
}

/// **NUMERIC-POLICY-3 guard (compute policy)** — the policy that should actually
/// run, given the requested one and whether a valid passing certificate exists.
///
/// - `require_cert == false` → the requested policy (unchanged opt-in behaviour).
/// - `require_cert == true` and requested is `Certified` → `Certified`.
/// - `require_cert == true`, requested is non-`Certified`, **valid cert** →
///   the requested policy (trusted).
/// - `require_cert == true`, requested is non-`Certified`, **no/invalid cert** →
///   **`Certified`** (safe fallback). The compute policy is runtime-switchable,
///   so this fallback is free; the **tier dtype** is build-time and is guarded
///   separately at load (a qint8 tier without a cert under `require_cert` is
///   refused, not silently downgraded).
pub fn governed_compute_policy(
    requested: NumericPolicy,
    require_cert: bool,
    has_valid_cert: bool,
) -> NumericPolicy {
    if !require_cert || requested == NumericPolicy::Certified || has_valid_cert {
        requested
    } else {
        NumericPolicy::Certified
    }
}

/// One-line description of the **effective** numeric mode for logs/stats.
pub fn numeric_mode_descriptor(
    effective_policy: NumericPolicy,
    tier_dtype: &str,
    cert_status: &str,
) -> String {
    format!(
        "policy={} tier={} cert={}",
        effective_policy.as_str(),
        tier_dtype,
        cert_status
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cert(model: &str, policy: NumericPolicy, tier: &str, pass: bool) -> NumericCertificate {
        NumericCertificate {
            certificate_version: CERTIFICATE_VERSION,
            created_at_unix: 1,
            code_version: "test".into(),
            model_id: model.into(),
            manifest_version: 5,
            numeric_policy: policy.as_str().into(),
            tier_dtype: tier.into(),
            validation_set_id: VALIDATION_SET_ID.into(),
            tolerance: STRICT_LOGIT_TOLERANCE,
            cases: vec![],
            pass,
            worst_max_abs_diff: 0.0,
        }
    }

    #[test]
    fn certificate_serialization_roundtrips() {
        let c = sample_cert("m1", NumericPolicy::Strict, "qint8", true);
        let json = serde_json::to_string(&c).unwrap();
        let back: NumericCertificate = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model_id, "m1");
        assert_eq!(back.numeric_policy, "strict");
        assert_eq!(back.tier_dtype, "qint8");
        assert!(back.passed());
    }

    #[test]
    fn is_valid_for_matches_and_rejects() {
        let c = sample_cert("m1", NumericPolicy::Strict, "qint8", true);
        assert!(c.is_valid_for("m1", NumericPolicy::Strict, "qint8", 5, VALIDATION_SET_ID));
        // Wrong model / policy / tier / manifest / set id → invalid.
        assert!(!c.is_valid_for("m2", NumericPolicy::Strict, "qint8", 5, VALIDATION_SET_ID));
        assert!(!c.is_valid_for("m1", NumericPolicy::Fast, "qint8", 5, VALIDATION_SET_ID));
        assert!(!c.is_valid_for("m1", NumericPolicy::Strict, "bf16", 5, VALIDATION_SET_ID));
        assert!(!c.is_valid_for("m1", NumericPolicy::Strict, "qint8", 4, VALIDATION_SET_ID));
        assert!(!c.is_valid_for("m1", NumericPolicy::Strict, "qint8", 5, "other-set"));
    }

    #[test]
    fn failed_certificate_is_never_valid() {
        let c = sample_cert("m1", NumericPolicy::Strict, "qint8", false);
        assert!(!c.is_valid_for("m1", NumericPolicy::Strict, "qint8", 5, VALIDATION_SET_ID));
    }

    #[test]
    fn guard_falls_back_without_cert_and_allows_with_cert() {
        // No governance → requested unchanged.
        assert_eq!(
            governed_compute_policy(NumericPolicy::Strict, false, false),
            NumericPolicy::Strict
        );
        // Governance on, no cert → Certified.
        assert_eq!(
            governed_compute_policy(NumericPolicy::Strict, true, false),
            NumericPolicy::Certified
        );
        // Governance on, valid cert → requested.
        assert_eq!(
            governed_compute_policy(NumericPolicy::Strict, true, true),
            NumericPolicy::Strict
        );
        // Certified is always Certified.
        assert_eq!(
            governed_compute_policy(NumericPolicy::Certified, true, false),
            NumericPolicy::Certified
        );
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = std::env::temp_dir().join(format!("atenia_cert_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = certificate_path(&dir, NumericPolicy::Strict, "qint8");
        let c = sample_cert("m1", NumericPolicy::Strict, "qint8", true);
        save_certificate(&c, &path).unwrap();
        let loaded = load_certificate(&path).expect("load");
        assert!(loaded.is_valid_for("m1", NumericPolicy::Strict, "qint8", 5, VALIDATION_SET_ID));
        // Missing file → None.
        assert!(load_certificate(&dir.join("nope.json")).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn descriptor_format() {
        let d = numeric_mode_descriptor(NumericPolicy::Strict, "qint8", "valid");
        assert_eq!(d, "policy=strict tier=qint8 cert=valid");
    }
}
