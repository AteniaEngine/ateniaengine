//! **AQS-1** — experimental quantisation-policy abstraction layer.
//!
//! This module is the substrate the AQS (Atenia Quantization Search)
//! design audit (`docs/AQS_ARCHITECTURE_AUDIT.md`) identifies as the
//! minimum extensible surface to prepare for AQS-2 / AQS-3 without
//! changing any productive runtime behaviour.
//!
//! ## Scope contract (AQS-1)
//!
//! * **Wraps**, does not replace, the existing M9 INT8 / β outlier /
//!   β-pivot AWQ / β-pivot hybrid helpers in
//!   [`crate::tensor::quantizer`].
//! * **No** search, no GPTQ, no CUDA, no tier-planner integration,
//!   no loader / manifest changes, no default-on policies.
//! * **Experimental, opt-in, CPU-only.** Nothing in this module is
//!   reachable from the productive path; consumers must construct a
//!   `&dyn QuantizationPolicy` explicitly.
//!
//! See `docs/HANDOFF_AQS_1.md` for the milestone summary.

pub mod certification;
pub mod end_to_end;
pub mod evaluator;
pub mod gptq;
pub mod policy;
pub mod search;

pub use certification::{
    AqsCertificationReport, AqsPolicyReport, CertificationStatus, ADR_004_GATE,
};
pub use search::{
    default_candidate_grid, policy_for_kind, search_from_end_to_end_results,
    search_tensor_local, AqsCandidateSpec, AqsLocalTensorRanking, AqsPolicyKind,
    AqsSearchConfig, AqsSearchResult,
};
pub use end_to_end::{
    logit_drift_metrics, render_result_table, EndToEndEvalResult, PolicyEvalCandidate,
};
pub use evaluator::{
    evaluate_tensor_policies, evaluate_tensor_policy, EvalError, TensorEvalInput,
    TensorEvalResult,
};
pub use gptq::{
    apply_gptq_reconstruction_inplace, approximate_hessian_diag, GptqConfig, GptqError,
};
pub use gptq::{
    apply_gptq_real_inplace, cholesky_decompose, cholesky_inverse, compute_hessian,
    GptqRealConfig,
};
pub use policy::{
    AwqPolicy, Bf16Fallback, CalibrationContext, GptqPolicy, GptqSurrogatePolicy, HybridPolicy,
    PlainInt8, PolicyError, QuantizationPolicy,
};
