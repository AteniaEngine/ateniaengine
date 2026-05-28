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

pub mod evaluator;
pub mod gptq;
pub mod policy;

pub use evaluator::{
    evaluate_tensor_policies, evaluate_tensor_policy, EvalError, TensorEvalInput,
    TensorEvalResult,
};
pub use gptq::{
    apply_gptq_reconstruction_inplace, approximate_hessian_diag, GptqConfig, GptqError,
};
pub use policy::{
    AwqPolicy, Bf16Fallback, CalibrationContext, GptqPolicy, HybridPolicy, PlainInt8,
    PolicyError, QuantizationPolicy,
};
