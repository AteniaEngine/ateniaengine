//! **MOE-1** — Mixture-of-Experts **certification substrate** (experimental).
//!
//! This module is **infrastructure only**. It does NOT execute, support, or
//! implement MoE in any form — no router, no top-k, no dispatch, no sparse
//! execution, no MoE family, no graph operators, no runtime change. Its sole
//! purpose is to prepare the *certification* groundwork so that, when a real
//! MoE path is eventually built, there is already a stable, well-defined way
//! to describe a certification fixture and decide how it can be certified.
//!
//! Background: the MOE-0 architecture audit concluded that the data plane
//! (loaders, tier planner, disk streaming) is mostly MoE-ready, the compute
//! plane (graph engine, runtime) is the principal blocker, and — critically —
//! **certification is gated on having a small MoE fixture**, because the
//! ADR-004 F64 reference (a PyTorch double-precision forward) is infeasible
//! for large MoE checkpoints. This module encodes that reasoning as code +
//! tests so it cannot drift.
//!
//! See `docs/MOE_CERTIFICATION_SUBSTRATE.md` for the full analysis.

pub mod binding;
pub mod convention;
pub mod data_plane;
pub mod decoder_layer;
pub mod dense;
pub mod detect;
/// **MOE-FULL-9** — MoE family recognition + loader fail-loud preparation.
pub mod family;
pub mod fixture;
pub mod full_forward;
/// **MOE-FULL-7** — experimental MoE generation (prefill + KV cache + decode).
pub mod generate;
/// **MOE-FULL-9** — Grouped-Query Attention via load-time K/V weight tiling.
pub mod gqa;
pub mod graph_op;
pub mod layer;
/// **MOE-FULL-14** — machine-readable MoE certification manifest.
pub mod manifest;
/// **MOE-FULL-12** — DeepSeek-V2 MLA attention + imperative DeepSeek forward.
pub mod mla;
/// **MOE-FULL-14** — controlled production MoE path (gating + dispatcher).
pub mod production;
/// **MOE-FULL-3** — experimental Mixtral family adapter (load-only metadata).
pub mod mixtral_adapter;
pub mod numerical;
/// **MOE-FULL-8** — experimental tiered expert residency (RAM / NVMe).
pub mod residency;
/// **MOE-FULL-10** — controlled productive Mixtral runtime (opt-in).
pub mod runtime;
pub mod smoke;
pub mod sparse;
pub mod stack;
pub mod validation;

pub use numerical::{MoeNumericalReport, NumericalMetrics, MOE_NUMERICAL_TOLERANCE};

pub use binding::{
    build_packed_layer, build_real_layer, packed_dims, BindingShape, MoeBindingError,
    PackedExpertDims, RealExpertTensorBinding,
};

pub use smoke::{
    discover_safetensors_files, LocalMoeCheckpoint, MinimalMoeConfig, MoeSmokeError,
};

pub use validation::{RealMoeCheckpointValidation, ValidationReport};

pub use mixtral_adapter::{
    MixtralAdapter, MixtralAdapterError, MixtralExpertLayout, MixtralMetadata, MixtralTensorSpec,
};

pub use decoder_layer::{
    build_experimental_decoder_layer, decoder_layer_reference, register_and_build_decoder_layer,
    ExpAttnWeights,
};

pub use layer::{MoeConfigError, MoeLayerConfig, MoeLayerError, RealMoeLayer};

pub use stack::{MoeStackConfig, MoeStackConfigError, MoeStackError, RealMoeStack};

pub use data_plane::{ExpertTensors, MoeLayerMap, MoeTensorEntry, MoeWeightMap};

pub use graph_op::{
    execute_conditional_expert, execute_dynamic_dispatch, execute_real_moe_layer,
    execute_real_moe_layer_with, execute_sparse_reference, expert_weight_in_selection,
    get_layer, get_real_moe_layer, register_layer, register_real_moe_layer, DynamicDispatchOutput,
};

pub use dense::{
    build_fixture_layer, softmax, MoeDenseError, MoeDenseExpert, MoeDenseLayer, MoeRouterOutput,
};
pub use sparse::{
    combine_selected, top_k_routing, top_k_routing_with, MoeSparseError, MoeSparseForwardOutput,
    TopKSelection,
};

pub use layer::MoeExecutionConvention;

pub use convention::MoeConventionResolver;

pub use detect::{
    classify_tensor_name, detect_moe, is_moe_expert_tensor, is_moe_router_tensor,
    unsupported_message, MoeDetection, TensorNameInfo, TensorRole,
};
pub use family::{
    classify_family, experimental_moe_enabled, moe_failloud_report, validate_family_config,
    FamilyConfigValidation, MoeFamily, MoeFamilyDescriptor, ENABLE_MOE_ENV, EXPERIMENTAL_MOE_ENV,
};
pub use runtime::{MixtralRuntime, MixtralRuntimeError, MoeRuntime, MoeRuntimeError};
pub use manifest::{MoeCertEntry, MoeCertManifest, MoeCertScope};
pub use production::{controlled_moe_generate, diagnose_moe, ControlledMoeError, MoeDiagnosis};
pub use fixture::{
    f64_reference_weight_bytes, recommend_strategy, FixtureMoESpec, FixtureSpecError,
    MoECertificationStrategy,
};
