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

pub mod fixture;

pub use fixture::{
    f64_reference_weight_bytes, recommend_strategy, FixtureMoESpec, FixtureSpecError,
    MoECertificationStrategy,
};
