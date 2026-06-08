//! **MOE-INTEGRATE-2** — opt-in **resolver bridge** from a declarative
//! [`MoeStructuralSpec`] to a runtime plan (and, opt-in, the certified
//! [`MoeRuntime`]).
//!
//! This is the bridge the roadmap reserved: it turns a declarative MoE spec
//! ([`super::moe_family_spec`]) into a typed [`ResolvedMoeRuntimePlan`] —
//! family, routing, attention, expert layout, dense-first, YaRN, disk-tier hint,
//! execution convention, and a **runnable** flag — and can, **opt-in only**, hand
//! a runnable plan off to the existing certified runtime.
//!
//! ## Contract (critical)
//!
//! - **Handwritten certified paths remain default.** This does NOT change the
//!   productive routing ([`crate::moe::production::decide_route`]), the dense
//!   loader fail-loud, the numerics, or any manifest. The certified
//!   [`MoeRuntime`] assembly is unchanged; this only *resolves + validates* a
//!   spec and, behind the opt-in, *delegates* to it.
//! - **Opt-in.** [`MoeSpecResolver::load_runtime`] requires the explicit MoE
//!   opt-in (`ATENIA_ENABLE_MOE=1`); without it, it fails loud before touching
//!   disk. Resolving a plan ([`MoeSpecResolver::resolve`]) is pure + always safe.
//! - **No new family support claimed.** The resolver describes/validates the
//!   four known archs; it does not enable any new family. **DeepSeek-V3 is
//!   mechanism-only, non-runnable** — asking for its runtime fails loud.
//! - **Equivalence guard.** A spec whose convention disagrees with the
//!   authoritative handwritten [`crate::moe::family::MoeFamily`] descriptor is a
//!   hard error (the spec can never silently diverge from the certified path).

use crate::moe::family::{experimental_moe_enabled, MoeFamily};
use crate::moe::manifest::{MoeCertManifest, MoeCertScope};
use crate::moe::runtime::{MoeRuntime, MoeRuntimeError};
use crate::moe::MoeExecutionConvention;
use crate::moe::v3_router::V3RouterConfig;

use super::moe_family_spec::{
    AttentionKind, DiskTierPolicy, MoeArch, MoeSpecError, MoeStructuralSpec, RoutingScheme,
};
use super::moe_spec::{ExpertLayout, SharedGating};
use std::path::Path;

/// The router plan resolved from a spec — what the runtime would do at routing.
#[derive(Debug, Clone, PartialEq)]
pub enum RoutingPlan {
    /// `softmax(W·x)` top-k (Mixtral / Qwen-MoE / DeepSeek-V2), with renorm flag.
    SoftmaxTopK { renormalize: bool },
    /// `sigmoid(W·x)` top-k (no group structure), with renorm flag.
    SigmoidTopK { renormalize: bool },
    /// DeepSeek-V3 `noaux_tc` group-limited routing (see [`V3RouterConfig`]).
    V3NoAuxGroupLimited,
}

/// A fully resolved runtime plan for a MoE spec. Describes how the spec maps to
/// the runtime; it does **not** execute.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedMoeRuntimePlan {
    pub arch: MoeArch,
    pub family: MoeFamily,
    /// The execution convention the certified runtime uses for this family
    /// (`Atenia` renorm+ungated, or `HuggingFaceQwen` no-renorm+sigmoid-shared).
    pub convention: MoeExecutionConvention,
    pub routing: RoutingPlan,
    pub attention: AttentionKind,
    pub expert_layout: ExpertLayout,
    pub shared_present: bool,
    pub shared_gating: SharedGating,
    pub renormalize_topk: bool,
    pub dense_first_layers: usize,
    pub yarn: bool,
    pub disk_tier_hint: DiskTierPolicy,
    /// A representative `V3RouterConfig` (for the V3 mechanism arch only).
    pub v3_router_config: Option<V3RouterConfig>,
    /// Whether this arch describes a runnable **model** (false for the V3
    /// routing mechanism, which is non-runnable).
    pub runnable: bool,
    /// The manifest certification scope for the family (informational; the
    /// resolver does not gate on it — the manifest is unchanged).
    pub cert_scope: MoeCertScope,
}

/// Errors from the resolver bridge — all fail-loud.
#[derive(Debug, Clone, PartialEq)]
pub enum MoeResolveError {
    /// The spec itself is structurally invalid.
    Spec(MoeSpecError),
    /// The spec's convention disagrees with the authoritative handwritten
    /// family descriptor (renorm / shared) — it would diverge from the certified
    /// path.
    EquivalenceMismatch { detail: String },
    /// A runtime was requested for a non-runnable (mechanism-only) arch.
    NonRunnable { arch: &'static str },
    /// A runtime was requested without the MoE opt-in (`ATENIA_ENABLE_MOE=1`).
    OptInRequired,
    /// The certified runtime failed to load / run.
    Runtime(String),
}

impl std::fmt::Display for MoeResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoeResolveError::Spec(e) => write!(f, "moe-resolver: {e}"),
            MoeResolveError::EquivalenceMismatch { detail } => {
                write!(f, "moe-resolver: spec diverges from the handwritten path: {detail}")
            }
            MoeResolveError::NonRunnable { arch } => write!(
                f,
                "moe-resolver: `{arch}` is mechanism-only / non-runnable (no real-model runtime)"
            ),
            MoeResolveError::OptInRequired => write!(
                f,
                "moe-resolver: the MoE runtime is opt-in — set ATENIA_ENABLE_MOE=1 to run it"
            ),
            MoeResolveError::Runtime(e) => write!(f, "moe-resolver: runtime: {e}"),
        }
    }
}

impl std::error::Error for MoeResolveError {}

impl From<MoeRuntimeError> for MoeResolveError {
    fn from(e: MoeRuntimeError) -> Self {
        MoeResolveError::Runtime(e.to_string())
    }
}

/// The opt-in resolver bridge.
pub struct MoeSpecResolver;

impl MoeSpecResolver {
    /// Resolve a declarative spec into a runtime plan. Pure + always safe (no
    /// I/O, no execution). Fail-loud on an invalid spec or a convention that
    /// disagrees with the authoritative handwritten family descriptor.
    pub fn resolve(spec: &MoeStructuralSpec) -> Result<ResolvedMoeRuntimePlan, MoeResolveError> {
        spec.validate().map_err(MoeResolveError::Spec)?;

        let family = spec.to_moe_family();
        let d = family.descriptor();

        // Equivalence guard: the spec's renorm + shared MUST match the
        // handwritten runtime convention, or it would diverge from the certified
        // path. (V3 routing is renorm=true / shared=true, which matches the
        // DeepSeek descriptor; this holds for all four presets.)
        if spec.renormalize_topk != d.renormalizes_topk {
            return Err(MoeResolveError::EquivalenceMismatch {
                detail: format!(
                    "renormalize_topk={} but {} descriptor expects {}",
                    spec.renormalize_topk,
                    family.name(),
                    d.renormalizes_topk
                ),
            });
        }
        if spec.shared_present != d.has_shared_expert {
            return Err(MoeResolveError::EquivalenceMismatch {
                detail: format!(
                    "shared_present={} but {} descriptor expects {}",
                    spec.shared_present,
                    family.name(),
                    d.has_shared_expert
                ),
            });
        }

        // Execution convention: HuggingFaceQwen iff a sigmoid-gated shared expert
        // (exactly what `RealMoeLayer::resolve_convention` infers); else Atenia.
        let convention = if spec.shared_present && spec.shared_gating == SharedGating::Sigmoid {
            MoeExecutionConvention::HuggingFaceQwen
        } else {
            MoeExecutionConvention::Atenia
        };

        let routing = match spec.routing {
            RoutingScheme::SoftmaxTopK => {
                RoutingPlan::SoftmaxTopK { renormalize: spec.renormalize_topk }
            }
            RoutingScheme::SigmoidTopK => {
                RoutingPlan::SigmoidTopK { renormalize: spec.renormalize_topk }
            }
            RoutingScheme::V3NoAuxGroupLimited => RoutingPlan::V3NoAuxGroupLimited,
        };

        // Representative V3 router config for the mechanism arch (public V3 scale).
        let v3_router_config = spec.v3_router_config(256, 8);

        let cert_scope = MoeCertManifest::builtin().scope_for(family);

        Ok(ResolvedMoeRuntimePlan {
            arch: spec.arch,
            family,
            convention,
            routing,
            attention: spec.attention,
            expert_layout: spec.expert_layout,
            shared_present: spec.shared_present,
            shared_gating: spec.shared_gating,
            renormalize_topk: spec.renormalize_topk,
            dense_first_layers: spec.dense_first_layers,
            yarn: spec.yarn,
            disk_tier_hint: spec.disk_tier,
            v3_router_config,
            runnable: spec.arch.is_runnable_model(),
            cert_scope,
        })
    }

    /// The runtime gate, factored out for deterministic testing: a runnable plan
    /// with the opt-in set may proceed; otherwise fail loud. `opt_in` is the MoE
    /// opt-in flag (`ATENIA_ENABLE_MOE=1`).
    pub fn runtime_gate(
        plan: &ResolvedMoeRuntimePlan,
        opt_in: bool,
    ) -> Result<(), MoeResolveError> {
        if !plan.runnable {
            return Err(MoeResolveError::NonRunnable { arch: plan.arch.label() });
        }
        if !opt_in {
            return Err(MoeResolveError::OptInRequired);
        }
        Ok(())
    }

    /// **Opt-in runtime bridge.** Resolve the spec, enforce the runtime gate
    /// (runnable + `ATENIA_ENABLE_MOE=1`), then delegate to the **unchanged**
    /// certified [`MoeRuntime::load_from_dir`]. Fail-loud for a non-runnable arch
    /// (e.g. DeepSeek-V3 mechanism) or a missing opt-in — before touching disk.
    /// The default productive path ([`crate::moe::production`]) is not affected.
    pub fn load_runtime(
        spec: &MoeStructuralSpec,
        dir: &Path,
    ) -> Result<MoeRuntime, MoeResolveError> {
        let plan = Self::resolve(spec)?;
        Self::runtime_gate(&plan, experimental_moe_enabled())?;
        Ok(MoeRuntime::load_from_dir(dir)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(arch: MoeArch) -> ResolvedMoeRuntimePlan {
        MoeSpecResolver::resolve(&arch.preset()).unwrap()
    }

    // ---- FASE 4: resolver(spec) == handwritten descriptor / expectations ----

    #[test]
    fn resolve_mixtral_matches_handwritten() {
        let p = plan(MoeArch::Mixtral);
        assert_eq!(p.family, MoeFamily::Mixtral);
        assert_eq!(p.convention, MoeExecutionConvention::Atenia);
        assert_eq!(p.routing, RoutingPlan::SoftmaxTopK { renormalize: true });
        assert_eq!(p.attention, AttentionKind::Gqa);
        assert_eq!(p.dense_first_layers, 0);
        assert!(!p.yarn);
        assert_eq!(p.disk_tier_hint, DiskTierPolicy::DiskStreamable);
        assert!(p.runnable);
        assert!(p.v3_router_config.is_none());
        assert!(p.cert_scope.is_runnable());
    }

    #[test]
    fn resolve_qwen_matches_handwritten() {
        let p = plan(MoeArch::QwenMoe);
        assert_eq!(p.family, MoeFamily::QwenMoe);
        // Qwen-MoE: sigmoid-gated shared → HuggingFaceQwen convention, no renorm.
        assert_eq!(p.convention, MoeExecutionConvention::HuggingFaceQwen);
        assert_eq!(p.routing, RoutingPlan::SoftmaxTopK { renormalize: false });
        assert_eq!(p.attention, AttentionKind::Gqa);
        assert!(p.runnable);
    }

    #[test]
    fn resolve_deepseek_v2_lite_matches_handwritten() {
        let p = plan(MoeArch::DeepSeekV2Lite);
        assert_eq!(p.family, MoeFamily::DeepSeekMoe);
        // DeepSeek-V2-Lite: ungated shared + renorm → Atenia convention; MLA.
        assert_eq!(p.convention, MoeExecutionConvention::Atenia);
        assert_eq!(p.routing, RoutingPlan::SoftmaxTopK { renormalize: true });
        assert_eq!(p.attention, AttentionKind::Mla);
        assert_eq!(p.dense_first_layers, 1);
        assert!(p.yarn);
        assert!(p.runnable);
    }

    #[test]
    fn resolve_v3_routing_is_mechanism_only_non_runnable() {
        let p = plan(MoeArch::DeepSeekV3Route);
        assert_eq!(p.family, MoeFamily::DeepSeekMoe);
        assert_eq!(p.routing, RoutingPlan::V3NoAuxGroupLimited);
        assert_eq!(p.attention, AttentionKind::Mla);
        assert_eq!(p.dense_first_layers, 3);
        assert!(!p.runnable, "V3 routing is mechanism-only / non-runnable");
        // It still carries a valid representative V3 router config.
        assert!(p.v3_router_config.is_some());
    }

    #[test]
    fn convention_matches_runtime_resolution_rule() {
        // RealMoeLayer infers HuggingFaceQwen iff a shared sigmoid gate exists.
        for a in [MoeArch::Mixtral, MoeArch::DeepSeekV2Lite] {
            assert_eq!(plan(a).convention, MoeExecutionConvention::Atenia);
        }
        assert_eq!(plan(MoeArch::QwenMoe).convention, MoeExecutionConvention::HuggingFaceQwen);
    }

    // ---- FASE 5: guardrails ----

    #[test]
    fn runtime_gate_blocks_non_runnable_v3() {
        let p = plan(MoeArch::DeepSeekV3Route);
        // Even with the opt-in, a non-runnable arch is refused.
        assert!(matches!(
            MoeSpecResolver::runtime_gate(&p, true),
            Err(MoeResolveError::NonRunnable { .. })
        ));
    }

    #[test]
    fn runtime_gate_requires_opt_in_for_runnable() {
        let p = plan(MoeArch::Mixtral);
        assert!(matches!(
            MoeSpecResolver::runtime_gate(&p, false),
            Err(MoeResolveError::OptInRequired)
        ));
        // Runnable + opt-in → gate passes (does not load anything here).
        assert!(MoeSpecResolver::runtime_gate(&p, true).is_ok());
    }

    #[test]
    fn load_runtime_v3_fails_loud_before_touching_disk() {
        // Non-runnable arch → NonRunnable, regardless of dir existence / opt-in.
        let err = MoeSpecResolver::load_runtime(
            &MoeArch::DeepSeekV3Route.preset(),
            Path::new("/nonexistent"),
        )
        .unwrap_err();
        assert!(matches!(err, MoeResolveError::NonRunnable { .. }));
    }

    #[test]
    fn equivalence_guard_rejects_divergent_spec() {
        // A Mixtral spec that does not renormalise contradicts the handwritten
        // Mixtral convention (renorm=true) → hard error.
        let mut s = MoeArch::Mixtral.preset();
        s.renormalize_topk = false;
        assert!(matches!(
            MoeSpecResolver::resolve(&s),
            Err(MoeResolveError::EquivalenceMismatch { .. })
        ));
    }

    #[test]
    fn invalid_spec_is_rejected_by_resolver() {
        // MLA on Mixtral is an invalid spec → propagated as a Spec error.
        let mut s = MoeArch::Mixtral.preset();
        s.attention = AttentionKind::Mla;
        assert!(matches!(MoeSpecResolver::resolve(&s), Err(MoeResolveError::Spec(_))));
    }
}
