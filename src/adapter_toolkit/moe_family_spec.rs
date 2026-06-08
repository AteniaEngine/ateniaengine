//! **MOE-ATK-DECL-1** — declarative **MoE family structural spec** (describe +
//! validate only).
//!
//! A **declarative spec layer parallel to the handwritten paths**: it expresses
//! the *structural* shape of a MoE family across every axis the runtime cares
//! about (expert layout, routing scheme incl. DeepSeek-V3 modern routing, shared
//! expert, attention incl. MLA/YaRN/GQA, dense-first layers, disk-tier policy,
//! tensor naming) as typed data, with `preset`s that **reproduce the certified
//! families** (Mixtral, Qwen-MoE, DeepSeek-V2-Lite) and the **DeepSeek-V3-like
//! routing mechanism** (L0). It extends the MOE-INTEGRATE-1 [`super::moe_spec`]
//! (Mixtral/Qwen YAML front-end) with the DeepSeek/MLA + V3 axes.
//!
//! ## Scope (critical)
//!
//! - **Not replacing the certified paths.** This only *describes and validates*;
//!   it does not load, route, execute, or change the runtime / loader / manifests.
//!   The handwritten `MoeRuntime` assembly stays the source of execution truth.
//! - **No new family support claimed.** A `preset` existing here is a *structural
//!   description*, not a support/certification claim. DeepSeek-V3 as a model is
//!   **not supported** (real-weight provisioning-blocked); only its **routing
//!   mechanism** is L0 (`crate::moe::v3_router`). Equivalence is asserted against
//!   the authoritative runtime sources (`crate::moe::family`, `v3_router`), so the
//!   spec can never silently diverge from the handwritten path.
//! - Reuses the typed enums from [`super::moe_spec`] ([`ExpertLayout`],
//!   [`RouterNaming`], [`SharedExpertNaming`], [`SharedGating`]).

use crate::moe::family::MoeFamily;
use crate::moe::v3_router::{ScoringFunc, V3RouterConfig};

use super::moe_spec::{ExpertLayout, RouterNaming, SharedExpertNaming, SharedGating};

/// The MoE architectures this structural spec can express. Each maps to an
/// authoritative [`MoeFamily`]; DeepSeek-V2-Lite and DeepSeek-V3-routing both
/// bridge to [`MoeFamily::DeepSeekMoe`] but differ in their routing scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeArch {
    /// Mixtral-8x7B: classic `w1/w3/w2`, softmax top-k, renorm, no shared, GQA.
    Mixtral,
    /// Qwen1.5/2-MoE: classic `gate/up/down`, softmax top-k, sigmoid-gated shared, GQA + qkv bias.
    QwenMoe,
    /// DeepSeek-V2-Lite: packed experts, softmax top-k, ungated shared, MLA + YaRN, dense-first.
    DeepSeekV2Lite,
    /// DeepSeek-V3-like **routing mechanism** (L0): packed experts, sigmoid +
    /// aux-loss-free + group-limited routing, ungated shared, MLA + YaRN, dense-first.
    DeepSeekV3Route,
}

impl MoeArch {
    pub fn label(self) -> &'static str {
        match self {
            MoeArch::Mixtral => "mixtral",
            MoeArch::QwenMoe => "qwen-moe",
            MoeArch::DeepSeekV2Lite => "deepseek-v2-lite",
            MoeArch::DeepSeekV3Route => "deepseek-v3-route",
        }
    }

    /// Parse a label (fail-loud on unknown).
    pub fn from_label(s: &str) -> Result<Self, MoeSpecError> {
        match s.to_ascii_lowercase().as_str() {
            "mixtral" | "mixtral-moe" => Ok(MoeArch::Mixtral),
            "qwen-moe" | "qwen_moe" | "qwen2-moe" | "qwen3-moe" => Ok(MoeArch::QwenMoe),
            "deepseek-v2-lite" | "deepseek_v2_lite" => Ok(MoeArch::DeepSeekV2Lite),
            "deepseek-v3-route" | "deepseek-v3" | "deepseek_v3" => Ok(MoeArch::DeepSeekV3Route),
            other => Err(MoeSpecError::UnknownArch { got: other.to_string() }),
        }
    }

    /// Bridge to the authoritative runtime family enum.
    pub fn to_moe_family(self) -> MoeFamily {
        match self {
            MoeArch::Mixtral => MoeFamily::Mixtral,
            MoeArch::QwenMoe => MoeFamily::QwenMoe,
            MoeArch::DeepSeekV2Lite | MoeArch::DeepSeekV3Route => MoeFamily::DeepSeekMoe,
        }
    }

    /// Whether this arch describes a *model* Atenia supports running today.
    /// DeepSeek-V3-routing is a **mechanism only** (model not supported —
    /// provisioning-blocked), so this is `false` for it. Mixtral / Qwen-MoE /
    /// DeepSeek-V2-Lite are the three real-weight MoE-certified-L3 families.
    pub fn is_runnable_model(self) -> bool {
        !matches!(self, MoeArch::DeepSeekV3Route)
    }

    /// The canonical structural spec for this arch (reproduces the certified
    /// family's structure / the V3 routing mechanism).
    pub fn preset(self) -> MoeStructuralSpec {
        match self {
            MoeArch::Mixtral => MoeStructuralSpec {
                arch: self,
                expert_layout: ExpertLayout::ClassicW1W3W2,
                router_naming: RouterNaming::BlockSparseGate,
                shared_present: false,
                shared_naming: SharedExpertNaming::None,
                shared_gating: SharedGating::Ungated,
                routing: RoutingScheme::SoftmaxTopK,
                renormalize_topk: true,
                v3_routing: None,
                attention: AttentionKind::Gqa,
                qkv_bias: false,
                yarn: false,
                dense_first_layers: 0,
                disk_tier: DiskTierPolicy::DiskStreamable,
            },
            MoeArch::QwenMoe => MoeStructuralSpec {
                arch: self,
                expert_layout: ExpertLayout::ClassicGateUpDown,
                router_naming: RouterNaming::MlpGate,
                shared_present: true,
                shared_naming: SharedExpertNaming::MlpSharedExpert,
                shared_gating: SharedGating::Sigmoid,
                routing: RoutingScheme::SoftmaxTopK,
                renormalize_topk: false,
                v3_routing: None,
                attention: AttentionKind::Gqa,
                qkv_bias: true,
                yarn: false,
                dense_first_layers: 0,
                disk_tier: DiskTierPolicy::DiskStreamable,
            },
            MoeArch::DeepSeekV2Lite => MoeStructuralSpec {
                arch: self,
                expert_layout: ExpertLayout::Packed,
                router_naming: RouterNaming::MlpGate,
                shared_present: true,
                shared_naming: SharedExpertNaming::MlpSharedExpert,
                shared_gating: SharedGating::Ungated,
                routing: RoutingScheme::SoftmaxTopK,
                renormalize_topk: true,
                v3_routing: None,
                attention: AttentionKind::Mla,
                qkv_bias: false,
                yarn: true,
                dense_first_layers: 1,
                disk_tier: DiskTierPolicy::DiskStreamable,
            },
            MoeArch::DeepSeekV3Route => MoeStructuralSpec {
                arch: self,
                expert_layout: ExpertLayout::Packed,
                router_naming: RouterNaming::MlpGate,
                shared_present: true,
                shared_naming: SharedExpertNaming::MlpSharedExpert,
                shared_gating: SharedGating::Ungated,
                routing: RoutingScheme::V3NoAuxGroupLimited,
                renormalize_topk: true,
                // Representative public DeepSeek-V3 router knobs (mechanism only).
                v3_routing: Some(V3RoutingParams {
                    n_group: 8,
                    topk_group: 4,
                    routed_scaling_factor: 2.5,
                    norm_topk_prob: true,
                }),
                attention: AttentionKind::Mla,
                qkv_bias: false,
                yarn: true,
                dense_first_layers: 3,
                disk_tier: DiskTierPolicy::DiskStreamable,
            },
        }
    }
}

/// The router scheme an arch uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingScheme {
    /// `softmax(W·x)` → top-k (Mixtral / Qwen-MoE / DeepSeek-V2).
    SoftmaxTopK,
    /// `sigmoid(W·x)` per-expert → top-k (no aux-loss-free / group structure).
    SigmoidTopK,
    /// DeepSeek-V3 `noaux_tc`: sigmoid + `e_score_correction_bias` selection +
    /// group-limited top-k + `routed_scaling_factor` (`crate::moe::v3_router`).
    V3NoAuxGroupLimited,
}

/// Attention block kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    /// Multi-head attention.
    Mha,
    /// Grouped-query attention (Mixtral / Qwen-MoE).
    Gqa,
    /// Multi-head latent attention (DeepSeek-V2/V3).
    Mla,
}

/// Expert-tier residency policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiskTierPolicy {
    /// Experts must stay RAM-resident.
    RamResident,
    /// Experts can stream from an NVMe disk tier (huge-MoE feasible) — the policy
    /// every certified family uses for the C5 active-path forward.
    DiskStreamable,
}

/// DeepSeek-V3 router knobs (present iff [`RoutingScheme::V3NoAuxGroupLimited`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct V3RoutingParams {
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
    pub norm_topk_prob: bool,
}

/// A fully typed, declarative structural description of a MoE family. Parallel to
/// the handwritten runtime assembly — it never executes.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeStructuralSpec {
    pub arch: MoeArch,
    pub expert_layout: ExpertLayout,
    pub router_naming: RouterNaming,
    pub shared_present: bool,
    pub shared_naming: SharedExpertNaming,
    pub shared_gating: SharedGating,
    pub routing: RoutingScheme,
    pub renormalize_topk: bool,
    pub v3_routing: Option<V3RoutingParams>,
    pub attention: AttentionKind,
    pub qkv_bias: bool,
    pub yarn: bool,
    pub dense_first_layers: usize,
    pub disk_tier: DiskTierPolicy,
}

/// Errors from the structural spec (all fail-loud — no silent acceptance).
#[derive(Debug, Clone, PartialEq)]
pub enum MoeSpecError {
    UnknownArch { got: String },
    /// `routing == V3NoAuxGroupLimited` but no `v3_routing` params (or vice versa).
    V3ParamsInconsistent { routing_is_v3: bool, params_present: bool },
    /// MLA attention declared for a non-DeepSeek arch (or a DeepSeek arch without MLA).
    AttentionInconsistent { arch: &'static str, detail: &'static str },
    /// `shared_gating == Sigmoid` but `shared_present == false`.
    SharedGatingWithoutShared,
    /// The V3 router params are structurally invalid for a representative expert
    /// count (e.g. `n_routed % n_group != 0`, `topk_group > n_group`).
    V3RouterInvalid { detail: String },
}

impl std::fmt::Display for MoeSpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoeSpecError::UnknownArch { got } => write!(f, "moe-spec: unknown arch `{got}`"),
            MoeSpecError::V3ParamsInconsistent { routing_is_v3, params_present } => write!(
                f,
                "moe-spec: V3 routing/params inconsistent (routing_is_v3={routing_is_v3}, params_present={params_present})"
            ),
            MoeSpecError::AttentionInconsistent { arch, detail } => {
                write!(f, "moe-spec: attention inconsistent for {arch}: {detail}")
            }
            MoeSpecError::SharedGatingWithoutShared => {
                write!(f, "moe-spec: shared_gating=sigmoid but shared_present=false")
            }
            MoeSpecError::V3RouterInvalid { detail } => {
                write!(f, "moe-spec: V3 router params invalid: {detail}")
            }
        }
    }
}

impl std::error::Error for MoeSpecError {}

impl MoeStructuralSpec {
    /// Bridge to the authoritative runtime family.
    pub fn to_moe_family(&self) -> MoeFamily {
        self.arch.to_moe_family()
    }

    /// Build a [`V3RouterConfig`] from this spec's V3 params for a representative
    /// `(n_routed_experts, top_k)` — used to cross-check the params against the
    /// authoritative `crate::moe::v3_router` validator. `None` when the spec is
    /// not a V3-routing spec.
    pub fn v3_router_config(&self, n_routed_experts: usize, top_k: usize) -> Option<V3RouterConfig> {
        self.v3_routing.map(|p| V3RouterConfig {
            n_routed_experts,
            top_k,
            n_group: p.n_group,
            topk_group: p.topk_group,
            routed_scaling_factor: p.routed_scaling_factor,
            norm_topk_prob: p.norm_topk_prob,
            scoring_func: ScoringFunc::Sigmoid,
        })
    }

    /// Fail-loud structural validation. Catches V3 routing/params disagreement,
    /// MLA on a non-DeepSeek arch (or a DeepSeek arch missing MLA), shared-gating
    /// without a shared expert, and structurally invalid V3 router params.
    pub fn validate(&self) -> Result<(), MoeSpecError> {
        // V3 routing ⇔ V3 params.
        let routing_is_v3 = matches!(self.routing, RoutingScheme::V3NoAuxGroupLimited);
        if routing_is_v3 != self.v3_routing.is_some() {
            return Err(MoeSpecError::V3ParamsInconsistent {
                routing_is_v3,
                params_present: self.v3_routing.is_some(),
            });
        }

        // MLA iff DeepSeek arch.
        let is_deepseek = matches!(self.to_moe_family(), MoeFamily::DeepSeekMoe);
        match (is_deepseek, self.attention) {
            (true, AttentionKind::Mla) => {}
            (false, AttentionKind::Mla) => {
                return Err(MoeSpecError::AttentionInconsistent {
                    arch: self.arch.label(),
                    detail: "MLA is DeepSeek-only",
                })
            }
            (true, _) => {
                return Err(MoeSpecError::AttentionInconsistent {
                    arch: self.arch.label(),
                    detail: "DeepSeek arch must use MLA attention",
                })
            }
            (false, _) => {}
        }

        // Shared gating requires a shared expert.
        if self.shared_gating == SharedGating::Sigmoid && !self.shared_present {
            return Err(MoeSpecError::SharedGatingWithoutShared);
        }

        // V3 params must be structurally valid (reuse the authoritative validator
        // via a representative expert count divisible by n_group).
        if let Some(p) = self.v3_routing {
            // Representative: experts = 4 per group, top_k = topk_group * 1.
            let n_routed = p.n_group * 4;
            let top_k = (p.topk_group).max(1);
            let cfg = self
                .v3_router_config(n_routed, top_k)
                .expect("v3 params present");
            // A 1-token route validates the config end-to-end (fail-loud).
            let logits = vec![0.0_f32; n_routed];
            let bias = vec![0.0_f32; n_routed];
            crate::moe::v3_router::v3_route(&logits, &bias, &cfg)
                .map_err(|e| MoeSpecError::V3RouterInvalid { detail: e.to_string() })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [MoeArch; 4] = [
        MoeArch::Mixtral,
        MoeArch::QwenMoe,
        MoeArch::DeepSeekV2Lite,
        MoeArch::DeepSeekV3Route,
    ];

    #[test]
    fn every_preset_validates() {
        for a in ALL {
            a.preset().validate().unwrap_or_else(|e| panic!("{} preset invalid: {e}", a.label()));
        }
    }

    #[test]
    fn label_round_trips() {
        for a in ALL {
            assert_eq!(MoeArch::from_label(a.label()).unwrap(), a);
        }
        assert!(MoeArch::from_label("llama").is_err());
    }

    // ---- FASE 4: equivalence to the authoritative handwritten runtime sources ----

    #[test]
    fn family_bridge_matches_runtime() {
        assert_eq!(MoeArch::Mixtral.to_moe_family(), MoeFamily::Mixtral);
        assert_eq!(MoeArch::QwenMoe.to_moe_family(), MoeFamily::QwenMoe);
        assert_eq!(MoeArch::DeepSeekV2Lite.to_moe_family(), MoeFamily::DeepSeekMoe);
        assert_eq!(MoeArch::DeepSeekV3Route.to_moe_family(), MoeFamily::DeepSeekMoe);
    }

    #[test]
    fn renorm_and_shared_match_runtime_descriptor() {
        // The spec's renormalize_topk / shared_present must equal the
        // authoritative `MoeFamily::descriptor()` — the handwritten convention.
        for a in ALL {
            let s = a.preset();
            let d = a.to_moe_family().descriptor();
            assert_eq!(
                s.renormalize_topk,
                d.renormalizes_topk,
                "{} renorm vs runtime descriptor",
                a.label()
            );
            assert_eq!(
                s.shared_present,
                d.has_shared_expert,
                "{} shared vs runtime descriptor",
                a.label()
            );
        }
    }

    #[test]
    fn mixtral_spec_matches_handwritten_expectations() {
        let s = MoeArch::Mixtral.preset();
        assert_eq!(s.expert_layout, ExpertLayout::ClassicW1W3W2);
        assert_eq!(s.router_naming, RouterNaming::BlockSparseGate);
        assert_eq!(s.shared_naming, SharedExpertNaming::None);
        assert_eq!(s.routing, RoutingScheme::SoftmaxTopK);
        assert!(s.renormalize_topk && !s.shared_present);
        assert_eq!(s.attention, AttentionKind::Gqa);
        assert_eq!(s.dense_first_layers, 0);
        assert!(s.v3_routing.is_none());
    }

    #[test]
    fn qwen_moe_spec_matches_handwritten_expectations() {
        let s = MoeArch::QwenMoe.preset();
        assert_eq!(s.expert_layout, ExpertLayout::ClassicGateUpDown);
        assert_eq!(s.router_naming, RouterNaming::MlpGate);
        assert!(s.shared_present);
        assert_eq!(s.shared_gating, SharedGating::Sigmoid);
        assert!(!s.renormalize_topk); // Qwen norm_topk_prob = false
        assert_eq!(s.attention, AttentionKind::Gqa);
        assert!(s.qkv_bias);
    }

    #[test]
    fn deepseek_v2_lite_spec_matches_handwritten_expectations() {
        let s = MoeArch::DeepSeekV2Lite.preset();
        assert_eq!(s.expert_layout, ExpertLayout::Packed);
        assert!(s.shared_present);
        assert_eq!(s.shared_gating, SharedGating::Ungated);
        assert_eq!(s.routing, RoutingScheme::SoftmaxTopK);
        assert_eq!(s.attention, AttentionKind::Mla);
        assert!(s.yarn);
        assert_eq!(s.dense_first_layers, 1); // first_k_dense_replace
        assert!(s.arch.is_runnable_model());
    }

    #[test]
    fn deepseek_v3_route_spec_matches_v3_router_module() {
        let s = MoeArch::DeepSeekV3Route.preset();
        assert_eq!(s.routing, RoutingScheme::V3NoAuxGroupLimited);
        assert_eq!(s.attention, AttentionKind::Mla);
        assert_eq!(s.dense_first_layers, 3);
        let p = s.v3_routing.expect("v3 params");
        assert_eq!((p.n_group, p.topk_group), (8, 4));
        // The spec's V3 params must build a VALID `v3_router::V3RouterConfig` and
        // route — i.e. the declarative spec agrees with the certified mechanism.
        let cfg = s.v3_router_config(256, 8).unwrap();
        let logits = vec![0.1_f32; 256];
        let bias = vec![0.0_f32; 256];
        let r = crate::moe::v3_router::v3_route(&logits, &bias, &cfg).unwrap();
        assert_eq!(r.indices.len(), 8);
        // DeepSeek-V3 as a *model* is NOT supported (mechanism only).
        assert!(!s.arch.is_runnable_model());
    }

    #[test]
    fn mla_is_deepseek_only() {
        for a in [MoeArch::Mixtral, MoeArch::QwenMoe] {
            assert_ne!(a.preset().attention, AttentionKind::Mla);
        }
        for a in [MoeArch::DeepSeekV2Lite, MoeArch::DeepSeekV3Route] {
            assert_eq!(a.preset().attention, AttentionKind::Mla);
        }
    }

    // ---- FASE 5: fail-loud on inconsistent specs ----

    #[test]
    fn fail_loud_v3_routing_without_params() {
        let mut s = MoeArch::DeepSeekV3Route.preset();
        s.v3_routing = None; // routing says V3 but params gone
        assert!(matches!(s.validate(), Err(MoeSpecError::V3ParamsInconsistent { .. })));
    }

    #[test]
    fn fail_loud_mla_on_mixtral() {
        let mut s = MoeArch::Mixtral.preset();
        s.attention = AttentionKind::Mla;
        assert!(matches!(s.validate(), Err(MoeSpecError::AttentionInconsistent { .. })));
    }

    #[test]
    fn fail_loud_shared_gating_without_shared() {
        let mut s = MoeArch::QwenMoe.preset();
        s.shared_present = false; // sigmoid gating but no shared expert
        assert!(matches!(s.validate(), Err(MoeSpecError::SharedGatingWithoutShared)));
    }

    #[test]
    fn fail_loud_invalid_v3_params() {
        let mut s = MoeArch::DeepSeekV3Route.preset();
        s.v3_routing = Some(V3RoutingParams {
            n_group: 8,
            topk_group: 9, // > n_group → invalid
            routed_scaling_factor: 2.5,
            norm_topk_prob: true,
        });
        assert!(matches!(s.validate(), Err(MoeSpecError::V3RouterInvalid { .. })));
    }
}
