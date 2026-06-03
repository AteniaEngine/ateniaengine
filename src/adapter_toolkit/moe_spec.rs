//! **MOE-INTEGRATE-1** — Adapter Toolkit **MoE Specification v1**.
//!
//! A declarative, **non-executing** front-end for Mixture-of-Experts families,
//! parallel to the dense [`super::spec::ResolvedAdapterSpec`]. It turns the
//! optional [`super::dsl::MoeSection`] into a typed [`ResolvedMoeSpec`] that
//! *describes and validates* a MoE family — it does **not** run, route, load,
//! or lift the dense loader's fail-loud guard (those are MOE-INTEGRATE-2).
//!
//! ## Design contract (from `docs/MOE_ADAPTER_SPEC_AUDIT.md`)
//!
//! - **No second source of truth.** Every DSL field defaults to `auto`,
//!   deferring to `config.json` via [`crate::nn::llama::moe_config::MoeConfig`].
//!   Explicit values are *expectations* checked against the model
//!   (`effective_*` combine declared ⊕ config), never injected into it.
//! - **Reuse, don't duplicate.** Family conventions (renormalise / shared) are
//!   read from the runtime's authoritative [`crate::moe::family::MoeFamily`]
//!   descriptor; checkpoint validation delegates to
//!   [`crate::moe::family::validate_family_config`].
//! - **Typed, not stringly.** The informational `&'static str` naming/layout in
//!   `moe::family` becomes the typed [`ExpertLayout`] / [`RouterNaming`] /
//!   [`SharedExpertNaming`] here.

use crate::moe::family::{validate_family_config, FamilyConfigValidation, MoeFamily};
use crate::nn::llama::moe_config::MoeConfig;

use super::dsl::{AdapterDsl, MoeCount, MoeFlag, MoeSection};
use super::ToolkitError;

/// The MoE families the spec recognises. DeepSeek-MoE is **deferred** (MLA is a
/// separate runtime, out of this milestone's charter).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeFamilyKind {
    Mixtral,
    QwenMoe,
}

impl MoeFamilyKind {
    /// Bridge to the runtime's authoritative family enum.
    pub fn to_moe_family(self) -> MoeFamily {
        match self {
            MoeFamilyKind::Mixtral => MoeFamily::Mixtral,
            MoeFamilyKind::QwenMoe => MoeFamily::QwenMoe,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            MoeFamilyKind::Mixtral => "mixtral",
            MoeFamilyKind::QwenMoe => "qwen-moe",
        }
    }
}

/// Resolve a DSL `family` string to a MoE family. DeepSeek is a typed,
/// explicit "deferred" error so the message is precise.
pub fn resolve_moe_family(s: &str) -> Result<MoeFamilyKind, ToolkitError> {
    match s.to_ascii_lowercase().as_str() {
        "mixtral" | "mixtral-moe" | "mixtral_moe" => Ok(MoeFamilyKind::Mixtral),
        "qwen-moe" | "qwen_moe" | "qwenmoe" | "qwen2-moe" | "qwen2_moe" | "qwen3-moe"
        | "qwen3_moe" => Ok(MoeFamilyKind::QwenMoe),
        "deepseek-moe" | "deepseek_moe" | "deepseekmoe" => Err(ToolkitError::Resolution(
            "DeepSeek-MoE is deferred (MLA attention is a separate runtime, out of \
             MOE-INTEGRATE-1 scope)"
                .into(),
        )),
        other => Err(ToolkitError::Resolution(format!(
            "unknown MoE family `{other}` (expected mixtral | qwen-moe; deepseek-moe deferred)"
        ))),
    }
}

/// Expert weight layout — the typed form of the informational layout strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertLayout {
    /// Classic per-expert `…experts.{e}.w1/w3/w2.weight` (Mixtral block-sparse).
    ClassicW1W3W2,
    /// Classic per-expert `…experts.{e}.{gate,up,down}_proj.weight` (Qwen-MoE).
    ClassicGateUpDown,
    /// Packed 3-D `…experts.gate_up_proj` + `…experts.down_proj`.
    Packed,
}

/// Router tensor naming — the typed form of the informational router strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterNaming {
    /// `…block_sparse_moe.gate` (Mixtral classic).
    BlockSparseGate,
    /// `…mlp.gate` (Qwen-MoE 1.5/2, Mixtral packed).
    MlpGate,
    /// `…mlp.router` (Qwen3-MoE).
    MlpRouter,
}

/// Shared-expert tensor naming — the typed form of the informational strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedExpertNaming {
    /// No shared expert (Mixtral).
    None,
    /// `…mlp.shared_expert.*` (+ `shared_expert_gate`) — Qwen-MoE.
    MlpSharedExpert,
}

/// Shared-expert gating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedGating {
    /// `sigmoid(shared_expert_gate · x)` scaling (Qwen-MoE).
    Sigmoid,
    /// Added ungated (Mixtral has no shared expert; ungated is the trivial case).
    Ungated,
}

/// The resolved, typed MoE specification. Declared values are `Option`
/// (`None` == `auto`, deferred to `config.json`); family conventions come from
/// the authoritative [`MoeFamily`] descriptor.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedMoeSpec {
    pub kind: MoeFamilyKind,
    pub family: MoeFamily,
    /// Declared routed-expert count (`None` = auto).
    pub experts: Option<usize>,
    /// Declared top-k (`None` = auto).
    pub top_k: Option<usize>,
    /// Declared shared-expert presence (`None` = auto).
    pub shared_present: Option<bool>,
    /// Declared shared-expert gating (`None` = auto).
    pub shared_gating: Option<SharedGating>,
    /// Declared top-k renormalisation (`None` = auto / family default).
    pub renormalize_topk: Option<bool>,
    /// Declared expert layout (`None` = auto — detect from tensors at load).
    pub expert_layout: Option<ExpertLayout>,
    /// Declared router naming (`None` = auto).
    pub router_naming: Option<RouterNaming>,
    /// Shared naming, derived from the family (not author-declared).
    pub shared_naming: SharedExpertNaming,
}

fn parse_count(c: Option<&MoeCount>, field: &str) -> Result<Option<usize>, ToolkitError> {
    match c {
        None => Ok(None),
        Some(MoeCount::Count(n)) => Ok(Some(*n)),
        Some(MoeCount::Keyword(k)) if k.eq_ignore_ascii_case("auto") => Ok(None),
        Some(MoeCount::Keyword(k)) => Err(ToolkitError::Resolution(format!(
            "moe.{field}: `{k}` is not valid (expected an integer or `auto`)"
        ))),
    }
}

fn parse_flag(f: Option<&MoeFlag>, field: &str) -> Result<Option<bool>, ToolkitError> {
    match f {
        None => Ok(None),
        Some(MoeFlag::Bool(b)) => Ok(Some(*b)),
        Some(MoeFlag::Keyword(k)) if k.eq_ignore_ascii_case("auto") => Ok(None),
        Some(MoeFlag::Keyword(k)) => Err(ToolkitError::Resolution(format!(
            "moe.{field}: `{k}` is not valid (expected true/false or `auto`)"
        ))),
    }
}

fn parse_layout(
    s: Option<&String>,
    kind: MoeFamilyKind,
) -> Result<Option<ExpertLayout>, ToolkitError> {
    match s.map(|s| s.to_ascii_lowercase()) {
        None => Ok(None),
        Some(ref k) if k == "auto" => Ok(None),
        Some(ref k) if k == "packed" => Ok(Some(ExpertLayout::Packed)),
        Some(ref k) if k == "classic" => Ok(Some(match kind {
            // "classic" is family-dependent: Mixtral = w1/w3/w2, Qwen = gate/up/down.
            MoeFamilyKind::Mixtral => ExpertLayout::ClassicW1W3W2,
            MoeFamilyKind::QwenMoe => ExpertLayout::ClassicGateUpDown,
        })),
        Some(other) => Err(ToolkitError::Resolution(format!(
            "moe.experts_layout: `{other}` is not valid (expected classic | packed | auto)"
        ))),
    }
}

fn parse_router(
    s: Option<&String>,
    kind: MoeFamilyKind,
) -> Result<Option<RouterNaming>, ToolkitError> {
    let parsed = match s.map(|s| s.to_ascii_lowercase()) {
        None => return Ok(None),
        Some(ref k) if k == "auto" => return Ok(None),
        Some(ref k) if k == "block_sparse" || k == "block_sparse_gate" => {
            RouterNaming::BlockSparseGate
        }
        Some(ref k) if k == "mlp_gate" => RouterNaming::MlpGate,
        Some(ref k) if k == "mlp_router" => RouterNaming::MlpRouter,
        Some(other) => {
            return Err(ToolkitError::Resolution(format!(
                "moe.router_naming: `{other}` is not valid (expected block_sparse | mlp_gate | mlp_router | auto)"
            )))
        }
    };
    // Family/naming contradictions are hard errors (no silent acceptance).
    match (kind, parsed) {
        (MoeFamilyKind::QwenMoe, RouterNaming::BlockSparseGate) => Err(ToolkitError::Resolution(
            "moe.router_naming `block_sparse` is Mixtral-only; family is qwen-moe".into(),
        )),
        (MoeFamilyKind::Mixtral, RouterNaming::MlpRouter) => Err(ToolkitError::Resolution(
            "moe.router_naming `mlp_router` is Qwen3-MoE-only; family is mixtral".into(),
        )),
        _ => Ok(Some(parsed)),
    }
}

fn parse_gating(s: Option<&String>) -> Result<Option<SharedGating>, ToolkitError> {
    match s.map(|s| s.to_ascii_lowercase()) {
        None => Ok(None),
        Some(ref k) if k == "auto" => Ok(None),
        Some(ref k) if k == "sigmoid" => Ok(Some(SharedGating::Sigmoid)),
        Some(ref k) if k == "none" || k == "ungated" => Ok(Some(SharedGating::Ungated)),
        Some(other) => Err(ToolkitError::Resolution(format!(
            "moe.shared_expert.gating: `{other}` is not valid (expected sigmoid | none | auto)"
        ))),
    }
}

impl ResolvedMoeSpec {
    /// Resolve a [`ResolvedMoeSpec`] from a parsed DSL. Errors if the DSL has no
    /// `moe` section, an unknown MoE family, an invalid keyword, or an internal
    /// contradiction (e.g. `shared_expert.present: false` + `gating: sigmoid`,
    /// or a router naming that contradicts the family).
    pub fn resolve(dsl: &AdapterDsl) -> Result<Self, ToolkitError> {
        let section: &MoeSection = dsl.moe.as_ref().ok_or_else(|| {
            ToolkitError::Resolution("no `moe` section present (not a MoE spec)".into())
        })?;
        let kind = resolve_moe_family(&dsl.family)?;
        let family = kind.to_moe_family();

        let experts = parse_count(section.experts.as_ref(), "experts")?;
        let top_k = parse_count(section.top_k.as_ref(), "top_k")?;

        let (shared_present, shared_gating) = match &section.shared_expert {
            None => (None, None),
            Some(se) => (
                parse_flag(se.present.as_ref(), "shared_expert.present")?,
                parse_gating(se.gating.as_ref())?,
            ),
        };

        let renormalize_topk = match &section.routing {
            None => None,
            Some(r) => parse_flag(r.renormalize_topk.as_ref(), "routing.renormalize_topk")?,
        };

        let expert_layout = parse_layout(section.experts_layout.as_ref(), kind)?;
        let router_naming = parse_router(section.router_naming.as_ref(), kind)?;
        let shared_naming = match family.descriptor().has_shared_expert {
            true => SharedExpertNaming::MlpSharedExpert,
            false => SharedExpertNaming::None,
        };

        // Internal consistency (hard errors, no silent acceptance).
        if shared_present == Some(false) && shared_gating == Some(SharedGating::Sigmoid) {
            return Err(ToolkitError::Resolution(
                "moe.shared_expert: gating `sigmoid` declared but present is `false`".into(),
            ));
        }
        if let (Some(k), Some(n)) = (top_k, experts) {
            if k > n {
                return Err(ToolkitError::Resolution(format!(
                    "moe: top_k ({k}) exceeds experts ({n})"
                )));
            }
        }

        Ok(ResolvedMoeSpec {
            kind,
            family,
            experts,
            top_k,
            shared_present,
            shared_gating,
            renormalize_topk,
            expert_layout,
            router_naming,
            shared_naming,
        })
    }

    // ---- `auto` resolution: declared ⊕ config.json (no second source of truth) ----

    /// Effective routed-expert count: the declared value, else `config.json`.
    pub fn effective_experts(&self, config: &MoeConfig) -> Option<usize> {
        self.experts.or(config.num_experts)
    }

    /// Effective top-k: the declared value, else `config.json` (clamped).
    pub fn effective_top_k(&self, config: &MoeConfig, default: usize) -> usize {
        match self.top_k {
            Some(k) => {
                let k = k.max(1);
                match self.effective_experts(config) {
                    Some(n) if n > 0 => k.min(n),
                    _ => k,
                }
            }
            None => config.experts_per_token_or(default),
        }
    }

    /// Effective top-k renormalisation: declared, else `config.json`, else the
    /// family default (Mixtral renormalises; Qwen does not).
    pub fn effective_renormalize_topk(&self, config: &MoeConfig) -> bool {
        self.renormalize_topk
            .or(config.norm_topk_prob)
            .unwrap_or(self.family.descriptor().renormalizes_topk)
    }

    /// Effective shared-expert presence: declared, else `config.json`.
    pub fn effective_shared_present(&self, config: &MoeConfig) -> bool {
        self.shared_present.unwrap_or(config.has_shared_experts)
    }

    // ---- checkpoint validation: delegate to the runtime's validator ----

    /// Validate this spec against a real checkpoint's tensor names + parsed
    /// [`MoeConfig`], **reusing** [`validate_family_config`] (no duplicated
    /// logic). Cross-checks expert count, top-k bound, and shared-expert
    /// agreement; returns the family-aware notes.
    pub fn validate_against<'a, I>(&self, names: I, config: &MoeConfig) -> FamilyConfigValidation
    where
        I: IntoIterator<Item = &'a str>,
    {
        validate_family_config(names, config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter_toolkit::dsl::AdapterDsl;

    fn dsl(text: &str) -> AdapterDsl {
        AdapterDsl::from_str(text, true).expect("dsl parses")
    }

    #[test]
    fn resolve_family_accepts_known_and_defers_deepseek() {
        assert_eq!(resolve_moe_family("mixtral").unwrap(), MoeFamilyKind::Mixtral);
        assert_eq!(resolve_moe_family("qwen-moe").unwrap(), MoeFamilyKind::QwenMoe);
        assert_eq!(resolve_moe_family("Qwen3_MoE").unwrap(), MoeFamilyKind::QwenMoe);
        let e = resolve_moe_family("deepseek-moe").unwrap_err();
        assert!(format!("{e}").contains("deferred"));
        assert!(resolve_moe_family("llama").is_err());
    }

    #[test]
    fn qwen_moe_full_section_resolves() {
        let spec = ResolvedMoeSpec::resolve(&dsl(
            "family: qwen-moe\nmoe:\n  experts: 60\n  top_k: 4\n  shared_expert:\n    present: true\n    gating: sigmoid\n  routing:\n    renormalize_topk: false\n  experts_layout: classic\n",
        ))
        .unwrap();
        assert_eq!(spec.kind, MoeFamilyKind::QwenMoe);
        assert_eq!(spec.experts, Some(60));
        assert_eq!(spec.top_k, Some(4));
        assert_eq!(spec.shared_present, Some(true));
        assert_eq!(spec.shared_gating, Some(SharedGating::Sigmoid));
        assert_eq!(spec.renormalize_topk, Some(false));
        assert_eq!(spec.expert_layout, Some(ExpertLayout::ClassicGateUpDown));
        assert_eq!(spec.shared_naming, SharedExpertNaming::MlpSharedExpert);
    }

    #[test]
    fn mixtral_minimal_section_resolves_with_family_defaults() {
        let spec = ResolvedMoeSpec::resolve(&dsl("family: mixtral\nmoe: {}\n")).unwrap();
        assert_eq!(spec.kind, MoeFamilyKind::Mixtral);
        assert_eq!(spec.experts, None); // auto
        assert_eq!(spec.shared_naming, SharedExpertNaming::None);
        // Family default: Mixtral renormalises.
        assert!(spec.effective_renormalize_topk(&MoeConfig::default()));
    }

    #[test]
    fn classic_layout_is_family_dependent() {
        let mix = ResolvedMoeSpec::resolve(&dsl("family: mixtral\nmoe:\n  experts_layout: classic\n")).unwrap();
        assert_eq!(mix.expert_layout, Some(ExpertLayout::ClassicW1W3W2));
        let qwen = ResolvedMoeSpec::resolve(&dsl("family: qwen-moe\nmoe:\n  experts_layout: classic\n")).unwrap();
        assert_eq!(qwen.expert_layout, Some(ExpertLayout::ClassicGateUpDown));
    }

    #[test]
    fn auto_defaults_defer_to_config() {
        let spec = ResolvedMoeSpec::resolve(&dsl("family: qwen-moe\nmoe:\n  experts: auto\n  top_k: auto\n")).unwrap();
        assert_eq!(spec.experts, None);
        let cfg = MoeConfig {
            num_experts: Some(128),
            experts_per_token: Some(8),
            norm_topk_prob: Some(true),
            has_shared_experts: false,
            ..Default::default()
        };
        assert_eq!(spec.effective_experts(&cfg), Some(128));
        assert_eq!(spec.effective_top_k(&cfg, 2), 8);
        assert!(spec.effective_renormalize_topk(&cfg)); // config wins over family default
        assert!(!spec.effective_shared_present(&cfg));
    }

    #[test]
    fn declared_value_overrides_but_does_not_mutate_config() {
        let spec = ResolvedMoeSpec::resolve(&dsl("family: qwen-moe\nmoe:\n  experts: 64\n")).unwrap();
        let cfg = MoeConfig { num_experts: Some(60), ..Default::default() };
        // Declared expectation is preferred by effective_*, but config is untouched.
        assert_eq!(spec.effective_experts(&cfg), Some(64));
        assert_eq!(cfg.num_experts, Some(60));
    }

    #[test]
    fn contradiction_shared_false_with_sigmoid_errors() {
        let e = ResolvedMoeSpec::resolve(&dsl(
            "family: qwen-moe\nmoe:\n  shared_expert:\n    present: false\n    gating: sigmoid\n",
        ))
        .unwrap_err();
        assert!(format!("{e}").contains("sigmoid"));
    }

    #[test]
    fn contradiction_router_naming_vs_family_errors() {
        let e = ResolvedMoeSpec::resolve(&dsl("family: qwen-moe\nmoe:\n  router_naming: block_sparse\n")).unwrap_err();
        assert!(format!("{e}").contains("Mixtral-only"));
        let e2 = ResolvedMoeSpec::resolve(&dsl("family: mixtral\nmoe:\n  router_naming: mlp_router\n")).unwrap_err();
        assert!(format!("{e2}").contains("Qwen3-MoE-only"));
    }

    #[test]
    fn top_k_exceeds_experts_errors() {
        let e = ResolvedMoeSpec::resolve(&dsl("family: mixtral\nmoe:\n  experts: 8\n  top_k: 9\n")).unwrap_err();
        assert!(format!("{e}").contains("exceeds"));
    }

    #[test]
    fn invalid_keyword_errors() {
        assert!(ResolvedMoeSpec::resolve(&dsl("family: mixtral\nmoe:\n  experts: lots\n")).is_err());
        assert!(ResolvedMoeSpec::resolve(&dsl("family: mixtral\nmoe:\n  experts_layout: weird\n")).is_err());
    }

    #[test]
    fn dense_spec_has_no_moe_section() {
        // Backward compatibility: a dense spec still parses and carries no moe.
        let d = dsl("family: llama\n");
        assert!(d.moe.is_none());
        // Resolving it as MoE fails loudly (it is not a MoE spec).
        assert!(ResolvedMoeSpec::resolve(&d).is_err());
    }

    #[test]
    fn validate_against_delegates_to_runtime_validator() {
        let spec = ResolvedMoeSpec::resolve(&dsl("family: mixtral\nmoe:\n  experts: 8\n")).unwrap();
        let names = vec![
            "model.layers.0.block_sparse_moe.gate.weight",
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.0.w2.weight",
            "model.layers.0.block_sparse_moe.experts.0.w3.weight",
            "model.layers.0.block_sparse_moe.experts.7.w1.weight",
        ];
        let cfg = MoeConfig {
            num_experts: Some(8),
            experts_per_token: Some(2),
            has_shared_experts: false,
            ..Default::default()
        };
        let v = spec.validate_against(names.iter().copied(), &cfg);
        assert!(v.consistent, "notes: {:?}", v.notes);
    }
}
