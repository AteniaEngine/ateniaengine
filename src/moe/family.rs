//! **MOE-FULL-9** — MoE family recognition + productive-loader preparation
//! (metadata + wiring only; **no load activation**).
//!
//! This module gives the loader enough information to (1) detect a MoE
//! checkpoint, (2) **identify its family** (Mixtral vs Qwen-MoE), and (3)
//! **validate** the declared config against the tensors — and then **still fail
//! loud** with a precise, family-aware message instead of the old generic
//! "MoE unsupported". It does **not** lift the fail-loud guard, enable MoE
//! loading, or touch the productive runtime / Adapter Toolkit / CLI. It is the
//! metadata/wiring layer a future productive path would build on.
//!
//! ```text
//!   detect_moe(names)  ──► is this MoE at all?
//!   classify_family(names)  ──► Mixtral | QwenMoe | (unrecognised)
//!   validate_family_config(names, &MoeConfig)  ──► consistent? notes?
//!   moe_failloud_report(names)  ──► the human-facing loader message:
//!        "MoE detected / Family: Mixtral / Productive support not enabled"
//! ```
//!
//! ## Family fingerprints (from real checkpoints, MIXTRAL-CERT-1 / QWEN-MOE-CERT-1)
//!
//! | Marker | Mixtral | Qwen-MoE |
//! |---|---|---|
//! | router | `block_sparse_moe.gate` / `mlp.gate` (packed) | `mlp.gate` / `mlp.router` |
//! | experts | `block_sparse_moe.experts.{e}.w1/w3/w2` or packed `mlp.experts.gate_up_proj` | `mlp.experts.{e}.gate_proj/up_proj/down_proj` |
//! | shared expert | none | `mlp.shared_expert.*` (gated by `shared_expert_gate`) |

use crate::nn::llama::moe_config::MoeConfig;

use super::detect::detect_moe;

/// Environment flag that opts in to the experimental controlled MoE path
/// (MOE-FULL-10). When **unset**, every MoE path fails loud exactly as before.
pub const EXPERIMENTAL_MOE_ENV: &str = "ATENIA_EXPERIMENTAL_MOE";

/// Whether the experimental controlled MoE path is opted in
/// (`ATENIA_EXPERIMENTAL_MOE=1`). This gates **only** the dedicated
/// experimental Mixtral runtime ([`crate::moe::runtime`]); it does **not**
/// lift the dense loader's fail-loud guard, which always refuses MoE.
pub fn experimental_moe_enabled() -> bool {
    std::env::var(EXPERIMENTAL_MOE_ENV).as_deref() == Ok("1")
}

/// A recognised MoE family. Metadata only — recognising a family does **not**
/// enable loading it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeFamily {
    /// Mixtral (`block_sparse_moe.*` classic, or packed `mlp.experts.gate_up_proj`
    /// without a shared expert). No shared expert.
    Mixtral,
    /// Qwen-MoE (`mlp.experts.{e}.gate_proj/...`, `mlp.gate` / `mlp.router`),
    /// usually with a sigmoid-gated shared expert.
    QwenMoe,
    /// DeepSeek-MoE (`kv_a_proj_with_mqa` MLA attention, `mlp.shared_experts.*`
    /// plural shared expert, packed routed experts). MLA distinguishes it from
    /// Qwen-MoE.
    DeepSeekMoe,
}

impl MoeFamily {
    /// Short human name.
    pub fn name(self) -> &'static str {
        match self {
            MoeFamily::Mixtral => "Mixtral",
            MoeFamily::QwenMoe => "Qwen-MoE",
            MoeFamily::DeepSeekMoe => "DeepSeek-MoE",
        }
    }
}

/// Static metadata describing a family's tensor conventions. Pure wiring data;
/// nothing here loads or executes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeFamilyDescriptor {
    pub family: MoeFamily,
    /// Router tensor naming (informational).
    pub router_naming: &'static str,
    /// Expert tensor layout (informational).
    pub expert_layout: &'static str,
    /// Whether the family carries a shared expert.
    pub has_shared_expert: bool,
    /// The execution convention this family follows (MOE-17/18).
    pub renormalizes_topk: bool,
}

impl MoeFamily {
    /// The static descriptor for this family.
    pub fn descriptor(self) -> MoeFamilyDescriptor {
        match self {
            MoeFamily::Mixtral => MoeFamilyDescriptor {
                family: self,
                router_naming: "block_sparse_moe.gate / mlp.gate (packed)",
                expert_layout: "block_sparse_moe.experts.{e}.w1/w3/w2 or packed mlp.experts.gate_up_proj+down_proj",
                has_shared_expert: false,
                renormalizes_topk: true,
            },
            MoeFamily::QwenMoe => MoeFamilyDescriptor {
                family: self,
                router_naming: "mlp.gate / mlp.router",
                expert_layout: "mlp.experts.{e}.gate_proj/up_proj/down_proj (+ mlp.shared_expert.*)",
                has_shared_expert: true,
                renormalizes_topk: false,
            },
            MoeFamily::DeepSeekMoe => MoeFamilyDescriptor {
                family: self,
                router_naming: "mlp.gate",
                expert_layout: "packed mlp.experts.gate_up_proj/down_proj (+ mlp.shared_experts.*); MLA attention",
                has_shared_expert: true,
                renormalizes_topk: true,
            },
        }
    }
}

/// Identify the MoE family from a checkpoint's tensor names. Returns `None` if
/// the checkpoint is not MoE **or** is MoE but of an unrecognised family.
///
/// Rules (checked in order):
/// 1. any `block_sparse_moe` → **Mixtral** (classic, unambiguous).
/// 2. else any `shared_expert` / `.mlp.router.` / per-expert `…gate_proj`
///    under `.mlp.experts.` → **Qwen-MoE**.
/// 3. else packed `mlp.experts.gate_up_proj` (no shared, no block_sparse) →
///    **Mixtral packed** (documented heuristic; Qwen2-MoE-packed is rarer and
///    would carry a shared expert, caught by rule 2).
pub fn classify_family<'a, I>(names: I) -> Option<MoeFamily>
where
    I: IntoIterator<Item = &'a str>,
{
    let owned: Vec<&str> = names.into_iter().collect();
    if !detect_moe(owned.iter().copied()).is_moe {
        return None;
    }

    let mut has_block_sparse = false;
    let mut has_shared = false;
    let mut has_mlp_router = false;
    let mut has_qwen_classic_expert = false;
    let mut has_packed = false;
    let mut has_deepseek = false;
    for n in &owned {
        if n.contains("block_sparse_moe") {
            has_block_sparse = true;
        }
        if n.contains("shared_expert") {
            has_shared = true;
        }
        if n.contains(".mlp.router.") {
            has_mlp_router = true;
        }
        // Qwen classic per-expert gate (NOT the packed "gate_up_proj", which
        // does not contain the substring "gate_proj").
        if n.contains(".mlp.experts.") && n.contains("gate_proj") {
            has_qwen_classic_expert = true;
        }
        if n.contains("mlp.experts.gate_up_proj") {
            has_packed = true;
        }
        // DeepSeek: MLA (`kv_a_proj_with_mqa`) or the plural `shared_experts`.
        if n.contains("kv_a_proj_with_mqa") || n.contains(".shared_experts.") {
            has_deepseek = true;
        }
    }

    if has_deepseek {
        Some(MoeFamily::DeepSeekMoe)
    } else if has_block_sparse {
        Some(MoeFamily::Mixtral)
    } else if has_shared || has_mlp_router || has_qwen_classic_expert {
        Some(MoeFamily::QwenMoe)
    } else if has_packed {
        Some(MoeFamily::Mixtral)
    } else {
        None
    }
}

/// Outcome of validating a checkpoint's declared MoE config against its tensors.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FamilyConfigValidation {
    /// True iff no inconsistency was found.
    pub consistent: bool,
    /// Human-readable notes (mismatches, missing fields). Empty when fully
    /// consistent with all fields present.
    pub notes: Vec<String>,
}

/// Validate a declared [`MoeConfig`] (MOE-FULL-2) against the tensor evidence:
/// expert count, experts-per-token bounds, and shared-expert presence. Never
/// executes anything; it only cross-checks metadata. A fully-consistent MoE
/// with all fields present returns `consistent = true, notes = []`.
pub fn validate_family_config<'a, I>(names: I, config: &MoeConfig) -> FamilyConfigValidation
where
    I: IntoIterator<Item = &'a str>,
{
    let owned: Vec<&str> = names.into_iter().collect();
    let det = detect_moe(owned.iter().copied());
    let mut notes = Vec::new();
    let mut consistent = true;

    if !det.is_moe {
        return FamilyConfigValidation {
            consistent: false,
            notes: vec!["not a MoE checkpoint".into()],
        };
    }

    // Expert count: config vs tensor-implied.
    match (config.num_experts, det.implied_expert_count()) {
        (Some(cfg_n), Some(impl_n)) if cfg_n != impl_n => {
            consistent = false;
            notes.push(format!(
                "config num_experts={cfg_n} but tensors imply {impl_n}"
            ));
        }
        (None, _) => notes.push("config num_experts absent".into()),
        _ => {}
    }

    // experts_per_token bound.
    match (config.experts_per_token, config.num_experts) {
        (Some(k), Some(n)) if k > n => {
            consistent = false;
            notes.push(format!("experts_per_token={k} exceeds num_experts={n}"));
        }
        (None, _) => notes.push("config experts_per_token absent".into()),
        _ => {}
    }

    // Shared-expert agreement.
    let tensors_have_shared = det.shared_expert_tensor_count > 0;
    if config.has_shared_experts != tensors_have_shared {
        consistent = false;
        notes.push(format!(
            "config has_shared_experts={} but tensors {} shared-expert weights",
            config.has_shared_experts,
            if tensors_have_shared { "contain" } else { "contain no" }
        ));
    }

    FamilyConfigValidation { consistent, notes }
}

/// Build the **family-aware fail-loud message** the productive loader emits
/// when it detects a MoE checkpoint. The loader still returns
/// `LoaderError::MoeUnsupported` (the fail-loud guard is **unchanged**); this
/// only makes the message precise:
///
/// ```text
/// MoE detected
/// Family: Mixtral (experts=8, router_tensors=1, shared_expert_tensors=0)
/// Productive support not enabled (experimental MoE path is opt-in / test-only;
/// see docs/MOE_OVERVIEW.md). Loading as dense would drop the expert weights.
/// ```
///
/// Pure; no I/O. Returns `None`-equivalent fallback wording if the names are
/// not MoE (the loader only calls this once detection has already fired).
pub fn moe_failloud_report<'a, I>(names: I) -> String
where
    I: IntoIterator<Item = &'a str>,
{
    let owned: Vec<&str> = names.into_iter().collect();
    let det = detect_moe(owned.iter().copied());
    let family = classify_family(owned.iter().copied())
        .map(|f| f.name())
        .unwrap_or("unrecognised");
    let experts = det
        .implied_expert_count()
        .map(|n| n.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    // MOE-FULL-10: a controlled Mixtral runtime exists behind an opt-in. The
    // dense loader still refuses MoE (it cannot execute experts); the opt-in
    // only enables the dedicated `moe::runtime::MixtralRuntime` entry.
    let opt_in_hint = if family == "Mixtral" {
        format!(
            " A controlled experimental Mixtral runtime is available via \
             moe::runtime::MixtralRuntime when {EXPERIMENTAL_MOE_ENV}=1 (opt-in); \
             the dense loader still refuses MoE."
        )
    } else {
        String::new()
    };
    format!(
        "MoE detected\n\
         Family: {family} (experts={experts}, router_tensors={}, expert_tensors={}, shared_expert_tensors={})\n\
         Productive support not enabled (the experimental MoE path is opt-in / test-only; \
         see docs/MOE_OVERVIEW.md and docs/HANDOFF_MOE_FULL_10.md). Loading this checkpoint \
         as dense would silently drop the expert weights and produce a broken model.{opt_in_hint}",
        det.router_tensor_count, det.expert_tensor_count, det.shared_expert_tensor_count,
    )
}

// ============================================================================
// Tests (synthetic tensor names + configs — no model, no loader, no execution)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn mixtral_classic() -> Vec<&'static str> {
        vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.block_sparse_moe.gate.weight",
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            "model.layers.0.block_sparse_moe.experts.0.w2.weight",
            "model.layers.0.block_sparse_moe.experts.0.w3.weight",
            "model.layers.0.block_sparse_moe.experts.7.w1.weight",
        ]
    }

    fn mixtral_packed() -> Vec<&'static str> {
        vec![
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.down_proj",
        ]
    }

    fn qwen_moe() -> Vec<&'static str> {
        vec![
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.0.up_proj.weight",
            "model.layers.0.mlp.experts.0.down_proj.weight",
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.layers.0.mlp.shared_expert_gate.weight",
        ]
    }

    fn dense() -> Vec<&'static str> {
        vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ]
    }

    #[test]
    fn classifies_mixtral_classic_and_packed() {
        assert_eq!(classify_family(mixtral_classic()), Some(MoeFamily::Mixtral));
        assert_eq!(classify_family(mixtral_packed()), Some(MoeFamily::Mixtral));
    }

    #[test]
    fn classifies_qwen_moe() {
        assert_eq!(classify_family(qwen_moe()), Some(MoeFamily::QwenMoe));
    }

    #[test]
    fn dense_is_not_a_family() {
        assert_eq!(classify_family(dense()), None);
    }

    #[test]
    fn descriptors_carry_family_metadata() {
        let m = MoeFamily::Mixtral.descriptor();
        assert!(!m.has_shared_expert);
        assert!(m.renormalizes_topk);
        let q = MoeFamily::QwenMoe.descriptor();
        assert!(q.has_shared_expert);
        assert!(!q.renormalizes_topk);
    }

    #[test]
    fn validate_config_consistent() {
        let cfg = MoeConfig {
            num_experts: Some(8),
            experts_per_token: Some(2),
            has_shared_experts: false,
            ..Default::default()
        };
        let v = validate_family_config(mixtral_classic(), &cfg);
        assert!(v.consistent, "notes: {:?}", v.notes);
        assert!(v.notes.is_empty());
    }

    #[test]
    fn validate_config_detects_expert_count_mismatch() {
        let cfg = MoeConfig {
            num_experts: Some(4), // tensors imply 8
            experts_per_token: Some(2),
            has_shared_experts: false,
            ..Default::default()
        };
        let v = validate_family_config(mixtral_classic(), &cfg);
        assert!(!v.consistent);
        assert!(v.notes.iter().any(|n| n.contains("num_experts")));
    }

    #[test]
    fn validate_config_detects_shared_expert_disagreement() {
        // Qwen tensors have a shared expert, but config says none.
        let cfg = MoeConfig {
            num_experts: Some(1),
            experts_per_token: Some(1),
            has_shared_experts: false,
            ..Default::default()
        };
        let v = validate_family_config(qwen_moe(), &cfg);
        assert!(!v.consistent);
        assert!(v.notes.iter().any(|n| n.contains("shared_experts")));
    }

    #[test]
    fn failloud_report_names_the_family() {
        let msg = moe_failloud_report(mixtral_classic());
        assert!(msg.contains("MoE detected"));
        assert!(msg.contains("Family: Mixtral"));
        assert!(msg.contains("experts=8"));
        assert!(msg.contains("Productive support not enabled"));

        let qmsg = moe_failloud_report(qwen_moe());
        assert!(qmsg.contains("Family: Qwen-MoE"));
    }
}
