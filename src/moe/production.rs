//! **MOE-FULL-14** — controlled production MoE path.
//!
//! A single gated entry that turns the experimental MoE runtime into a
//! **controlled** product path, **without** declaring general support. It
//! dispatches a model directory through the certification gate:
//!
//! ```text
//!   model dir → detect MoE → classify family → unsupported-variant check
//!             → certification scope (manifest) → opt-in flag
//!             → MoeRuntime → generate
//! ```
//!
//! Every gate has a clear error. The dense path is untouched; the dense
//! loader's fail-loud guard still refuses to load MoE as a dense model.
//! Execution requires **both** a certified family **and** the explicit opt-in
//! (`ATENIA_ENABLE_MOE=1` or `--experimental-moe`).

use std::path::{Path, PathBuf};

use super::detect::detect_moe;
use super::family::{classify_family, experimental_moe_enabled, MoeFamily, ENABLE_MOE_ENV};
use super::manifest::{MoeCertManifest, MoeCertScope};
use super::runtime::{MoeRuntime, MoeRuntimeError};
use crate::v17::loader::safetensors_reader::SafetensorsReader;

/// Errors from the controlled MoE production path.
#[derive(Debug)]
pub enum ControlledMoeError {
    /// The directory has no readable `*.safetensors`, or I/O failed.
    Io(String),
    /// Not a MoE checkpoint (or an unrecognised MoE family).
    NotMoe,
    /// A recognised-but-unsupported variant (e.g. Qwen3 QK-norm, DeepSeek
    /// Q-LoRA). Carries the explanation.
    UnsupportedVariant(String),
    /// The family is recognised but not at a runnable certification scope.
    NotCertified { family: MoeFamily, scope: MoeCertScope },
    /// The opt-in flag is not set.
    NotEnabled { family: MoeFamily, scope: MoeCertScope },
    /// The MoE runtime failed to load / run.
    Runtime(MoeRuntimeError),
}

impl std::fmt::Display for ControlledMoeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ControlledMoeError::Io(m) => write!(f, "moe-production: {m}"),
            ControlledMoeError::NotMoe => {
                write!(f, "moe-production: not a recognised MoE checkpoint")
            }
            ControlledMoeError::UnsupportedVariant(m) => {
                write!(f, "MoE checkpoint detected.\n{m}")
            }
            ControlledMoeError::NotCertified { family, scope } => write!(
                f,
                "MoE checkpoint detected.\nFamily: {}.\nStatus: {} (not a runnable certification \
                 scope). The controlled MoE runtime only runs certified families.",
                family.name(),
                scope.as_str()
            ),
            ControlledMoeError::NotEnabled { family, scope } => write!(
                f,
                "MoE checkpoint detected.\nFamily: {}.\nStatus: {}.\nThe controlled MoE runtime is \
                 opt-in: set {ENABLE_MOE_ENV}=1 (or pass --experimental-moe) to run it. The dense \
                 loader still refuses MoE.",
                family.name(),
                scope.as_str()
            ),
            ControlledMoeError::Runtime(e) => write!(f, "moe-production: {e}"),
        }
    }
}

impl std::error::Error for ControlledMoeError {}

/// Read-only diagnosis of a model directory's MoE status.
#[derive(Debug, Clone)]
pub struct MoeDiagnosis {
    pub is_moe: bool,
    pub family: Option<MoeFamily>,
    pub scope: Option<MoeCertScope>,
    pub unsupported_variant: Option<String>,
    /// True iff a controlled `generate` would be allowed *given the opt-in*.
    pub certified_runnable: bool,
    /// Whether the opt-in flag is currently set.
    pub opt_in_set: bool,
    /// A human-facing one-paragraph status line.
    pub message: String,
}

fn first_safetensors(dir: &Path) -> Result<PathBuf, ControlledMoeError> {
    std::fs::read_dir(dir)
        .map_err(|e| ControlledMoeError::Io(format!("read_dir {dir:?}: {e}")))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.extension().and_then(|x| x.to_str()) == Some("safetensors"))
        .ok_or_else(|| ControlledMoeError::Io(format!("no .safetensors in {dir:?}")))
}

fn read_names(dir: &Path) -> Result<Vec<String>, ControlledMoeError> {
    let st = first_safetensors(dir)?;
    let reader = SafetensorsReader::open(&st)
        .map_err(|e| ControlledMoeError::Io(format!("open {st:?}: {e:?}")))?;
    Ok(reader.iter().map(|e| e.name.to_string()).collect())
}

/// Detect an unsupported MoE *variant* from tensor names (vs the manifest's
/// `unsupported_variants`). Returns the explanation if matched.
fn unsupported_variant(names: &[String], manifest: &MoeCertManifest) -> Option<String> {
    // QK-norm (Qwen3-MoE): per-head q/k RMSNorm in attention.
    let has_qk_norm = names
        .iter()
        .any(|n| n.contains("self_attn.q_norm") || n.contains("self_attn.k_norm"));
    // DeepSeek Q-LoRA: low-rank query compression.
    let has_q_lora = names
        .iter()
        .any(|n| n.contains("self_attn.q_a_proj") || n.contains("self_attn.q_b_proj"));
    if has_qk_norm {
        let u = manifest.unsupported_variants.iter().find(|u| u.marker.contains("q_norm"));
        return Some(
            u.map(|u| format!("Family: {} variant.\nStatus: unsupported ({}).", u.variant, u.reason))
                .unwrap_or_else(|| "Status: unsupported variant (QK-norm attention).".into()),
        );
    }
    if has_q_lora {
        let u = manifest.unsupported_variants.iter().find(|u| u.marker.contains("q_a_proj"));
        return Some(
            u.map(|u| format!("Family: {} variant.\nStatus: unsupported ({}).", u.variant, u.reason))
                .unwrap_or_else(|| "Status: unsupported variant (Q-LoRA).".into()),
        );
    }
    // **MOE-PRODUCT-2** — DeepSeek-V3 modern routing (sigmoid + aux-loss-free):
    // the per-expert `e_score_correction_bias` tensor. The V3 routing mechanism is
    // L0-certified (`src/moe/v3_router.rs`) but **mechanism-only / non-runnable** —
    // refuse it as a real model (defence-in-depth; real V3 is also caught above by
    // Q-LoRA). DeepSeek-V2-Lite has no such tensor, so it is unaffected.
    let has_v3_router = names.iter().any(|n| n.contains("e_score_correction_bias"));
    if has_v3_router {
        return Some(
            "Family: DeepSeek-V3 routing variant.\nStatus: unsupported (modern routing \
             sigmoid + aux-loss-free + group-limited is L0 mechanism-only / non-runnable)."
                .into(),
        );
    }
    None
}

/// **Read-only** diagnosis of a model directory (never executes). Safe to call
/// regardless of the opt-in flag — it only reports status.
pub fn diagnose_moe(dir: &Path) -> MoeDiagnosis {
    let manifest = MoeCertManifest::builtin();
    let names = match read_names(dir) {
        Ok(n) => n,
        Err(e) => {
            return MoeDiagnosis {
                is_moe: false,
                family: None,
                scope: None,
                unsupported_variant: None,
                certified_runnable: false,
                opt_in_set: experimental_moe_enabled(),
                message: format!("could not inspect model: {e}"),
            };
        }
    };
    let is_moe = detect_moe(names.iter().map(|s| s.as_str())).is_moe;
    if !is_moe {
        return MoeDiagnosis {
            is_moe: false,
            family: None,
            scope: None,
            unsupported_variant: None,
            certified_runnable: false,
            opt_in_set: experimental_moe_enabled(),
            message: "dense checkpoint (not MoE).".into(),
        };
    }
    if let Some(u) = unsupported_variant(&names, &manifest) {
        return MoeDiagnosis {
            is_moe: true,
            family: classify_family(names.iter().map(|s| s.as_str())),
            scope: Some(MoeCertScope::Unsupported),
            unsupported_variant: Some(u.clone()),
            certified_runnable: false,
            opt_in_set: experimental_moe_enabled(),
            message: format!("MoE detected. {u}"),
        };
    }
    let family = classify_family(names.iter().map(|s| s.as_str()));
    let scope = family.map(|f| manifest.scope_for(f));
    let runnable = scope.map(|s| s.is_runnable()).unwrap_or(false);
    let opt_in = experimental_moe_enabled();
    let message = match (family, scope) {
        (Some(fam), Some(sc)) => {
            if sc.is_runnable() {
                if opt_in {
                    format!(
                        "MoE detected. Family: {}. Status: {}. Controlled runtime ENABLED ({ENABLE_MOE_ENV}=1).",
                        fam.name(), sc.as_str()
                    )
                } else {
                    format!(
                        "MoE detected. Family: {}. Status: {}. Set {ENABLE_MOE_ENV}=1 (or --experimental-moe) to run the controlled MoE runtime.",
                        fam.name(), sc.as_str()
                    )
                }
            } else {
                format!(
                    "MoE detected. Family: {}. Status: {} (not runnable).",
                    fam.name(), sc.as_str()
                )
            }
        }
        _ => "MoE detected, but the family is not recognised.".into(),
    };
    MoeDiagnosis {
        is_moe: true,
        family,
        scope,
        unsupported_variant: None,
        certified_runnable: runnable,
        opt_in_set: opt_in,
        message,
    }
}

/// **MOE-INTEGRATE-2** — the routing decision the normal `generate` path makes
/// from a read-only [`MoeDiagnosis`]. Pure (no I/O); the CLI maps it to an
/// action. The default for anything that is not a runnable, opted-in MoE is to
/// **not** run MoE — dense passes through, everything else fails loud.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoeRoute {
    /// Not a (recognised) MoE checkpoint → use the dense pipeline, unchanged.
    Dense,
    /// A runnable, certified MoE family **with the opt-in set** → route to the
    /// controlled MoE runtime.
    RunMoe { family: MoeFamily },
    /// A runnable, certified MoE family but the **opt-in is not set** → fail
    /// loud (the default protection).
    NeedsOptIn { family: MoeFamily },
    /// MoE but an unsupported variant / not a runnable certification scope /
    /// unrecognised family → fail loud.
    Refused,
}

/// Decide how the normal `generate` path should handle a diagnosed model.
/// Fail-loud by default: only a runnable, certified MoE family with the opt-in
/// explicitly set is routed to the MoE runtime; dense passes through unchanged.
pub fn decide_route(diag: &MoeDiagnosis) -> MoeRoute {
    if !diag.is_moe {
        return MoeRoute::Dense;
    }
    if diag.unsupported_variant.is_some() || !diag.certified_runnable {
        return MoeRoute::Refused;
    }
    match diag.family {
        // **MOE-PRODUCT-2** — DeepSeek-MoE (DeepSeek-V2-Lite, MLA) is now routable
        // behind the opt-in, like Mixtral / Qwen-MoE. The unsupported-variant gate
        // above already refuses DeepSeek-V2 (Q-LoRA) and DeepSeek-V3 (Q-LoRA + the
        // V3 routing marker), so only the certified, runnable V2-Lite shape reaches
        // here. The DeepSeek-V3 routing *mechanism* is never a runnable model.
        Some(family) if diag.opt_in_set => MoeRoute::RunMoe { family },
        Some(family) => MoeRoute::NeedsOptIn { family },
        None => MoeRoute::Refused,
    }
}

/// **Controlled production generate**: gate a model directory through the
/// certification matrix + opt-in, then run the MoE runtime. Returns the
/// generated token ids. Every gate is a clear error; the dense path is never
/// touched.
pub fn controlled_moe_generate(
    dir: &Path,
    prompt_ids: &[u32],
    max_new_tokens: usize,
) -> Result<Vec<u32>, ControlledMoeError> {
    let manifest = MoeCertManifest::builtin();
    let names = read_names(dir)?;

    if !detect_moe(names.iter().map(|s| s.as_str())).is_moe {
        return Err(ControlledMoeError::NotMoe);
    }
    if let Some(u) = unsupported_variant(&names, &manifest) {
        return Err(ControlledMoeError::UnsupportedVariant(u));
    }
    let family = classify_family(names.iter().map(|s| s.as_str()))
        .ok_or(ControlledMoeError::NotMoe)?;
    let scope = manifest.scope_for(family);
    if !scope.is_runnable() {
        return Err(ControlledMoeError::NotCertified { family, scope });
    }
    if !experimental_moe_enabled() {
        return Err(ControlledMoeError::NotEnabled { family, scope });
    }

    let rt = MoeRuntime::load_from_dir(dir).map_err(ControlledMoeError::Runtime)?;
    Ok(rt.generate(prompt_ids, max_new_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnose_nonexistent_is_not_moe() {
        let d = diagnose_moe(Path::new("/nonexistent/model/dir"));
        assert!(!d.is_moe);
    }

    // ---- MOE-INTEGRATE-2: routing decision (pure; no model needed) ----

    fn diag(
        is_moe: bool,
        family: Option<MoeFamily>,
        runnable: bool,
        opt_in: bool,
        unsupported: Option<&str>,
    ) -> MoeDiagnosis {
        MoeDiagnosis {
            is_moe,
            family,
            scope: None,
            unsupported_variant: unsupported.map(str::to_string),
            certified_runnable: runnable,
            opt_in_set: opt_in,
            message: String::new(),
        }
    }

    #[test]
    fn dense_routes_to_dense() {
        assert_eq!(
            decide_route(&diag(false, None, false, false, None)),
            MoeRoute::Dense
        );
        // A dense checkpoint with the opt-in set is still dense.
        assert_eq!(
            decide_route(&diag(false, None, false, true, None)),
            MoeRoute::Dense
        );
    }

    #[test]
    fn runnable_moe_requires_opt_in() {
        // Mixtral, runnable, no opt-in → fail loud (NeedsOptIn).
        assert_eq!(
            decide_route(&diag(true, Some(MoeFamily::Mixtral), true, false, None)),
            MoeRoute::NeedsOptIn { family: MoeFamily::Mixtral }
        );
        // Qwen-MoE, runnable, opt-in set → route to MoE.
        assert_eq!(
            decide_route(&diag(true, Some(MoeFamily::QwenMoe), true, true, None)),
            MoeRoute::RunMoe { family: MoeFamily::QwenMoe }
        );
    }

    #[test]
    fn unsupported_or_uncertified_is_refused_even_with_opt_in() {
        // Unsupported variant → Refused regardless of opt-in.
        assert_eq!(
            decide_route(&diag(true, Some(MoeFamily::QwenMoe), false, true, Some("QK-norm"))),
            MoeRoute::Refused
        );
        // Recognised but not runnable scope → Refused.
        assert_eq!(
            decide_route(&diag(true, Some(MoeFamily::DeepSeekMoe), false, true, None)),
            MoeRoute::Refused
        );
        // MoE but unrecognised family → Refused.
        assert_eq!(
            decide_route(&diag(true, None, false, true, None)),
            MoeRoute::Refused
        );
    }

    // ---- MOE-PRODUCT-2: DeepSeek-V2-Lite routable; V3 routing refused ----

    #[test]
    fn deepseek_v2_lite_runnable_routes_with_opt_in() {
        // DeepSeek-MoE (V2-Lite), runnable, no opt-in → NeedsOptIn.
        assert_eq!(
            decide_route(&diag(true, Some(MoeFamily::DeepSeekMoe), true, false, None)),
            MoeRoute::NeedsOptIn { family: MoeFamily::DeepSeekMoe }
        );
        // With the opt-in → routes to the MoE runtime.
        assert_eq!(
            decide_route(&diag(true, Some(MoeFamily::DeepSeekMoe), true, true, None)),
            MoeRoute::RunMoe { family: MoeFamily::DeepSeekMoe }
        );
    }

    #[test]
    fn v3_router_marker_is_unsupported_variant() {
        let manifest = MoeCertManifest::builtin();
        // DeepSeek-V3 routing tensor → unsupported (mechanism-only, non-runnable).
        let v3 = vec![
            "model.layers.0.mlp.gate.weight".to_string(),
            "model.layers.0.mlp.gate.e_score_correction_bias".to_string(),
        ];
        let u = unsupported_variant(&v3, &manifest).expect("V3 router marker is unsupported");
        assert!(u.contains("non-runnable") || u.contains("V3"));
        // DeepSeek-V2-Lite (no Q-LoRA, no V3 marker) is NOT flagged.
        let v2lite = vec![
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight".to_string(),
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "model.layers.0.mlp.gate.weight".to_string(),
            "model.layers.0.mlp.shared_experts.gate_proj.weight".to_string(),
        ];
        assert!(unsupported_variant(&v2lite, &manifest).is_none());
    }
}
