//! **MODEL-INTAKE-1** — architecture compatibility layer
//! ("Say Yes More Often, Safely").
//!
//! The native adapter registry ([`super::resolve_adapter`]) matches a
//! checkpoint's `architectures[0]` / `model_type` against the seven
//! registered families. Anything it does not recognise is, by itself,
//! a **hard reject** at load (`pipeline.rs`): many genuinely
//! Llama-compatible checkpoints never even get a chance to run.
//!
//! This module adds an **explicit, auditable** second stage that runs
//! **only when the native registry returns `None`** — so every
//! certified / natively-supported model keeps its exact current
//! behaviour (this layer is never consulted for them). For a
//! non-native architecture it produces one of two outcomes:
//!
//! - **Accept** — either the architecture is on the curated,
//!   evidence-gated [`LLAMA_COMPATIBLE_ALLOWLIST`], or the operator
//!   opted into the **generic decoder path** (`ATENIA_INTAKE_GENERIC=1`).
//!   In both cases the checkpoint's config must first pass the
//!   [`check_llama_topology`] structural checks, and the acceptance is
//!   loudly logged as **UNCERTIFIED**.
//! - **Reject** — with an explicit, actionable message. **There is no
//!   silent fallback**: an unknown architecture without opt-in, or any
//!   config that fails the topology checks, is refused.
//!
//! Design rules honoured (MODEL-INTAKE-1): safety > coverage; never
//! run a clearly-incompatible model; certified families unchanged;
//! never degrade correctness; every fallback explicit, auditable, and
//! visible.

use crate::nn::llama::config::LlamaConfig;

use super::{resolve_adapter, AteniaModelAdapter, ModelFormat, ModelMetadata};

/// Environment opt-in for the generic Llama-compatible decoder path.
/// Default off → unknown architectures are rejected (current
/// behaviour preserved). Set to `1` to attempt the generic path for
/// an unknown architecture, subject to the topology checks.
pub const GENERIC_INTAKE_ENV: &str = "ATENIA_INTAKE_GENERIC";

/// One curated, **evidence-gated** allowlist entry: an
/// `architectures[0]` string that is *not* natively registered but is
/// documented to be the same decoder topology as `base_architecture`.
#[derive(Clone, Copy, Debug)]
pub struct AllowlistEntry {
    /// The non-native `architectures[0]` string this entry matches.
    pub architecture: &'static str,
    /// The registered base architecture it maps to (e.g.
    /// `"LlamaForCausalLM"`). Resolved through the native registry so
    /// the base family's weight mapper / graph builder are reused.
    pub base_architecture: &'static str,
    /// Why this mapping is considered safe — recorded so the decision
    /// is auditable, never "by intuition".
    pub evidence: &'static str,
}

/// **Curated, evidence-gated allowlist.** Deliberately small: the vast
/// majority of Llama-compatible checkpoints (Vicuna, NousHermes,
/// SmolLM, OpenLLaMA, TinyLlama, Yi's HF releases, …) already declare
/// `architectures: ["LlamaForCausalLM"]` and therefore resolve
/// **natively** — they never reach this table. This list only covers
/// the residual checkpoints that ship a *distinct* architecture string
/// yet are, by documented evidence, the identical Llama decoder. Add a
/// new entry only with a concrete topology reference, never on a hunch.
pub const LLAMA_COMPATIBLE_ALLOWLIST: &[AllowlistEntry] = &[
    AllowlistEntry {
        architecture: "LLaMAForCausalLM",
        base_architecture: "LlamaForCausalLM",
        evidence: "legacy capitalisation of LlamaForCausalLM emitted by the original \
                   LLaMA release and some older converters — byte-identical decoder \
                   and tensor names",
    },
    AllowlistEntry {
        architecture: "YiForCausalLM",
        base_architecture: "LlamaForCausalLM",
        evidence: "Yi (01.AI) Technical Report adopts the Llama architecture \
                   (RMSNorm + RoPE + SwiGLU + GQA) with standard Llama tensor names; \
                   the trust_remote_code variant exposes this distinct arch string \
                   while HF-format Yi releases declare LlamaForCausalLM directly",
    },
];

/// Which intake path accepted (or would accept) an architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntakeStatus {
    /// Matched the curated evidence-gated allowlist.
    Allowlisted,
    /// Unknown architecture accepted via the `ATENIA_INTAKE_GENERIC`
    /// opt-in after passing the topology checks.
    Generic,
}

impl IntakeStatus {
    pub fn label(self) -> &'static str {
        match self {
            IntakeStatus::Allowlisted => "allowlist (known-compatible)",
            IntakeStatus::Generic => "generic opt-in (ATENIA_INTAKE_GENERIC)",
        }
    }
}

/// The decision the compat layer hands back to the loader.
pub enum IntakeOutcome {
    /// Run the checkpoint with `adapter`, after emitting `warnings`
    /// (already operator-facing strings). `status` records the path.
    Accept {
        adapter: &'static dyn AteniaModelAdapter,
        status: IntakeStatus,
        warnings: Vec<String>,
    },
    /// Refuse the checkpoint with an explicit, actionable `message`.
    Reject { message: String },
}

/// Resolve a **non-native** architecture through the compatibility
/// layer. Call this only when [`resolve_adapter`] returned `None`.
///
/// `generic_opt_in` is normally `std::env::var(GENERIC_INTAKE_ENV) ==
/// Ok("1")`; it is a parameter so the decision is pure and testable.
pub fn resolve_intake(
    architecture: &str,
    model_type: Option<&str>,
    config: &LlamaConfig,
    generic_opt_in: bool,
) -> IntakeOutcome {
    // 1. Curated allowlist — known-compatible distinct arch strings.
    if let Some(entry) = allowlist_lookup(architecture) {
        let Some(adapter) = base_adapter(entry.base_architecture) else {
            // Defensive: an allowlist entry whose base is not itself
            // natively registered is a programming error, not an
            // operator error. Refuse rather than mis-route.
            return IntakeOutcome::Reject {
                message: format!(
                    "internal: allowlist base architecture {:?} is not registered",
                    entry.base_architecture
                ),
            };
        };
        return match check_llama_topology(config) {
            Ok(warnings) => {
                let mut all = vec![format!(
                    "architecture \"{architecture}\" is on the known-compatible allowlist \
                     (mapped to {} — UNCERTIFIED). Evidence: {}",
                    entry.base_architecture, entry.evidence
                )];
                all.extend(warnings);
                IntakeOutcome::Accept {
                    adapter,
                    status: IntakeStatus::Allowlisted,
                    warnings: all,
                }
            }
            Err(reasons) => IntakeOutcome::Reject {
                message: reject_message(
                    architecture,
                    &format!(
                        "it is allowlisted as {}-compatible but its config fails the \
                         compatibility checks",
                        entry.base_architecture
                    ),
                    &reasons,
                ),
            },
        };
    }

    // 2. Unknown architecture — only via explicit opt-in.
    if !generic_opt_in {
        return IntakeOutcome::Reject {
            message: format!(
                "unsupported architecture \"{architecture}\" (model_type={model_type:?}). \
                 {}. If you believe this is a standard Llama-compatible decoder, re-run \
                 with {GENERIC_INTAKE_ENV}=1 to attempt the generic decoder path \
                 (UNCERTIFIED, topology-checked). Otherwise it is rejected.",
                super::supported_architectures_message()
            ),
        };
    }

    let Some(adapter) = base_adapter("LlamaForCausalLM") else {
        return IntakeOutcome::Reject {
            message: "internal: Llama base adapter is not registered".to_string(),
        };
    };
    match check_llama_topology(config) {
        Ok(warnings) => {
            let mut all = vec![format!(
                "architecture \"{architecture}\" accepted via the GENERIC opt-in decoder \
                 path ({GENERIC_INTAKE_ENV}=1) as a Llama-compatible decoder — UNCERTIFIED. \
                 Correctness is the operator's responsibility; weight loading still fails \
                 loudly on any tensor-name/shape mismatch."
            )];
            all.extend(warnings);
            IntakeOutcome::Accept {
                adapter,
                status: IntakeStatus::Generic,
                warnings: all,
            }
        }
        Err(reasons) => IntakeOutcome::Reject {
            message: reject_message(
                architecture,
                "it failed the generic Llama-compatibility checks and will not be run",
                &reasons,
            ),
        },
    }
}

/// Structural compatibility checks for the generic / allowlisted Llama
/// path (**FASE 5**). Returns `Ok(warnings)` when the config is a
/// plain Llama-topology decoder (the `warnings` are non-fatal
/// divergences worth surfacing), or `Err(reasons)` listing every hard
/// incompatibility found.
///
/// Hard failures are config shapes a generic Llama build **cannot**
/// reproduce faithfully — accepting them would silently corrupt
/// numerics, violating the "never degrade correctness" rule. Anything
/// requiring a dedicated adapter (Gemma soft-caps, Gemma 3 dual-RoPE,
/// Phi-4 partial rotary) is refused here rather than mis-run as Llama.
pub fn check_llama_topology(c: &LlamaConfig) -> Result<Vec<String>, Vec<String>> {
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // --- positive dimensions ---
    if c.hidden_size == 0 {
        errors.push("hidden_size must be > 0".into());
    }
    if c.num_attention_heads == 0 {
        errors.push("num_attention_heads must be > 0".into());
    }
    if c.num_key_value_heads == 0 {
        errors.push("num_key_value_heads must be > 0".into());
    }
    if c.intermediate_size == 0 {
        errors.push("intermediate_size must be > 0".into());
    }
    if c.vocab_size == 0 {
        errors.push("vocab_size must be > 0".into());
    }
    if c.num_hidden_layers == 0 {
        errors.push("num_hidden_layers must be > 0".into());
    }
    if c.rope_theta == 0 {
        errors.push("rope_theta must be > 0".into());
    }
    if !(c.rms_norm_eps.is_finite() && c.rms_norm_eps > 0.0) {
        errors.push(format!(
            "rms_norm_eps must be finite and > 0 (got {})",
            c.rms_norm_eps
        ));
    }

    // --- head geometry (only when the dims are sane) ---
    if c.hidden_size > 0 && c.num_attention_heads > 0 {
        if c.hidden_size % c.num_attention_heads != 0 && c.head_dim.is_none() {
            errors.push(format!(
                "hidden_size ({}) is not divisible by num_attention_heads ({}) and no \
                 explicit head_dim is set",
                c.hidden_size, c.num_attention_heads
            ));
        }
    }
    if c.num_attention_heads > 0 && c.num_key_value_heads > 0 {
        if c.num_attention_heads % c.num_key_value_heads != 0 {
            errors.push(format!(
                "num_attention_heads ({}) is not divisible by num_key_value_heads ({}) — \
                 GQA grouping is undefined",
                c.num_attention_heads, c.num_key_value_heads
            ));
        }
    }
    if let Some(hd) = c.head_dim {
        if hd == 0 {
            errors.push("explicit head_dim must be > 0".into());
        }
    }

    // --- specialised-family fields require a dedicated adapter ---
    // A generic Llama build would silently drop these → wrong numerics.
    if c.attn_logit_softcapping.is_some() {
        errors.push(
            "config carries attn_logit_softcapping (Gemma-2 attention soft-cap) which the \
             generic Llama path cannot reproduce — a dedicated adapter is required"
                .into(),
        );
    }
    if c.final_logit_softcapping.is_some() {
        errors.push(
            "config carries final_logit_softcapping (Gemma-2 final soft-cap) which the \
             generic Llama path cannot reproduce — a dedicated adapter is required"
                .into(),
        );
    }
    if c.query_pre_attn_scalar.is_some() {
        errors.push(
            "config carries query_pre_attn_scalar (Gemma-2 attention scaling) which the \
             generic Llama path cannot reproduce — a dedicated adapter is required"
                .into(),
        );
    }
    if c.rope_local_base_freq.is_some() || c.sliding_window_pattern.is_some() {
        errors.push(
            "config carries Gemma-3 dual-RoPE / sliding-window-pattern fields which the \
             generic Llama path cannot reproduce — a dedicated adapter is required"
                .into(),
        );
    }
    if c.partial_rotary_factor.is_some() {
        errors.push(
            "config carries partial_rotary_factor (partial rotary, e.g. Phi-4) which the \
             generic Llama path cannot reproduce — a dedicated adapter is required"
                .into(),
        );
    }

    // --- non-fatal divergences (accept, but surface) ---
    if let Some(w) = c.sliding_window {
        warnings.push(format!(
            "config declares sliding_window={w}; the generic Llama path uses full \
             attention, so outputs diverge for contexts longer than {w} tokens"
        ));
    }
    if c.rope_scaling.is_some() {
        warnings.push(
            "config declares rope_scaling; the generic path applies the Llama-family \
             scaling rules — verify they match the source model"
                .into(),
        );
    }

    if errors.is_empty() {
        Ok(warnings)
    } else {
        Err(errors)
    }
}

/// Look up a non-native architecture string in the curated allowlist.
pub fn allowlist_lookup(architecture: &str) -> Option<&'static AllowlistEntry> {
    LLAMA_COMPATIBLE_ALLOWLIST
        .iter()
        .find(|e| e.architecture == architecture)
}

/// Resolve a base architecture string through the native registry.
fn base_adapter(base_architecture: &str) -> Option<&'static dyn AteniaModelAdapter> {
    let metadata = ModelMetadata {
        architecture: base_architecture,
        model_type: None,
        format: ModelFormat::HfSafetensors,
    };
    resolve_adapter(&metadata)
}

fn reject_message(architecture: &str, lead: &str, reasons: &[String]) -> String {
    format!(
        "architecture \"{architecture}\": {lead}: [{}]",
        reasons.join("; ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal, structurally-valid plain-Llama config. Built field
    /// by field (not via `from_json_str`) so tests can deliberately
    /// inject incompatible shapes that the parser/validator would
    /// otherwise reject up front.
    fn base_cfg() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 32000,
            hidden_size: 2048,
            num_hidden_layers: 22,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            intermediate_size: 5632,
            max_position_embeddings: 2048,
            rope_theta: 10000,
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
            attention_bias: Some(false),
            model_type: Some("llama".to_string()),
            bos_token_id: 1,
            eos_token_id: 2,
            eos_token_ids: vec![2],
            pad_token_id: None,
            head_dim: None,
            rope_scaling: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            sliding_window: None,
            query_pre_attn_scalar: None,
            rope_local_base_freq: None,
            sliding_window_pattern: None,
            partial_rotary_factor: None,
        }
    }

    #[test]
    fn allowlisted_arch_accepts_without_opt_in() {
        let out = resolve_intake("YiForCausalLM", Some("Yi"), &base_cfg(), false);
        match out {
            IntakeOutcome::Accept { status, warnings, .. } => {
                assert_eq!(status, IntakeStatus::Allowlisted);
                assert!(warnings.iter().any(|w| w.contains("allowlist")));
            }
            IntakeOutcome::Reject { message } => panic!("expected accept, got reject: {message}"),
        }
    }

    #[test]
    fn unknown_arch_rejected_without_opt_in() {
        let out = resolve_intake("TotallyUnknownForCausalLM", None, &base_cfg(), false);
        match out {
            IntakeOutcome::Reject { message } => {
                assert!(message.contains("unsupported architecture"));
                assert!(message.contains(GENERIC_INTAKE_ENV));
            }
            IntakeOutcome::Accept { .. } => panic!("unknown arch must be rejected without opt-in"),
        }
    }

    #[test]
    fn unknown_arch_accepts_with_opt_in_and_clean_config() {
        let out = resolve_intake("TotallyUnknownForCausalLM", None, &base_cfg(), true);
        match out {
            IntakeOutcome::Accept { status, warnings, .. } => {
                assert_eq!(status, IntakeStatus::Generic);
                assert!(warnings.iter().any(|w| w.contains("GENERIC")));
            }
            IntakeOutcome::Reject { message } => panic!("expected accept, got reject: {message}"),
        }
    }

    #[test]
    fn generic_rejects_indivisible_heads() {
        let mut cfg = base_cfg();
        cfg.num_attention_heads = 30; // 2048 % 30 != 0, no head_dim
        let out = resolve_intake("TotallyUnknownForCausalLM", None, &cfg, true);
        assert!(matches!(out, IntakeOutcome::Reject { .. }));
    }

    #[test]
    fn generic_rejects_gqa_misgrouping() {
        let mut cfg = base_cfg();
        cfg.num_key_value_heads = 5; // 32 % 5 != 0
        let out = resolve_intake("TotallyUnknownForCausalLM", None, &cfg, true);
        assert!(matches!(out, IntakeOutcome::Reject { .. }));
    }

    #[test]
    fn specialised_fields_force_rejection_even_when_allowlisted() {
        let mut cfg = base_cfg();
        cfg.final_logit_softcapping = Some(30.0); // Gemma-2 → needs adapter
        let out = resolve_intake("YiForCausalLM", Some("Yi"), &cfg, false);
        match out {
            IntakeOutcome::Reject { message } => assert!(message.contains("soft-cap")),
            IntakeOutcome::Accept { .. } => panic!("soft-cap config must be rejected"),
        }
    }

    #[test]
    fn sliding_window_is_a_warning_not_a_rejection() {
        let mut cfg = base_cfg();
        cfg.sliding_window = Some(4096);
        let warnings = check_llama_topology(&cfg).expect("sliding_window must not be fatal");
        assert!(warnings.iter().any(|w| w.contains("sliding_window")));
    }

    #[test]
    fn topology_accepts_plain_llama() {
        assert!(check_llama_topology(&base_cfg()).is_ok());
    }

    #[test]
    fn allowlist_bases_are_all_natively_registered() {
        for entry in LLAMA_COMPATIBLE_ALLOWLIST {
            assert!(
                base_adapter(entry.base_architecture).is_some(),
                "allowlist base {:?} must resolve natively",
                entry.base_architecture
            );
            // And the allowlisted arch itself must NOT be native
            // (else the compat layer would never see it).
            let meta = ModelMetadata {
                architecture: entry.architecture,
                model_type: None,
                format: ModelFormat::HfSafetensors,
            };
            assert!(
                resolve_adapter(&meta).is_none(),
                "allowlisted arch {:?} must be non-native",
                entry.architecture
            );
        }
    }
}
