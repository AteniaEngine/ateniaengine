//! **MODEL-INTAKE-1** — architecture compatibility layer, exercised
//! through the crate's *public* surface with a real `config.json`
//! parse (`LlamaConfig::from_json_str`). Complements the in-lib unit
//! tests by proving the public API is reachable and behaves on
//! genuinely-parsed configs.
//!
//! Decision matrix covered:
//!   - allowlisted distinct arch  → Accept (no opt-in needed)
//!   - unknown arch, no opt-in    → Reject (with the env hint)
//!   - unknown arch, opt-in       → Accept (generic, uncertified)
//!   - native arch                → never in the allowlist (compat
//!     layer is bypassed for it)

use atenia_engine::model_adapters::compat::{
    allowlist_lookup, check_llama_topology, resolve_intake, IntakeOutcome, IntakeStatus,
    GENERIC_INTAKE_ENV, LLAMA_COMPATIBLE_ALLOWLIST,
};
use atenia_engine::nn::llama::config::LlamaConfig;

/// A real, parser-validated plain-Llama config (TinyLlama-shaped).
fn llama_config() -> LlamaConfig {
    let json = r#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 32000,
        "hidden_size": 2048,
        "num_hidden_layers": 22,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "intermediate_size": 5632,
        "max_position_embeddings": 2048,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": false,
        "bos_token_id": 1,
        "eos_token_id": 2
    }"#;
    LlamaConfig::from_json_str(json).expect("valid llama config must parse")
}

#[test]
fn allowlisted_architecture_is_accepted() {
    let cfg = llama_config();
    // Every allowlist entry must accept the plain-Llama config without
    // requiring the generic opt-in.
    for entry in LLAMA_COMPATIBLE_ALLOWLIST {
        let out = resolve_intake(entry.architecture, None, &cfg, false);
        match out {
            IntakeOutcome::Accept { status, .. } => {
                assert_eq!(status, IntakeStatus::Allowlisted, "{}", entry.architecture);
            }
            IntakeOutcome::Reject { message } => {
                panic!("allowlisted {} rejected: {message}", entry.architecture)
            }
        }
    }
}

#[test]
fn unknown_architecture_without_opt_in_is_rejected_with_hint() {
    let cfg = llama_config();
    let out = resolve_intake("SomeBrandNewForCausalLM", Some("brandnew"), &cfg, false);
    match out {
        IntakeOutcome::Reject { message } => {
            assert!(message.contains("unsupported architecture"));
            assert!(message.contains(GENERIC_INTAKE_ENV));
        }
        IntakeOutcome::Accept { .. } => panic!("unknown arch must be rejected without opt-in"),
    }
}

#[test]
fn unknown_architecture_with_opt_in_and_clean_config_is_accepted() {
    let cfg = llama_config();
    let out = resolve_intake("SomeBrandNewForCausalLM", None, &cfg, true);
    match out {
        IntakeOutcome::Accept { status, warnings, .. } => {
            assert_eq!(status, IntakeStatus::Generic);
            assert!(warnings.iter().any(|w| w.contains("UNCERTIFIED")));
        }
        IntakeOutcome::Reject { message } => panic!("expected accept, got: {message}"),
    }
}

#[test]
fn native_architectures_are_not_in_the_allowlist() {
    // The compat layer must never shadow a natively-supported family.
    for native in [
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "MistralForCausalLM",
        "Phi3ForCausalLM",
        "Gemma2ForCausalLM",
        "Gemma3ForCausalLM",
    ] {
        assert!(
            allowlist_lookup(native).is_none(),
            "{native} must not be in the compat allowlist (it is native)"
        );
    }
}

#[test]
fn plain_llama_topology_passes() {
    assert!(check_llama_topology(&llama_config()).is_ok());
}
