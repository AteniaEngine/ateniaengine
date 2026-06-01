//! **MOE-FULL-14** — controlled production path: gating + dispatcher.
//!
//! Exercises `moe::controlled_moe_generate` and `moe::diagnose_moe` end to end
//! on real model directories built from the committed fixtures:
//!  - certified Mixtral → runs only with the opt-in;
//!  - dense → NotMoe;
//!  - unsupported variant (Qwen3 QK-norm) → UnsupportedVariant (regardless of
//!    the flag);
//!  - read-only diagnosis reports the right status.
//! The dense path is never touched.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use atenia_engine::moe::{
    controlled_moe_generate, diagnose_moe, ControlledMoeError, MoeCertScope, MoeFamily,
};
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn tmp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("atenia_moe14_{}_{}", std::process::id(), name));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Build a model dir from the committed Mixtral fixture (config.json + weights).
fn mixtral_model_dir(label: &str) -> PathBuf {
    let d = tmp_dir(label);
    std::fs::copy(fixture_dir().join("mixtral_tiny_config.json"), d.join("config.json")).unwrap();
    std::fs::copy(fixture_dir().join("full_mixtral.safetensors"), d.join("model.safetensors"))
        .unwrap();
    d
}

fn write_st(dir: &Path, tensors: &[(&str, Vec<usize>)]) {
    let buffers: Vec<Vec<u8>> =
        tensors.iter().map(|(_, s)| vec![0u8; s.iter().product::<usize>() * 4]).collect();
    let mut views: HashMap<String, TensorView> = HashMap::new();
    for (i, (name, shape)) in tensors.iter().enumerate() {
        views.insert((*name).to_string(), TensorView::new(StDtype::F32, shape.clone(), &buffers[i]).unwrap());
    }
    std::fs::write(dir.join("model.safetensors"), safetensors::serialize(&views, &None).unwrap())
        .unwrap();
    std::fs::write(dir.join("config.json"), r#"{"vocab_size":12,"hidden_size":8,"num_hidden_layers":1,
        "num_attention_heads":1,"num_key_value_heads":1,"intermediate_size":16,"num_local_experts":4,
        "num_experts_per_tok":2}"#).unwrap();
}

#[test]
fn controlled_path_gates_on_opt_in_and_runs_certified_mixtral() {
    let dir = mixtral_model_dir("gate");

    // (a) Without the opt-in → NotEnabled (clear, actionable).
    // SAFETY: this test owns the flag across both branches.
    unsafe {
        std::env::remove_var("ATENIA_ENABLE_MOE");
        std::env::remove_var("ATENIA_EXPERIMENTAL_MOE");
    }
    match controlled_moe_generate(&dir, &[22, 25, 29], 8) {
        Err(ControlledMoeError::NotEnabled { family, scope }) => {
            assert_eq!(family, MoeFamily::Mixtral);
            assert!(scope.is_runnable());
        }
        other => panic!("expected NotEnabled, got {other:?}"),
    }

    // (b) With the opt-in → runs and stops at EOS.
    unsafe {
        std::env::set_var("ATENIA_ENABLE_MOE", "1");
    }
    let out = controlled_moe_generate(&dir, &[22, 25, 29], 8).expect("certified Mixtral must run");
    eprintln!("CONTROLLED Mixtral generate = {out:?}");
    assert_eq!(out, vec![17, 20]);
}

#[test]
fn dense_dir_is_not_moe() {
    let d = tmp_dir("dense");
    write_st(&d, &[
        ("model.embed_tokens.weight", vec![12, 8]),
        ("model.layers.0.mlp.gate_proj.weight", vec![16, 8]),
        ("model.layers.0.mlp.up_proj.weight", vec![16, 8]),
        ("model.layers.0.mlp.down_proj.weight", vec![8, 16]),
        ("lm_head.weight", vec![12, 8]),
    ]);
    assert!(matches!(controlled_moe_generate(&d, &[1], 4), Err(ControlledMoeError::NotMoe)));
    let diag = diagnose_moe(&d);
    assert!(!diag.is_moe);
}

#[test]
fn qwen3_qk_norm_is_unsupported_variant() {
    // A Qwen3-MoE marker (QK-norm) must be refused as an unsupported variant,
    // regardless of the opt-in flag.
    let d = tmp_dir("qwen3");
    write_st(&d, &[
        ("model.embed_tokens.weight", vec![12, 8]),
        ("model.layers.0.self_attn.q_norm.weight", vec![8]),
        ("model.layers.0.self_attn.k_norm.weight", vec![8]),
        ("model.layers.0.mlp.gate.weight", vec![4, 8]),
        ("model.layers.0.mlp.experts.gate_up_proj", vec![4, 32, 8]),
        ("model.layers.0.mlp.experts.down_proj", vec![4, 8, 16]),
        ("lm_head.weight", vec![12, 8]),
    ]);
    // No opt-in needed: the unsupported-variant gate fires before the flag check.
    match controlled_moe_generate(&d, &[1], 4) {
        Err(ControlledMoeError::UnsupportedVariant(msg)) => {
            assert!(msg.contains("unsupported"), "msg: {msg}");
        }
        other => panic!("expected UnsupportedVariant, got {other:?}"),
    }
    let diag = diagnose_moe(&d);
    assert_eq!(diag.scope, Some(MoeCertScope::Unsupported));
    assert!(!diag.certified_runnable);
}

#[test]
fn diagnose_reports_certified_mixtral_status() {
    let dir = mixtral_model_dir("diag");
    let diag = diagnose_moe(&dir);
    assert!(diag.is_moe);
    assert_eq!(diag.family, Some(MoeFamily::Mixtral));
    assert!(diag.scope.map(|s| s.is_runnable()).unwrap_or(false));
    assert!(diag.message.contains("Mixtral"));
}
