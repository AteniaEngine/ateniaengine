//! **MOE-FULL-11** — robustness: the controlled MoE runtime returns clear,
//! specific errors for wrong families, invalid config, missing tensors, and
//! inconsistent expert counts. All behind the opt-in (so we exercise the real
//! error paths, not the gate).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use atenia_engine::moe::{MoeRuntime, MoeRuntimeError};
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn opt_in() {
    unsafe {
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
    }
}

/// Write a safetensors file with the given `(name, shape)` tensors, zero-filled.
fn write_st(path: &Path, tensors: &[(&str, Vec<usize>)]) {
    let buffers: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, shape)| vec![0u8; shape.iter().product::<usize>() * 4])
        .collect();
    let mut views: HashMap<String, TensorView> = HashMap::new();
    for (i, (name, shape)) in tensors.iter().enumerate() {
        views.insert(
            (*name).to_string(),
            TensorView::new(StDtype::F32, shape.clone(), &buffers[i]).unwrap(),
        );
    }
    let bytes = safetensors::serialize(&views, &None).unwrap();
    std::fs::write(path, bytes).unwrap();
}

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("atenia_moe11_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

/// A structurally valid tiny classic-Mixtral tensor listing (zeros).
fn mixtral_tensors() -> Vec<(&'static str, Vec<usize>)> {
    let dm = 8usize;
    let dff = 16usize;
    let mut v: Vec<(&'static str, Vec<usize>)> = vec![
        ("model.embed_tokens.weight", vec![12, dm]),
        ("lm_head.weight", vec![12, dm]),
        ("model.norm.weight", vec![dm]),
        ("model.layers.0.input_layernorm.weight", vec![dm]),
        ("model.layers.0.post_attention_layernorm.weight", vec![dm]),
        ("model.layers.0.self_attn.q_proj.weight", vec![dm, dm]),
        ("model.layers.0.self_attn.k_proj.weight", vec![dm, dm]),
        ("model.layers.0.self_attn.v_proj.weight", vec![dm, dm]),
        ("model.layers.0.self_attn.o_proj.weight", vec![dm, dm]),
        ("model.layers.0.block_sparse_moe.gate.weight", vec![4, dm]),
    ];
    for e in 0..4 {
        let g: &'static str = Box::leak(format!("model.layers.0.block_sparse_moe.experts.{e}.w1.weight").into_boxed_str());
        let u: &'static str = Box::leak(format!("model.layers.0.block_sparse_moe.experts.{e}.w3.weight").into_boxed_str());
        let d: &'static str = Box::leak(format!("model.layers.0.block_sparse_moe.experts.{e}.w2.weight").into_boxed_str());
        v.push((g, vec![dff, dm]));
        v.push((u, vec![dff, dm]));
        v.push((d, vec![dm, dff]));
    }
    v
}

fn mixtral_config_json(num_local_experts: usize) -> String {
    format!(
        r#"{{"vocab_size":12,"hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,
        "num_attention_heads":1,"num_key_value_heads":1,"head_dim":8,"num_local_experts":{num_local_experts},
        "num_experts_per_tok":2,"rope_theta":10000.0,"rms_norm_eps":1e-5,"eos_token_id":0}}"#
    )
}

#[test]
fn dense_checkpoint_refused_as_not_moe() {
    opt_in();
    let w = tmp("dense.safetensors");
    write_st(&w, &[
        ("model.embed_tokens.weight", vec![12, 8]),
        ("model.layers.0.self_attn.q_proj.weight", vec![8, 8]),
        ("model.layers.0.mlp.gate_proj.weight", vec![16, 8]),
        ("model.layers.0.mlp.up_proj.weight", vec![16, 8]),
        ("model.layers.0.mlp.down_proj.weight", vec![8, 16]),
        ("lm_head.weight", vec![12, 8]),
    ]);
    let c = tmp("dense_config.json");
    std::fs::write(&c, mixtral_config_json(4)).unwrap();
    let err = MoeRuntime::load_from_files(&c, &w).unwrap_err();
    assert!(matches!(err, MoeRuntimeError::NotMoe), "got {err:?}");
}

#[test]
fn malformed_deepseek_mla_reports_clear_error() {
    // A DeepSeek (MLA) checkpoint is now recognised (MOE-FULL-12), but this one
    // is incomplete (shared expert missing up/down; config lacks MLA fields) →
    // a clear structural error (not OptIn/NotMoe), not a silent success.
    opt_in();
    let w = tmp("mla.safetensors");
    write_st(&w, &[
        ("model.embed_tokens.weight", vec![12, 8]),
        ("model.layers.0.self_attn.kv_a_proj_with_mqa.weight", vec![12, 8]),
        ("model.layers.0.self_attn.kv_b_proj.weight", vec![16, 8]),
        ("model.layers.0.mlp.gate.weight", vec![4, 8]),
        ("model.layers.0.mlp.experts.gate_up_proj", vec![4, 32, 8]),
        ("model.layers.0.mlp.experts.down_proj", vec![4, 8, 16]),
        ("model.layers.0.mlp.shared_experts.gate_proj.weight", vec![16, 8]),
        ("lm_head.weight", vec![12, 8]),
    ]);
    let c = tmp("mla_config.json");
    std::fs::write(&c, mixtral_config_json(4)).unwrap();
    let err = MoeRuntime::load_from_files(&c, &w).unwrap_err();
    assert!(
        matches!(
            err,
            MoeRuntimeError::Load(_)
                | MoeRuntimeError::Config(_)
                | MoeRuntimeError::ConfigInconsistent(_)
        ),
        "expected a clear structural error, got {err:?}"
    );
}

#[test]
fn missing_tensor_reports_load_error() {
    opt_in();
    // Valid Mixtral minus `model.norm.weight`.
    let mut tensors = mixtral_tensors();
    tensors.retain(|(n, _)| *n != "model.norm.weight");
    let w = tmp("missing.safetensors");
    write_st(&w, &tensors);
    let c = tmp("missing_config.json");
    std::fs::write(&c, mixtral_config_json(4)).unwrap();
    let err = MoeRuntime::load_from_files(&c, &w).unwrap_err();
    assert!(matches!(err, MoeRuntimeError::Load(_)), "got {err:?}");
}

#[test]
fn invalid_config_reports_config_error() {
    opt_in();
    let w = tmp("validcfg.safetensors");
    write_st(&w, &mixtral_tensors());
    // config.json missing num_attention_heads.
    let c = tmp("bad_config.json");
    std::fs::write(
        &c,
        r#"{"vocab_size":12,"hidden_size":8,"num_hidden_layers":1,"num_key_value_heads":1,
        "intermediate_size":16,"num_local_experts":4,"num_experts_per_tok":2}"#,
    )
    .unwrap();
    let err = MoeRuntime::load_from_files(&c, &w).unwrap_err();
    assert!(matches!(err, MoeRuntimeError::Config(_)), "got {err:?}");
}

#[test]
fn config_expert_count_mismatch_reports_inconsistent() {
    opt_in();
    let w = tmp("mismatch.safetensors");
    write_st(&w, &mixtral_tensors()); // 4 experts in tensors
    let c = tmp("mismatch_config.json");
    std::fs::write(&c, mixtral_config_json(8)).unwrap(); // config claims 8
    let err = MoeRuntime::load_from_files(&c, &w).unwrap_err();
    assert!(matches!(err, MoeRuntimeError::ConfigInconsistent(_)), "got {err:?}");
}

#[test]
fn real_mixtral_still_loads_with_bad_then_good_config() {
    // Sanity: the committed real Mixtral still loads (regression guard that the
    // robustness branches did not break the happy path).
    opt_in();
    let rt = MoeRuntime::load_from_files(
        &fixture_dir().join("mixtral_tiny_config.json"),
        &fixture_dir().join("full_mixtral.safetensors"),
    )
    .expect("real Mixtral must still load");
    assert_eq!(rt.num_layers(), 2);
}
