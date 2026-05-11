//! Tests for `LlamaConfig` parsing, validation, and derived helpers
//! (M4.5-b1, Paso 1).

use atenia_engine::nn::llama::{LlamaConfig, RopeScaling};
use std::path::PathBuf;

/// Embedded copy of `models/tinyllama-1.1b/config.json` so the test
/// runs without depending on the model directory being checked out.
const TINYLLAMA_CONFIG_JSON: &str = r#"{
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 5632,
  "max_position_embeddings": 2048,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 22,
  "num_key_value_heads": 4,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.35.0",
  "use_cache": true,
  "vocab_size": 32000
}"#;

#[test]
fn parse_embedded_tinyllama_config_matches_known_values() {
    let cfg = LlamaConfig::from_json_str(TINYLLAMA_CONFIG_JSON)
        .expect("embedded TinyLlama config must parse cleanly");

    assert_eq!(cfg.vocab_size, 32000);
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_hidden_layers, 22);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 4);
    assert_eq!(cfg.intermediate_size, 5632);
    assert_eq!(cfg.max_position_embeddings, 2048);
    assert_eq!(cfg.rope_theta, 10_000);
    assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-9);
    assert_eq!(cfg.tie_word_embeddings, false);
    assert_eq!(cfg.attention_bias, Some(false));
    assert_eq!(cfg.effective_attention_bias(), false);
    assert_eq!(cfg.model_type.as_deref(), Some("llama"));
    assert_eq!(cfg.bos_token_id, 1);
    assert_eq!(cfg.eos_token_id, 2);
    assert_eq!(cfg.pad_token_id, None);
}

#[test]
fn derived_helpers_match_tinyllama_arithmetic() {
    let cfg = LlamaConfig::from_json_str(TINYLLAMA_CONFIG_JSON).unwrap();
    assert_eq!(cfg.head_dim(), 64); // 2048 / 32
    assert_eq!(cfg.kv_head_dim(), 64);
    assert_eq!(cfg.kv_groups(), 8); // 32 / 4
}

#[test]
fn total_params_estimate_is_within_one_percent_of_one_point_one_billion() {
    let cfg = LlamaConfig::from_json_str(TINYLLAMA_CONFIG_JSON).unwrap();
    let est = cfg.total_params_estimate();
    // TinyLlama is advertised as 1.1B; allow a 1% band around it.
    let lower = 1_080_000_000_usize;
    let upper = 1_120_000_000_usize;
    assert!(
        (lower..=upper).contains(&est),
        "param estimate {} not in [{}, {}]",
        est,
        lower,
        upper
    );
}

#[test]
fn validate_rejects_hidden_size_not_divisible_by_heads() {
    // Take the canonical config, surgically corrupt one field.
    let mut v: serde_json::Value = serde_json::from_str(TINYLLAMA_CONFIG_JSON).unwrap();
    v["hidden_size"] = serde_json::json!(2050); // not divisible by 32
    let s = serde_json::to_string(&v).unwrap();
    let err = LlamaConfig::from_json_str(&s).expect_err("should reject");
    let msg = format!("{}", err);
    assert!(
        msg.contains("divisible by num_attention_heads"),
        "got error: {}",
        msg
    );
}

#[test]
fn validate_rejects_q_heads_not_multiple_of_kv_heads() {
    let mut v: serde_json::Value = serde_json::from_str(TINYLLAMA_CONFIG_JSON).unwrap();
    v["num_attention_heads"] = serde_json::json!(33); // 33 % 4 != 0
    v["hidden_size"] = serde_json::json!(33 * 64); // keep head_dim valid
    let s = serde_json::to_string(&v).unwrap();
    let err = LlamaConfig::from_json_str(&s).expect_err("should reject");
    assert!(format!("{}", err).contains("multiple of num_key_value_heads"));
}

#[test]
fn validate_rejects_odd_head_dim() {
    let mut v: serde_json::Value = serde_json::from_str(TINYLLAMA_CONFIG_JSON).unwrap();
    // Pick hidden_size / num_attention_heads → odd: 32 heads, hidden 32*63 = 2016.
    v["hidden_size"] = serde_json::json!(2016);
    v["num_attention_heads"] = serde_json::json!(32);
    let s = serde_json::to_string(&v).unwrap();
    let err = LlamaConfig::from_json_str(&s).expect_err("should reject");
    assert!(format!("{}", err).contains("head_dim"));
}

#[test]
fn validate_accepts_rope_theta_as_integer_or_float() {
    // Float form (canonical).
    assert!(LlamaConfig::from_json_str(TINYLLAMA_CONFIG_JSON).is_ok());

    // Integer form (some configs in the wild).
    let mut v: serde_json::Value = serde_json::from_str(TINYLLAMA_CONFIG_JSON).unwrap();
    v["rope_theta"] = serde_json::json!(10000);
    let s = serde_json::to_string(&v).unwrap();
    let cfg = LlamaConfig::from_json_str(&s).expect("integer rope_theta accepted");
    assert_eq!(cfg.rope_theta, 10_000);
}

#[test]
fn parse_synthetic_minimal_config() {
    // A miniature config to make sure the parser does not require any
    // of the metadata fields TinyLlama happens to set.
    let json = r#"{
        "vocab_size": 100,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": true,
        "attention_bias": false,
        "bos_token_id": 0,
        "eos_token_id": 1
    }"#;
    let cfg = LlamaConfig::from_json_str(json).expect("synthetic config should parse");
    assert_eq!(cfg.vocab_size, 100);
    assert_eq!(cfg.head_dim(), 4);
    assert_eq!(cfg.kv_groups(), 2);
    assert_eq!(cfg.tie_word_embeddings, true);
    assert_eq!(cfg.pad_token_id, None);
}

/// File-system check against the real config in models/tinyllama-1.1b/.
/// `#[ignore]` so CI / sandbox runs don't require the model directory;
/// run locally with `cargo test --test tinyllama_config_test -- --ignored`.
#[test]
#[ignore]
fn parse_real_config_file_on_disk() {
    let path = PathBuf::from("models/tinyllama-1.1b/config.json");
    let cfg =
        LlamaConfig::from_json_file(&path).expect("real on-disk config should parse and validate");
    assert_eq!(cfg.vocab_size, 32000);
    assert_eq!(cfg.num_hidden_layers, 22);
}

/// M4.6 Phase B.1 — Qwen2's official `config.json` omits the
/// `attention_bias` field entirely. The parser must accept that
/// absence and resolve the effective value via `model_type`.
#[test]
fn parse_qwen2_config_without_attention_bias_field() {
    // Verbatim snapshot of the Qwen 2.5 1.5B Instruct config.json
    // shipped by Qwen — the `attention_bias` key is intentionally
    // missing.
    let json = r#"{
      "architectures": ["Qwen2ForCausalLM"],
      "attention_dropout": 0.0,
      "bos_token_id": 151643,
      "eos_token_id": 151645,
      "hidden_act": "silu",
      "hidden_size": 1536,
      "initializer_range": 0.02,
      "intermediate_size": 8960,
      "max_position_embeddings": 32768,
      "max_window_layers": 21,
      "model_type": "qwen2",
      "num_attention_heads": 12,
      "num_hidden_layers": 28,
      "num_key_value_heads": 2,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000.0,
      "sliding_window": 32768,
      "tie_word_embeddings": true,
      "torch_dtype": "bfloat16",
      "transformers_version": "4.43.1",
      "use_cache": true,
      "use_sliding_window": false,
      "vocab_size": 151936
    }"#;
    let cfg =
        LlamaConfig::from_json_str(json).expect("Qwen2 config without attention_bias must parse");
    assert_eq!(cfg.model_type.as_deref(), Some("qwen2"));
    assert_eq!(cfg.attention_bias, None);
    // qwen2-aware default: hard-coded QKV biases on.
    assert!(cfg.effective_attention_bias());
    assert_eq!(cfg.vocab_size, 151_936);
    assert_eq!(cfg.num_hidden_layers, 28);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert_eq!(cfg.kv_groups(), 6);
    assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);
    assert_eq!(cfg.rope_theta, 1_000_000);
    assert_eq!(cfg.tie_word_embeddings, true);
}

/// Sanity check on the helper alone: an explicit `false` overrides
/// the qwen2 default.
#[test]
fn effective_attention_bias_explicit_overrides_qwen2_default() {
    let json = r#"{
        "vocab_size": 100,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": true,
        "model_type": "qwen2",
        "attention_bias": false,
        "bos_token_id": 0,
        "eos_token_id": 1
    }"#;
    let cfg = LlamaConfig::from_json_str(json).expect("must parse");
    assert_eq!(cfg.attention_bias, Some(false));
    assert_eq!(cfg.effective_attention_bias(), false);
}

/// Sanity check: model_type "llama" with no attention_bias defaults
/// to false (covers the legacy / loose-config case).
#[test]
fn effective_attention_bias_defaults_false_for_llama() {
    let json = r#"{
        "vocab_size": 100,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": true,
        "model_type": "llama",
        "bos_token_id": 0,
        "eos_token_id": 1
    }"#;
    let cfg = LlamaConfig::from_json_str(json).expect("must parse");
    assert_eq!(cfg.attention_bias, None);
    assert_eq!(cfg.effective_attention_bias(), false);
}

/// Real on-disk Qwen2 config — same `#[ignore]` convention as the
/// TinyLlama variant.
#[test]
#[ignore]
fn parse_qwen2_real_config_file_on_disk() {
    let path = PathBuf::from("models/qwen2.5-1.5b-instruct/config.json");
    let cfg = LlamaConfig::from_json_file(&path)
        .expect("real on-disk Qwen2 config should parse and validate");
    assert_eq!(cfg.model_type.as_deref(), Some("qwen2"));
    assert!(cfg.effective_attention_bias());
    assert_eq!(cfg.vocab_size, 151_936);
    assert_eq!(cfg.num_hidden_layers, 28);
}

// ---------------------------------------------------------------------------
// M4.6 Phase C.1 — Llama 3.2 config additions: rope_scaling + head_dim
// ---------------------------------------------------------------------------

/// Llama 3.2 1B's `config.json` carries an explicit `head_dim` field
/// AND a populated `rope_scaling` block with `rope_type: "llama3"`.
/// The parser must ingest both without rejecting unsupported fields.
#[test]
fn parse_llama_3_2_config_with_rope_scaling_and_head_dim() {
    let json = r#"{
      "architectures": ["LlamaForCausalLM"],
      "attention_bias": false,
      "attention_dropout": 0.0,
      "bos_token_id": 128000,
      "eos_token_id": 128009,
      "head_dim": 64,
      "hidden_act": "silu",
      "hidden_size": 2048,
      "initializer_range": 0.02,
      "intermediate_size": 8192,
      "max_position_embeddings": 131072,
      "mlp_bias": false,
      "model_type": "llama",
      "num_attention_heads": 32,
      "num_hidden_layers": 16,
      "num_key_value_heads": 8,
      "pad_token_id": 128004,
      "rms_norm_eps": 1e-05,
      "rope_scaling": {
        "factor": 32.0,
        "high_freq_factor": 4.0,
        "low_freq_factor": 1.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
      },
      "rope_theta": 500000.0,
      "tie_word_embeddings": true,
      "torch_dtype": "bfloat16",
      "use_cache": true,
      "vocab_size": 128256
    }"#;
    let cfg = LlamaConfig::from_json_str(json)
        .expect("Llama 3.2 config must parse with rope_scaling and head_dim");

    assert_eq!(cfg.vocab_size, 128_256);
    assert_eq!(cfg.num_hidden_layers, 16);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.kv_groups(), 4);
    assert_eq!(cfg.rope_theta, 500_000);
    assert_eq!(cfg.tie_word_embeddings, true);
    assert_eq!(cfg.model_type.as_deref(), Some("llama"));

    // Phase C.1 specifics:
    assert_eq!(cfg.head_dim, Some(64));
    assert_eq!(cfg.effective_head_dim(), 64);
    let scaling = cfg
        .effective_rope_scaling()
        .expect("rope_scaling should be present");
    match scaling {
        RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
        } => {
            assert!((*factor - 32.0).abs() < 1e-6);
            assert!((*low_freq_factor - 1.0).abs() < 1e-6);
            assert!((*high_freq_factor - 4.0).abs() < 1e-6);
            assert_eq!(*original_max_position_embeddings, 8192);
        }
    }
}

/// Legacy `type` alias (transformers <4.43) must be honoured the
/// same way as the modern `rope_type` field.
#[test]
fn parse_rope_scaling_accepts_legacy_type_alias() {
    let json = r#"{
        "vocab_size": 100,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 64,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": true,
        "attention_bias": false,
        "rope_scaling": {
            "type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192
        },
        "bos_token_id": 0,
        "eos_token_id": 1
    }"#;
    let cfg = LlamaConfig::from_json_str(json).expect("legacy `type` alias must work");
    let scaling = cfg
        .effective_rope_scaling()
        .expect("rope_scaling should be parsed via legacy `type` alias");
    assert!(matches!(scaling, RopeScaling::Llama3 { factor, .. } if (*factor - 8.0).abs() < 1e-6));
}

/// Unknown `rope_type` (e.g. yarn, longrope) is silently downgraded
/// to no-scaling — the parser is permissive so future scaling
/// variants land as opt-in extensions, not breaking changes.
#[test]
fn parse_unknown_rope_type_silently_yields_no_scaling() {
    let json = r#"{
        "vocab_size": 100,
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 32,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": true,
        "attention_bias": false,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4.0
        },
        "bos_token_id": 0,
        "eos_token_id": 1
    }"#;
    let cfg = LlamaConfig::from_json_str(json).expect("unknown rope_type must not fail parse");
    assert!(cfg.effective_rope_scaling().is_none());
}

/// Models without `rope_scaling` (TinyLlama / SmolLM2 / Qwen) must
/// continue to report `None`. Same for the implicit `head_dim`.
#[test]
fn legacy_models_have_no_rope_scaling_and_implicit_head_dim() {
    let cfg = LlamaConfig::from_json_str(TINYLLAMA_CONFIG_JSON).unwrap();
    assert!(cfg.effective_rope_scaling().is_none());
    assert_eq!(cfg.head_dim, None);
    assert_eq!(cfg.effective_head_dim(), 64); // 2048 / 32
}

/// Real on-disk Llama 3.2 1B config — same `#[ignore]` convention.
#[test]
#[ignore]
fn parse_llama_3_2_real_config_file_on_disk() {
    let path = PathBuf::from("models/llama-3.2-1b-instruct/config.json");
    let cfg = LlamaConfig::from_json_file(&path)
        .expect("real on-disk Llama 3.2 config should parse and validate");
    assert_eq!(cfg.model_type.as_deref(), Some("llama"));
    assert_eq!(cfg.vocab_size, 128_256);
    assert_eq!(cfg.num_hidden_layers, 16);
    assert_eq!(cfg.effective_head_dim(), 64);
    let scaling = cfg
        .effective_rope_scaling()
        .expect("must have llama3 scaling");
    match scaling {
        RopeScaling::Llama3 { factor, .. } => assert!((*factor - 32.0).abs() < 1e-6),
    }
}
