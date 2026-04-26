//! Tests for `TinyLlamaConfig` parsing, validation, and derived helpers
//! (M4.5-b1, Paso 1).

use atenia_engine::nn::tinyllama::TinyLlamaConfig;
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
    let cfg = TinyLlamaConfig::from_json_str(TINYLLAMA_CONFIG_JSON)
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
    assert_eq!(cfg.attention_bias, false);
    assert_eq!(cfg.bos_token_id, 1);
    assert_eq!(cfg.eos_token_id, 2);
    assert_eq!(cfg.pad_token_id, None);
}

#[test]
fn derived_helpers_match_tinyllama_arithmetic() {
    let cfg = TinyLlamaConfig::from_json_str(TINYLLAMA_CONFIG_JSON).unwrap();
    assert_eq!(cfg.head_dim(), 64); // 2048 / 32
    assert_eq!(cfg.kv_head_dim(), 64);
    assert_eq!(cfg.kv_groups(), 8); // 32 / 4
}

#[test]
fn total_params_estimate_is_within_one_percent_of_one_point_one_billion() {
    let cfg = TinyLlamaConfig::from_json_str(TINYLLAMA_CONFIG_JSON).unwrap();
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
    let err = TinyLlamaConfig::from_json_str(&s).expect_err("should reject");
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
    let err = TinyLlamaConfig::from_json_str(&s).expect_err("should reject");
    assert!(format!("{}", err).contains("multiple of num_key_value_heads"));
}

#[test]
fn validate_rejects_odd_head_dim() {
    let mut v: serde_json::Value = serde_json::from_str(TINYLLAMA_CONFIG_JSON).unwrap();
    // Pick hidden_size / num_attention_heads → odd: 32 heads, hidden 32*63 = 2016.
    v["hidden_size"] = serde_json::json!(2016);
    v["num_attention_heads"] = serde_json::json!(32);
    let s = serde_json::to_string(&v).unwrap();
    let err = TinyLlamaConfig::from_json_str(&s).expect_err("should reject");
    assert!(format!("{}", err).contains("head_dim"));
}

#[test]
fn validate_accepts_rope_theta_as_integer_or_float() {
    // Float form (canonical).
    assert!(TinyLlamaConfig::from_json_str(TINYLLAMA_CONFIG_JSON).is_ok());

    // Integer form (some configs in the wild).
    let mut v: serde_json::Value = serde_json::from_str(TINYLLAMA_CONFIG_JSON).unwrap();
    v["rope_theta"] = serde_json::json!(10000);
    let s = serde_json::to_string(&v).unwrap();
    let cfg = TinyLlamaConfig::from_json_str(&s).expect("integer rope_theta accepted");
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
    let cfg = TinyLlamaConfig::from_json_str(json).expect("synthetic config should parse");
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
    let cfg = TinyLlamaConfig::from_json_file(&path)
        .expect("real on-disk config should parse and validate");
    assert_eq!(cfg.vocab_size, 32000);
    assert_eq!(cfg.num_hidden_layers, 22);
}
