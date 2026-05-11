//! M4.7.6.a — Llama 2 13B Chat config + builder smoke validation.
//!
//! Locks the architectural compatibility surface between Atenia's
//! `LlamaConfig` parser / `build_llama` graph builder and the
//! freshly-downloaded `meta-llama/Llama-2-13b-chat-hf` checkpoint.
//!
//! Coverage:
//!
//!   1. `llama2_13b_config_parses_with_expected_values` —
//!      parses `config.json` from disk, asserts every field
//!      against the public HF model card.
//!   2. `llama2_13b_safetensors_index_is_consistent` —
//!      reads `model.safetensors.index.json`, asserts 3 shards
//!      and 403 named tensors (40 layers × 10 + 3 = 403, where
//!      the 10th is `rotary_emb.inv_freq` per layer that Atenia
//!      computes at runtime and skips on load).
//!   3. `build_full_llama2_13b_yields_363_parameters` — builds
//!      the graph at `LlamaRuntime { batch: 1, seq: 4 }`, asserts
//!      `param_names.len() == 363` and the embedding / LM head /
//!      final norm anchors are present. **No forward**, no
//!      weight loading.
//!   4. `total_params_estimate_is_within_one_percent_of_thirteen_billion` —
//!      arithmetic check that the M4.6.1 `total_params_estimate`
//!      heuristic places the model within ±1 % of 13.0 B params.
//!
//! `#[ignore]`-gated because it requires the safetensors files
//! on disk. Run with:
//!
//! ```powershell
//! cargo test --test m4_7_6_a_llama2_13b_config_test --release \
//!     -- --ignored --nocapture
//! ```

use std::path::Path;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::llama::{LlamaConfig, LlamaRuntime, build_llama};

/// Llama 2 13B Chat lives on the **internal NVMe (D:)** for the
/// demo per the M4.7.4 / M4.7.5 drive policy — F: USB HDD is
/// too slow. Override via `ATENIA_LLAMA2_13B_DIR` env var.
const MODEL_DIR: &str = "D:/Atenia/models/llama-2-13b-chat";

#[test]
#[ignore = "requires Llama 2 13B Chat checkpoint at MODEL_DIR"]
fn llama2_13b_config_parses_with_expected_values() {
    let cfg_path = Path::new(MODEL_DIR).join("config.json");
    assert!(cfg_path.exists(), "config.json not found at {:?}", cfg_path);

    let cfg = LlamaConfig::from_json_file(&cfg_path).expect("config must parse");

    // Reference values from the public HuggingFace model card.
    assert_eq!(cfg.vocab_size, 32_000);
    assert_eq!(cfg.hidden_size, 5120);
    assert_eq!(cfg.intermediate_size, 13_824);
    assert_eq!(cfg.num_hidden_layers, 40);
    assert_eq!(cfg.num_attention_heads, 40);
    assert_eq!(
        cfg.num_key_value_heads, 40,
        "Llama 2 13B Chat is full MHA (no GQA) — kv_heads should equal q_heads"
    );
    assert_eq!(cfg.max_position_embeddings, 4096);
    assert_eq!(cfg.rope_theta, 10_000);
    assert_eq!(cfg.rms_norm_eps, 1e-5);
    assert!(!cfg.tie_word_embeddings);
    // attention_bias is absent from Llama 2's config.json (Llama
    // 2 hard-codes QKV biases off, like all non-Qwen Llamas).
    // Atenia's parser leaves it as None and resolves at runtime
    // via `effective_attention_bias()`.
    assert_eq!(cfg.attention_bias, None);
    assert!(
        !cfg.effective_attention_bias(),
        "Llama 2 must have no QKV bias"
    );
    assert_eq!(cfg.model_type.as_deref(), Some("llama"));
    assert!(cfg.rope_scaling.is_none(), "Llama 2 uses plain RoPE");
    assert_eq!(cfg.head_dim, None);
    // hidden_size / num_attention_heads = 5120 / 40 = 128
    assert_eq!(cfg.effective_head_dim(), 128);
    // No GQA → kv_groups = 1
    assert_eq!(cfg.kv_groups(), 1);
    assert_eq!(cfg.bos_token_id, 1);
    assert_eq!(cfg.eos_token_id, 2);
}

#[test]
#[ignore = "requires Llama 2 13B Chat checkpoint at MODEL_DIR"]
fn llama2_13b_safetensors_index_is_consistent() {
    use std::collections::HashSet;

    let idx_path = Path::new(MODEL_DIR).join("model.safetensors.index.json");
    assert!(idx_path.exists(), "index.json missing at {:?}", idx_path);

    let raw = std::fs::read_to_string(&idx_path).expect("read index");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("parse index");

    let weight_map = v["weight_map"]
        .as_object()
        .expect("weight_map must be an object");
    assert_eq!(
        weight_map.len(),
        403,
        "Llama 2 13B Chat ships 403 named tensors (40 layers × 10 + \
         3 globals; the 10th per-layer is `rotary_emb.inv_freq` which \
         Atenia computes at runtime)"
    );

    let shards: HashSet<String> = weight_map
        .values()
        .filter_map(|s| s.as_str().map(|x| x.to_string()))
        .collect();
    assert_eq!(shards.len(), 3, "expected 3 shards, got {:?}", shards);
    for n in 1..=3 {
        let fname = format!("model-0000{n}-of-00003.safetensors");
        assert!(shards.contains(&fname), "missing shard {}", fname);
        // And the shard file itself exists on disk.
        let shard_path = Path::new(MODEL_DIR).join(&fname);
        assert!(shard_path.exists(), "shard {} not on disk", fname);
    }

    // Total size from metadata = 26.03 GB, also recorded.
    let total_size = v["metadata"]["total_size"].as_u64().expect("total_size");
    assert!(
        (total_size as i64 - 26_031_738_880_i64).abs() < 1_000_000,
        "total_size {} is far from the expected 26.03 GB ({} bytes)",
        total_size,
        26_031_738_880_u64
    );
}

#[test]
#[ignore = "requires Llama 2 13B Chat checkpoint at MODEL_DIR"]
fn build_full_llama2_13b_yields_363_parameters() {
    let cfg_path = Path::new(MODEL_DIR).join("config.json");
    let cfg = LlamaConfig::from_json_file(&cfg_path).expect("config");

    let runtime = LlamaRuntime { batch: 1, seq: 4 };

    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama(&mut gb, &cfg, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    let _graph = gb.build();

    // 1 (embed) + 40 layers × 9 (per-layer Atenia parameters,
    // excluding the runtime-computed `rotary_emb.inv_freq`)
    // + 1 (final RMSNorm) + 1 (lm_head) = 363.
    assert_eq!(handles.param_names.len(), 363);
    assert_eq!(handles.param_ids.len(), 363);

    // Anchors at boundary positions.
    let names: Vec<&str> = handles.param_names.iter().map(|s| s.as_str()).collect();
    for required in &[
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.39.mlp.down_proj.weight", // last layer
        "model.norm.weight",
        "lm_head.weight",
    ] {
        assert!(
            names.contains(required),
            "missing required param name: {}",
            required
        );
    }

    // No `rotary_emb.inv_freq` in Atenia's param list — Atenia
    // computes it at runtime in the RoPE arm.
    for name in &names {
        assert!(
            !name.contains("rotary_emb"),
            "Atenia must not load rotary_emb buffers: found {}",
            name
        );
    }
}

#[test]
#[ignore = "requires Llama 2 13B Chat checkpoint at MODEL_DIR"]
fn total_params_estimate_is_within_one_percent_of_thirteen_billion() {
    let cfg_path = Path::new(MODEL_DIR).join("config.json");
    let cfg = LlamaConfig::from_json_file(&cfg_path).expect("config");

    // Manual computation (matches the M4.6.1 heuristic):
    //   embed     = vocab × hidden                                     = 32000 × 5120
    //   per layer = 4 × hidden² (QKVO) + 3 × hidden × intermediate (FFN) + 2 × hidden (norms)
    //   final     = hidden + vocab × hidden                            (final norm + lm_head)
    let embed = cfg.vocab_size * cfg.hidden_size;
    let per_layer = 4 * cfg.hidden_size * cfg.hidden_size
        + 3 * cfg.hidden_size * cfg.intermediate_size
        + 2 * cfg.hidden_size;
    let lm_head = cfg.vocab_size * cfg.hidden_size;
    let final_norm = cfg.hidden_size;

    let total = (embed + cfg.num_hidden_layers * per_layer + final_norm + lm_head) as f64;
    println!("total_params_estimate = {:.4} B", total / 1e9);

    // Reference: Llama 2 13B has 13.0 B parameters per the
    // public model card. Allow ±1 % to accommodate small
    // tokeniser-specific overhead and our heuristic's blind
    // spots (no positional / dropout / etc. on Llama 2).
    let target = 13.0e9;
    let rel_err = ((total - target) / target).abs();
    assert!(
        rel_err < 0.01,
        "total params {:.4} B differs from 13.0 B by {:.2} % (>1 % budget)",
        total / 1e9,
        rel_err * 100.0
    );
}
