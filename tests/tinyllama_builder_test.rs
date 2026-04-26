//! Tests for `build_tinyllama` (M4.5-b1 Paso 3.1).
//!
//! Coverage:
//!   1. Tiny config builds without panic.
//!   2. Full TinyLlama config produces 201 parameters.
//!   3. HF parameter naming convention.
//!   4. Per-parameter shape sanity.
//!   5. Tiny-config forward smoke test (no numerical validation).
//!   6. Full TinyLlama config builds and validates without execution.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::nn::tinyllama::{
    build_tinyllama, TinyLlamaConfig, TinyLlamaHandles, TinyLlamaRuntime,
};
use atenia_engine::tensor::Tensor;

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

/// A small but architecturally faithful config for fast tests.
fn tiny_config() -> TinyLlamaConfig {
    TinyLlamaConfig::from_json_str(
        r#"{
          "vocab_size": 100,
          "hidden_size": 64,
          "num_hidden_layers": 1,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "intermediate_size": 128,
          "max_position_embeddings": 32,
          "rope_theta": 10000.0,
          "rms_norm_eps": 1e-5,
          "tie_word_embeddings": false,
          "attention_bias": false,
          "bos_token_id": 0,
          "eos_token_id": 1
        }"#,
    )
    .expect("tiny config must parse")
}

fn build_with(
    cfg: &TinyLlamaConfig,
    runtime: TinyLlamaRuntime,
) -> (atenia_engine::amg::graph::Graph, TinyLlamaHandles) {
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_tinyllama(&mut gb, cfg, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    (gb.build(), handles)
}

#[test]
fn build_tiny_config_does_not_panic_and_aligns_param_lists() {
    let cfg = tiny_config();
    let runtime = TinyLlamaRuntime { batch: 1, seq: 4 };
    let (_graph, handles) = build_with(&cfg, runtime);

    assert_eq!(
        handles.param_names.len(),
        handles.param_ids.len(),
        "param_names and param_ids must be index-aligned"
    );
    // 1 (embed) + 1 layer × 9 (input_ln + q + k + v + o + post_ln + gate + up + down)
    // + 1 (final norm) + 1 (lm_head) = 12.
    assert_eq!(handles.param_names.len(), 12);
}

#[test]
fn build_full_tinyllama_yields_201_parameters() {
    let cfg = TinyLlamaConfig::from_json_str(TINYLLAMA_CONFIG_JSON).unwrap();
    let runtime = TinyLlamaRuntime { batch: 1, seq: 4 };
    let (_graph, handles) = build_with(&cfg, runtime);
    // 1 + 22 × 9 + 1 + 1 = 201
    assert_eq!(handles.param_names.len(), 201);
    assert_eq!(handles.param_ids.len(), 201);
}

#[test]
fn param_names_match_huggingface_convention() {
    let cfg = TinyLlamaConfig::from_json_str(TINYLLAMA_CONFIG_JSON).unwrap();
    let runtime = TinyLlamaRuntime { batch: 1, seq: 4 };
    let (_graph, handles) = build_with(&cfg, runtime);

    let names: Vec<&str> = handles.param_names.iter().map(|s| s.as_str()).collect();

    // Canonical anchors must all be present.
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
        "model.layers.21.mlp.down_proj.weight", // last layer
        "model.norm.weight",
        "lm_head.weight",
    ] {
        assert!(
            names.contains(required),
            "missing required param name: {}",
            required
        );
    }

    // Negative check: HF Llama uses `model.layers`, not `transformer.`.
    for name in &names {
        assert!(
            !name.starts_with("transformer."),
            "unexpected GPT-style prefix on `{}`",
            name
        );
    }
}

#[test]
fn param_shapes_match_expected_post_transform_layouts() {
    let cfg = TinyLlamaConfig::from_json_str(TINYLLAMA_CONFIG_JSON).unwrap();
    let runtime = TinyLlamaRuntime { batch: 1, seq: 4 };
    let (graph, handles) = build_with(&cfg, runtime);

    let lookup = |name: &str| -> &Tensor {
        let idx = handles
            .param_names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("missing param `{}`", name));
        let id = handles.param_ids[idx];
        graph.nodes[id]
            .output
            .as_ref()
            .unwrap_or_else(|| panic!("node {} has no tensor", id))
    };

    assert_eq!(lookup("model.embed_tokens.weight").shape, vec![32_000, 2048]);
    assert_eq!(
        lookup("model.layers.0.self_attn.q_proj.weight").shape,
        vec![2048, 2048]
    );
    assert_eq!(
        lookup("model.layers.0.self_attn.k_proj.weight").shape,
        vec![2048, 2048],
        "k_proj graph shape is post-tile post-transpose"
    );
    assert_eq!(
        lookup("model.layers.0.input_layernorm.weight").shape,
        vec![1, 1, 2048],
        "RMSNorm gamma is reshaped to [1, 1, hidden] for BroadcastMul"
    );
    assert_eq!(
        lookup("model.layers.0.mlp.gate_proj.weight").shape,
        vec![2048, 5632]
    );
    assert_eq!(
        lookup("model.layers.0.mlp.down_proj.weight").shape,
        vec![5632, 2048]
    );
    assert_eq!(lookup("model.norm.weight").shape, vec![1, 1, 2048]);
    assert_eq!(lookup("lm_head.weight").shape, vec![2048, 32_000]);
}

#[test]
fn build_tiny_executes_forward_smoke() {
    let cfg = tiny_config();
    let runtime = TinyLlamaRuntime { batch: 1, seq: 4 };
    let (mut graph, _handles) = build_with(&cfg, runtime);

    // Token IDs as f32 (Atenia IndexSelect cast convention).
    let tokens = Tensor::new_cpu(vec![1, 4], vec![1.0_f32, 2.0, 3.0, 4.0]);
    let outputs = graph.execute(vec![tokens]);
    assert_eq!(outputs.len(), 1, "expected one Output tensor (logits)");

    let logits = &outputs[0];
    assert_eq!(
        logits.shape,
        vec![1, 4, cfg.vocab_size],
        "logits shape mismatch"
    );

    // Sanity: every logit must be finite. Weights are zero-initialized
    // here, so the masked-softmax row may yield a uniform distribution
    // and the overall logits be all zeros — both fine, neither
    // is non-finite.
    for &v in logits.as_cpu_slice() {
        assert!(v.is_finite(), "non-finite logit produced: {}", v);
    }
}

#[test]
fn build_full_tinyllama_no_execute_validates_and_exposes_handles() {
    let cfg = TinyLlamaConfig::from_json_str(TINYLLAMA_CONFIG_JSON).unwrap();
    let runtime = TinyLlamaRuntime { batch: 1, seq: 4 };
    let (_graph, handles) = build_with(&cfg, runtime);

    // Building the full 22-layer graph with zero-initialized weights
    // for inference is heavy but tractable; we deliberately skip
    // `graph.execute` here because it would allocate intermediate
    // activations on the order of ~1.1 GB and dominate test runtime.
    assert_eq!(handles.param_names.len(), 201);
    assert!(handles.param_ids.iter().all(|&id| id > 0));
}
