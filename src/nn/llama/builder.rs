//! Llama-family graph builder (M4.5-b1, Paso 3).
//!
//! Constructs the full Llama-family inference graph from a
//! [`LlamaConfig`] (mirroring the HF `config.json`) and a
//! [`LlamaRuntime`] (concrete batch and seq for the run).
//!
//! ## Architecture
//! ```text
//! tokens [batch, seq]
//!   → IndexSelect(embed_tokens)               [batch, seq, hidden]
//!   → 22× transformer block:                                    │
//!       ┌─ RMSNorm (× input_layernorm γ)                        │
//!       │  Q/K/V proj  (Linear-over-flat)                       │
//!       │  reshape to [b, s, n_heads, head_dim]                 │
//!       │  RoPE (Q, K)                                          │
//!       │  permute → [b, n_heads, s, head_dim]                  │
//!       │  scores = Q @ K.T   (4D BatchMatMul, K is pre-scaled) │
//!       │  + causal_mask  (BroadcastAdd on [1,1,s,s])           │
//!       │  softmax (last dim)                                   │
//!       │  attn = softmax @ V                                   │
//!       │  permute back, reshape, o_proj                        │
//!       │  + residual                                           │
//!       ├─ RMSNorm (× post_attention_layernorm γ)               │
//!       │  SwiGLU: silu(gate(x)) ⊙ up(x) → down                 │
//!       │  + residual                                           │
//!   → final RMSNorm (× model.norm γ)
//!   → LM head Linear → logits [batch, seq, vocab]
//! ```
//!
//! ## Position handling
//! Positions are implicit `[0..seq)`. KV cache, paged attention,
//! and dynamic seq are M5+.
//!
//! ## Shape-specific graph
//! Reshape requires concrete dims; switching `(batch, seq)` means
//! rebuilding the graph. This is a deliberate M4.5 simplification.
//!
//! ## SiLU note
//! [`GraphBuilder::silu`] currently uses `NodeType::Activation(SiLU)`,
//! which only registers a forward path. Suitable for inference;
//! switch to `NodeType::SiLU` when training support lands.
//!
//! ## GQA via load-time tile (no graph cost)
//! K/V projections are tiled by `kv_groups` at load time
//! (see [`crate::nn::llama::weight_loading`]), so this graph
//! sees pure 32-head MHA. KV-shared attention is M5+.

use crate::amg::builder::GraphBuilder;
use crate::nn::llama::config::LlamaConfig;
use crate::tensor::Tensor;

/// Concrete runtime dimensions for graph construction. Different
/// `(batch, seq)` combinations require rebuilding the graph.
#[derive(Debug, Clone, Copy)]
pub struct LlamaRuntime {
    pub batch: usize,
    pub seq: usize,
}

/// Handles produced by [`build_llama`] for downstream
/// integration with [`crate::v17::loader::weight_mapper::WeightMapper`].
pub struct LlamaHandles {
    pub token_input_id: usize,
    pub logits_id: usize,
    /// Index-aligned with [`Self::param_names`].
    pub param_ids: Vec<usize>,
    /// Parameter names in HuggingFace convention
    /// (`"model.layers.{i}.self_attn.q_proj.weight"` etc.).
    pub param_names: Vec<String>,
}

fn register_param_named(
    gb: &mut GraphBuilder,
    full_name: &str,
    shape: Vec<usize>,
    param_ids: &mut Vec<usize>,
    param_names: &mut Vec<String>,
) -> usize {
    let numel: usize = shape.iter().product();
    let data = vec![0.0_f32; numel];
    let tensor = Tensor::new_cpu(shape, data);
    let node_id = gb.parameter(tensor);
    param_ids.push(node_id);
    param_names.push(full_name.to_string());
    node_id
}

/// Build a single Llama transformer block. Returns the output
/// node_id (input shape `[batch, seq, hidden]`, output same).
fn build_transformer_block_llama(
    gb: &mut GraphBuilder,
    layer_idx: usize,
    x: usize,
    causal_mask_id: usize,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    param_ids: &mut Vec<usize>,
    param_names: &mut Vec<String>,
) -> usize {
    let prefix = format!("model.layers.{}", layer_idx);
    let hidden = config.hidden_size;
    let n_heads = config.num_attention_heads;
    let head_dim = config.effective_head_dim();
    let intermediate = config.intermediate_size;
    let batch = runtime.batch;
    let seq = runtime.seq;
    let bs = (batch * seq) as isize;

    // ---- 1. Input layernorm (RMSNorm + γ) ----
    let input_ln_gamma = register_param_named(
        gb,
        &format!("{}.input_layernorm.weight", prefix),
        vec![1, 1, hidden],
        param_ids,
        param_names,
    );
    let h_normed = gb.rms_norm(x, config.rms_norm_eps);
    let h = gb.broadcast_mul(h_normed, input_ln_gamma);

    // ---- 2. Q/K/V projections ----
    let q_proj_w = register_param_named(
        gb,
        &format!("{}.self_attn.q_proj.weight", prefix),
        vec![hidden, hidden],
        param_ids,
        param_names,
    );
    let k_proj_w = register_param_named(
        gb,
        &format!("{}.self_attn.k_proj.weight", prefix),
        vec![hidden, hidden], // post-tile + transpose
        param_ids,
        param_names,
    );
    let v_proj_w = register_param_named(
        gb,
        &format!("{}.self_attn.v_proj.weight", prefix),
        vec![hidden, hidden],
        param_ids,
        param_names,
    );

    // matmul is 2D — flatten [b, s, h] → [b*s, h] before each Linear.
    let h_flat = gb.reshape(h, vec![bs, hidden as isize]);
    let q_flat_raw = gb.matmul(h_flat, q_proj_w);
    let k_flat_raw = gb.matmul(h_flat, k_proj_w);
    let v_flat_raw = gb.matmul(h_flat, v_proj_w);

    // ---- 2.b QKV biases (Qwen 2.5 family) ----
    //
    // Llama, TinyLlama, and SmolLM2 omit Q/K/V biases. Qwen 2.5
    // hard-codes them (`bias=True` in `Qwen2Attention`); the
    // resolved value lives in `LlamaConfig::effective_attention_bias`
    // so per-family quirks stay out of the builder.
    //
    // Bias parameter shapes are `[1, hidden]` (post-GQA-tile) so a
    // same-rank `BroadcastAdd` against the matmul output `[bs, hidden]`
    // satisfies the AMG broadcast rule (rank match required at
    // `graph.rs::broadcast_add`). The tile + reshape happens at load
    // time via the weight-mapper transform pipeline (B.3); the builder
    // sees the bias parameter at its final `[1, hidden]` shape.
    let (q_flat, k_flat, v_flat) = if config.effective_attention_bias() {
        let q_proj_b = register_param_named(
            gb,
            &format!("{}.self_attn.q_proj.bias", prefix),
            vec![1, hidden],
            param_ids,
            param_names,
        );
        let k_proj_b = register_param_named(
            gb,
            &format!("{}.self_attn.k_proj.bias", prefix),
            vec![1, hidden],
            param_ids,
            param_names,
        );
        let v_proj_b = register_param_named(
            gb,
            &format!("{}.self_attn.v_proj.bias", prefix),
            vec![1, hidden],
            param_ids,
            param_names,
        );
        (
            gb.broadcast_add(q_flat_raw, q_proj_b),
            gb.broadcast_add(k_flat_raw, k_proj_b),
            gb.broadcast_add(v_flat_raw, v_proj_b),
        )
    } else {
        (q_flat_raw, k_flat_raw, v_flat_raw)
    };

    // ---- 3. Multi-head reshape ----
    let split_shape = vec![
        batch as isize,
        seq as isize,
        n_heads as isize,
        head_dim as isize,
    ];
    let q = gb.reshape(q_flat, split_shape.clone());
    let k = gb.reshape(k_flat, split_shape.clone());
    let v = gb.reshape(v_flat, split_shape);

    // ---- 4. RoPE on Q and K (V unchanged) ----
    //
    // Llama 3.x checkpoints carry a `rope_scaling` block in the
    // config that reshapes the inverse-frequency schedule (Llama 3
    // piecewise scaling). Plain RoPE (TinyLlama, SmolLM2, Qwen)
    // takes the `None` branch and remains bit-identical.
    let (q_rope, k_rope) = match config.effective_rope_scaling() {
        None => (
            gb.rope(q, head_dim, config.rope_theta),
            gb.rope(k, head_dim, config.rope_theta),
        ),
        Some(crate::nn::llama::RopeScaling::Llama3 {
            factor,
            low_freq_factor,
            high_freq_factor,
            original_max_position_embeddings,
        }) => {
            let scaling = crate::amg::nodes::RopeScalingLlama3::new(
                *factor,
                *low_freq_factor,
                *high_freq_factor,
                *original_max_position_embeddings,
            );
            (
                gb.rope_scaled(q, head_dim, config.rope_theta, scaling.clone()),
                gb.rope_scaled(k, head_dim, config.rope_theta, scaling),
            )
        }
        // **M11.B** — Phi-3 / Phi-3.5 LongRope is parsed and the
        // `compute_inv_freqs_longrope` primitive lives in
        // `nn/rope.rs`, but the executor / AMG-node wire-up + the
        // Phi3 builder land in follow-up commits. The Llama
        // builder only emits Llama-family checkpoints, so a
        // LongRope-bearing config reaching this match is a caller
        // bug — Phi-3 checkpoints must route through the Phi3
        // builder once it lands.
        Some(crate::nn::llama::RopeScaling::LongRope { .. }) => {
            panic!(
                "LongRope rope_scaling reached the Llama builder. \
                 Phi-3 / Phi-3.5 checkpoints must use the Phi3 builder \
                 (M11.B in progress; LongRope executor wire-up still pending)."
            );
        }
    };

    // ---- 5. [b, s, h, d] → [b, h, s, d] ----
    let q_perm = gb.permute(q_rope, vec![0, 2, 1, 3]);
    let k_perm = gb.permute(k_rope, vec![0, 2, 1, 3]);
    let v_perm = gb.permute(v, vec![0, 2, 1, 3]);

    // ---- 6. Attention scores: Q @ K^T  (4D BMM) ----
    // K_proj weights already absorbed the 1/sqrt(head_dim) scale at
    // load-time, so no explicit scale node is needed here.
    let k_perm_t = gb.transpose_last_two(k_perm);
    let scores = gb.batch_matmul(q_perm, k_perm_t);

    // ---- 7. Causal mask (broadcast over batch and heads) ----
    let scores_masked = gb.broadcast_add(scores, causal_mask_id);

    // ---- 8. Softmax over last dim ----
    let attn_weights = gb.softmax(scores_masked);

    // ---- 9. weights @ V ----
    let attn_out = gb.batch_matmul(attn_weights, v_perm);
    // [b, n_heads, s, head_dim] → [b, s, n_heads, head_dim]
    let attn_out_back = gb.permute(attn_out, vec![0, 2, 1, 3]);

    // ---- 10. Output projection ----
    let attn_out_flat = gb.reshape(attn_out_back, vec![bs, hidden as isize]);
    let o_proj_w = register_param_named(
        gb,
        &format!("{}.self_attn.o_proj.weight", prefix),
        vec![hidden, hidden],
        param_ids,
        param_names,
    );
    let attn_proj_flat = gb.matmul(attn_out_flat, o_proj_w);
    let attn_proj = gb.reshape(
        attn_proj_flat,
        vec![batch as isize, seq as isize, hidden as isize],
    );

    // ---- 11. Attention residual ----
    let x_residual_1 = gb.add(x, attn_proj);

    // ---- 12. Post-attention layernorm ----
    let post_ln_gamma = register_param_named(
        gb,
        &format!("{}.post_attention_layernorm.weight", prefix),
        vec![1, 1, hidden],
        param_ids,
        param_names,
    );
    let h2_normed = gb.rms_norm(x_residual_1, config.rms_norm_eps);
    let h2 = gb.broadcast_mul(h2_normed, post_ln_gamma);

    // ---- 13. SwiGLU FFN ----
    let gate_proj_w = register_param_named(
        gb,
        &format!("{}.mlp.gate_proj.weight", prefix),
        vec![hidden, intermediate],
        param_ids,
        param_names,
    );
    let up_proj_w = register_param_named(
        gb,
        &format!("{}.mlp.up_proj.weight", prefix),
        vec![hidden, intermediate],
        param_ids,
        param_names,
    );
    let down_proj_w = register_param_named(
        gb,
        &format!("{}.mlp.down_proj.weight", prefix),
        vec![intermediate, hidden],
        param_ids,
        param_names,
    );

    let h2_flat = gb.reshape(h2, vec![bs, hidden as isize]);
    let gate_flat = gb.matmul(h2_flat, gate_proj_w);
    let up_flat = gb.matmul(h2_flat, up_proj_w);
    let silu_gate_flat = gb.silu(gate_flat);
    let ffn_pre_down_flat = gb.mul(silu_gate_flat, up_flat);
    let ffn_out_flat = gb.matmul(ffn_pre_down_flat, down_proj_w);
    let ffn_out = gb.reshape(
        ffn_out_flat,
        vec![batch as isize, seq as isize, hidden as isize],
    );

    // ---- 14. FFN residual ----
    gb.add(x_residual_1, ffn_out)
}

/// Build the complete Llama-family graph.
///
/// `token_input_id` must be a previously-registered `Input` of
/// shape `[batch, seq]` containing token IDs as f32.
pub fn build_llama(
    gb: &mut GraphBuilder,
    config: &LlamaConfig,
    runtime: &LlamaRuntime,
    token_input_id: usize,
) -> LlamaHandles {
    config.validate().expect("invalid LlamaConfig");


    let mut param_ids: Vec<usize> = Vec::new();
    let mut param_names: Vec<String> = Vec::new();

    // ---- Causal mask Parameter ----
    // Shape [1, 1, seq, seq] so it broadcasts against attention
    // scores [batch, n_heads, seq, seq] under BroadcastAdd's
    // same-rank rule.
    let seq = runtime.seq;
    let mut mask_data = vec![0.0_f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            mask_data[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    let mask_tensor = Tensor::new_cpu(vec![1, 1, seq, seq], mask_data);
    let causal_mask_id = gb.parameter(mask_tensor);

    // ---- Embedding lookup ----
    let embed_w = register_param_named(
        gb,
        "model.embed_tokens.weight",
        vec![config.vocab_size, config.hidden_size],
        &mut param_ids,
        &mut param_names,
    );
    let mut x = gb.index_select(embed_w, token_input_id);

    // ---- Transformer blocks ----
    for layer_idx in 0..config.num_hidden_layers {
        x = build_transformer_block_llama(
            gb,
            layer_idx,
            x,
            causal_mask_id,
            config,
            runtime,
            &mut param_ids,
            &mut param_names,
        );
    }

    // ---- Final RMSNorm ----
    let final_ln_gamma = register_param_named(
        gb,
        "model.norm.weight",
        vec![1, 1, config.hidden_size],
        &mut param_ids,
        &mut param_names,
    );
    let x_normed = gb.rms_norm(x, config.rms_norm_eps);
    let x_final = gb.broadcast_mul(x_normed, final_ln_gamma);

    // ---- LM head ----
    //
    // When `config.tie_word_embeddings == true`, the LM head reuses the
    // `embed_tokens` weight matrix transposed, instead of having its own
    // Parameter. This is standard practice in modern Llama-family models
    // (Llama 3.2, Qwen 2.5, SmolLM2, ...) and matches HuggingFace's
    // behavior of NOT storing a separate `lm_head.weight` in the
    // safetensors checkpoint when tied.
    //
    // Approach: insert a `Transpose2D` node in the graph that converts
    // `embed_w` from `[vocab, hidden]` to `[hidden, vocab]` at execution
    // time, then matmul against the normalized hidden state. This avoids
    // duplicating the weight in memory.
    //
    // Performance note: `Transpose2D` performs a data copy of size
    // `vocab × hidden` floats per forward pass. For TinyLlama
    // (`vocab=32000, hidden=2048`) this is ~256 MB. For models with
    // larger vocabularies (Llama 3.2: 128256, Qwen 2.5: 151936) the cost
    // grows accordingly. Acceptable for M4.6 forward-only validation;
    // future optimization (e.g., transpose-free matmul via operand
    // swapping) is a known follow-up.
    let bs = (runtime.batch * runtime.seq) as isize;
    let x_flat = gb.reshape(x_final, vec![bs, config.hidden_size as isize]);

    let lm_head_input = if config.tie_word_embeddings {
        // Reuse `embed_w` as the transposed lm_head weight. After
        // Transpose2D the shape is `[hidden, vocab]` — exactly what a
        // separate `lm_head.weight` parameter would have after the
        // `LoadTransform::Transpose2D` applied at load time.
        gb.transpose_2d(embed_w)
    } else {
        // Register a dedicated `lm_head.weight` Parameter.
        register_param_named(
            gb,
            "lm_head.weight",
            vec![config.hidden_size, config.vocab_size],
            &mut param_ids,
            &mut param_names,
        )
    };

    let logits_flat = gb.matmul(x_flat, lm_head_input);
    let logits = gb.reshape(
        logits_flat,
        vec![
            runtime.batch as isize,
            runtime.seq as isize,
            config.vocab_size as isize,
        ],
    );

    LlamaHandles {
        token_input_id,
        logits_id: logits,
        param_ids,
        param_names,
    }
}
