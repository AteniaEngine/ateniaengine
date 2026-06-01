//! **MOE-FULL-6** — experimental tiny full MoE transformer forward (graph).
//!
//! Composes a *whole* tiny MoE transformer in the AMG graph and runs a
//! multi-token prefill forward to logits:
//!
//! ```text
//!   token ids [1, seq]
//!    → IndexSelect(embed_tokens)                 [1, seq, hidden]
//!    → for each decoder layer:
//!        RMSNorm·γ → self-attention (Q/K/V/O, RoPE, causal mask, softmax)
//!                  → + residual
//!        RMSNorm·γ → MoeRealLayerReference (position-wise, MOE-FULL-4/6)
//!                  → + residual
//!    → final RMSNorm·γ
//!    → lm_head matmul                            [1, seq, vocab]
//! ```
//!
//! ## Scope (Ruta A — honest, bounded)
//!
//! * **Multi-token prefill, real causal mask, real RoPE.** No decode loop, no
//!   generation, no KV cache, no batching.
//! * **MHA only (`num_key_value_heads == num_attention_heads`).** GQA is out
//!   of scope here (in the productive dense path GQA is resolved by a load-
//!   time K/V tile; reusing that is a larger refactor). The fixture is
//!   generated with `num_key_value_heads == num_attention_heads`, so this is a
//!   real Mixtral, just configured without GQA — documented, not faked.
//! * Built **only from existing AMG primitives** + the MOE-FULL-4/6 MoE node.
//!   No new graph op. The attention `1/sqrt(head_dim)` score scale is absorbed
//!   by pre-scaling `w_q` in Rust (GraphBuilder has no `scale` op), which is
//!   numerically exact.
//! * CPU-only, test/opt-in only. No productive loader/runtime/Adapter-Toolkit/
//!   CUDA/CLI. The MOE-2 fail-loud guard is unchanged.
//!
//! Validated against an offline HuggingFace `MixtralForCausalLM` f64 reference
//! (see `fixtures/moe/generate_full_forward_reference.py`).

use crate::amg::builder::GraphBuilder;
use crate::tensor::Tensor;

use super::graph_op::register_real_moe_layer;
use super::layer::RealMoeLayer;
use super::residency::ResidentExpertLayer;
use std::sync::Arc;

/// **MOE-PROD-2** — how a layer's MoE block is held for the graph backend:
///
/// * `Owned` — a RAM-f32 [`RealMoeLayer`], registered per forward. The default
///   (every existing fixture / test): byte-identical to MOE-FULL-6.
/// * `Registered` — a pre-registered, tier-able [`ResidentExpertLayer`]
///   (experts in bf16-RAM or on NVMe) referenced by its registry `layer_id`.
///   Registered **once** at load so cloning `TinyMixtralWeights` per forward
///   costs a `u32`, not the expert weights, and the experts never have to be
///   f32-resident in RAM. `ResidentExpertLayer::forward` is certified
///   bit-identical to `RealMoeLayer::forward_auto` (MOE-FULL-8).
#[derive(Debug, Clone)]
pub enum MoeBlock {
    Owned(RealMoeLayer),
    Registered(u32),
}

impl MoeBlock {
    /// Pre-register a resident layer once and wrap its id.
    pub fn registered(layer: Arc<ResidentExpertLayer>, cache_capacity: usize) -> Self {
        MoeBlock::Registered(super::graph_op::register_resident_moe_layer(layer, cache_capacity))
    }

    /// Resolve to a graph `layer_id`: `Owned` registers (consuming) on demand;
    /// `Registered` returns the existing id.
    pub(crate) fn into_layer_id(self) -> u32 {
        match self {
            MoeBlock::Owned(m) => register_real_moe_layer(m),
            MoeBlock::Registered(id) => id,
        }
    }
}

/// Tiny Mixtral hyperparameters (MHA, no GQA).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TinyMixtralConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub rope_theta: u32,
}

/// **MOE-FULL-11** — optional Q/K/V attention biases (Qwen-MoE has them;
/// Mixtral does not). Each vector is length `hidden` (`n_heads * head_dim`);
/// the K/V biases must already be tiled to MHA shape (via `gqa::to_mha_kv`)
/// when the model uses GQA, exactly like the K/V weights.
#[derive(Debug, Clone)]
pub struct QkvBias {
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

/// Per-layer dense weights (attention + norms). Row-major HF layout
/// `[out, in]` for the projections; γ vectors length `hidden`.
#[derive(Debug, Clone)]
pub struct TinyDecoderWeights {
    pub input_ln: Vec<f32>,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
    pub post_ln: Vec<f32>,
    /// **MOE-FULL-11** — optional Q/K/V biases (`None` = Mixtral / MHA-no-bias,
    /// byte-identical to MOE-FULL-6). `Some` adds them post-projection.
    pub attn_bias: Option<QkvBias>,
    /// **MOE-PROD-2** — the MoE block, either RAM-f32 `Owned` (default) or a
    /// pre-registered tier-able `Registered` resident layer.
    pub moe: MoeBlock,
}

/// Add a `[1, hidden]` bias to a `[seq, hidden]` (or `[1, hidden]`) projection,
/// optionally pre-scaled (the Q bias absorbs the `1/sqrt(head_dim)` score scale
/// the same way `w_q` does). No-op-free: only called when a bias is present.
pub(crate) fn add_proj_bias(
    gb: &mut GraphBuilder,
    flat: usize,
    bias: &[f32],
    hidden: usize,
    scale: f32,
) -> usize {
    let data: Vec<f32> =
        if scale == 1.0 { bias.to_vec() } else { bias.iter().map(|v| v * scale).collect() };
    let bp = gb.parameter(Tensor::new_cpu(vec![1, hidden], data));
    gb.broadcast_add(flat, bp)
}

/// All weights of the tiny MoE transformer.
#[derive(Debug, Clone)]
pub struct TinyMixtralWeights {
    pub config: TinyMixtralConfig,
    pub embed_tokens: Vec<f32>, // [vocab, hidden]
    pub layers: Vec<TinyDecoderWeights>,
    pub final_norm: Vec<f32>, // [hidden]
    pub lm_head: Vec<f32>,    // [vocab, hidden]
    pub rms_eps: f32,
}

/// Build the tiny full MoE transformer graph. `token_input_id` is an `Input`
/// of shape `[1, seq]` holding token ids as f32. Returns the logits node id
/// (shape `[1, seq, vocab]`). Consumes `w` (registers each layer's MoE).
pub fn build_tiny_mixtral_graph(
    gb: &mut GraphBuilder,
    token_input_id: usize,
    seq: usize,
    w: TinyMixtralWeights,
) -> usize {
    let c = w.config;
    let hidden = c.hidden_size;
    let n_heads = c.num_attention_heads;
    let head_dim = c.head_dim;
    let vocab = c.vocab_size;
    let hi = hidden as isize;
    let si = seq as isize;
    let inv_sqrt = 1.0_f32 / (head_dim as f32).sqrt();

    // ---- Causal mask [1, 1, seq, seq] ----
    let mut mask = vec![0.0_f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            mask[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    let causal_mask = gb.parameter(Tensor::new_cpu(vec![1, 1, seq, seq], mask));

    // ---- Embeddings ----
    let embed_w = gb.parameter(Tensor::new_cpu(vec![vocab, hidden], w.embed_tokens));
    let mut x = gb.index_select(embed_w, token_input_id); // [1, seq, hidden]

    // ---- Decoder layers ----
    for lw in w.layers {
        // 1. input RMSNorm·γ
        let g1 = gb.parameter(Tensor::new_cpu(vec![1, 1, hidden], lw.input_ln));
        let h_n = gb.rms_norm(x, w.rms_eps);
        let h = gb.broadcast_mul(h_n, g1);

        // 2. Q/K/V projections. HF weights are [out, in]; use
        //    matmul_rhs_transposed so y = h @ Wᵀ. Pre-scale w_q by
        //    1/sqrt(head_dim) to absorb the attention score scale.
        let wq_scaled: Vec<f32> = lw.w_q.iter().map(|v| v * inv_sqrt).collect();
        let wq = gb.parameter(Tensor::new_cpu(vec![hidden, hidden], wq_scaled));
        let wk = gb.parameter(Tensor::new_cpu(vec![hidden, hidden], lw.w_k));
        let wv = gb.parameter(Tensor::new_cpu(vec![hidden, hidden], lw.w_v));
        let wo = gb.parameter(Tensor::new_cpu(vec![hidden, hidden], lw.w_o));

        let h_flat = gb.reshape(h, vec![si, hi]);
        let q_flat0 = gb.matmul_rhs_transposed(h_flat, wq); // [seq, hidden]
        let k_flat0 = gb.matmul_rhs_transposed(h_flat, wk);
        let v_flat0 = gb.matmul_rhs_transposed(h_flat, wv);
        // MOE-FULL-11: optional Q/K/V biases (Qwen-MoE). Q bias absorbs the
        // 1/sqrt(head_dim) score scale (mirrors the pre-scaled w_q).
        let (q_flat, k_flat, v_flat) = match &lw.attn_bias {
            Some(b) => (
                add_proj_bias(gb, q_flat0, &b.q, hidden, inv_sqrt),
                add_proj_bias(gb, k_flat0, &b.k, hidden, 1.0),
                add_proj_bias(gb, v_flat0, &b.v, hidden, 1.0),
            ),
            None => (q_flat0, k_flat0, v_flat0),
        };

        // 3. multi-head reshape [1, seq, n_heads, head_dim]
        let mh = vec![1, si, n_heads as isize, head_dim as isize];
        let q = gb.reshape(q_flat, mh.clone());
        let k = gb.reshape(k_flat, mh.clone());
        let v = gb.reshape(v_flat, mh);

        // 4. RoPE on Q and K (V unchanged)
        let q_r = gb.rope(q, head_dim, c.rope_theta);
        let k_r = gb.rope(k, head_dim, c.rope_theta);

        // 5. [1, n_heads, seq, head_dim]
        let q_p = gb.permute(q_r, vec![0, 2, 1, 3]);
        let k_p = gb.permute(k_r, vec![0, 2, 1, 3]);
        let v_p = gb.permute(v, vec![0, 2, 1, 3]);

        // 6. scores = Q @ Kᵀ  (scale already in q) → [1, n_heads, seq, seq]
        let k_pt = gb.transpose_last_two(k_p);
        let scores = gb.batch_matmul(q_p, k_pt);
        // 7. causal mask
        let scores_m = gb.broadcast_add(scores, causal_mask);
        // 8. softmax
        let attn = gb.softmax(scores_m);
        // 9. context = attn @ V → [1, n_heads, seq, head_dim]
        let ctx = gb.batch_matmul(attn, v_p);
        // 10. back to [1, seq, n_heads, head_dim] → [seq, hidden]
        let ctx_b = gb.permute(ctx, vec![0, 2, 1, 3]);
        let ctx_flat = gb.reshape(ctx_b, vec![si, hi]);
        // 11. output projection
        let o_flat = gb.matmul_rhs_transposed(ctx_flat, wo); // [seq, hidden]
        let o = gb.reshape(o_flat, vec![1, si, hi]);
        // 12. attention residual
        let x1 = gb.add(x, o);

        // 13. post-attention RMSNorm·γ
        let g2 = gb.parameter(Tensor::new_cpu(vec![1, 1, hidden], lw.post_ln));
        let h2_n = gb.rms_norm(x1, w.rms_eps);
        let h2 = gb.broadcast_mul(h2_n, g2);

        // 14. MoE block, position-wise: flatten [1,seq,hidden] → [seq*hidden],
        //     one MoE node applies the layer per row (MOE-FULL-6), reshape back.
        let moe_id = lw.moe.into_layer_id();
        let h2_flat = gb.reshape(h2, vec![(seq * hidden) as isize]);
        let moe_out = gb.moe_real_layer_reference(h2_flat, moe_id);
        let moe_3d = gb.reshape(moe_out, vec![1, si, hi]);

        // 15. MoE residual
        x = gb.add(x1, moe_3d);
    }

    // ---- Final RMSNorm·γ ----
    let gf = gb.parameter(Tensor::new_cpu(vec![1, 1, hidden], w.final_norm));
    let x_n = gb.rms_norm(x, w.rms_eps);
    let x_f = gb.broadcast_mul(x_n, gf);

    // ---- lm_head: [seq, hidden] @ lm_headᵀ([vocab, hidden]) → [seq, vocab] ----
    let lm_w = gb.parameter(Tensor::new_cpu(vec![vocab, hidden], w.lm_head));
    let x_flat = gb.reshape(x_f, vec![si, hi]);
    let logits_flat = gb.matmul_rhs_transposed(x_flat, lm_w); // [seq, vocab]
    gb.reshape(logits_flat, vec![1, si, vocab as isize])
}

// ============================================================================
// Tests (synthetic weights — the real-fixture + HF comparison is the
// integration test in tests/moe_full_forward_test.rs).
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::dense::build_fixture_layer;
    use crate::moe::MoeLayerConfig;

    fn seeded(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as u32;
            out.push((u as f32 / u32::MAX as f32) * 2.0 - 1.0);
        }
        out
    }

    /// A synthetic tiny model whose MoE block is the d_model=8 fixture layer.
    /// Weights are independent of `seq`.
    fn synthetic() -> (TinyMixtralWeights, usize) {
        let hidden = 8;
        let n_heads = 2;
        let head_dim = 4;
        let vocab = 10;
        let n_layers = 2;
        let cfg = TinyMixtralConfig {
            vocab_size: vocab,
            hidden_size: hidden,
            num_hidden_layers: n_layers,
            num_attention_heads: n_heads,
            head_dim,
            rope_theta: 10000,
        };
        let layers: Vec<TinyDecoderWeights> = (0..n_layers)
            .map(|l| {
                let routed = build_fixture_layer(); // d_model 8, 4 experts
                let moe_cfg =
                    MoeLayerConfig::new(routed.num_experts(), 2, false, routed.d_model, routed.d_ff)
                        .unwrap();
                let moe = RealMoeLayer {
                    config: moe_cfg,
                    routed,
                    shared: None,
                    shared_gate: None,
                };
                TinyDecoderWeights {
                    input_ln: seeded(l as u64 * 10 + 1, hidden),
                    w_q: seeded(l as u64 * 10 + 2, hidden * hidden),
                    w_k: seeded(l as u64 * 10 + 3, hidden * hidden),
                    w_v: seeded(l as u64 * 10 + 4, hidden * hidden),
                    w_o: seeded(l as u64 * 10 + 5, hidden * hidden),
                    post_ln: seeded(l as u64 * 10 + 6, hidden),
                    attn_bias: None,
                    moe: MoeBlock::Owned(moe),
                }
            })
            .collect();
        let w = TinyMixtralWeights {
            config: cfg,
            embed_tokens: seeded(900, vocab * hidden),
            layers,
            final_norm: seeded(901, hidden),
            lm_head: seeded(902, vocab * hidden),
            rms_eps: 1e-5,
        };
        (w, vocab)
    }

    fn run(seq: usize, tokens: &[f32], w: TinyMixtralWeights) -> Vec<f32> {
        let mut gb = GraphBuilder::new();
        let tok = gb.input();
        let logits = build_tiny_mixtral_graph(&mut gb, tok, seq, w);
        gb.output(logits);
        let mut g = gb.build();
        let t = Tensor::new_cpu(vec![1, seq], tokens.to_vec());
        g.execute(vec![t])[0].as_cpu_slice().to_vec()
    }

    #[test]
    fn builds_and_logits_shape() {
        let seq = 4;
        let (w, vocab) = synthetic();
        let tokens = vec![1.0, 3.0, 0.0, 2.0];
        let logits = run(seq, &tokens, w);
        assert_eq!(logits.len(), seq * vocab);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn is_deterministic() {
        let seq = 4;
        let tokens = vec![1.0, 3.0, 0.0, 2.0];
        let a = run(seq, &tokens, synthetic().0);
        let b = run(seq, &tokens, synthetic().0);
        assert_eq!(a, b);
    }

    #[test]
    fn causal_mask_hides_future() {
        // Changing the LAST token must not change earlier positions' logits
        // (causality). Compare position 0..seq-1 logits for two inputs that
        // differ only in the final token.
        let seq = 4;
        let vocab = synthetic().1;
        let t1 = vec![1.0, 3.0, 0.0, 2.0];
        let t2 = vec![1.0, 3.0, 0.0, 5.0]; // only last token differs
        let l1 = run(seq, &t1, synthetic().0);
        let l2 = run(seq, &t2, synthetic().0);
        // positions 0..seq-1 (rows 0,1,2) must be identical.
        for pos in 0..(seq - 1) {
            for v in 0..vocab {
                let i = pos * vocab + v;
                assert!(
                    (l1[i] - l2[i]).abs() < 1e-6,
                    "causal: pos {pos} changed when only the last token changed"
                );
            }
        }
        // the last position SHOULD differ (it sees the changed token).
        let last = (seq - 1) * vocab;
        assert!(
            (0..vocab).any(|v| (l1[last + v] - l2[last + v]).abs() > 1e-6),
            "last position should react to the changed last token"
        );
    }
}
