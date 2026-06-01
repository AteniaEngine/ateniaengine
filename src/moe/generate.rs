//! **MOE-FULL-7** — experimental MoE generation: prefill + KV cache + decode.
//!
//! Builds the first end-to-end *generation* path for the experimental tiny MoE
//! transformer (MOE-FULL-6): a multi-token **prefill** seeds a per-layer **KV
//! cache**, then an incremental **decode** loop produces tokens one at a time,
//! reusing the cached K/V instead of recomputing the whole prefix.
//!
//! ```text
//!   prompt ──► prefill (seq = prompt_len, cached_len = 0)
//!              │  harvest per-layer post-RoPE K, V  ──► KV cache (seed)
//!              ▼
//!   greedy argmax ─► token₀
//!              │
//!   decode step s (seq = 1, cached_len = prompt_len + s):
//!     embed ─► [norm ─► Q/K/V ─► RoPE(offset = cached_len)
//!              ─► concat(cache_K, K_new) / concat(cache_V, V_new)
//!              ─► attention (single query vs cached_len+1 keys, NO mask)
//!              ─► O ─► +res ─► norm ─► MoE ─► +res] × layers
//!     ─► final norm ─► lm_head ─► logits[1,1,vocab]
//!     greedy argmax ─► tokenₛ₊₁ ;  harvest K_full/V_full ─► next cache
//! ```
//!
//! ## Scope (experimental, honest, bounded)
//!
//! * **Greedy only** (argmax; no sampling), CPU-only, test/opt-in only.
//! * **MHA, no GQA** (same tiny Mixtral fixture as MOE-FULL-6). Same documented
//!   simplifications: the `1/sqrt(head_dim)` score scale is absorbed by pre-
//!   scaling `w_q` in Rust; the MoE block is the MOE-FULL-4 certified node.
//! * **No new graph op.** Decode reuses `rope_with_offset`, `concat`,
//!   `batch_matmul`, `softmax`, `permute`, `transpose_last_two` — all existing.
//! * The decode attention needs **no causal mask**: the single new query lives
//!   at the last position, so it legitimately attends to every cached key.
//! * The KV cache lives in plain `Tensor`s harvested from graph node outputs
//!   and re-injected via `Graph::overwrite_parameter` — the exact runtime
//!   state-machine the productive dense generator uses, but here entirely
//!   inside the experimental MoE module. **No productive loader / runtime /
//!   Adapter Toolkit / CLI / WeightStore / CUDA / fail-loud change.**
//!
//! Correctness is locked two ways: (1) `prefill + decode == full recompute`
//! (the MOE-FULL-6 `build_tiny_mixtral_graph` is the oracle), and (2) the
//! generated ids + per-step logits match an offline HuggingFace f64 greedy
//! reference (`fixtures/moe/generate_decode_reference.py`).

use crate::amg::builder::GraphBuilder;
use crate::amg::graph::Graph;
use crate::tensor::Tensor;

use super::full_forward::TinyMixtralWeights;
use super::graph_op::register_real_moe_layer;

/// Per-layer KV cache wiring for a decode-step graph: the two cache parameter
/// slots that the runtime patches before the forward, and the two post-concat
/// node ids that are harvested after the forward to become the next step's
/// cache.
#[derive(Debug, Clone, Copy)]
pub struct KvSlotHandle {
    pub cache_k_param_id: usize,
    pub cache_v_param_id: usize,
    pub k_full_node_id: usize,
    pub v_full_node_id: usize,
}

/// Output of [`generate_greedy_tiny`].
#[derive(Debug, Clone)]
pub struct GreedyGeneration {
    /// Generated token ids in order (length `max_new_tokens`).
    pub tokens: Vec<u32>,
    /// The full vocab logits row that produced each generated token
    /// (length `max_new_tokens`, each inner vec length `vocab`).
    pub step_logits: Vec<Vec<f32>>,
}

/// Build the **prefill** graph (multi-token, `cached_len = 0`) and additionally
/// return, per layer, the node ids of the post-RoPE-permute K and post-permute
/// V — shape `[1, n_heads, seq, head_dim]` — which seed the KV cache.
///
/// Structurally identical to `build_tiny_mixtral_graph` (MOE-FULL-6); the only
/// addition is exposing the K/V nodes. Consumes `w` (registers each MoE layer).
/// Returns `(logits_node_id, per_layer_kv_nodes)`.
pub fn build_tiny_mixtral_prefill(
    gb: &mut GraphBuilder,
    token_input_id: usize,
    seq: usize,
    w: TinyMixtralWeights,
) -> (usize, Vec<(usize, usize)>) {
    let c = w.config;
    let hidden = c.hidden_size;
    let n_heads = c.num_attention_heads;
    let head_dim = c.head_dim;
    let vocab = c.vocab_size;
    let hi = hidden as isize;
    let si = seq as isize;
    let inv_sqrt = 1.0_f32 / (head_dim as f32).sqrt();

    // Causal mask [1, 1, seq, seq].
    let mut mask = vec![0.0_f32; seq * seq];
    for i in 0..seq {
        for j in (i + 1)..seq {
            mask[i * seq + j] = f32::NEG_INFINITY;
        }
    }
    let causal_mask = gb.parameter(Tensor::new_cpu(vec![1, 1, seq, seq], mask));

    let embed_w = gb.parameter(Tensor::new_cpu(vec![vocab, hidden], w.embed_tokens));
    let mut x = gb.index_select(embed_w, token_input_id); // [1, seq, hidden]

    let mut kv_nodes: Vec<(usize, usize)> = Vec::with_capacity(w.layers.len());

    for lw in w.layers {
        let g1 = gb.parameter(Tensor::new_cpu(vec![1, 1, hidden], lw.input_ln));
        let h_n = gb.rms_norm(x, w.rms_eps);
        let h = gb.broadcast_mul(h_n, g1);

        let wq_scaled: Vec<f32> = lw.w_q.iter().map(|v| v * inv_sqrt).collect();
        let wq = gb.parameter(Tensor::new_cpu(vec![hidden, hidden], wq_scaled));
        let wk = gb.parameter(Tensor::new_cpu(vec![hidden, hidden], lw.w_k));
        let wv = gb.parameter(Tensor::new_cpu(vec![hidden, hidden], lw.w_v));
        let wo = gb.parameter(Tensor::new_cpu(vec![hidden, hidden], lw.w_o));

        let h_flat = gb.reshape(h, vec![si, hi]);
        let q_flat = gb.matmul_rhs_transposed(h_flat, wq);
        let k_flat = gb.matmul_rhs_transposed(h_flat, wk);
        let v_flat = gb.matmul_rhs_transposed(h_flat, wv);

        let mh = vec![1, si, n_heads as isize, head_dim as isize];
        let q = gb.reshape(q_flat, mh.clone());
        let k = gb.reshape(k_flat, mh.clone());
        let v = gb.reshape(v_flat, mh);

        let q_r = gb.rope(q, head_dim, c.rope_theta);
        let k_r = gb.rope(k, head_dim, c.rope_theta);

        let q_p = gb.permute(q_r, vec![0, 2, 1, 3]);
        let k_p = gb.permute(k_r, vec![0, 2, 1, 3]); // [1, n_heads, seq, head_dim]
        let v_p = gb.permute(v, vec![0, 2, 1, 3]); // [1, n_heads, seq, head_dim]

        // Harvest these as the cache seed.
        kv_nodes.push((k_p, v_p));

        let k_pt = gb.transpose_last_two(k_p);
        let scores = gb.batch_matmul(q_p, k_pt);
        let scores_m = gb.broadcast_add(scores, causal_mask);
        let attn = gb.softmax(scores_m);
        let ctx = gb.batch_matmul(attn, v_p);
        let ctx_b = gb.permute(ctx, vec![0, 2, 1, 3]);
        let ctx_flat = gb.reshape(ctx_b, vec![si, hi]);
        let o_flat = gb.matmul_rhs_transposed(ctx_flat, wo);
        let o = gb.reshape(o_flat, vec![1, si, hi]);
        let x1 = gb.add(x, o);

        let g2 = gb.parameter(Tensor::new_cpu(vec![1, 1, hidden], lw.post_ln));
        let h2_n = gb.rms_norm(x1, w.rms_eps);
        let h2 = gb.broadcast_mul(h2_n, g2);

        let moe_id = register_real_moe_layer(lw.moe);
        let h2_flat = gb.reshape(h2, vec![(seq * hidden) as isize]);
        let moe_out = gb.moe_real_layer_reference(h2_flat, moe_id);
        let moe_3d = gb.reshape(moe_out, vec![1, si, hi]);
        x = gb.add(x1, moe_3d);
    }

    let gf = gb.parameter(Tensor::new_cpu(vec![1, 1, hidden], w.final_norm));
    let x_n = gb.rms_norm(x, w.rms_eps);
    let x_f = gb.broadcast_mul(x_n, gf);

    let lm_w = gb.parameter(Tensor::new_cpu(vec![vocab, hidden], w.lm_head));
    let x_flat = gb.reshape(x_f, vec![si, hi]);
    let logits_flat = gb.matmul_rhs_transposed(x_flat, lm_w);
    let logits = gb.reshape(logits_flat, vec![1, si, vocab as isize]);
    (logits, kv_nodes)
}

/// Build a **decode-step** graph: a single new token (`seq = 1`) attends to a
/// resident KV cache of `cached_len` tokens. The cache enters as two parameter
/// slots per layer (patched at runtime via `overwrite_parameter`); the post-
/// concat `K_full`/`V_full` (length `cached_len + 1`) are harvested for the
/// next step. Consumes `w`. Returns `(logits_node_id, per_layer_handles)`.
///
/// RoPE is applied at `position_offset = cached_len` so the new token rotates
/// at its absolute position. No causal mask is needed (one query, last
/// position, sees all keys).
pub fn build_tiny_mixtral_decode(
    gb: &mut GraphBuilder,
    token_input_id: usize,
    cached_len: usize,
    w: TinyMixtralWeights,
) -> (usize, Vec<KvSlotHandle>) {
    let c = w.config;
    let hidden = c.hidden_size;
    let n_heads = c.num_attention_heads;
    let head_dim = c.head_dim;
    let vocab = c.vocab_size;
    let hi = hidden as isize;
    let inv_sqrt = 1.0_f32 / (head_dim as f32).sqrt();
    let nh = n_heads as isize;
    let hd = head_dim as isize;
    let cl = cached_len as isize;

    let embed_w = gb.parameter(Tensor::new_cpu(vec![vocab, hidden], w.embed_tokens));
    let mut x = gb.index_select(embed_w, token_input_id); // [1, 1, hidden]

    let mut handles: Vec<KvSlotHandle> = Vec::with_capacity(w.layers.len());

    for lw in w.layers {
        let g1 = gb.parameter(Tensor::new_cpu(vec![1, 1, hidden], lw.input_ln));
        let h_n = gb.rms_norm(x, w.rms_eps);
        let h = gb.broadcast_mul(h_n, g1);

        let wq_scaled: Vec<f32> = lw.w_q.iter().map(|v| v * inv_sqrt).collect();
        let wq = gb.parameter(Tensor::new_cpu(vec![hidden, hidden], wq_scaled));
        let wk = gb.parameter(Tensor::new_cpu(vec![hidden, hidden], lw.w_k));
        let wv = gb.parameter(Tensor::new_cpu(vec![hidden, hidden], lw.w_v));
        let wo = gb.parameter(Tensor::new_cpu(vec![hidden, hidden], lw.w_o));

        let h_flat = gb.reshape(h, vec![1, hi]);
        let q_flat = gb.matmul_rhs_transposed(h_flat, wq); // [1, hidden]
        let k_flat = gb.matmul_rhs_transposed(h_flat, wk);
        let v_flat = gb.matmul_rhs_transposed(h_flat, wv);

        let mh = vec![1, 1, nh, hd];
        let q = gb.reshape(q_flat, mh.clone());
        let k = gb.reshape(k_flat, mh.clone());
        let v = gb.reshape(v_flat, mh);

        // RoPE at absolute position `cached_len` for the new token.
        let q_r = gb.rope_with_offset(q, head_dim, c.rope_theta, cached_len as u32);
        let k_r = gb.rope_with_offset(k, head_dim, c.rope_theta, cached_len as u32);

        let q_p = gb.permute(q_r, vec![0, 2, 1, 3]); // [1, n_heads, 1, head_dim]
        let k_new = gb.permute(k_r, vec![0, 2, 1, 3]); // [1, n_heads, 1, head_dim]
        let v_new = gb.permute(v, vec![0, 2, 1, 3]); // [1, n_heads, 1, head_dim]

        // Cache parameter slots [1, n_heads, cached_len, head_dim]. Placeholder
        // zeros at build time; the runtime patches them via overwrite_parameter.
        let zeros = vec![0.0_f32; n_heads * cached_len * head_dim];
        let cache_k = gb.parameter(Tensor::new_cpu(vec![1, n_heads, cached_len, head_dim], zeros.clone()));
        let cache_v = gb.parameter(Tensor::new_cpu(vec![1, n_heads, cached_len, head_dim], zeros));

        // Concat cache ⊕ new along the seq axis (axis 2) → [1, n_heads, cached_len+1, head_dim].
        let k_full = gb.concat(cache_k, k_new, 2);
        let v_full = gb.concat(cache_v, v_new, 2);

        // Single query vs cached_len+1 keys → scores [1, n_heads, 1, cached_len+1].
        let k_full_t = gb.transpose_last_two(k_full);
        let scores = gb.batch_matmul(q_p, k_full_t);
        let attn = gb.softmax(scores);
        let ctx = gb.batch_matmul(attn, v_full); // [1, n_heads, 1, head_dim]
        let ctx_b = gb.permute(ctx, vec![0, 2, 1, 3]); // [1, 1, n_heads, head_dim]
        let ctx_flat = gb.reshape(ctx_b, vec![1, hi]);
        let o_flat = gb.matmul_rhs_transposed(ctx_flat, wo);
        let o = gb.reshape(o_flat, vec![1, 1, hi]);
        let x1 = gb.add(x, o);

        let g2 = gb.parameter(Tensor::new_cpu(vec![1, 1, hidden], lw.post_ln));
        let h2_n = gb.rms_norm(x1, w.rms_eps);
        let h2 = gb.broadcast_mul(h2_n, g2);

        let moe_id = register_real_moe_layer(lw.moe);
        let h2_flat = gb.reshape(h2, vec![hi]);
        let moe_out = gb.moe_real_layer_reference(h2_flat, moe_id);
        let moe_3d = gb.reshape(moe_out, vec![1, 1, hi]);
        x = gb.add(x1, moe_3d);

        let _ = cl; // documents the cache length used for the slot shape
        handles.push(KvSlotHandle {
            cache_k_param_id: cache_k,
            cache_v_param_id: cache_v,
            k_full_node_id: k_full,
            v_full_node_id: v_full,
        });
    }

    let gf = gb.parameter(Tensor::new_cpu(vec![1, 1, hidden], w.final_norm));
    let x_n = gb.rms_norm(x, w.rms_eps);
    let x_f = gb.broadcast_mul(x_n, gf);

    let lm_w = gb.parameter(Tensor::new_cpu(vec![vocab, hidden], w.lm_head));
    let x_flat = gb.reshape(x_f, vec![1, hi]);
    let logits_flat = gb.matmul_rhs_transposed(x_flat, lm_w);
    let logits = gb.reshape(logits_flat, vec![1, 1, vocab as isize]);
    (logits, handles)
}

fn argmax_row(logits: &[f32], position: usize, vocab: usize) -> u32 {
    let row = &logits[position * vocab..(position + 1) * vocab];
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn harvest(g: &Graph, ids: impl Iterator<Item = usize>) -> Vec<Tensor> {
    ids.map(|id| g.nodes[id].output.as_ref().expect("kv node not materialised").clone())
        .collect()
}

/// **MOE-FULL-7** — greedy generation with prefill + KV cache + incremental
/// decode. Runs prefill on `prompt`, then `max_new_tokens - 1` decode steps,
/// reusing the harvested KV cache each step. Returns the generated ids and the
/// per-step logits rows. Deterministic (greedy argmax). CPU-only.
pub fn generate_greedy_tiny(
    w: &TinyMixtralWeights,
    prompt: &[u32],
    max_new_tokens: usize,
) -> GreedyGeneration {
    assert!(!prompt.is_empty(), "generate_greedy_tiny: empty prompt");
    assert!(max_new_tokens >= 1, "generate_greedy_tiny: max_new_tokens must be >= 1");
    let vocab = w.config.vocab_size;
    let prompt_len = prompt.len();

    // ---- Prefill ----
    let mut gb = GraphBuilder::new();
    let tok = gb.input();
    let (logits_id, kv_nodes) = build_tiny_mixtral_prefill(&mut gb, tok, prompt_len, w.clone());
    gb.output(logits_id);
    let mut g = gb.build();
    let pin = Tensor::new_cpu(vec![1, prompt_len], prompt.iter().map(|&t| t as f32).collect());
    let prefill_logits = g.execute(vec![pin])[0].as_cpu_slice().to_vec();

    let first = argmax_row(&prefill_logits, prompt_len - 1, vocab);
    let first_row = prefill_logits[(prompt_len - 1) * vocab..prompt_len * vocab].to_vec();

    let mut cache_k = harvest(&g, kv_nodes.iter().map(|(k, _)| *k));
    let mut cache_v = harvest(&g, kv_nodes.iter().map(|(_, v)| *v));
    drop(g);

    let mut tokens = vec![first];
    let mut step_logits = vec![first_row];
    let mut next = first;

    // ---- Decode loop ----
    for step in 0..(max_new_tokens - 1) {
        let cached_len = prompt_len + step;

        let mut gb_d = GraphBuilder::new();
        let tin = gb_d.input();
        let (lid, handles) = build_tiny_mixtral_decode(&mut gb_d, tin, cached_len, w.clone());
        gb_d.output(lid);
        let mut g_d = gb_d.build();

        for (li, h) in handles.iter().enumerate() {
            g_d.overwrite_parameter(h.cache_k_param_id, cache_k[li].clone())
                .expect("decode: overwrite cache_K");
            g_d.overwrite_parameter(h.cache_v_param_id, cache_v[li].clone())
                .expect("decode: overwrite cache_V");
        }

        let tt = Tensor::new_cpu(vec![1, 1], vec![next as f32]);
        let ld = g_d.execute(vec![tt])[0].as_cpu_slice().to_vec(); // [vocab]
        next = argmax_row(&ld, 0, vocab);

        cache_k = harvest(&g_d, handles.iter().map(|h| h.k_full_node_id));
        cache_v = harvest(&g_d, handles.iter().map(|h| h.v_full_node_id));

        tokens.push(next);
        step_logits.push(ld);
    }

    GreedyGeneration { tokens, step_logits }
}

// ============================================================================
// Tests (synthetic weights — the real-fixture + HF greedy comparison is the
// integration test in tests/moe_decode_generation_test.rs).
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moe::dense::build_fixture_layer;
    use crate::moe::full_forward::{
        build_tiny_mixtral_graph, TinyDecoderWeights, TinyMixtralConfig,
    };
    use crate::moe::{MoeLayerConfig, RealMoeLayer};

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

    fn synthetic() -> TinyMixtralWeights {
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
                let routed = build_fixture_layer();
                let moe_cfg =
                    MoeLayerConfig::new(routed.num_experts(), 2, false, routed.d_model, routed.d_ff)
                        .unwrap();
                let moe = RealMoeLayer { config: moe_cfg, routed, shared: None, shared_gate: None };
                TinyDecoderWeights {
                    input_ln: seeded(l as u64 * 10 + 1, hidden),
                    w_q: seeded(l as u64 * 10 + 2, hidden * hidden),
                    w_k: seeded(l as u64 * 10 + 3, hidden * hidden),
                    w_v: seeded(l as u64 * 10 + 4, hidden * hidden),
                    w_o: seeded(l as u64 * 10 + 5, hidden * hidden),
                    post_ln: seeded(l as u64 * 10 + 6, hidden),
                    moe,
                }
            })
            .collect();
        TinyMixtralWeights {
            config: cfg,
            embed_tokens: seeded(900, vocab * hidden),
            layers,
            final_norm: seeded(901, hidden),
            lm_head: seeded(902, vocab * hidden),
            rms_eps: 1e-5,
        }
    }

    /// Full-recompute logits for a whole sequence via the MOE-FULL-6 oracle.
    fn full_recompute(w: &TinyMixtralWeights, seq_tokens: &[f32]) -> Vec<f32> {
        let seq = seq_tokens.len();
        let mut gb = GraphBuilder::new();
        let tok = gb.input();
        let logits = build_tiny_mixtral_graph(&mut gb, tok, seq, w.clone());
        gb.output(logits);
        let mut g = gb.build();
        let t = Tensor::new_cpu(vec![1, seq], seq_tokens.to_vec());
        g.execute(vec![t])[0].as_cpu_slice().to_vec()
    }

    #[test]
    fn generation_is_deterministic() {
        let w = synthetic();
        let prompt = [1u32, 3, 0];
        let a = generate_greedy_tiny(&w, &prompt, 4);
        let b = generate_greedy_tiny(&w, &prompt, 4);
        assert_eq!(a.tokens, b.tokens);
        assert_eq!(a.step_logits, b.step_logits);
        assert_eq!(a.tokens.len(), 4);
    }

    #[test]
    fn kv_cache_matches_full_recompute() {
        // The R2 falsifier: prefill + incremental decode must produce exactly
        // the same per-step logits as recomputing the full prefix every step.
        let w = synthetic();
        let vocab = w.config.vocab_size;
        let prompt = [2u32, 5, 1, 4];
        let max_new = 4;
        let out = generate_greedy_tiny(&w, &prompt, max_new);

        // Full sequence the decode loop effectively walked:
        // prompt ++ generated[..max_new-1] (the last generated token is never
        // fed back, so it is not part of any forward's input).
        let mut full: Vec<f32> = prompt.iter().map(|&t| t as f32).collect();
        for &t in &out.tokens[..max_new - 1] {
            full.push(t as f32);
        }
        let recompute = full_recompute(&w, &full);
        let prompt_len = prompt.len();

        for i in 0..max_new {
            let pos = prompt_len - 1 + i;
            let row = &recompute[pos * vocab..(pos + 1) * vocab];
            for v in 0..vocab {
                assert!(
                    (out.step_logits[i][v] - row[v]).abs() < 1e-4,
                    "kv decode step {i} logit {v} diverged from full recompute: {} vs {}",
                    out.step_logits[i][v],
                    row[v]
                );
            }
        }
    }

    #[test]
    fn argmax_tokens_match_full_recompute_greedy() {
        // The greedy ids the decode loop emits must equal a full-recompute
        // greedy walk (argmax at each step on the freshly recomputed prefix).
        let w = synthetic();
        let vocab = w.config.vocab_size;
        let prompt = [1u32, 7, 2];
        let max_new = 5;
        let out = generate_greedy_tiny(&w, &prompt, max_new);

        let mut seq: Vec<f32> = prompt.iter().map(|&t| t as f32).collect();
        let mut expected = Vec::new();
        for _ in 0..max_new {
            let logits = full_recompute(&w, &seq);
            let pos = seq.len() - 1;
            let tok = argmax_row(&logits, pos, vocab);
            expected.push(tok);
            seq.push(tok as f32);
        }
        assert_eq!(out.tokens, expected);
    }
}
