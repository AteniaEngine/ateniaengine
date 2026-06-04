//! **MOE-FULL-12** — DeepSeek-V2 **MLA** (multi-head latent attention) +
//! imperative DeepSeek-MoE forward / decode (CPU, f64 accumulation).
//!
//! DeepSeek-V2/V3 attention is architecturally different from the Mixtral/Qwen
//! path (MHA-with-bias on the AMG graph):
//!
//! * **Low-rank KV** — `kv_a_proj_with_mqa` compresses the hidden to
//!   `kv_lora_rank (+ qk_rope_head_dim)`, an RMSNorm (`kv_a_layernorm`) is
//!   applied, then `kv_b_proj` expands to per-head `k_nope` and `value`.
//! * **Decoupled RoPE** — query/key carry a `qk_nope` part (no RoPE) and a
//!   `qk_rope` part (RoPE). The key's RoPE part `k_pe` is **shared across
//!   heads** (one per token). The RoPE is **interleaved (GPT-J)**: adjacent
//!   dims `(2i, 2i+1)` form a rotated pair — *different* from the half-split
//!   (NeoX) RoPE used by Llama/Mixtral/Qwen (`gb.rope`).
//! * **Asymmetric head dims** — `qk_head_dim = qk_nope + qk_rope` for the
//!   scores; `v_head_dim` for the value. Score scale = `qk_head_dim ** -0.5`.
//!
//! Because none of that maps onto the existing MHA graph, MLA is implemented
//! **imperatively** here (correctness first, no optimisation), reusing the
//! certified MoE block (`RealMoeLayer::forward_auto`). It supports a prefill +
//! incremental KV-cache decode loop. Experimental, CPU-only, opt-in via the
//! runtime. No graph / fail-loud / dense change.

use super::layer::{MoeExecutionConvention, RealMoeLayer};

/// **MLA-0** — YaRN RoPE scaling parameters (DeepSeek-V2/V3). When present, the
/// RoPE `inv_freq` is reparametrised (NTK-by-parts) **and** the attention
/// softmax scale carries `mscale^2` — both active at *every* position, not just
/// long context. `mscale` for the softmax scale uses `mscale_all_dim` (matching
/// HuggingFace `DeepseekV2Attention`). For DeepSeek-V2-Lite `mscale ==
/// mscale_all_dim`, so the cos/sin `_mscale` cancels to 1.0 and only inv_freq +
/// the softmax scale change (the only two effects modelled here).
#[derive(Debug, Clone)]
pub struct YarnParams {
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    pub beta_fast: f64,
    pub beta_slow: f64,
    pub mscale: f64,
    pub mscale_all_dim: f64,
}

/// DeepSeek-V2 MLA + MoE config (the subset the imperative forward needs).
#[derive(Debug, Clone)]
pub struct DeepseekConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    /// **MLA-0** — `Some` when `config.rope_scaling.type == "yarn"`. `None`
    /// reproduces the pre-MLA-0 plain-RoPE + plain-scale behaviour exactly.
    pub yarn: Option<YarnParams>,
    /// **MLA-0** — `norm_topk_prob`: renormalise the selected top-k weights
    /// (DeepSeek `norm_topk_prob=true`, e.g. the existing fixtures) vs not
    /// (DeepSeek-V2-Lite `norm_topk_prob=false`). DeepSeek's shared expert is
    /// ungated either way; `false` ⇒ `HuggingFaceQwen` (no-renorm, ungated
    /// shared), `true` ⇒ `Atenia` (renorm, ungated shared).
    pub renorm_topk: bool,
}

impl DeepseekConfig {
    /// Query/key head dim used for the attention scores.
    pub fn qk_head_dim(&self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }

    /// MoE combine convention implied by `renorm_topk`.
    fn moe_convention(&self) -> MoeExecutionConvention {
        if self.renorm_topk {
            MoeExecutionConvention::Atenia
        } else {
            MoeExecutionConvention::HuggingFaceQwen
        }
    }

    /// RoPE `inv_freq` for the `qk_rope_head_dim` pairs (f64). YaRN-reparametrised
    /// when `self.yarn` is `Some`, else the plain `base^(-2i/dim)`.
    fn rope_inv_freqs(&self) -> Vec<f64> {
        rope_inv_freqs(self.qk_rope_head_dim, self.rope_theta as f64, self.yarn.as_ref())
    }

    /// Attention softmax scale: `qk_head_dim^-0.5`, times `mscale^2` under YaRN
    /// (`mscale = yarn_get_mscale(factor, mscale_all_dim)`).
    fn attn_scale(&self) -> f64 {
        let base = 1.0_f64 / (self.qk_head_dim() as f64).sqrt();
        match &self.yarn {
            None => base,
            Some(y) => {
                let m = yarn_get_mscale(y.factor, y.mscale_all_dim);
                base * m * m
            }
        }
    }
}

/// **MLA-0** — a plain dense SwiGLU FFN (`first_k_dense_replace` layers).
/// `down( silu(gate·x) ⊙ (up·x) )`. Weights row-major, same layout as an expert.
#[derive(Debug, Clone)]
pub struct DenseFfn {
    pub d_model: usize,
    pub d_ff: usize,
    pub gate: Vec<f32>, // [d_ff, d_model]
    pub up: Vec<f32>,   // [d_ff, d_model]
    pub down: Vec<f32>, // [d_model, d_ff]
}

impl DenseFfn {
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let g = matvec(&self.gate, self.d_ff, self.d_model, x);
        let u = matvec(&self.up, self.d_ff, self.d_model, x);
        let mut h = vec![0.0_f32; self.d_ff];
        for i in 0..self.d_ff {
            let gi = g[i] as f64;
            h[i] = ((gi / (1.0 + (-gi).exp())) as f32) * u[i]; // silu(gate)*up
        }
        matvec(&self.down, self.d_model, self.d_ff, &h)
    }
}

/// **MLA-0** — a DeepSeek layer's FFN is either the MoE block (routed + shared)
/// or, for the first `first_k_dense_replace` layers, a plain dense SwiGLU MLP.
#[derive(Debug, Clone)]
pub enum DeepseekFfn {
    Moe(RealMoeLayer),
    Dense(DenseFfn),
}

/// Per-layer DeepSeek weights (MLA attention + norms + the FFN: MoE or dense).
#[derive(Debug, Clone)]
pub struct DeepseekLayer {
    pub input_ln: Vec<f32>, // [hidden]
    pub w_q: Vec<f32>,      // [n_heads * qk_head_dim, hidden]
    pub w_kv_a: Vec<f32>,   // [kv_lora_rank + qk_rope_head_dim, hidden]
    pub kv_a_ln: Vec<f32>,  // [kv_lora_rank]
    pub w_kv_b: Vec<f32>,   // [n_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank]
    pub w_o: Vec<f32>,      // [hidden, n_heads * v_head_dim]
    pub post_ln: Vec<f32>,
    /// **MLA-0** — MoE block or dense SwiGLU (`first_k_dense_replace`).
    pub ffn: DeepseekFfn,
}

/// All DeepSeek weights.
#[derive(Debug, Clone)]
pub struct DeepseekWeights {
    pub config: DeepseekConfig,
    pub embed_tokens: Vec<f32>, // [vocab, hidden]
    pub layers: Vec<DeepseekLayer>,
    pub final_norm: Vec<f32>,
    pub lm_head: Vec<f32>, // [vocab, hidden]
}

/// Per-layer KV cache: materialised per-token, per-head key (`qk_head_dim`) and
/// value (`v_head_dim`). Correctness-first (not the latent-compressed cache).
#[derive(Debug, Clone, Default)]
pub struct MlaLayerCache {
    /// `[token][head][qk_head_dim]` — key with RoPE already applied at the
    /// token's absolute position.
    pub k: Vec<Vec<Vec<f32>>>,
    /// `[token][head][v_head_dim]`.
    pub v: Vec<Vec<Vec<f32>>>,
}

impl MlaLayerCache {
    pub fn len(&self) -> usize {
        self.k.len()
    }
    pub fn is_empty(&self) -> bool {
        self.k.is_empty()
    }
}

// ----------------------------------------------------------------------------
// Math helpers (f64 accumulation).
// ----------------------------------------------------------------------------

/// `y = W · x`, `W` row-major `[rows, cols]`, `x` length `cols`. f64 accum.
///
/// **MOE-PROD-8** — parallelised across output rows (rayon) for large weights,
/// bit-identical to the serial path (each `y[r]` is the same sequential f64
/// reduction; only the thread assignment changes). Same transform as
/// `dense::matvec`; benefits the DeepSeek/MLA attention projections.
fn matvec(w: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    const PAR_THRESHOLD: usize = 1 << 16;
    let row = |base: usize| -> f32 {
        let mut acc = 0.0_f64;
        for c in 0..cols {
            acc += (w[base + c] as f64) * (x[c] as f64);
        }
        acc as f32
    };
    let mut y = vec![0.0_f32; rows];
    if rows.saturating_mul(cols) >= PAR_THRESHOLD {
        use rayon::prelude::*;
        y.par_iter_mut().enumerate().for_each(|(r, yr)| *yr = row(r * cols));
    } else {
        for (r, yr) in y.iter_mut().enumerate() {
            *yr = row(r * cols);
        }
    }
    y
}

/// RMSNorm(x) · γ. f64 accumulation.
fn rmsnorm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ms = 0.0_f64;
    for &v in x {
        ms += (v as f64) * (v as f64);
    }
    ms /= n as f64;
    let inv = 1.0 / (ms + eps as f64).sqrt();
    (0..n).map(|i| ((x[i] as f64) * inv) as f32 * gamma[i]).collect()
}

/// **MLA-0** — YaRN helpers (port of HuggingFace `DeepseekV2` rotary).
fn yarn_get_mscale(scale: f64, mscale: f64) -> f64 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

fn yarn_find_correction_dim(num_rot: f64, dim: usize, base: f64, max_pos: usize) -> f64 {
    (dim as f64 * (max_pos as f64 / (num_rot * 2.0 * std::f64::consts::PI)).ln()) / (2.0 * base.ln())
}

/// `(low, high)` correction range, clamped to `[0, dim-1]`.
fn yarn_find_correction_range(
    low_rot: f64,
    high_rot: f64,
    dim: usize,
    base: f64,
    max_pos: usize,
) -> (f64, f64) {
    let low = yarn_find_correction_dim(low_rot, dim, base, max_pos).floor().max(0.0);
    let high = yarn_find_correction_dim(high_rot, dim, base, max_pos)
        .ceil()
        .min((dim - 1) as f64);
    (low, high)
}

/// **MLA-0** — RoPE `inv_freq` over the `rope_dim/2` pairs. Plain
/// `base^(-2i/dim)` without YaRN; the NTK-by-parts blend of interpolated /
/// extrapolated frequencies with YaRN (exactly HF `DeepseekV2YarnRotaryEmbedding`).
fn rope_inv_freqs(rope_dim: usize, base: f64, yarn: Option<&YarnParams>) -> Vec<f64> {
    let half = rope_dim / 2;
    let extra: Vec<f64> = (0..half)
        .map(|i| base.powf(-(2.0 * i as f64) / rope_dim as f64))
        .collect();
    match yarn {
        None => extra,
        Some(y) => {
            let (low, high) = yarn_find_correction_range(
                y.beta_fast,
                y.beta_slow,
                rope_dim,
                base,
                y.original_max_position_embeddings,
            );
            let denom = if (high - low).abs() < f64::EPSILON { 0.001 } else { high - low };
            (0..half)
                .map(|i| {
                    let inter = extra[i] / y.factor;
                    let ramp = (((i as f64) - low) / denom).clamp(0.0, 1.0);
                    let inv_freq_mask = 1.0 - ramp;
                    inter * (1.0 - inv_freq_mask) + extra[i] * inv_freq_mask
                })
                .collect()
        }
    }
}

/// **Interleaved (GPT-J) RoPE** on a `dim`-length vector at absolute `pos`:
/// rotates each adjacent pair `(x[2i], x[2i+1])` by `pos · inv_freqs[i]`.
/// Matches DeepSeek's `apply_rotary_emb`; `inv_freqs` is YaRN-aware (MLA-0).
fn rope_interleaved(x: &[f32], pos: usize, inv_freqs: &[f64]) -> Vec<f32> {
    let mut out = x.to_vec();
    for (i, &inv_freq) in inv_freqs.iter().enumerate() {
        let angle = pos as f64 * inv_freq;
        let (c, s) = (angle.cos(), angle.sin());
        let a = x[2 * i] as f64;
        let b = x[2 * i + 1] as f64;
        out[2 * i] = (a * c - b * s) as f32;
        out[2 * i + 1] = (a * s + b * c) as f32;
    }
    out
}

/// Project one token's normed hidden state into per-head `(q, k, value)` with
/// RoPE applied at absolute `pos`. `q`/`k` length `qk_head_dim`, `value` length
/// `v_head_dim`; the key's RoPE part is shared across heads.
fn project_token(
    h: &[f32],
    lw: &DeepseekLayer,
    cfg: &DeepseekConfig,
    pos: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let nh = cfg.num_attention_heads;
    let nope = cfg.qk_nope_head_dim;
    let rope = cfg.qk_rope_head_dim;
    let vd = cfg.v_head_dim;
    let qkh = cfg.qk_head_dim();
    let kvl = cfg.kv_lora_rank;
    let hidden = cfg.hidden_size;
    let inv_freqs = cfg.rope_inv_freqs(); // MLA-0: YaRN-aware

    let q = matvec(&lw.w_q, nh * qkh, hidden, h); // [nh*qkh]
    let ckv = matvec(&lw.w_kv_a, kvl + rope, hidden, h); // [kv_lora + qk_rope]
    let compressed = &ckv[0..kvl];
    let k_pe = &ckv[kvl..kvl + rope];
    let compressed_n = rmsnorm(compressed, &lw.kv_a_ln, cfg.rms_eps);
    let kv = matvec(&lw.w_kv_b, nh * (nope + vd), kvl, &compressed_n); // [nh*(nope+vd)]
    let k_pe_roped = rope_interleaved(k_pe, pos, &inv_freqs); // shared across heads

    let mut q_heads = vec![vec![0.0_f32; qkh]; nh];
    let mut k_heads = vec![vec![0.0_f32; qkh]; nh];
    let mut v_heads = vec![vec![0.0_f32; vd]; nh];
    for hd in 0..nh {
        let qh = &q[hd * qkh..(hd + 1) * qkh];
        let q_pe = rope_interleaved(&qh[nope..qkh], pos, &inv_freqs);
        for i in 0..nope {
            q_heads[hd][i] = qh[i];
            k_heads[hd][i] = kv[hd * (nope + vd) + i];
        }
        for i in 0..rope {
            q_heads[hd][nope + i] = q_pe[i];
            k_heads[hd][nope + i] = k_pe_roped[i];
        }
        for i in 0..vd {
            v_heads[hd][i] = kv[hd * (nope + vd) + nope + i];
        }
    }
    (q_heads, k_heads, v_heads)
}

/// Attention for one query token against cached keys/values `0..=cur` (causal),
/// returning the per-head context concatenated `[n_heads * v_head_dim]`.
fn attend(
    q_heads: &[Vec<f32>],
    cache: &MlaLayerCache,
    cfg: &DeepseekConfig,
) -> Vec<f32> {
    let nh = cfg.num_attention_heads;
    let qkh = cfg.qk_head_dim();
    let vd = cfg.v_head_dim;
    let n = cache.len();
    let scale = cfg.attn_scale(); // MLA-0: qk_head_dim^-0.5 * mscale^2 under YaRN
    let mut ctx = vec![0.0_f32; nh * vd];
    for hd in 0..nh {
        let mut sc = vec![0.0_f64; n];
        let mut m = f64::NEG_INFINITY;
        for s in 0..n {
            let mut d = 0.0_f64;
            for i in 0..qkh {
                d += (q_heads[hd][i] as f64) * (cache.k[s][hd][i] as f64);
            }
            sc[s] = d * scale;
            if sc[s] > m {
                m = sc[s];
            }
        }
        let mut den = 0.0_f64;
        for s in 0..n {
            sc[s] = (sc[s] - m).exp();
            den += sc[s];
        }
        for i in 0..vd {
            let mut acc = 0.0_f64;
            for s in 0..n {
                acc += sc[s] / den * (cache.v[s][hd][i] as f64);
            }
            ctx[hd * vd + i] = acc as f32;
        }
    }
    ctx
}

/// One decoder layer over a single token at `pos`, appending to the layer's KV
/// cache. Returns the updated residual hidden state `[hidden]`.
fn layer_step(
    x: &[f32],
    lw: &DeepseekLayer,
    cfg: &DeepseekConfig,
    pos: usize,
    cache: &mut MlaLayerCache,
) -> Vec<f32> {
    let hidden = cfg.hidden_size;
    let nh = cfg.num_attention_heads;
    let vd = cfg.v_head_dim;

    // 1. input RMSNorm → MLA projection → append to cache → attend.
    let h = rmsnorm(x, &lw.input_ln, cfg.rms_eps);
    let (q_heads, k_heads, v_heads) = project_token(&h, lw, cfg, pos);
    cache.k.push(k_heads);
    cache.v.push(v_heads);
    let ctx = attend(&q_heads, cache, cfg); // [nh*vd]
    let attn_out = matvec(&lw.w_o, hidden, nh * vd, &ctx);
    let _ = vd;
    let x1: Vec<f32> = (0..hidden).map(|i| x[i] + attn_out[i]).collect();

    // 2. post-attention RMSNorm → FFN (MoE block or dense SwiGLU) → residual.
    let h2 = rmsnorm(&x1, &lw.post_ln, cfg.rms_eps);
    let ffn_out = match &lw.ffn {
        // MLA-0: drive renorm from `norm_topk_prob` (DeepSeek shared is ungated,
        // so HuggingFaceQwen-with-no-gate == no-renorm + ungated shared).
        DeepseekFfn::Moe(m) => m
            .forward_with(&h2, cfg.moe_convention())
            .expect("deepseek moe forward"),
        DeepseekFfn::Dense(d) => d.forward(&h2),
    };
    (0..hidden).map(|i| x1[i] + ffn_out[i]).collect()
}

impl DeepseekWeights {
    /// **Debug / certification** — run ONLY layer `layer`'s MLA attention
    /// (projection + causal attend + o_proj) over a sequence of **already
    /// input-layernorm'd** hidden states, returning the per-token attention
    /// output `[seq][hidden]`. No residual, no MoE, no input-norm — used to
    /// certify the MLA attention in isolation against HuggingFace.
    pub fn debug_attention(&self, layer: usize, normed: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let cfg = &self.config;
        let lw = &self.layers[layer];
        let hidden = cfg.hidden_size;
        let nh = cfg.num_attention_heads;
        let vd = cfg.v_head_dim;
        let mut cache = MlaLayerCache::default();
        let mut out = Vec::with_capacity(normed.len());
        for (pos, h) in normed.iter().enumerate() {
            let (q_heads, k_heads, v_heads) = project_token(h, lw, cfg, pos);
            cache.k.push(k_heads);
            cache.v.push(v_heads);
            let ctx = attend(&q_heads, &cache, cfg);
            out.push(matvec(&lw.w_o, hidden, nh * vd, &ctx));
        }
        out
    }

    /// Final-norm + lm_head over one hidden state → logits `[vocab]`.
    fn head(&self, x: &[f32]) -> Vec<f32> {
        let h = rmsnorm(x, &self.final_norm, self.config.rms_eps);
        matvec(&self.lm_head, self.config.vocab_size, self.config.hidden_size, &h)
    }

    /// **Prefill**: run the whole prompt, returning `[seq * vocab]` logits and
    /// the per-layer KV caches (seeded with all prompt tokens).
    pub fn forward_prefill(&self, tokens: &[u32]) -> (Vec<f32>, Vec<MlaLayerCache>) {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let seq = tokens.len();
        let mut caches: Vec<MlaLayerCache> = vec![MlaLayerCache::default(); cfg.num_hidden_layers];

        // Running hidden state per token.
        let mut xs: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&t| {
                self.embed_tokens[(t as usize) * hidden..(t as usize + 1) * hidden].to_vec()
            })
            .collect();

        for (li, lw) in self.layers.iter().enumerate() {
            let mut new_xs = Vec::with_capacity(seq);
            for (pos, x) in xs.iter().enumerate() {
                new_xs.push(layer_step(x, lw, cfg, pos, &mut caches[li]));
            }
            xs = new_xs;
        }

        let mut logits = Vec::with_capacity(seq * cfg.vocab_size);
        for x in &xs {
            logits.extend(self.head(x));
        }
        (logits, caches)
    }

    /// **Decode** one new token at absolute `pos`, extending the caches.
    /// Returns the next-token logits `[vocab]`.
    pub fn forward_decode(
        &self,
        token: u32,
        pos: usize,
        caches: &mut [MlaLayerCache],
    ) -> Vec<f32> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let mut x =
            self.embed_tokens[(token as usize) * hidden..(token as usize + 1) * hidden].to_vec();
        for (li, lw) in self.layers.iter().enumerate() {
            x = layer_step(&x, lw, cfg, pos, &mut caches[li]);
        }
        self.head(&x)
    }

    /// **Greedy generation** with prefill + incremental KV-cache decode, stopping
    /// at EOS or after `max_new_tokens`. Returns the generated token ids.
    pub fn generate_greedy_eos(
        &self,
        prompt: &[u32],
        max_new_tokens: usize,
        eos_token_ids: &[u32],
    ) -> Vec<u32> {
        assert!(!prompt.is_empty());
        let vocab = self.config.vocab_size;
        let (logits, mut caches) = self.forward_prefill(prompt);
        let mut next = argmax(&logits[(prompt.len() - 1) * vocab..prompt.len() * vocab]);
        let mut out = vec![next];
        if eos_token_ids.contains(&next) {
            return out;
        }
        for step in 0..(max_new_tokens - 1) {
            let pos = prompt.len() + step;
            let row = self.forward_decode(next, pos, &mut caches);
            next = argmax(&row);
            out.push(next);
            if eos_token_ids.contains(&next) {
                break;
            }
        }
        out
    }
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaved_rope_pos_zero_is_identity() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let inv = rope_inv_freqs(4, 10000.0, None);
        assert_eq!(rope_interleaved(&x, 0, &inv), x);
    }

    #[test]
    fn interleaved_rope_rotates_pairs() {
        // pos=1, dim=2, base=10000: inv_freq[0]=1 → angle 1 rad on pair (x0,x1).
        let x = vec![1.0, 0.0];
        let inv = rope_inv_freqs(2, 10000.0, None);
        let r = rope_interleaved(&x, 1, &inv);
        assert!((r[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((r[1] - 1.0_f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn yarn_inv_freqs_differ_from_plain_and_no_yarn_is_unchanged() {
        let plain = rope_inv_freqs(8, 10000.0, None);
        let plain2 = rope_inv_freqs(8, 10000.0, None);
        assert_eq!(plain, plain2, "no-yarn must be deterministic + unchanged");
        let y = YarnParams {
            factor: 8.0,
            original_max_position_embeddings: 16,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 0.707,
            mscale_all_dim: 0.707,
        };
        let yarned = rope_inv_freqs(8, 10000.0, Some(&y));
        assert_ne!(plain, yarned, "yarn must reparametrise inv_freq");
        // mscale^2 > 1 for factor 8 (so the attention scale grows).
        let m = yarn_get_mscale(8.0, 0.707);
        assert!(m > 1.0 && yarn_get_mscale(1.0, 0.707) == 1.0);
    }

    #[test]
    fn rmsnorm_unit_when_gamma_one() {
        let x = vec![3.0, 4.0, 0.0, 0.0];
        let g = vec![1.0; 4];
        let y = rmsnorm(&x, &g, 1e-5);
        let ms: f32 = y.iter().map(|v| v * v).sum::<f32>() / 4.0;
        assert!((ms - 1.0).abs() < 1e-3);
    }
}
