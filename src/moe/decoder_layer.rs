//! **MOE-FULL-5** — experimental MoE decoder-layer composition (graph).
//!
//! Composes a single decoder layer in the AMG graph to prove that the
//! MOE-FULL-4 `MoeRealLayerReference` node integrates correctly with the
//! surrounding dense pieces (norm + attention + residuals). Structure:
//!
//! ```text
//!   x
//!    → RMSNorm·γ₁  → self-attention (Q/K/V/O) → +x          (residual 1)
//!    → RMSNorm·γ₂  → MoeRealLayerReference     → +residual1 (residual 2)
//!    → output
//! ```
//!
//! ## Scope (deliberately minimal, experimental)
//!
//! * **Single token (`seq = 1`), single-head, no RoPE, no GQA, no KV cache,
//!   no multi-token, no batching.** Every attention op runs structurally
//!   (Q/K/V projection, scaled scores, softmax, weighted-V, output
//!   projection), but with one token the goal is to validate **graph
//!   composition**, not multi-token attention dynamics (that is MOE-FULL-6).
//! * It is built from **existing AMG primitives** (`rms_norm`, `matmul`,
//!   `matmul_rhs_transposed`, `softmax`, `add`, `broadcast_mul`, `scale`) plus
//!   the MOE-FULL-4 MoE node. It does **not** reuse the productive
//!   `build_transformer_block_llama` (private + coupled to the productive
//!   runtime/loader; reusing it would require refactoring the dense path,
//!   which is out of scope) and it adds **no new graph op**.
//! * CPU-only, registry-backed, test/opt-in only. No generation, no loader,
//!   no Adapter Toolkit, no CUDA, no CLI. The MOE-2 fail-loud guard is
//!   unchanged.
//!
//! The graph layer is validated against an independent imperative reference
//! ([`decoder_layer_reference`]) computing exactly the same operation, within
//! 1e-5 (the MOE-16 methodology).

use crate::amg::builder::GraphBuilder;
use crate::tensor::Tensor;

use super::graph_op::register_real_moe_layer;
use super::layer::{MoeLayerError, RealMoeLayer};

/// Dense weights for the experimental single-head attention sub-block. All
/// matrices are row-major `[d_model, d_model]` (applied as `x @ W`), and the
/// two RMSNorm γ vectors are `[d_model]`.
#[derive(Debug, Clone)]
pub struct ExpAttnWeights {
    pub d_model: usize,
    pub norm1_gamma: Vec<f32>,
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
    pub norm2_gamma: Vec<f32>,
    pub rms_eps: f32,
}

impl ExpAttnWeights {
    /// Validate that every matrix is `d_model * d_model` and γ is `d_model`.
    pub fn validate(&self) -> Result<(), String> {
        let dm = self.d_model;
        let sq = dm * dm;
        for (name, w, want) in [
            ("norm1_gamma", &self.norm1_gamma, dm),
            ("w_q", &self.w_q, sq),
            ("w_k", &self.w_k, sq),
            ("w_v", &self.w_v, sq),
            ("w_o", &self.w_o, sq),
            ("norm2_gamma", &self.norm2_gamma, dm),
        ] {
            if w.len() != want {
                return Err(format!("{name}: expected {want} elems, got {}", w.len()));
            }
        }
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Imperative reference (f32, f64-accumulating matvec) — the oracle.
// ----------------------------------------------------------------------------

/// `y = x · W` where `W` is row-major `[d_model, d_model]`, `x` length
/// `d_model`. Standard `x @ W` (NOT transposed): `y[j] = Σ_i x[i] · W[i*dm+j]`.
/// f64 accumulation.
fn matvec_xw(x: &[f32], w: &[f32], dm: usize) -> Vec<f32> {
    let mut y = vec![0.0_f32; dm];
    for j in 0..dm {
        let mut acc = 0.0_f64;
        for i in 0..dm {
            acc += (x[i] as f64) * (w[i * dm + j] as f64);
        }
        y[j] = acc as f32;
    }
    y
}

/// RMSNorm(x) · γ, f64 accumulation. `rms = sqrt(mean(x²) + eps)`.
fn rms_norm_gamma(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ms = 0.0_f64;
    for &v in x {
        ms += (v as f64) * (v as f64);
    }
    ms /= n as f64;
    let inv = 1.0 / (ms + eps as f64).sqrt();
    (0..n).map(|i| ((x[i] as f64) * inv) as f32 * gamma[i]).collect()
}

/// Imperative reference for the experimental decoder layer (single token).
/// Computes exactly what [`build_experimental_decoder_layer`] assembles in the
/// graph, using `layer.forward_auto` for the MoE sub-block.
pub fn decoder_layer_reference(
    x: &[f32],
    attn: &ExpAttnWeights,
    moe: &RealMoeLayer,
) -> Result<Vec<f32>, MoeLayerError> {
    let dm = attn.d_model;
    // 1. norm1
    let h = rms_norm_gamma(x, &attn.norm1_gamma, attn.rms_eps);
    // 2. Q/K/V projections
    let q = matvec_xw(&h, &attn.w_q, dm);
    let k = matvec_xw(&h, &attn.w_k, dm);
    let v = matvec_xw(&h, &attn.w_v, dm);
    // 3. scores = (q · kᵀ) / sqrt(dm)  — single token → scalar
    let mut dot = 0.0_f64;
    for i in 0..dm {
        dot += (q[i] as f64) * (k[i] as f64);
    }
    let _score = (dot / (dm as f64).sqrt()) as f32;
    // 4. softmax over a single element → 1.0 → context = v
    //    (kept explicit so the graph and reference share the structure)
    let ctx = v; // softmax([s]) = [1.0]; ctx = 1.0 * v
    // 5. output projection
    let attn_out = matvec_xw(&ctx, &attn.w_o, dm);
    // 6. residual 1
    let x1: Vec<f32> = (0..dm).map(|i| x[i] + attn_out[i]).collect();
    // 7. norm2
    let h2 = rms_norm_gamma(&x1, &attn.norm2_gamma, attn.rms_eps);
    // 8. MoE sub-block
    let moe_out = moe.forward_auto(&h2)?;
    // 9. residual 2
    Ok((0..dm).map(|i| x1[i] + moe_out[i]).collect())
}

// ----------------------------------------------------------------------------
// Graph builder.
// ----------------------------------------------------------------------------

/// Build the experimental decoder layer into `gb`. `x_id` is the input token
/// node (shape `[1, d_model]`); `moe_layer_id` indexes a `RealMoeLayer` in the
/// MOE-FULL-4 registry. Returns the output node id (shape `[1, d_model]`).
///
/// Uses only existing AMG primitives + the `MoeRealLayerReference` node.
pub fn build_experimental_decoder_layer(
    gb: &mut GraphBuilder,
    x_id: usize,
    attn: &ExpAttnWeights,
    moe_layer_id: u32,
) -> usize {
    let dm = attn.d_model;
    let dmi = dm as isize;

    // Constant weight nodes (GraphBuilder has no `constant`; a `parameter`
    // node holding a fixed tensor is the constant mechanism).
    let g1 = gb.parameter(Tensor::new_cpu(vec![1, dm], attn.norm1_gamma.clone()));
    let wq = gb.parameter(Tensor::new_cpu(vec![dm, dm], attn.w_q.clone()));
    let wk = gb.parameter(Tensor::new_cpu(vec![dm, dm], attn.w_k.clone()));
    let wv = gb.parameter(Tensor::new_cpu(vec![dm, dm], attn.w_v.clone()));
    let wo = gb.parameter(Tensor::new_cpu(vec![dm, dm], attn.w_o.clone()));
    let g2 = gb.parameter(Tensor::new_cpu(vec![1, dm], attn.norm2_gamma.clone()));

    // 1. norm1·γ₁
    let h_normed = gb.rms_norm(x_id, attn.rms_eps);
    let h = gb.broadcast_mul(h_normed, g1);

    // 2. Q/K/V projections ([1,dm] @ [dm,dm] = [1,dm])
    let q = gb.matmul(h, wq);
    let k = gb.matmul(h, wk);
    let v = gb.matmul(h, wv);

    // 3. scores = q · kᵀ → [1,1]. NOTE: with a single token, softmax over a
    //    one-element row is always 1.0, so the usual 1/sqrt(dm) score scale
    //    does not affect the output and is omitted (GraphBuilder has no
    //    `scale` op). The imperative reference mirrors this (ctx = v).
    let qk = gb.matmul_rhs_transposed(q, k); // [1,1]

    // 4. softmax (single element → 1.0)
    let attn_w = gb.softmax(qk);

    // 5. context = attn_w @ v ([1,1] @ [1,dm] = [1,dm])
    let ctx = gb.matmul(attn_w, v);

    // 6. output projection
    let attn_out = gb.matmul(ctx, wo);

    // 7. residual 1
    let x1 = gb.add(x_id, attn_out);

    // 8. norm2·γ₂
    let h2_normed = gb.rms_norm(x1, attn.rms_eps);
    let h2 = gb.broadcast_mul(h2_normed, g2);

    // 9. MoE sub-block (input flat [dm]; output [dm] → reshape to [1,dm])
    let h2_flat = gb.reshape(h2, vec![dmi]);
    let moe = gb.moe_real_layer_reference(h2_flat, moe_layer_id);
    let moe_2d = gb.reshape(moe, vec![1, dmi]);

    // 10. residual 2
    gb.add(x1, moe_2d)
}

/// Convenience: register `moe` in the MOE-FULL-4 registry and build the layer.
/// Returns `(graph_output_id, moe_layer_id)`.
pub fn register_and_build_decoder_layer(
    gb: &mut GraphBuilder,
    x_id: usize,
    attn: &ExpAttnWeights,
    moe: RealMoeLayer,
) -> (usize, u32) {
    let id = register_real_moe_layer(moe);
    let out = build_experimental_decoder_layer(gb, x_id, attn, id);
    (out, id)
}

// ============================================================================
// Tests (synthetic attention weights + synthetic MoE fixture; no files)
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

    /// A small real MoE layer from the synthetic fixture (4 experts, top-2,
    /// d_model 8, no shared expert).
    fn fixture_moe() -> RealMoeLayer {
        let routed = build_fixture_layer();
        let config = MoeLayerConfig::new(routed.num_experts(), 2, false, routed.d_model, routed.d_ff)
            .unwrap();
        RealMoeLayer { config, routed, shared: None, shared_gate: None }
    }

    fn fixture_attn(dm: usize) -> ExpAttnWeights {
        ExpAttnWeights {
            d_model: dm,
            norm1_gamma: seeded(1, dm),
            w_q: seeded(2, dm * dm),
            w_k: seeded(3, dm * dm),
            w_v: seeded(4, dm * dm),
            w_o: seeded(5, dm * dm),
            norm2_gamma: seeded(6, dm),
            rms_eps: 1e-6,
        }
    }

    fn run_graph(attn: &ExpAttnWeights, moe: RealMoeLayer, x: &[f32]) -> Vec<f32> {
        let mut gb = GraphBuilder::new();
        let x_id = gb.input();
        let (out_id, _id) = register_and_build_decoder_layer(&mut gb, x_id, attn, moe);
        gb.output(out_id);
        let mut g = gb.build();
        let t = Tensor::new_cpu(vec![1, attn.d_model], x.to_vec());
        let outs = g.execute(vec![t]);
        outs[0].as_cpu_slice().to_vec()
    }

    #[test]
    fn weights_validate() {
        assert!(fixture_attn(8).validate().is_ok());
        let mut bad = fixture_attn(8);
        bad.w_q.pop();
        assert!(bad.validate().is_err());
    }

    #[test]
    fn graph_matches_reference() {
        let dm = 8;
        let attn = fixture_attn(dm);
        let moe = fixture_moe();
        let x = seeded(42, dm);
        let reference = decoder_layer_reference(&x, &attn, &moe).unwrap();
        let got = run_graph(&attn, moe, &x);
        assert_eq!(got.len(), dm);
        for d in 0..dm {
            assert!(
                (got[d] - reference[d]).abs() < 1e-5,
                "decoder layer graph vs reference at {d}: {} vs {}",
                got[d],
                reference[d]
            );
        }
    }

    #[test]
    fn graph_is_deterministic() {
        let dm = 8;
        let attn = fixture_attn(dm);
        let x = seeded(7, dm);
        let a = run_graph(&attn, fixture_moe(), &x);
        let b = run_graph(&attn, fixture_moe(), &x);
        assert_eq!(a, b);
        assert!(a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn residual_changes_output() {
        // The output must differ from a bare MoE forward on x (residuals +
        // attention contribute), proving real composition (not a pass-through).
        let dm = 8;
        let attn = fixture_attn(dm);
        let moe = fixture_moe();
        let x = seeded(11, dm);
        let bare_moe = moe.forward_auto(&x).unwrap();
        let layer = decoder_layer_reference(&x, &attn, &moe).unwrap();
        assert!((0..dm).any(|d| (layer[d] - bare_moe[d]).abs() > 1e-4));
    }
}
