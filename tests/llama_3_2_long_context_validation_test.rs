//! Long-context falsifier for the Llama 3 RoPE wiring (M4.6 Phase C.6).
//!
//! Background: phases C.4 and C.5 exercise Llama 3.2 1B end-to-end at
//! `seq_len = 4`. With such a short context every angle
//! `t · inv_freq[i]` lives in the high-frequency band, where the
//! Llama 3 piecewise scaling is the identity. Both phases would pass
//! green even if the scaling were silently turned off — they cannot
//! distinguish "scaling correctly applied" from "scaling never
//! applied at all".
//!
//! This test is the decisive falsifier on hypothesis H7 of the Phase C
//! investigation. It builds a graph containing a single RoPE node with
//! `scaling = Some(RopeScalingLlama3)` at `seq_len = 2048` — large
//! enough that:
//!   * dim i=16 has wavelength ~4443, sitting in the mid-band where
//!     the smooth interpolation kicks in;
//!   * dim i=31 has wavelength ~2 × 10⁶, sitting in the low-band
//!     where the inverse frequency is divided by `factor` (32).
//!
//! Two assertions:
//!
//!   A. **Graph wiring is correct.** The graph output matches a
//!      direct call to `apply_rope_with_inv_freqs(x, head_dim,
//!      &compute_inv_freqs_llama3(...))` bit-for-bit. Confirms the
//!      `NodeType::RoPE { scaling: Some(_) }` branch in graph.rs is
//!      threading the right `inv_freq` vector into the forward.
//!
//!   B. **Scaling is materially active.** The graph output is NOT
//!      bit-equivalent to plain `apply_rope`. At positions in the
//!      mid/low band, the rotation angle differs by O(1) radians, so
//!      the activations after RoPE differ by macroscopic amounts. If
//!      this assertion failed it would mean the graph silently
//!      degraded to the legacy path despite our `scaling = Some`
//!      input.
//!
//! Only assertion A is a tight numerical test; B is a sanity check
//! that the scaling actually changes the result.
//!
//! The full-model F64 ground truth at long context is intentionally
//! out of scope here: regenerating that fixture costs tens of GB of
//! RAM and dozens of minutes, while the scaling math itself is
//! already verified mathematically by the unit tests in C.2 and the
//! per-position drift trends in C.5. Combining A and B therefore
//! covers the only remaining gap: that the scaled `inv_freq` vector
//! actually reaches the rotation kernel through the AMG pipeline.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::nodes::RopeScalingLlama3;
use atenia_engine::nn::rope::{
    apply_rope, apply_rope_with_inv_freqs, compute_inv_freqs, compute_inv_freqs_llama3,
};
use atenia_engine::tensor::Tensor;

/// Llama 3.2 1B production parameters.
const HEAD_DIM: usize = 64;
const BASE_FREQ: u32 = 500_000;
const FACTOR: f32 = 32.0;
const LOW: f32 = 1.0;
const HIGH: f32 = 4.0;
const ORIGINAL_MAX_POS: u32 = 8192;

/// Long enough that some positions sit in the low / mid frequency
/// bands of the Llama 3 piecewise schedule (low_wavelen = 8192,
/// high_wavelen = 2048; positions near s=2047 with low-band dims
/// give angles ~6e-3 unscaled vs ~2e-4 scaled, and mid-band dims
/// give angles ~3 vs ~0.9 — entirely different rotations).
const SEQ_LEN: usize = 2048;
const N_HEADS: usize = 4;
const BATCH: usize = 1;

/// Deterministic synthetic input. Values are bounded so the L2 norm
/// preservation property of RoPE keeps everything in F32 range, and
/// non-zero everywhere so a bit-equality test would catch any
/// regression that nulls out one component.
fn make_input() -> Tensor {
    let total = BATCH * SEQ_LEN * N_HEADS * HEAD_DIM;
    let mut data = Vec::with_capacity(total);
    for k in 0..total {
        // Spread values across [-1, 1] in a non-trivial pattern
        // (alternating sign, slowly varying magnitude).
        let raw = ((k as f32) * 0.013).sin() * 0.7
            + ((k as f32) * 0.27).cos() * 0.3;
        data.push(raw);
    }
    Tensor::new_cpu(vec![BATCH, SEQ_LEN, N_HEADS, HEAD_DIM], data)
}

#[test]
fn llama_3_long_context_graph_matches_direct_scaled_rope_bit_exact() {
    let input = make_input();

    // ---- Reference: direct call to the kernel with the scaled inv_freq.
    let inv_freqs_scaled = compute_inv_freqs_llama3(
        HEAD_DIM,
        BASE_FREQ,
        FACTOR,
        LOW,
        HIGH,
        ORIGINAL_MAX_POS,
    );
    let direct_scaled = apply_rope_with_inv_freqs(&input, HEAD_DIM, &inv_freqs_scaled);

    // ---- Graph path: NodeType::RoPE { scaling: Some(_) } via builder.
    let scaling = RopeScalingLlama3::new(FACTOR, LOW, HIGH, ORIGINAL_MAX_POS);
    let mut gb = GraphBuilder::new();
    let in_id = gb.input();
    let rope_id = gb.rope_scaled(in_id, HEAD_DIM, BASE_FREQ, scaling);
    let _ = gb.output(rope_id);
    let mut graph = gb.build();
    let outputs = graph.execute(vec![input.clone()]);
    let graph_out: &[f32] = outputs[0].as_cpu_slice();
    let direct_out: &[f32] = direct_scaled.as_cpu_slice();

    assert_eq!(graph_out.len(), direct_out.len());

    // ---- Assertion A: bit-exact equality with the direct kernel call.
    // No floating-point tolerance: the graph path and the direct call
    // both feed the identical `inv_freqs_scaled` vector through the
    // same arithmetic. Any difference would mean graph wiring routed
    // a different schedule through the kernel.
    for (i, (g, d)) in graph_out.iter().zip(direct_out.iter()).enumerate() {
        assert_eq!(
            g, d,
            "graph vs direct mismatch at index {} (pos={}, head={}, dim={}): {} vs {}",
            i,
            i / (N_HEADS * HEAD_DIM) % SEQ_LEN,
            i / HEAD_DIM % N_HEADS,
            i % HEAD_DIM,
            g,
            d
        );
    }

    // ---- Assertion B: scaling is materially different from plain RoPE.
    // For tokens near s=2047 with mid-band dims, the rotation angle
    // shifts by O(1) radians — `cos`/`sin` differ by O(1). Therefore
    // the L_inf and L_2 distances between scaled and unscaled outputs
    // must be macroscopic.
    let unscaled = apply_rope(&input, HEAD_DIM, BASE_FREQ);
    let unscaled_slice: &[f32] = unscaled.as_cpu_slice();
    let max_abs_diff = graph_out
        .iter()
        .zip(unscaled_slice.iter())
        .map(|(s, u)| (s - u).abs())
        .fold(0.0_f32, f32::max);

    // Empirical lower bound: scaling at seq=2048 produces O(0.1)–O(1)
    // pointwise differences. Use 0.1 as a conservative floor so a
    // genuine regression (zero diff = scaling silently dropped)
    // trips the assertion immediately.
    assert!(
        max_abs_diff > 0.1,
        "scaled and unscaled RoPE outputs are suspiciously close \
         (max abs diff = {:.6e}); the Llama 3 scaling appears to \
         have NOT been applied — falsifier of H7 from the C \
         investigation has triggered.",
        max_abs_diff
    );

    println!(
        "Long-context falsifier passed: graph bit-exactly matches \
         direct scaled kernel; scaled-vs-unscaled max abs diff = \
         {:.4} (well above the 0.1 floor, scaling is materially \
         active at seq={}).",
        max_abs_diff, SEQ_LEN
    );
}

/// Sanity: at seq=4 the falsifier would NOT trip (scaling is identity
/// in the high-frequency band). This documents WHY C.4 / C.5 cannot
/// detect a broken scaling and motivates this file's existence.
///
/// The test exists as a meta-explanation — if we ever change either
/// `compute_inv_freqs` or `compute_inv_freqs_llama3` such that the
/// high-band identity is broken at seq=4, this canary would fire and
/// surface the change.
#[test]
fn llama_3_short_context_scaled_and_unscaled_almost_coincide() {
    let short_seq = 4;
    let total = BATCH * short_seq * N_HEADS * HEAD_DIM;
    let data: Vec<f32> = (0..total)
        .map(|k| ((k as f32) * 0.013).sin() * 0.7 + ((k as f32) * 0.27).cos() * 0.3)
        .collect();
    let input = Tensor::new_cpu(vec![BATCH, short_seq, N_HEADS, HEAD_DIM], data);

    let unscaled = apply_rope(&input, HEAD_DIM, BASE_FREQ);
    let scaled_inv_freqs = compute_inv_freqs_llama3(
        HEAD_DIM,
        BASE_FREQ,
        FACTOR,
        LOW,
        HIGH,
        ORIGINAL_MAX_POS,
    );
    let unscaled_inv_freqs = compute_inv_freqs(HEAD_DIM, BASE_FREQ);
    // Sanity for the test itself: every dim where the scaled vector
    // diverges from the unscaled one must produce angle differences
    // that are tiny at s=3 (the largest position in the short test).
    let mut max_inv_diff = 0.0_f32;
    for (s, u) in scaled_inv_freqs.iter().zip(unscaled_inv_freqs.iter()) {
        max_inv_diff = max_inv_diff.max((s - u).abs());
    }
    println!("max inv_freq divergence at scaling boundary = {:.6e}", max_inv_diff);

    let scaled = apply_rope_with_inv_freqs(&input, HEAD_DIM, &scaled_inv_freqs);

    let max_abs_diff = unscaled
        .as_cpu_slice()
        .iter()
        .zip(scaled.as_cpu_slice().iter())
        .map(|(u, s)| (u - s).abs())
        .fold(0.0_f32, f32::max);

    println!(
        "seq={} max abs diff scaled-vs-unscaled = {:.6e}",
        short_seq, max_abs_diff
    );
    // Loose assertion: at small seq the diff is at most a few times
    // 1e-3 because the angle differences propagate through cos/sin
    // with `s` as small as 3.
    assert!(
        max_abs_diff < 0.05,
        "even at seq=4 the scaled RoPE diverged unexpectedly far \
         from unscaled ({:.4}); investigate the high-band identity",
        max_abs_diff
    );
}
