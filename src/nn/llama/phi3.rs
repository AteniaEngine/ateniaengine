//! **M11.B step 3** — Phi-3 / Phi-3.5 weight-splitting utilities.
//!
//! Microsoft Phi-3 / Phi-3.5 ship two fused weight tensors that
//! Atenia's existing `LoadTransform` system cannot natively
//! decompose into the per-projection tensors the Llama-family
//! transformer block expects:
//!
//! - `qkv_proj` of shape
//!   `[hidden, (n_heads_q + 2 * n_heads_kv) * head_dim]` →
//!   q_proj, k_proj, v_proj (each `[hidden, n_heads_x * head_dim]`)
//! - `gate_up_proj` of shape `[hidden, 2 * intermediate]` →
//!   gate_proj, up_proj (each `[hidden, intermediate]`)
//!
//! `LoadTransform` is a strict 1-source → 1-target pipeline
//! (`WeightMapping { node_id, transforms }`). Rather than widen
//! that abstraction with a new 1-to-N concept (invasive changes
//! across `weight_mapper.rs` plus its BF16 / Cuda / Disk slow-path
//! mirrors), this module exposes pure split functions that the
//! Phi-3 builder calls inline before the regular load. The split
//! happens once on the host F32 buffer; the resulting per-
//! projection slices feed `Tensor::new_cpu` calls that hydrate
//! q_proj / k_proj / v_proj / gate_proj / up_proj parameter
//! nodes directly. The standard `WeightMapper` then sees those
//! parameter slots as already-populated and skips them.
//!
//! Every function in this module is a pure value transform — no
//! tensor / graph machinery, no I/O. Invariants are validated up
//! front and surface as `String` errors that the caller wraps
//! into a proper `LoaderError`.

/// **M11.B step 3** — split a fused `qkv_proj` weight tensor
/// along its last (output-features) axis into three per-
/// projection tensors `(q, k, v)`.
///
/// ## Layout contract
///
/// HuggingFace stores `qkv_proj` in the linear layer's native
/// row-major form `[out_features, in_features]` where
/// `out_features = (n_heads_q + 2 * n_heads_kv) * head_dim`.
/// This function consumes the tensor in that layout and
/// returns three slices each of shape
/// `[n_heads_x * head_dim, in_features]`.
///
/// **Note**: Atenia's loader applies a `LoadTransform::Transpose2D`
/// to projection weights to reach `[in_features, out_features]`
/// (Atenia's matmul convention). This function is meant to run
/// **before** the transpose — so it operates on the HF layout
/// where the fused dimension is `out_features` (rows). The
/// caller can either transpose post-split or pre-split; both
/// work because slicing rows in `[out, in]` is equivalent to
/// slicing columns in `[in, out]` after transpose, modulo
/// memory layout.
///
/// ## Phi-3.5 Mini specifics
///
/// Phi-3.5 Mini: hidden = 3072, n_heads_q = n_heads_kv = 32,
/// head_dim = 96. Fused shape = `[(32 + 64) * 96, 3072] =
/// [9216, 3072]`. Output q / k / v shapes = `[3072, 3072]` each.
///
/// ## Returns
///
/// `(q, k, v)` — three `Vec<f32>` buffers in the input's row-
/// major layout. The output shapes are
/// `[n_heads_q * head_dim, in_features]`,
/// `[n_heads_kv * head_dim, in_features]`,
/// `[n_heads_kv * head_dim, in_features]`.
///
/// ## Errors
///
/// - `fused.len() != fused_shape[0] * fused_shape[1]`
/// - `fused_shape.len() != 2`
/// - `fused_shape[0] != (n_heads_q + 2 * n_heads_kv) * head_dim`
pub fn split_fused_qkv(
    fused: &[f32],
    fused_shape: &[usize],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), String> {
    if fused_shape.len() != 2 {
        return Err(format!(
            "split_fused_qkv: expected rank-2 tensor, got shape {fused_shape:?}"
        ));
    }
    let out_features = fused_shape[0];
    let in_features = fused_shape[1];
    if fused.len() != out_features * in_features {
        return Err(format!(
            "split_fused_qkv: data length {} does not match shape {fused_shape:?} (numel {})",
            fused.len(),
            out_features * in_features
        ));
    }
    let expected_out = (n_heads_q + 2 * n_heads_kv) * head_dim;
    if out_features != expected_out {
        return Err(format!(
            "split_fused_qkv: expected out_features = (n_heads_q + 2*n_heads_kv) * head_dim = {expected_out}, got {out_features}"
        ));
    }
    let q_rows = n_heads_q * head_dim;
    let kv_rows = n_heads_kv * head_dim;
    // Row ranges in [out, in] layout.
    //   q : 0                 .. q_rows
    //   k : q_rows             .. q_rows + kv_rows
    //   v : q_rows + kv_rows   .. q_rows + 2 * kv_rows
    let q_end = q_rows;
    let k_end = q_rows + kv_rows;
    let v_end = q_rows + 2 * kv_rows;
    let q = fused[0 .. q_end * in_features].to_vec();
    let k = fused[q_end * in_features .. k_end * in_features].to_vec();
    let v = fused[k_end * in_features .. v_end * in_features].to_vec();
    Ok((q, k, v))
}

/// **M11.B step 3** — split a fused `gate_up_proj` weight tensor
/// along its last (output-features) axis into two per-projection
/// tensors `(gate, up)`. Phi-3.5 Mini concatenates the two FFN
/// up-projections (gate_proj + up_proj) into a single weight
/// `gate_up_proj` of shape `[2 * intermediate, hidden]`; this
/// function halves the row count.
///
/// ## Returns
///
/// `(gate, up)` — two `Vec<f32>` buffers, each of shape
/// `[intermediate, in_features]` in the input's row-major
/// layout. The first half goes to gate_proj, the second to
/// up_proj — the order is the HF convention used by Phi-3.
///
/// ## Errors
///
/// - `fused_shape.len() != 2`
/// - `fused_shape[0] % 2 != 0`
/// - `fused.len() != fused_shape[0] * fused_shape[1]`
pub fn split_fused_gate_up(
    fused: &[f32],
    fused_shape: &[usize],
) -> Result<(Vec<f32>, Vec<f32>), String> {
    if fused_shape.len() != 2 {
        return Err(format!(
            "split_fused_gate_up: expected rank-2 tensor, got shape {fused_shape:?}"
        ));
    }
    let out_features = fused_shape[0];
    let in_features = fused_shape[1];
    if out_features % 2 != 0 {
        return Err(format!(
            "split_fused_gate_up: out_features {out_features} is not divisible by 2"
        ));
    }
    if fused.len() != out_features * in_features {
        return Err(format!(
            "split_fused_gate_up: data length {} does not match shape {fused_shape:?} (numel {})",
            fused.len(),
            out_features * in_features
        ));
    }
    let half = out_features / 2;
    let split_idx = half * in_features;
    let gate = fused[0 .. split_idx].to_vec();
    let up = fused[split_idx ..].to_vec();
    Ok((gate, up))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Phi-3.5 Mini production shape** — `qkv_proj` row-count
    /// equals `(32 + 32 + 32) * 96 = 9216`. With `n_heads_q =
    /// n_heads_kv = 32` and `head_dim = 96` the resulting q / k
    /// / v slices each have row-count `32 * 96 = 3072`.
    #[test]
    fn split_qkv_phi35_mini_shape() {
        let n_q = 32usize;
        let n_kv = 32usize;
        let head_dim = 96usize;
        let in_features = 3072usize;
        let out_features = (n_q + 2 * n_kv) * head_dim;
        let fused: Vec<f32> = (0..out_features * in_features)
            .map(|i| i as f32 * 1e-4)
            .collect();
        let (q, k, v) = split_fused_qkv(
            &fused,
            &[out_features, in_features],
            n_q,
            n_kv,
            head_dim,
        )
        .expect("split_fused_qkv must accept the Phi-3.5 Mini shape");
        assert_eq!(q.len(), n_q * head_dim * in_features);
        assert_eq!(k.len(), n_kv * head_dim * in_features);
        assert_eq!(v.len(), n_kv * head_dim * in_features);
        // First element of q must equal the first element of fused.
        assert_eq!(q[0], fused[0]);
        // First element of k starts after q's rows.
        let k_offset = n_q * head_dim * in_features;
        assert_eq!(k[0], fused[k_offset]);
        // First element of v starts after q + k rows.
        let v_offset = (n_q + n_kv) * head_dim * in_features;
        assert_eq!(v[0], fused[v_offset]);
        // Last element of v is the last element of fused.
        assert_eq!(v[v.len() - 1], fused[fused.len() - 1]);
    }

    /// GQA case: n_kv smaller than n_q. Verify the split still
    /// produces correctly-sized slices and the row offsets match.
    #[test]
    fn split_qkv_gqa_case() {
        let n_q = 32usize;
        let n_kv = 8usize;
        let head_dim = 64usize;
        let in_features = 2048usize;
        let out_features = (n_q + 2 * n_kv) * head_dim;
        let fused: Vec<f32> = (0..out_features * in_features)
            .map(|i| i as f32)
            .collect();
        let (q, k, v) = split_fused_qkv(
            &fused,
            &[out_features, in_features],
            n_q,
            n_kv,
            head_dim,
        )
        .expect("split should succeed");
        assert_eq!(q.len(), 32 * 64 * 2048);
        assert_eq!(k.len(), 8 * 64 * 2048);
        assert_eq!(v.len(), 8 * 64 * 2048);
        // Spot-check element values at boundaries.
        assert_eq!(q[q.len() - 1], fused[q.len() - 1]);
        assert_eq!(k[0], fused[q.len()]);
        assert_eq!(v[0], fused[q.len() + k.len()]);
    }

    /// Shape mismatch: data length inconsistent with declared shape.
    #[test]
    fn split_qkv_rejects_inconsistent_data_length() {
        let fused: Vec<f32> = vec![0.0; 100];
        let err = split_fused_qkv(&fused, &[200, 1], 1, 1, 1)
            .expect_err("inconsistent data length must fail");
        assert!(err.contains("data length"), "got: {err}");
    }

    /// Out-features mismatch: shape claims 100 rows but the
    /// (n_q + 2*n_kv) * head_dim equation gives a different
    /// number.
    #[test]
    fn split_qkv_rejects_unexpected_out_features() {
        let fused: Vec<f32> = vec![0.0; 100 * 8];
        let err = split_fused_qkv(&fused, &[100, 8], 32, 32, 96)
            .expect_err("out_features mismatch must fail");
        assert!(err.contains("out_features"), "got: {err}");
    }

    /// **Phi-3.5 Mini production shape** — `gate_up_proj` has
    /// row-count `2 * 8192 = 16384` and column-count 3072. The
    /// half-split produces two `[8192, 3072]` slices.
    #[test]
    fn split_gate_up_phi35_mini_shape() {
        let intermediate = 8192usize;
        let in_features = 3072usize;
        let out_features = 2 * intermediate;
        let fused: Vec<f32> = (0..out_features * in_features)
            .map(|i| i as f32 * 1e-5)
            .collect();
        let (gate, up) = split_fused_gate_up(&fused, &[out_features, in_features])
            .expect("split must succeed");
        assert_eq!(gate.len(), intermediate * in_features);
        assert_eq!(up.len(), intermediate * in_features);
        assert_eq!(gate[0], fused[0]);
        assert_eq!(up[0], fused[intermediate * in_features]);
        assert_eq!(up[up.len() - 1], fused[fused.len() - 1]);
    }

    /// Odd row count must fail — the half-split is undefined.
    #[test]
    fn split_gate_up_rejects_odd_out_features() {
        let fused: Vec<f32> = vec![0.0; 7 * 4];
        let err = split_fused_gate_up(&fused, &[7, 4])
            .expect_err("odd out_features must fail");
        assert!(err.contains("divisible by 2"), "got: {err}");
    }

    /// Round-trip: re-concatenating the splits along the row
    /// axis must reproduce the original buffer bit-exactly.
    #[test]
    fn split_qkv_round_trip_concat_matches_input() {
        let n_q = 8usize;
        let n_kv = 4usize;
        let head_dim = 16usize;
        let in_features = 32usize;
        let out_features = (n_q + 2 * n_kv) * head_dim;
        let fused: Vec<f32> = (0..out_features * in_features)
            .map(|i| (i as f32).sin())
            .collect();
        let (q, k, v) = split_fused_qkv(
            &fused,
            &[out_features, in_features],
            n_q,
            n_kv,
            head_dim,
        )
        .expect("split should succeed");
        let mut concatenated = q;
        concatenated.extend_from_slice(&k);
        concatenated.extend_from_slice(&v);
        assert_eq!(concatenated.len(), fused.len());
        for (i, (a, b)) in concatenated.iter().zip(fused.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "split round-trip mismatch at index {i}: {a} vs {b}"
            );
        }
    }
}
