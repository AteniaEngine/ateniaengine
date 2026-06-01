//! **MOE-FULL-9** — Grouped-Query Attention (GQA) support for the experimental
//! MoE transformer, via load-time K/V weight tiling.
//!
//! Mixtral-8x7B (and most modern MoE models) use GQA: there are fewer K/V heads
//! than query heads (`num_key_value_heads < num_attention_heads`), so each K/V
//! head is shared by `kv_groups = num_attention_heads / num_key_value_heads`
//! query heads. HuggingFace implements this at runtime with `repeat_kv`, which
//! expands `[batch, n_kv, seq, head_dim]` to `[batch, n_heads, seq, head_dim]`
//! by repeating each K/V head `kv_groups` times: query head `h` attends to K/V
//! head `h / kv_groups`.
//!
//! The productive dense path resolves GQA "by a load-time K/V tile" (see
//! `nn::llama::builder`). This module does the same for the MoE path: it tiles
//! the **K and V projection weights** so they project directly to
//! `n_heads * head_dim` rows in the same head order HF's `repeat_kv` produces.
//! The MOE-FULL-6/7 MHA attention graph (`build_tiny_mixtral_graph`,
//! `build_tiny_mixtral_prefill`, `build_tiny_mixtral_decode`) is then reused
//! **unchanged** — no new graph op, no attention-topology change.
//!
//! ## Tiling contract
//!
//! `w_kv` is row-major `[n_kv_heads * head_dim, d_model]`, head `j` occupying
//! rows `[j*head_dim, (j+1)*head_dim)`. The tiled output is
//! `[n_heads * head_dim, d_model]` where output head `h` copies input head
//! `h / kv_groups` — exactly HF's `repeat_kv` ordering. With
//! `n_kv_heads == n_heads` (MHA) the tiling is the identity.
//!
//! Experimental, CPU-only, test/opt-in only. No loader / runtime / Adapter
//! Toolkit / CLI / CUDA change; the MOE-2 fail-loud guard is unchanged.

/// Error from [`tile_kv_weight`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GqaError {
    /// `n_kv_heads` is 0, or does not divide `n_heads`.
    BadHeadCounts { n_heads: usize, n_kv_heads: usize },
    /// `w_kv.len()` is not `n_kv_heads * head_dim * d_model`.
    BadShape { expected: usize, actual: usize },
}

impl std::fmt::Display for GqaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GqaError::BadHeadCounts { n_heads, n_kv_heads } => write!(
                f,
                "gqa: num_attention_heads ({n_heads}) must be a positive multiple of \
                 num_key_value_heads ({n_kv_heads})"
            ),
            GqaError::BadShape { expected, actual } => {
                write!(f, "gqa: K/V weight has {actual} elems, expected {expected}")
            }
        }
    }
}

impl std::error::Error for GqaError {}

/// Number of query heads each K/V head serves. `n_heads / n_kv_heads`.
pub fn kv_groups(n_heads: usize, n_kv_heads: usize) -> Result<usize, GqaError> {
    if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
        return Err(GqaError::BadHeadCounts { n_heads, n_kv_heads });
    }
    Ok(n_heads / n_kv_heads)
}

/// Tile a K or V projection weight from `[n_kv_heads*head_dim, d_model]` to
/// `[n_heads*head_dim, d_model]`, replicating HF `repeat_kv`: output head `h`
/// is a copy of input head `h / kv_groups`. With `n_kv_heads == n_heads` this
/// is an exact clone (MHA — the identity).
///
/// Row-major layout throughout. Returns the tiled buffer (length
/// `n_heads * head_dim * d_model`).
pub fn tile_kv_weight(
    w_kv: &[f32],
    n_kv_heads: usize,
    n_heads: usize,
    head_dim: usize,
    d_model: usize,
) -> Result<Vec<f32>, GqaError> {
    let groups = kv_groups(n_heads, n_kv_heads)?;
    let expected = n_kv_heads * head_dim * d_model;
    if w_kv.len() != expected {
        return Err(GqaError::BadShape { expected, actual: w_kv.len() });
    }
    let head_rows = head_dim * d_model; // elements per head block
    let mut out = Vec::with_capacity(n_heads * head_rows);
    for h in 0..n_heads {
        let src_head = h / groups; // HF repeat_kv: q head h ← kv head h/groups
        let base = src_head * head_rows;
        out.extend_from_slice(&w_kv[base..base + head_rows]);
    }
    Ok(out)
}

/// Convenience: tile K/V weights to MHA shape if needed. Returns the original
/// (cloned) buffer unchanged when `n_kv_heads == n_heads`. Pure helper for the
/// load-time path that prepares `TinyDecoderWeights`.
pub fn to_mha_kv(
    w_kv: &[f32],
    n_kv_heads: usize,
    n_heads: usize,
    head_dim: usize,
    d_model: usize,
) -> Result<Vec<f32>, GqaError> {
    if n_kv_heads == n_heads {
        // MHA: nothing to tile.
        let expected = n_heads * head_dim * d_model;
        if w_kv.len() != expected {
            return Err(GqaError::BadShape { expected, actual: w_kv.len() });
        }
        return Ok(w_kv.to_vec());
    }
    tile_kv_weight(w_kv, n_kv_heads, n_heads, head_dim, d_model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_groups_math() {
        assert_eq!(kv_groups(4, 2).unwrap(), 2);
        assert_eq!(kv_groups(8, 8).unwrap(), 1); // MHA
        assert_eq!(kv_groups(32, 8).unwrap(), 4); // Mixtral-8x7B
        assert!(kv_groups(4, 0).is_err());
        assert!(kv_groups(4, 3).is_err()); // 3 does not divide 4
    }

    #[test]
    fn tile_replicates_kv_heads_in_hf_order() {
        // 2 kv heads, head_dim 2, d_model 1 → easy to read.
        // kv head 0 rows = [10, 11], kv head 1 rows = [20, 21].
        let w_kv = vec![10.0, 11.0, 20.0, 21.0];
        let tiled = tile_kv_weight(&w_kv, 2, 4, 2, 1).unwrap();
        // groups = 2: head0←kv0, head1←kv0, head2←kv1, head3←kv1.
        assert_eq!(tiled, vec![10.0, 11.0, 10.0, 11.0, 20.0, 21.0, 20.0, 21.0]);
    }

    #[test]
    fn mha_tiling_is_identity() {
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // n_kv == n_heads == 2, head_dim 2, d_model 2.
        let same = to_mha_kv(&w, 2, 2, 2, 2).unwrap();
        assert_eq!(same, w);
        // Explicit tile with groups=1 also identity.
        let same2 = tile_kv_weight(&w, 2, 2, 2, 2).unwrap();
        assert_eq!(same2, w);
    }

    #[test]
    fn rejects_bad_shape() {
        let w = vec![0.0; 10];
        assert!(matches!(
            tile_kv_weight(&w, 2, 4, 8, 32),
            Err(GqaError::BadShape { .. })
        ));
    }

    #[test]
    fn mixtral_8x7b_shape_tiling() {
        // n_kv=8, n_heads=32, head_dim=128, d_model=4096 — just the shape math.
        let n_kv = 8;
        let n_heads = 32;
        let head_dim = 4;
        let d_model = 2;
        let w_kv = vec![0.5_f32; n_kv * head_dim * d_model];
        let tiled = tile_kv_weight(&w_kv, n_kv, n_heads, head_dim, d_model).unwrap();
        assert_eq!(tiled.len(), n_heads * head_dim * d_model);
    }
}
