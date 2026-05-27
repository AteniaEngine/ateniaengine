//! **β.2** — tests for the experimental
//! [`TensorStorage::CpuInt8Outlier`] variant: constructor
//! validation, CPU reconstruction, `ensure_cpu` transition, and
//! no-regression assertions against the existing [`TensorStorage::CpuInt8`]
//! behaviour.
//!
//! Lives in `tests/` (integration) rather than inside `tensor.rs`
//! because the existing storage-test pattern (`tensor_storage_test.rs`,
//! `cpu_bf16_storage_test.rs`) puts these as integration tests, and
//! some of the assertions reach into the β.1 quantizer module which
//! is more natural to import as a downstream consumer.

use atenia_engine::tensor::quantizer::{
    self, absmax_per_group_symmetric, decompose_outliers_topk_by_absmax,
};
use atenia_engine::tensor::{DType, Tensor, TensorStorage};

/// Helper: build a small `[K, N]` matrix with a few high-magnitude
/// outlier columns embedded in a "bulk" of small values. The exact
/// numbers are chosen so that the per-column absmax difference
/// between outlier and non-outlier columns is ~3 orders of
/// magnitude, which makes the reconstruction asserts unambiguous.
fn build_outlier_matrix(k: usize, n: usize, outlier_cols: &[usize]) -> Vec<f32> {
    let mut w = vec![0.0_f32; k * n];
    for row in 0..k {
        for col in 0..n {
            let base = ((row * n + col) as f32 * 0.01) - 0.5;
            w[row * n + col] = if outlier_cols.contains(&col) {
                base * 1000.0
            } else {
                base
            };
        }
    }
    w
}

#[test]
fn cpu_int8_outlier_constructor_accepts_valid_decomposition() {
    let w = build_outlier_matrix(16, 8, &[1, 5]);
    let decomp = decompose_outliers_topk_by_absmax(&w, &[16, 8], 8, 2)
        .expect("decomposition must succeed on a well-formed input");
    let t = Tensor::from_outlier_decomposition(decomp);
    assert_eq!(t.shape, vec![16, 8]);
    assert!(matches!(t.storage, TensorStorage::CpuInt8Outlier { .. }));
}

#[test]
fn cpu_int8_outlier_copy_to_cpu_vec_reconstructs_values() {
    // Outlier columns are reconstructed bit-exact via the sidecar;
    // non-outlier columns inherit the per-group INT8 envelope.
    let w = build_outlier_matrix(16, 8, &[0, 7]);
    let decomp = decompose_outliers_topk_by_absmax(&w, &[16, 8], 8, 2).unwrap();
    let t = Tensor::from_outlier_decomposition(decomp);

    let recon = t.copy_to_cpu_vec();
    assert_eq!(recon.len(), 16 * 8);

    for row in 0..16 {
        for &col in &[0_usize, 7] {
            assert_eq!(
                recon[row * 8 + col],
                w[row * 8 + col],
                "outlier column {col} row {row} must reconstruct bit-exact"
            );
        }
    }

    // And the reconstruction must beat plain INT8 on the same input.
    let (q_plain, scales_plain) = absmax_per_group_symmetric(&w, &[16, 8], 8);
    let n = 8;
    let mut int8_recon = vec![0.0_f32; w.len()];
    for idx in 0..w.len() {
        let row = idx / n;
        let col = idx % n;
        let g = row / 8;
        int8_recon[idx] = (q_plain[idx] as f32) * scales_plain[g * n + col];
    }
    let max_err = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    };
    let plain_err = max_err(&int8_recon, &w);
    let outlier_err = max_err(&recon, &w);
    assert!(
        outlier_err * 5.0 < plain_err,
        "expected outlier reconstruction at least 5x better than plain INT8 \
         (plain={plain_err}, outlier={outlier_err})"
    );
}

#[test]
fn cpu_int8_outlier_ensure_cpu_converts_to_f32_cpu_storage() {
    let w = build_outlier_matrix(8, 4, &[2]);
    let decomp = decompose_outliers_topk_by_absmax(&w, &[8, 4], 4, 1).unwrap();
    let mut t = Tensor::from_outlier_decomposition(decomp);

    let expected = t.copy_to_cpu_vec();
    t.ensure_cpu().expect("ensure_cpu must succeed on a CPU variant");
    assert!(matches!(t.storage, TensorStorage::Cpu(_)));
    assert_eq!(t.dtype, DType::F32);
    assert_eq!(t.as_cpu_slice(), expected.as_slice());
}

#[test]
#[should_panic(expected = "β.2 only supports 2D weights")]
fn cpu_int8_outlier_rejects_invalid_shape() {
    Tensor::new_cpu_int8_outlier(
        vec![2, 2, 2], // 3D — rejected
        vec![0_i8; 8],
        vec![1.0_f32; 2],
        8,
        vec![],
        vec![],
    );
}

#[test]
#[should_panic(expected = "outlier_cols contains a duplicate")]
fn cpu_int8_outlier_rejects_duplicate_outlier_cols() {
    // 2x4 weight, group_size=2 → 1 group → scales length = 1 * 4 = 4
    Tensor::new_cpu_int8_outlier(
        vec![2, 4],
        vec![0_i8; 8],
        vec![1.0_f32; 4],
        2,
        vec![1, 1],            // duplicate
        vec![0.0_f32; 2 * 2],  // K * M = 2 * 2
    );
}

#[test]
#[should_panic(expected = "out of range")]
fn cpu_int8_outlier_rejects_out_of_range_outlier_cols() {
    Tensor::new_cpu_int8_outlier(
        vec![2, 4],
        vec![0_i8; 8],
        vec![1.0_f32; 4],
        2,
        vec![4], // out of range (N = 4 → valid indices [0, 4))
        vec![0.0_f32; 2],
    );
}

#[test]
#[should_panic(expected = "outlier_values.len()")]
fn cpu_int8_outlier_rejects_wrong_outlier_value_len() {
    Tensor::new_cpu_int8_outlier(
        vec![2, 4],
        vec![0_i8; 8],
        vec![1.0_f32; 4],
        2,
        vec![0, 1],
        vec![0.0_f32; 3], // should be K * M = 4
    );
}

#[test]
fn cpu_int8_outlier_dtype_remains_int8() {
    // Design decision: β.2 reuses DType::Int8 (the sidecar is
    // metadata about the same conceptual quantised weight, not a
    // new element type). Pin the invariant so a future refactor
    // notices if the dtype tag drifts.
    let w = build_outlier_matrix(4, 4, &[0]);
    let decomp = decompose_outliers_topk_by_absmax(&w, &[4, 4], 4, 1).unwrap();
    let t = Tensor::from_outlier_decomposition(decomp);
    assert_eq!(t.dtype, DType::Int8);
}

#[test]
fn cpu_int8_outlier_does_not_change_existing_cpu_int8_behavior() {
    // Pin: round-tripping a tensor through the plain CpuInt8 path
    // produces exactly the same reconstruction it did before β.2,
    // even when the input would also have produced outlier-heavy
    // columns.
    let w = build_outlier_matrix(16, 8, &[2, 6]);
    let plain = quantizer::quantize_int8_per_group(&w, &[16, 8], 8);
    assert!(matches!(plain.storage, TensorStorage::CpuInt8 { .. }));

    let recon = plain.copy_to_cpu_vec();
    // Reconstruction must match the direct dequant math for CpuInt8.
    let (q, scales) = absmax_per_group_symmetric(&w, &[16, 8], 8);
    let n = 8;
    let mut expected = vec![0.0_f32; w.len()];
    for idx in 0..w.len() {
        let row = idx / n;
        let col = idx % n;
        let g = row / 8;
        expected[idx] = (q[idx] as f32) * scales[g * n + col];
    }
    assert_eq!(recon, expected);
}

#[test]
fn cpu_int8_outlier_as_cpu_slice_panics_decode_on_access_contract() {
    // CpuInt8Outlier is decode-on-access by design (same contract as
    // CpuInt8 and CpuBf16). `as_cpu_slice` must panic with the β.2
    // tag rather than silently expose the i8 buffer as f32.
    let w = build_outlier_matrix(4, 4, &[0]);
    let decomp = decompose_outliers_topk_by_absmax(&w, &[4, 4], 4, 1).unwrap();
    let t = Tensor::from_outlier_decomposition(decomp);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = t.as_cpu_slice();
    }));
    assert!(result.is_err(), "as_cpu_slice must panic on CpuInt8Outlier");
}
