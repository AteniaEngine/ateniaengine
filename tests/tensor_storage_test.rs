//! APX v20 M3-a — tests for the new `TensorStorage` enum and the
//! vendor-neutral accessor API on `Tensor`.
//!
//! Scope: validate behavior of the 7 canonical accessors
//! (`new_cpu`, `set_cpu_data`, `as_cpu_slice`, `as_cpu_slice_mut`,
//! `copy_to_cpu_vec`, `ensure_cpu`, `numel`, `storage`) and the
//! pre-0.20 compatibility shims (`data`, `data_mut`, `num_elements`).
//!
//! M3-a only ships `TensorStorage::Cpu`; subsequent sub-milestones
//! add other backends. The tests below exercise the CPU path and
//! the invariants that must hold regardless of the number of variants.

use atenia_engine::tensor::{Tensor, TensorStorage};

#[test]
fn test_cpu_storage_roundtrip() {
    let t = Tensor::new_cpu(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    assert_eq!(t.as_cpu_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        t.copy_to_cpu_vec(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "copy_to_cpu_vec must return the same values"
    );
}

#[test]
fn test_numel() {
    // Rank-2 tensor: numel = product of shape.
    let t = Tensor::new_cpu(vec![2, 3], vec![0.0; 6]);
    assert_eq!(t.numel(), 6);

    // Rank-1 tensor.
    let t1 = Tensor::new_cpu(vec![5], vec![0.0; 5]);
    assert_eq!(t1.numel(), 5);

    // Rank-4 NCHW-shaped tensor (common in M1 conv ops).
    let t4 = Tensor::new_cpu(vec![1, 3, 4, 4], vec![0.0; 48]);
    assert_eq!(t4.numel(), 48);
}

#[test]
fn test_storage_accessor() {
    let t = Tensor::new_cpu(vec![3], vec![7.0, 8.0, 9.0]);

    // In M3-a only `Cpu` exists; the match is exhaustive.
    match t.storage() {
        TensorStorage::Cpu(v) => {
            assert_eq!(v, &vec![7.0, 8.0, 9.0]);
        }
    }
}

#[test]
fn test_set_cpu_data() {
    let mut t = Tensor::new_cpu(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

    t.set_cpu_data(vec![10.0, 20.0, 30.0, 40.0]);

    assert_eq!(t.as_cpu_slice(), &[10.0, 20.0, 30.0, 40.0]);
    assert_eq!(t.shape, vec![2, 2], "set_cpu_data must not touch shape");
    assert_eq!(t.numel(), 4);
}

#[test]
fn test_ensure_cpu_noop() {
    let mut t = Tensor::new_cpu(vec![4], vec![1.0, 2.0, 3.0, 4.0]);

    // ensure_cpu on a CPU-resident tensor is a no-op in M3-a.
    t.ensure_cpu();

    // Contents are preserved bit-for-bit.
    assert_eq!(t.as_cpu_slice(), &[1.0, 2.0, 3.0, 4.0]);
    // And chaining returns &mut Self for fluent usage.
    t.ensure_cpu().as_cpu_slice_mut()[0] = 99.0;
    assert_eq!(t.as_cpu_slice()[0], 99.0);
}

#[test]
fn test_clone_preserves_storage() {
    let t = Tensor::new_cpu(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let c = t.clone();

    // The clone has an equivalent Cpu storage with the same values.
    match c.storage() {
        TensorStorage::Cpu(v) => {
            assert_eq!(v, &vec![1.0, 2.0, 3.0, 4.0]);
        }
    }

    // And the clone is independent: mutating it does not touch the
    // original.
    let mut c_mut = c;
    c_mut.as_cpu_slice_mut()[0] = 999.0;
    assert_eq!(t.as_cpu_slice()[0], 1.0, "original must be unchanged");
    assert_eq!(c_mut.as_cpu_slice()[0], 999.0);
}
