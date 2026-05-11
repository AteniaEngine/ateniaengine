//! Unit tests for `TensorStorage::CpuBf16` and the BF16 helpers
//! (M4.7.2.a).
//!
//! Covers the minimum behavioural contract introduced by M4.7.2.a:
//!
//! - `bf16_bits_to_f32` and `f32_to_bf16_bits` are inverse on the BF16
//!   subset of F32 (values whose lower 16 mantissa bits are zero).
//! - `Tensor::new_cpu_bf16` constructs a `CpuBf16`-backed tensor with
//!   the right dtype and shape.
//! - `copy_to_cpu_vec` decodes BF16 → F32 on access and is non-mutating.
//! - `ensure_cpu` materialises BF16 → F32 by transitioning the storage
//!   variant in place.
//! - `as_cpu_slice` panics on `CpuBf16` storage with a clear message
//!   pointing callers to `copy_to_cpu_vec` or `ensure_cpu`.
//! - `set_cpu_bf16_bits` updates both the storage and the dtype.

use atenia_engine::tensor::tensor::{
    DType, Tensor, TensorStorage, bf16_bits_to_f32, f32_to_bf16_bits,
};

/// A handful of representative F32 values whose lower 16 mantissa
/// bits are already zero — every value here round-trips through BF16
/// bit-exact.
fn bf16_exact_samples() -> Vec<f32> {
    vec![
        0.0_f32, -0.0_f32, 1.0, -1.0, 0.5, -0.25, 2.0, -16.0, 0.0078125, // 2^-7, exact BF16
        -1024.0,
    ]
}

#[test]
fn bf16_bits_round_trip_is_bit_exact_on_bf16_subset() {
    for v in bf16_exact_samples() {
        let bits = f32_to_bf16_bits(v);
        let back = bf16_bits_to_f32(bits);
        assert_eq!(
            v.to_bits(),
            back.to_bits(),
            "bf16 round-trip lost bits for {} (orig 0x{:08X}, back 0x{:08X})",
            v,
            v.to_bits(),
            back.to_bits()
        );
    }
}

#[test]
fn f32_to_bf16_truncates_lower_mantissa() {
    // Pi has non-zero lower mantissa bits; the down-convert drops
    // them, and the round-trip is *not* bit-exact but stays within
    // the BF16 grid (max relative error ~2^-8).
    let pi = std::f32::consts::PI;
    let bits = f32_to_bf16_bits(pi);
    let back = bf16_bits_to_f32(bits);
    assert_ne!(
        pi.to_bits(),
        back.to_bits(),
        "lossy down-convert must alter mantissa for pi"
    );
    let rel_err = (pi - back).abs() / pi.abs();
    assert!(
        rel_err < 1.0 / 256.0,
        "BF16 relative error on pi = {} exceeds 2^-8",
        rel_err
    );
    // The quantised value is exactly recoverable on a second
    // round-trip (idempotent on the BF16 grid).
    let bits2 = f32_to_bf16_bits(back);
    assert_eq!(bits, bits2, "second round-trip must be bit-exact");
}

#[test]
fn new_cpu_bf16_constructs_tensor_with_bf16_dtype() {
    // Encode a small fixed pattern: F32 values that all live on the
    // BF16 grid, so we can compare decoded output bit-exact.
    let originals = bf16_exact_samples();
    let bits: Vec<u16> = originals.iter().copied().map(f32_to_bf16_bits).collect();
    let shape = vec![bits.len()];
    let t = Tensor::new_cpu_bf16(shape.clone(), bits.clone());

    assert_eq!(t.shape, shape);
    assert_eq!(t.numel(), bits.len());
    assert_eq!(t.dtype, DType::BF16);
    assert!(matches!(t.storage(), TensorStorage::CpuBf16(_)));
}

#[test]
fn copy_to_cpu_vec_decodes_on_access_without_mutating_storage() {
    let originals = bf16_exact_samples();
    let bits: Vec<u16> = originals.iter().copied().map(f32_to_bf16_bits).collect();
    let t = Tensor::new_cpu_bf16(vec![bits.len()], bits);

    let decoded = t.copy_to_cpu_vec();
    assert_eq!(decoded.len(), originals.len());
    for (i, (decoded_v, orig)) in decoded.iter().zip(originals.iter()).enumerate() {
        assert_eq!(
            decoded_v.to_bits(),
            orig.to_bits(),
            "decoded[{}] = {} != original {}",
            i,
            decoded_v,
            orig
        );
    }

    // Storage variant is unchanged after the access.
    assert!(matches!(t.storage(), TensorStorage::CpuBf16(_)));

    // Calling decode again returns a fresh allocation each time
    // (no-cache semantics).
    let decoded2 = t.copy_to_cpu_vec();
    assert_eq!(decoded, decoded2);
}

#[test]
fn ensure_cpu_transitions_bf16_to_f32_storage() {
    let originals = bf16_exact_samples();
    let bits: Vec<u16> = originals.iter().copied().map(f32_to_bf16_bits).collect();
    let mut t = Tensor::new_cpu_bf16(vec![bits.len()], bits);

    t.ensure_cpu().expect("ensure_cpu on CpuBf16 must succeed");

    // Storage variant flipped; data still equals the originals.
    assert!(matches!(t.storage(), TensorStorage::Cpu(_)));
    // M4.7.2.c bugfix: dtype tag must also flip to F32 so
    // downstream ops that read `dtype` to construct outputs or
    // to enforce mixed-precision rejection see a consistent F32
    // tensor.
    assert_eq!(
        t.dtype,
        DType::F32,
        "ensure_cpu must flip dtype to F32 on CpuBf16 → Cpu"
    );
    let view = t.as_cpu_slice();
    assert_eq!(view.len(), originals.len());
    for (got, want) in view.iter().zip(originals.iter()) {
        assert_eq!(got.to_bits(), want.to_bits());
    }

    // Calling ensure_cpu again is a no-op on the now-Cpu variant.
    t.ensure_cpu().unwrap();
    assert!(matches!(t.storage(), TensorStorage::Cpu(_)));
}

#[test]
#[should_panic(expected = "CpuBf16 storage requires decode-on-access")]
fn as_cpu_slice_panics_on_bf16_storage() {
    let bits: Vec<u16> = vec![f32_to_bf16_bits(1.0); 4];
    let t = Tensor::new_cpu_bf16(vec![4], bits);
    let _ = t.as_cpu_slice();
}

#[test]
fn set_cpu_bf16_bits_updates_storage_and_dtype() {
    // Start from an F32 tensor, then swap to BF16 in place.
    let mut t = Tensor::new_cpu(vec![4], vec![1.0_f32, 2.0, 3.0, 4.0]);
    assert_eq!(t.dtype, DType::F32);

    let new_bits: Vec<u16> = (1..=4_u16).map(|i| f32_to_bf16_bits(i as f32)).collect();
    t.set_cpu_bf16_bits(new_bits.clone());

    assert_eq!(t.dtype, DType::BF16);
    match t.storage() {
        TensorStorage::CpuBf16(b) => assert_eq!(b, &new_bits),
        other => panic!("expected CpuBf16, got {:?}", other),
    }

    let decoded = t.copy_to_cpu_vec();
    assert_eq!(decoded, vec![1.0, 2.0, 3.0, 4.0]);
}
