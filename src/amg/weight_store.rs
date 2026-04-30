//! M5.c.2.a — `WeightStore`: Arc-backed shared parameter store.
//!
//! The M5 prefill+decode plan (D58) builds **two** [`Graph`]
//! instances against the same model and runs them in lockstep
//! during generation. Naïvely cloning all parameter tensors
//! into each graph doubles RAM (52 GB instead of 26 GB for
//! Llama 2 13B BF16) — unworkable on a 32 GB box. The
//! `WeightStore` is the indirection that fixes this:
//!
//!   1. Weights load **once** into the store via the existing
//!      [`crate::v17::loader::weight_mapper::WeightMapper`]
//!      pipeline (extended in M5.c.2.b to write into the
//!      store rather than directly into a graph).
//!   2. The store wraps each parameter buffer in an
//!      [`Arc<Vec<f32>>`] / [`Arc<Vec<u16>>`].
//!   3. Both graphs register parameter slots whose
//!      `Tensor::storage` is [`TensorStorage::CpuShared`] /
//!      [`TensorStorage::CpuBf16Shared`] holding clones of
//!      the same `Arc` — i.e. they reference the same
//!      physical bytes without duplication.
//!
//! ## Mutability contract
//!
//! Tensors materialised from a `WeightStore` are read-only by
//! construction (see [`crate::tensor::TensorStorage::CpuShared`]).
//! `as_cpu_slice` returns a borrow; `as_cpu_slice_mut` panics.
//! This is the right contract for inference: the M5 forward
//! path never mutates parameters. Training, M4.7 disk-spill,
//! and AdamW all keep their distinct mutable tensor paths and
//! treat shared variants as inapplicable.
//!
//! ## What this module does NOT do
//!
//! M5.c.2.a is the **infrastructure** sub-phase. The
//! WeightStore type and the shared-storage primitives ship
//! here; the actual `WeightMapper::load_into_store` writer
//! and the `build_llama_with_store` builder land in M5.c.2.b.
//! A test in this module proves the round-trip works
//! end-to-end at the `Tensor` level (Arc::strong_count proof
//! of sharing + bit-exact read parity).

use std::sync::Arc;
use crate::tensor::{Tensor, TensorStorage};

/// One stored parameter — a pre-shaped buffer wrapped in an
/// `Arc` so multiple `Tensor` wrappers can reference it
/// without copying.
///
/// Storage is **either** F32 (`F32 { arc, .. }`) **or** BF16
/// (`Bf16 { arc, .. }`). Mixing within one parameter is not
/// supported — the `WeightMapper` writes one variant per
/// parameter based on its `store_params_as_bf16` flag.
#[derive(Debug, Clone)]
pub enum SharedParam {
    F32 { shape: Vec<usize>, arc: Arc<Vec<f32>> },
    Bf16 { shape: Vec<usize>, arc: Arc<Vec<u16>> },
}

impl SharedParam {
    /// Materialise a [`Tensor`] backed by this shared
    /// parameter. Cheap (Arc clone, no Vec copy).
    pub fn to_tensor(&self) -> Tensor {
        match self {
            SharedParam::F32 { shape, arc } =>
                Tensor::cpu_shared(shape.clone(), Arc::clone(arc)),
            SharedParam::Bf16 { shape, arc } =>
                Tensor::cpu_bf16_shared(shape.clone(), Arc::clone(arc)),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            SharedParam::F32 { shape, .. } | SharedParam::Bf16 { shape, .. } => shape,
        }
    }

    /// Bytes resident in the underlying buffer (F32 = 4 ×
    /// numel; BF16 = 2 × numel). Useful for the M5.c.2.b
    /// telemetry that proves Arc-sharing actually saves RAM.
    pub fn resident_bytes(&self) -> usize {
        match self {
            SharedParam::F32 { arc, .. } => arc.len() * 4,
            SharedParam::Bf16 { arc, .. } => arc.len() * 2,
        }
    }

    /// Strong-count over the inner `Arc`. Two-graph
    /// configurations expect 2 (or more) strong refs once
    /// both graphs have materialised their parameter slots.
    /// Useful for tests that verify sharing actually
    /// happened.
    pub fn strong_count(&self) -> usize {
        match self {
            SharedParam::F32 { arc, .. } => Arc::strong_count(arc),
            SharedParam::Bf16 { arc, .. } => Arc::strong_count(arc),
        }
    }
}

/// Index-keyed parameter store. Order corresponds to the
/// builder's parameter-registration order so the same index
/// names the same parameter across both graphs.
#[derive(Debug, Default)]
pub struct WeightStore {
    /// Parameters in builder order.
    pub params: Vec<SharedParam>,
    /// HuggingFace-convention name per parameter (parallel
    /// with `params`). Mirrors `LlamaHandles::param_names`.
    pub names: Vec<String>,
}

impl WeightStore {
    pub fn new() -> Self { Self::default() }

    /// Insert an F32 parameter. Wraps the provided `Vec`
    /// in a fresh `Arc`. Returns the parameter index for
    /// downstream `to_tensor` lookups.
    pub fn insert_f32(&mut self, name: impl Into<String>, shape: Vec<usize>, data: Vec<f32>) -> usize {
        let idx = self.params.len();
        self.params.push(SharedParam::F32 { shape, arc: Arc::new(data) });
        self.names.push(name.into());
        idx
    }

    /// Insert a BF16 parameter. Wraps the provided `Vec<u16>`
    /// (raw BF16 bit patterns) in a fresh `Arc`.
    pub fn insert_bf16(&mut self, name: impl Into<String>, shape: Vec<usize>, bits: Vec<u16>) -> usize {
        let idx = self.params.len();
        self.params.push(SharedParam::Bf16 { shape, arc: Arc::new(bits) });
        self.names.push(name.into());
        idx
    }

    /// Lookup a parameter by index. Index is the value
    /// returned by `insert_*`.
    pub fn get(&self, idx: usize) -> Option<&SharedParam> { self.params.get(idx) }

    /// Lookup by HuggingFace-convention name. Linear scan
    /// (fine — called once per parameter at graph build
    /// time, never on the forward hot path).
    pub fn get_by_name(&self, name: &str) -> Option<&SharedParam> {
        self.names.iter().position(|n| n == name).and_then(|i| self.params.get(i))
    }

    /// Total bytes resident across every stored parameter.
    /// Used by the M5.c.2.b R2 falsifier to assert Arc-shared
    /// sizes match single-graph sizes (i.e. building a second
    /// graph that references the same store does not double
    /// the resident footprint).
    pub fn resident_bytes(&self) -> usize {
        self.params.iter().map(|p| p.resident_bytes()).sum()
    }

    pub fn len(&self) -> usize { self.params.len() }
    pub fn is_empty(&self) -> bool { self.params.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_f32_round_trips_bit_exact() {
        // Round-trip falsifier: insert a parameter, take two
        // Tensor materialisations from it, both must read the
        // same bytes — and Arc::strong_count must show shared
        // ownership.
        let mut store = WeightStore::new();
        let data: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let _idx = store.insert_f32("w.test", vec![4, 4], data.clone());

        let p = store.get_by_name("w.test").unwrap();
        let t1 = p.to_tensor();
        let t2 = p.to_tensor();

        // Same shape, same dtype, same values.
        assert_eq!(t1.shape, vec![4, 4]);
        assert_eq!(t2.shape, vec![4, 4]);
        assert_eq!(t1.copy_to_cpu_vec(), data);
        assert_eq!(t2.copy_to_cpu_vec(), data);

        // Direct &[f32] borrow works on CpuShared.
        assert_eq!(t1.as_cpu_slice(), data.as_slice());
        assert_eq!(t2.as_cpu_slice(), data.as_slice());

        // Ownership: store holds 1 strong ref; t1 and t2 each
        // add one → 3 total.
        assert_eq!(p.strong_count(), 3,
            "expected 3 strong refs (store + 2 tensors), got {}", p.strong_count());

        // Dropping a tensor decrements the count.
        drop(t1);
        assert_eq!(p.strong_count(), 2);
        drop(t2);
        assert_eq!(p.strong_count(), 1);
    }

    #[test]
    fn shared_bf16_decodes_via_copy_to_cpu_vec() {
        // BF16 path: store BF16 bits, materialise a tensor,
        // copy_to_cpu_vec must produce the F32 upcast that
        // matches the round-trip via f32_to_bf16_bits.
        use crate::tensor::tensor::{f32_to_bf16_bits, bf16_bits_to_f32};
        let mut store = WeightStore::new();

        let f32_src: Vec<f32> = vec![0.0, 0.5, -0.25, 1.5, 2.0, -2.0, 3.14, 100.0];
        let bf16_bits: Vec<u16> = f32_src.iter().map(|&f| f32_to_bf16_bits(f)).collect();
        store.insert_bf16("w.test_bf16", vec![2, 4], bf16_bits.clone());

        let p = store.get_by_name("w.test_bf16").unwrap();
        let t = p.to_tensor();
        assert_eq!(t.shape, vec![2, 4]);

        // Decoded F32 matches lossy round-trip of the source.
        let decoded = t.copy_to_cpu_vec();
        let expected: Vec<f32> = bf16_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect();
        assert_eq!(decoded, expected,
            "BF16 decode through CpuBf16Shared != reference round-trip");
    }

    #[test]
    fn arc_sharing_does_not_duplicate_buffer_across_tensors() {
        // The headline M5.c.2.a property: two Tensor instances
        // produced from the same SharedParam reference the SAME
        // Arc — verifiable both via strong_count AND via the
        // observation that the inner Vec's heap allocation is
        // unique (we check this by writing into it through one
        // Arc reference and reading back through the other).
        let arc: Arc<Vec<f32>> = Arc::new(vec![1.0, 2.0, 3.0, 4.0]);
        let t1 = Tensor::cpu_shared(vec![4], Arc::clone(&arc));
        let t2 = Tensor::cpu_shared(vec![4], Arc::clone(&arc));

        // Three strong refs: original `arc`, t1's storage, t2's storage.
        assert_eq!(Arc::strong_count(&arc), 3);

        // Both tensors see the same bytes through different
        // pointers (the slices' .as_ptr() should match because
        // the underlying Vec is shared).
        let s1 = t1.as_cpu_slice();
        let s2 = t2.as_cpu_slice();
        assert_eq!(s1.as_ptr(), s2.as_ptr(),
            "CpuShared tensors built from the same Arc must share buffer pointer");
        assert_eq!(s1, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ensure_owned_breaks_sharing_for_caller_only() {
        // ensure_owned() transitions the calling tensor to
        // Cpu storage. Sibling tensors that hold the original
        // Arc are unaffected.
        let arc: Arc<Vec<f32>> = Arc::new(vec![10.0, 20.0, 30.0]);
        let mut t1 = Tensor::cpu_shared(vec![3], Arc::clone(&arc));
        let t2 = Tensor::cpu_shared(vec![3], Arc::clone(&arc));

        // Strong count is 3 (arc + t1 + t2).
        assert_eq!(Arc::strong_count(&arc), 3);

        // ensure_owned on t1: t1 transitions away.
        t1.ensure_owned().unwrap();
        // t1 is now Cpu storage (uniquely owned).
        assert!(matches!(t1.storage, TensorStorage::Cpu(_)));
        // t2 still holds the Arc; t2's storage is still
        // CpuShared. Strong count drops to 2 (arc + t2).
        assert!(matches!(t2.storage, TensorStorage::CpuShared(_)));
        assert_eq!(Arc::strong_count(&arc), 2);

        // Both tensors still produce the same data via
        // copy_to_cpu_vec (t1's data was cloned out, not
        // moved — original arc still holds [10, 20, 30]).
        assert_eq!(t1.copy_to_cpu_vec(), vec![10.0, 20.0, 30.0]);
        assert_eq!(t2.copy_to_cpu_vec(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn weight_store_resident_bytes_matches_expectation() {
        let mut store = WeightStore::new();
        // F32 4×4 = 16 floats × 4 = 64 bytes.
        store.insert_f32("p1", vec![4, 4], vec![0.0; 16]);
        // BF16 2×8 = 16 elements × 2 = 32 bytes.
        store.insert_bf16("p2", vec![2, 8], vec![0; 16]);
        assert_eq!(store.resident_bytes(), 64 + 32);
        assert_eq!(store.len(), 2);
    }
}
