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
use crate::tensor::Tensor;
#[cfg(test)]
use crate::tensor::TensorStorage;

/// One stored parameter — a pre-shaped buffer wrapped in
/// an `Arc` (host) or [`crate::gpu::tensor::tensor_gpu::TensorGPU`]
/// (device) so multiple `Tensor` wrappers can reference it
/// without copying.
///
/// Storage is one of:
///   - **F32** — host-resident, `Arc<Vec<f32>>`.
///   - **Bf16** — host-resident, `Arc<Vec<u16>>` (raw BF16
///     bit patterns).
///   - **Gpu** *(M6.c.3)* — VRAM-resident, refcounted
///     [`TensorGPU`]. Materialised by
///     [`WeightStore::upload_resident_layers`] for
///     parameters whose layer the residency planner picked.
///
/// Mixing within one parameter is not supported. The
/// initial state is set by
/// [`WeightMapper::store_params_as_bf16`]; the M6.c.3
/// residency upload path is monotonic — host-side variants
/// transition to `Gpu` and never come back (a session that
/// teardowns the pipeline frees both sides).
#[derive(Debug, Clone)]
pub enum SharedParam {
    F32 { shape: Vec<usize>, arc: Arc<Vec<f32>> },
    Bf16 { shape: Vec<usize>, arc: Arc<Vec<u16>> },
    /// **M6.c.3** — VRAM-resident. The `TensorGPU`'s inner
    /// `Arc<InnerGpuPtr>` refcounts the device allocation
    /// across every `to_tensor()` call.
    Gpu { shape: Vec<usize>, gpu: crate::gpu::tensor::tensor_gpu::TensorGPU },
}

impl SharedParam {
    /// Materialise a [`Tensor`] backed by this shared
    /// parameter. Cheap (Arc clone, no Vec copy).
    /// M6.c.3 — `Gpu` variant returns a `TensorStorage::Cuda`
    /// tensor sharing the underlying `TensorGPU` Arc.
    pub fn to_tensor(&self) -> Tensor {
        match self {
            SharedParam::F32 { shape, arc } =>
                Tensor::cpu_shared(shape.clone(), Arc::clone(arc)),
            SharedParam::Bf16 { shape, arc } =>
                Tensor::cpu_bf16_shared(shape.clone(), Arc::clone(arc)),
            SharedParam::Gpu { shape, gpu } => {
                // `TensorGPU::clone` is an Arc bump (cheap;
                // M5.c.2.a-style sharing semantics). The
                // resulting tensor's storage is
                // `TensorStorage::Cuda`, ready for the
                // residency-aware dispatch path in
                // `try_gpu_matmul`.
                let strides = Tensor::compute_strides(shape, &crate::tensor::Layout::Contiguous);
                Tensor {
                    shape: shape.clone(),
                    storage: crate::tensor::TensorStorage::Cuda(gpu.clone()),
                    device: crate::tensor::Device::GPU,
                    dtype: crate::tensor::DType::F32,
                    layout: crate::tensor::Layout::Contiguous,
                    strides,
                    grad: None,
                    op: None,
                }
            }
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            SharedParam::F32 { shape, .. }
            | SharedParam::Bf16 { shape, .. }
            | SharedParam::Gpu { shape, .. } => shape,
        }
    }

    /// Bytes resident in the underlying buffer.
    /// F32: 4×numel. BF16: 2×numel. Gpu: 4×numel (kernel
    /// ABI is F32; BF16 GPU kernel is M6.f / v21).
    pub fn resident_bytes(&self) -> usize {
        match self {
            SharedParam::F32 { arc, .. } => arc.len() * 4,
            SharedParam::Bf16 { arc, .. } => arc.len() * 2,
            SharedParam::Gpu { gpu, .. } => gpu.size_bytes(),
        }
    }

    /// True iff the parameter currently lives on GPU.
    pub fn is_gpu(&self) -> bool {
        matches!(self, SharedParam::Gpu { .. })
    }

    /// Strong-count over the inner `Arc` for host variants;
    /// strong-count over the `TensorGPU` inner Arc for the
    /// GPU variant. Used by tests that verify sharing.
    pub fn strong_count(&self) -> usize {
        match self {
            SharedParam::F32 { arc, .. } => Arc::strong_count(arc),
            SharedParam::Bf16 { arc, .. } => Arc::strong_count(arc),
            // TensorGPU's Arc is private; clone is a bump.
            // Strong-count introspection is not exposed —
            // callers that need it should use the host
            // variants. Return 1 as a defensive default.
            SharedParam::Gpu { .. } => 1,
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

    /// **M6.c.3** — upload a subset of parameters to VRAM
    /// in-place, replacing the matching `SharedParam::F32`
    /// or `Bf16` entries with `SharedParam::Gpu`.
    ///
    /// `should_upload` is a predicate keyed on the
    /// HuggingFace parameter name. The
    /// [`crate::gpu::residency_planner::ResidencyPlan`]
    /// + `extract_layer_index_from_param_name` together
    /// produce the predicate at the call site (the planner
    /// knows layer indices, the name parser maps a name
    /// like `"model.layers.27.self_attn.q_proj.weight"` to
    /// `Some(27)`).
    ///
    /// Skips entries that are already `Gpu` (idempotent).
    /// Returns `(uploaded_count, total_bytes_uploaded)`.
    ///
    /// Errors when the GPU upload fails (driver unavailable,
    /// VRAM exhausted, host→device transfer failure). The
    /// store is left in a consistent state — entries that
    /// uploaded successfully stay `Gpu`; the failing entry
    /// stays in its host variant; the operator can retry
    /// with a smaller resident set or fall back to CPU
    /// dispatch (`ATENIA_GPU=0`).
    pub fn upload_resident_layers<F>(
        &mut self,
        mut should_upload: F,
    ) -> Result<(usize, u64), WeightStoreError>
    where
        F: FnMut(&str) -> bool,
    {
        let mut count: usize = 0;
        let mut bytes_uploaded: u64 = 0;
        for i in 0..self.params.len() {
            if !should_upload(&self.names[i]) { continue; }
            if matches!(&self.params[i], SharedParam::Gpu { .. }) {
                // Idempotent — already on GPU.
                continue;
            }

            // Materialise a Tensor from the current
            // host-resident SharedParam, ensure_gpu it,
            // extract the TensorGPU.
            let mut t = self.params[i].to_tensor();
            t.ensure_gpu()
                .map_err(|e| WeightStoreError::UploadFailed {
                    name: self.names[i].clone(),
                    message: format!("{:?}", e),
                })?;
            let shape = t.shape.clone();
            let gpu = match t.storage {
                crate::tensor::TensorStorage::Cuda(g) => g,
                _ => {
                    return Err(WeightStoreError::UploadFailed {
                        name: self.names[i].clone(),
                        message: "ensure_gpu did not transition to TensorStorage::Cuda".into(),
                    });
                }
            };

            // Replace the entry in place with the GPU variant.
            // This drops the host-side Arc (since the store
            // was the only strong ref pre-upload — the post-
            // M5 generation loop builds fresh tensors per
            // step from `to_tensor()`, never holding strong
            // refs across calls).
            bytes_uploaded += gpu.size_bytes() as u64;
            self.params[i] = SharedParam::Gpu { shape, gpu };
            count += 1;
        }
        Ok((count, bytes_uploaded))
    }

    /// Number of params currently materialised on the GPU.
    pub fn num_gpu_resident(&self) -> usize {
        self.params.iter().filter(|p| p.is_gpu()).count()
    }

    /// Total bytes resident on the GPU across every
    /// `SharedParam::Gpu` entry.
    pub fn gpu_resident_bytes(&self) -> u64 {
        self.params.iter()
            .filter_map(|p| match p {
                SharedParam::Gpu { gpu, .. } => Some(gpu.size_bytes() as u64),
                _ => None,
            })
            .sum()
    }

    /// **M5.c.2.b** — extract loaded parameter tensors from a
    /// `Graph` into a fresh `WeightStore`, replacing the
    /// graph-side storage with `CpuShared` / `CpuBf16Shared`
    /// views over the same `Arc`. The graph itself stays
    /// usable; both the original graph and any subsequent
    /// graph that materialises tensors from the store reference
    /// the same physical bytes.
    ///
    /// `param_ids` and `param_names` must be index-aligned —
    /// pass the corresponding fields of `LlamaHandles` /
    /// equivalent. Tensors with `Cpu(_)` storage become
    /// `CpuShared`; `CpuBf16(_)` becomes `CpuBf16Shared`.
    /// Other variants (`Cuda`, `Disk`, already-`Shared`) are
    /// passed through as a fresh entry whose backing buffer is
    /// taken via `copy_to_cpu_vec` (Disk) or skipped (Cuda not
    /// supported in M5).
    ///
    /// Returns the populated store. The original graph is
    /// mutated in place: parameter slots are replaced with
    /// `Shared` storage. This is the reverse of
    /// `WeightMapper::load_into` — instead of writing weights
    /// INTO the graph, we hoist them OUT of the graph into
    /// shared storage that a sibling graph can reference.
    pub fn extract_from_graph(
        graph: &mut crate::amg::graph::Graph,
        param_ids: &[usize],
        param_names: &[String],
    ) -> Result<WeightStore, WeightStoreError> {
        if param_ids.len() != param_names.len() {
            return Err(WeightStoreError::IndexMismatch {
                ids_len: param_ids.len(), names_len: param_names.len(),
            });
        }

        let mut store = WeightStore::new();
        for (idx, (&node_id, name)) in param_ids.iter().zip(param_names.iter()).enumerate() {
            // Borrow the node's Tensor mutably so we can
            // replace its storage.
            let node = graph.nodes.get_mut(node_id)
                .ok_or(WeightStoreError::NodeOutOfRange {
                    node_id, len: idx, name: name.clone(),
                })?;
            let tensor = node.output.as_mut()
                .ok_or(WeightStoreError::NodeHasNoTensor {
                    node_id, name: name.clone(),
                })?;
            let shape = tensor.shape.clone();

            // Take ownership of the storage: replace with a
            // placeholder, then route the original through the
            // F32/BF16 hoist path.
            let original = std::mem::replace(
                &mut tensor.storage,
                crate::tensor::TensorStorage::Cpu(Vec::new()),
            );
            match original {
                crate::tensor::TensorStorage::Cpu(v) => {
                    let arc = Arc::new(v);
                    tensor.storage = crate::tensor::TensorStorage::CpuShared(Arc::clone(&arc));
                    store.params.push(SharedParam::F32 { shape, arc });
                    store.names.push(name.clone());
                }
                crate::tensor::TensorStorage::CpuBf16(bits) => {
                    let arc = Arc::new(bits);
                    tensor.storage =
                        crate::tensor::TensorStorage::CpuBf16Shared(Arc::clone(&arc));
                    store.params.push(SharedParam::Bf16 { shape, arc });
                    store.names.push(name.clone());
                }
                crate::tensor::TensorStorage::CpuShared(arc) => {
                    // Already shared (idempotent — extract twice
                    // is a no-op except for re-listing in the new store).
                    tensor.storage = crate::tensor::TensorStorage::CpuShared(Arc::clone(&arc));
                    store.params.push(SharedParam::F32 { shape, arc });
                    store.names.push(name.clone());
                }
                crate::tensor::TensorStorage::CpuBf16Shared(arc) => {
                    tensor.storage = crate::tensor::TensorStorage::CpuBf16Shared(Arc::clone(&arc));
                    store.params.push(SharedParam::Bf16 { shape, arc });
                    store.names.push(name.clone());
                }
                other => {
                    // Cuda / Disk: out of M5.c.2.b scope. Restore
                    // and surface the variant to the caller.
                    tensor.storage = other;
                    return Err(WeightStoreError::UnsupportedStorage {
                        node_id, name: name.clone(),
                    });
                }
            }
        }

        Ok(store)
    }
}

/// Errors produced by [`WeightStore`] hoist/extract operations.
#[derive(Debug)]
pub enum WeightStoreError {
    IndexMismatch { ids_len: usize, names_len: usize },
    NodeOutOfRange { node_id: usize, len: usize, name: String },
    NodeHasNoTensor { node_id: usize, name: String },
    UnsupportedStorage { node_id: usize, name: String },
    /// **M6.c.3** — host→device upload failed during
    /// `upload_resident_layers`. The store's other entries
    /// are left in a consistent state.
    UploadFailed { name: String, message: String },
}

impl std::fmt::Display for WeightStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightStoreError::IndexMismatch { ids_len, names_len } =>
                write!(f, "weight_store: param_ids.len={ids_len} != param_names.len={names_len}"),
            WeightStoreError::NodeOutOfRange { node_id, len, name } =>
                write!(f, "weight_store: node id {node_id} out of range at index {len} for '{name}'"),
            WeightStoreError::NodeHasNoTensor { node_id, name } =>
                write!(f, "weight_store: node {node_id} for '{name}' has no materialised tensor"),
            WeightStoreError::UnsupportedStorage { node_id, name } =>
                write!(f, "weight_store: node {node_id} for '{name}' has unsupported (Cuda/Disk) storage"),
            WeightStoreError::UploadFailed { name, message } =>
                write!(f, "weight_store: GPU upload of '{name}' failed: {message}"),
        }
    }
}

impl std::error::Error for WeightStoreError {}

/// **M6.c.3** — extract the transformer layer index from a
/// HuggingFace parameter name, if any.
///
/// Returns `Some(i)` for names like:
///   - `model.layers.{i}.self_attn.q_proj.weight`
///   - `model.layers.{i}.mlp.gate_proj.weight`
///   - `model.layers.{i}.input_layernorm.weight`
///
/// Returns `None` for names that don't belong to a
/// transformer layer (`model.embed_tokens.weight`,
/// `model.norm.weight`, `lm_head.weight`).
///
/// Used by the residency upload predicate at the
/// `GenerationPipeline` level to map "layer i is resident"
/// → "every parameter whose name parses to `Some(i)` is
/// resident".
pub fn extract_layer_index_from_param_name(name: &str) -> Option<usize> {
    // HF convention: `model.layers.{i}.<rest>`. Strict prefix
    // match keeps us safe from collisions with hypothetical
    // future name schemes (e.g. `model.encoder.layers.X` if
    // a new architecture ever ships).
    let suffix = name.strip_prefix("model.layers.")?;
    let dot = suffix.find('.')?;
    suffix[..dot].parse::<usize>().ok()
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
    fn extract_from_graph_hoists_owned_storage_to_arc_shared() {
        // M5.c.2.b — the headline test: build a graph with
        // owned Cpu / CpuBf16 parameter tensors, extract them
        // into a WeightStore, verify that:
        //   1. The graph's parameter slots are now CpuShared /
        //      CpuBf16Shared.
        //   2. The store's params reference the SAME Arcs
        //      (strong_count == 2: graph slot + store entry).
        //   3. Reads through both produce identical bytes.
        use crate::amg::builder::GraphBuilder;
        use crate::tensor::{Tensor, TensorStorage};
        use crate::tensor::tensor::f32_to_bf16_bits;

        let mut gb = GraphBuilder::new();
        // Two parameters: one F32, one BF16.
        let f32_data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let f32_id = gb.parameter(Tensor::new_cpu(vec![2, 3], f32_data.clone()));

        let bf16_src = vec![0.5_f32, 1.5, -0.25, 2.75];
        let bf16_bits: Vec<u16> = bf16_src.iter().map(|&f| f32_to_bf16_bits(f)).collect();
        let bf16_id = gb.parameter(Tensor::new_cpu_bf16(vec![2, 2], bf16_bits.clone()));

        let nodes = std::mem::take(&mut gb.nodes);
        let mut g = crate::amg::graph::Graph::build(nodes);

        // Pre-extraction: storage is owned (Cpu / CpuBf16).
        assert!(matches!(
            g.nodes[f32_id].output.as_ref().unwrap().storage,
            TensorStorage::Cpu(_)
        ));
        assert!(matches!(
            g.nodes[bf16_id].output.as_ref().unwrap().storage,
            TensorStorage::CpuBf16(_)
        ));

        // Extract.
        let names = vec!["w.f32".to_string(), "w.bf16".to_string()];
        let ids = vec![f32_id, bf16_id];
        let store = WeightStore::extract_from_graph(&mut g, &ids, &names).unwrap();

        // Post-extraction: graph storage is now Shared.
        assert!(matches!(
            g.nodes[f32_id].output.as_ref().unwrap().storage,
            TensorStorage::CpuShared(_)
        ));
        assert!(matches!(
            g.nodes[bf16_id].output.as_ref().unwrap().storage,
            TensorStorage::CpuBf16Shared(_)
        ));

        // Strong count: graph slot + store entry = 2.
        assert_eq!(store.params[0].strong_count(), 2,
            "F32 param should have 2 strong refs (graph + store)");
        assert_eq!(store.params[1].strong_count(), 2,
            "BF16 param should have 2 strong refs (graph + store)");

        // Reads through the graph match the original data.
        let g_f32_tensor = g.nodes[f32_id].output.as_ref().unwrap();
        assert_eq!(g_f32_tensor.copy_to_cpu_vec(), f32_data);

        // The store's tensors share the same Arc — extract a
        // tensor from the store and confirm its CpuShared
        // pointer matches the graph slot's CpuShared pointer.
        let store_tensor = store.params[0].to_tensor();
        let g_slice = g_f32_tensor.as_cpu_slice();
        let s_slice = store_tensor.as_cpu_slice();
        assert_eq!(g_slice.as_ptr(), s_slice.as_ptr(),
            "graph slot and store tensor must reference same buffer");

        // Names round-trip.
        assert_eq!(store.names, names);
    }

    #[test]
    fn extract_idempotent_on_already_shared_storage() {
        // Calling extract_from_graph on a graph whose params
        // are already CpuShared (e.g. from a prior extract or
        // a build_llama_with_store call) must not panic and
        // must preserve sharing.
        use crate::amg::builder::GraphBuilder;
        use crate::tensor::Tensor;

        let mut gb = GraphBuilder::new();
        let arc = std::sync::Arc::new(vec![1.0_f32, 2.0, 3.0]);
        let id = gb.parameter(Tensor::cpu_shared(vec![3], std::sync::Arc::clone(&arc)));
        let nodes = std::mem::take(&mut gb.nodes);
        let mut g = crate::amg::graph::Graph::build(nodes);

        let store = WeightStore::extract_from_graph(
            &mut g, &[id], &["w.shared".to_string()]).unwrap();

        // Original Arc still alive (reachable through graph
        // slot AND store entry AND original `arc` binding) → 3 refs.
        assert_eq!(std::sync::Arc::strong_count(&arc), 3);
        assert_eq!(store.params[0].strong_count(), 3);
    }

    #[test]
    fn extract_rejects_index_mismatch() {
        use crate::amg::builder::GraphBuilder;
        let mut gb = GraphBuilder::new();
        let nodes = std::mem::take(&mut gb.nodes);
        let mut g = crate::amg::graph::Graph::build(nodes);
        let result = WeightStore::extract_from_graph(
            &mut g, &[0, 1], &["only_one".to_string()]);
        assert!(matches!(result, Err(WeightStoreError::IndexMismatch { .. })));
    }

    #[test]
    fn extract_layer_index_parses_hf_names_correctly() {
        // **M6.c.3** — predicate parser must round-trip the
        // HuggingFace name convention without false positives.
        assert_eq!(
            extract_layer_index_from_param_name(
                "model.layers.0.self_attn.q_proj.weight"),
            Some(0));
        assert_eq!(
            extract_layer_index_from_param_name(
                "model.layers.27.mlp.gate_proj.weight"),
            Some(27));
        assert_eq!(
            extract_layer_index_from_param_name(
                "model.layers.39.input_layernorm.weight"),
            Some(39));
        // Non-layer names → None.
        assert_eq!(
            extract_layer_index_from_param_name("model.embed_tokens.weight"),
            None);
        assert_eq!(
            extract_layer_index_from_param_name("model.norm.weight"),
            None);
        assert_eq!(
            extract_layer_index_from_param_name("lm_head.weight"),
            None);
        // Defensive: empty / malformed.
        assert_eq!(extract_layer_index_from_param_name(""), None);
        assert_eq!(extract_layer_index_from_param_name("model.layers."), None);
        assert_eq!(extract_layer_index_from_param_name("model.layers.abc.weight"), None);
    }

    #[test]
    fn upload_resident_layers_skips_when_predicate_returns_false() {
        // **M6.c.3** — predicate-driven control. With a
        // predicate that always returns false, the store
        // must not try to upload anything (no GPU calls
        // made, no error possible).
        let mut store = WeightStore::new();
        store.insert_f32("p1".into(), vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        store.insert_bf16("p2".into(), vec![2, 2], vec![0; 4]);

        let result = store.upload_resident_layers(|_| false);
        let (count, bytes) = result.expect("predicate=false should never error");
        assert_eq!(count, 0);
        assert_eq!(bytes, 0);

        // Both entries still on host.
        assert!(matches!(store.params[0], SharedParam::F32 { .. }));
        assert!(matches!(store.params[1], SharedParam::Bf16 { .. }));
        assert_eq!(store.num_gpu_resident(), 0);
        assert_eq!(store.gpu_resident_bytes(), 0);
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
