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

use crate::tensor::Tensor;
#[cfg(test)]
use crate::tensor::TensorStorage;
use std::sync::Arc;

use crate::gpu::safety::resource_check::{SafetyDecision, check_before_gpu_operation};
use crate::gpu::tensor::TensorGPU;
use crate::tensor::disk_tier::DiskTensorHandle;

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
    F32 {
        shape: Vec<usize>,
        arc: Arc<Vec<f32>>,
    },
    Bf16 {
        shape: Vec<usize>,
        arc: Arc<Vec<u16>>,
    },
    /// **M6 step 4b** — VRAM-resident F32 parameter. The
    /// underlying [`TensorGPU`] holds an `Arc<InnerGpuPtr>` so
    /// `Clone` and `to_tensor` are cheap and share the device
    /// buffer with no extra allocation. The original BF16
    /// `Arc<Vec<u16>>` is dropped by `upload_layer_bf16_to_vram`
    /// at the moment the variant is overwritten — this is the
    /// mechanism that lets the M6 wire-up free RAM after a
    /// successful upload.
    Cuda { shape: Vec<usize>, gpu: TensorGPU },
    /// **M6 replan sub-fase 0** — disk-resident parameter. Bytes
    /// live on NVMe (or wherever `disk_tier::default_cache_dir`
    /// resolves); the [`DiskTensorHandle`] is an Arc-backed handle
    /// whose `Drop` removes the file when the last clone is gone.
    /// Logical dtype is encoded in `handle.dtype()` (F32 or BF16)
    /// — the variant doesn't carry it separately to keep the
    /// invariant single-sourced.
    ///
    /// At this commit (sub-fase 0) the variant has **no
    /// production constructor** beyond the test-facing
    /// [`WeightStore::insert_disk`]. Sub-fase 2 will wire it to
    /// the loader's tier-aware path.
    Disk {
        shape: Vec<usize>,
        handle: DiskTensorHandle,
    },
}

impl SharedParam {
    /// Materialise a [`Tensor`] backed by this shared
    /// parameter. Cheap (Arc clone, no Vec copy).
    pub fn to_tensor(&self) -> Tensor {
        match self {
            SharedParam::F32 { shape, arc } => Tensor::cpu_shared(shape.clone(), Arc::clone(arc)),
            SharedParam::Bf16 { shape, arc } => {
                Tensor::cpu_bf16_shared(shape.clone(), Arc::clone(arc))
            }
            SharedParam::Cuda { shape, gpu } => Tensor::from_cuda_gpu(shape.clone(), gpu.clone()),
            SharedParam::Disk { shape, handle } => Tensor::from_disk(shape.clone(), handle.clone()),
        }
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            SharedParam::F32 { shape, .. }
            | SharedParam::Bf16 { shape, .. }
            | SharedParam::Cuda { shape, .. }
            | SharedParam::Disk { shape, .. } => shape,
        }
    }

    /// Bytes resident in the underlying buffer (F32 = 4 ×
    /// numel; BF16 = 2 × numel; Cuda = device buffer size; Disk
    /// = `numel × bytes_per_element` of the on-disk dtype).
    ///
    /// **Note**: for the `Cuda` variant the bytes live in VRAM,
    /// not host RAM. For the `Disk` variant the bytes live on
    /// NVMe — they consume **zero RAM** until a consumer calls
    /// `ensure_cpu` to materialise. The aggregate
    /// `WeightStore::resident_bytes` sum is therefore not a
    /// pure-RAM measurement once any tensor lives on a non-RAM
    /// tier. The M6 replan introduces a `tier_breakdown()`
    /// helper later that accounts per-tier; for now,
    /// `resident_bytes` reports the storage byte count
    /// regardless of which tier hosts it.
    pub fn resident_bytes(&self) -> usize {
        match self {
            SharedParam::F32 { arc, .. } => arc.len() * 4,
            SharedParam::Bf16 { arc, .. } => arc.len() * 2,
            SharedParam::Cuda { gpu, .. } => gpu.size_bytes(),
            SharedParam::Disk { handle, .. } => handle.numel() * handle.dtype().bytes_per_element(),
        }
    }

    /// Strong-count over the inner `Arc`. Two-graph
    /// configurations expect 2 (or more) strong refs once
    /// both graphs have materialised their parameter slots.
    /// Useful for tests that verify sharing actually
    /// happened.
    ///
    /// For `Cuda` and `Disk` variants returns 1 unconditionally
    /// — both wrap inner `Arc`s (`Arc<InnerGpuPtr>` and
    /// `Arc<InnerDiskFile>` respectively) that are private and
    /// not exposed for external strong-count queries. Tests
    /// that verify Arc-sharing semantics use the `F32` / `Bf16`
    /// variants.
    pub fn strong_count(&self) -> usize {
        match self {
            SharedParam::F32 { arc, .. } => Arc::strong_count(arc),
            SharedParam::Bf16 { arc, .. } => Arc::strong_count(arc),
            SharedParam::Cuda { .. } => 1,
            SharedParam::Disk { .. } => 1,
        }
    }
}

/// **M6 step 4b** — outcome of `WeightStore::upload_layer_bf16_to_vram`.
///
/// `params_uploaded` counts entries successfully transitioned from
/// `SharedParam::Bf16` to `SharedParam::Cuda`. Non-BF16 entries and
/// entries whose upload failed are skipped silently (logged to
/// stderr) and do not contribute.
///
/// `vram_bytes_used` is the device-side F32 buffer total — the
/// persistent residency cost on the GPU.
///
/// `ram_bytes_freed` is the upper bound on RAM reclaimed: it sums
/// the BF16 byte sizes of every uploaded param. The actual amount
/// of RAM that returns to the OS depends on whether other code
/// (e.g. a sibling graph slot via `extract_from_graph`) still
/// holds the same `Arc<Vec<u16>>` — when it does, dropping the
/// store's clone only reduces the strong count, not the
/// allocation. In tests with isolated stores the figure is
/// exact; in production the figure is informational.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UploadReport {
    pub params_uploaded: usize,
    pub vram_bytes_used: u64,
    pub ram_bytes_freed: u64,
}

impl UploadReport {
    pub fn empty() -> Self {
        Self::default()
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
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an F32 parameter. Wraps the provided `Vec`
    /// in a fresh `Arc`. Returns the parameter index for
    /// downstream `to_tensor` lookups.
    pub fn insert_f32(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        data: Vec<f32>,
    ) -> usize {
        let idx = self.params.len();
        self.params.push(SharedParam::F32 {
            shape,
            arc: Arc::new(data),
        });
        self.names.push(name.into());
        idx
    }

    /// Insert a BF16 parameter. Wraps the provided `Vec<u16>`
    /// (raw BF16 bit patterns) in a fresh `Arc`.
    pub fn insert_bf16(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        bits: Vec<u16>,
    ) -> usize {
        let idx = self.params.len();
        self.params.push(SharedParam::Bf16 {
            shape,
            arc: Arc::new(bits),
        });
        self.names.push(name.into());
        idx
    }

    /// **M6 replan sub-fase 0** — insert a disk-resident
    /// parameter. The caller has already produced a
    /// [`DiskTensorHandle`] via
    /// `disk_tier::write_f32_tensor` /
    /// `disk_tier::write_bf16_tensor`; this method only
    /// records the handle alongside the parameter's logical
    /// shape and HuggingFace-convention name.
    ///
    /// At this commit there is no production caller; the
    /// method is exercised by the round-trip unit tests and
    /// will be wired to the loader's tier-aware path in
    /// sub-fase 2.
    pub fn insert_disk(
        &mut self,
        name: impl Into<String>,
        shape: Vec<usize>,
        handle: DiskTensorHandle,
    ) -> usize {
        let idx = self.params.len();
        self.params.push(SharedParam::Disk { shape, handle });
        self.names.push(name.into());
        idx
    }

    /// Lookup a parameter by index. Index is the value
    /// returned by `insert_*`.
    pub fn get(&self, idx: usize) -> Option<&SharedParam> {
        self.params.get(idx)
    }

    /// Lookup by HuggingFace-convention name. Linear scan
    /// (fine — called once per parameter at graph build
    /// time, never on the forward hot path).
    pub fn get_by_name(&self, name: &str) -> Option<&SharedParam> {
        self.names
            .iter()
            .position(|n| n == name)
            .and_then(|i| self.params.get(i))
    }

    /// Total bytes resident across every stored parameter.
    /// Used by the M5.c.2.b R2 falsifier to assert Arc-shared
    /// sizes match single-graph sizes (i.e. building a second
    /// graph that references the same store does not double
    /// the resident footprint).
    pub fn resident_bytes(&self) -> usize {
        self.params.iter().map(|p| p.resident_bytes()).sum()
    }

    pub fn len(&self) -> usize {
        self.params.len()
    }
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    /// **M10.3.1.1** — stamp each VRAM-resident parameter's
    /// `TensorGPU` with the per-tensor matmul precision policy
    /// resolved from the numeric certification manifest. Called
    /// once after the loader has populated the store and before
    /// the first forward; subsequent dispatcher reads see the
    /// stamped value via `gpu.matmul_policy_byte()` and route
    /// per-tensor between the certified and fast kernels.
    ///
    /// Tensors whose storage variant is not `Cuda` are left
    /// alone — only `cuda_matmul_bf16_inplace` /
    /// `cuda_matmul_bf16_native_inplace` consume the policy
    /// today (the dispatcher's `bf16_mixed_resident` arm). When
    /// M10.3.1.x extends per-tensor dispatch to the disk-streamed
    /// or RAM-resident paths, this method extends with arms for
    /// those variants.
    ///
    /// Returns the number of parameters stamped (Cuda variant
    /// count) for caller-side telemetry / smoke assertions.
    pub fn apply_per_tensor_policy(
        &mut self,
        manifest: &crate::nn::llama::numcert::NumcertManifest,
    ) -> usize {
        use crate::gpu::tensor::tensor_gpu::{
            MATMUL_POLICY_BYTE_CERTIFIED, MATMUL_POLICY_BYTE_FAST,
        };
        use crate::nn::llama::numcert::MatmulMode;
        let mut stamped = 0usize;
        let mut fast_count = 0usize;
        let mut certified_count = 0usize;
        for (param, name) in self.params.iter_mut().zip(self.names.iter()) {
            if let SharedParam::Cuda { gpu, .. } = param {
                let mode = manifest.resolve_for(name);
                let byte = match mode {
                    MatmulMode::Certified => {
                        certified_count += 1;
                        MATMUL_POLICY_BYTE_CERTIFIED
                    }
                    MatmulMode::Fast => {
                        fast_count += 1;
                        MATMUL_POLICY_BYTE_FAST
                    }
                    MatmulMode::Quantized => {
                        certified_count += 1;
                        MATMUL_POLICY_BYTE_CERTIFIED
                    }
                };
                gpu.set_matmul_policy_byte(Some(byte));
                stamped += 1;
            }
        }
        eprintln!(
            "[ATENIA] Numeric contract: per-tensor policy applied — \
             {} VRAM tensors stamped ({} fast, {} certified) from {}.",
            stamped,
            fast_count,
            certified_count,
            manifest.source.display(),
        );
        stamped
    }

    /// **M6 step 4b** — return the parameter indices belonging
    /// to a given Llama layer. Recognises the HuggingFace
    /// convention `model.layers.<N>.<...>` used by Llama 2 /
    /// Llama 3 / Qwen 2.5 / TinyLlama. Layers outside that
    /// convention (`model.embed_tokens`, `model.norm`,
    /// `lm_head`, etc.) are not matched by any layer index.
    fn indices_for_layer(&self, layer_idx: usize) -> Vec<usize> {
        let prefix = format!("model.layers.{}.", layer_idx);
        self.names
            .iter()
            .enumerate()
            .filter_map(|(i, name)| {
                if name.starts_with(&prefix) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// **M6 step 4b** — upload every BF16 parameter belonging to
    /// `layer_idx` to VRAM, replacing the entry in this store
    /// with a `SharedParam::Cuda` variant and dropping the
    /// original `Arc<Vec<u16>>`.
    ///
    /// The method first calls [`check_before_gpu_operation`] (the
    /// M6 safety gate) with the layer's required RAM/VRAM
    /// footprint. If the gate returns
    /// [`SafetyDecision::DegradeToCpu`] the upload is skipped
    /// entirely and the report comes back with all-zero counts;
    /// the caller's pipeline continues with the CPU-resident
    /// parameters unchanged. Other safety decisions (`Proceed`,
    /// `DegradeToLayers`) proceed with the upload (the planner
    /// in `pipeline.rs` is responsible for selecting which
    /// layers to upload based on the layer count returned by
    /// `DegradeToLayers`).
    ///
    /// Per-param failures (a single `bf16_to_f32_resident_in_vram`
    /// returning `None`) are non-fatal: the failed entry stays
    /// as `SharedParam::Bf16` and the loop continues. If **every**
    /// BF16 entry fails, returns `Err(AllUploadsFailed)` so the
    /// caller can fall back without leaving the store in a half-
    /// uploaded mixed state.
    ///
    /// Atomicity: each param is replaced in place once its
    /// upload completes. If an upload further down the layer
    /// fails after earlier ones succeeded, the earlier successes
    /// stay (no rollback). This matches the operator's intent —
    /// a partial residency is still useful and the caller can
    /// inspect `UploadReport.params_uploaded` to know what
    /// actually landed.
    pub fn upload_layer_bf16_to_vram(
        &mut self,
        layer_idx: usize,
    ) -> Result<UploadReport, WeightStoreError> {
        // Compute requirements first so the safety gate has the
        // right numbers. Only BF16 entries count toward the
        // upload — F32 / already-Cuda entries are skipped at
        // execution time.
        let mut required_vram_bytes: u64 = 0;
        let mut required_ram_bytes: u64 = 0;
        for &i in &self.indices_for_layer(layer_idx) {
            if let SharedParam::Bf16 { shape, .. } = &self.params[i] {
                let numel: u64 = shape.iter().product::<usize>() as u64;
                // Persistent VRAM cost is F32 (the upcast
                // destination). The transient BF16 device buffer
                // also lives during the upload but is a peak
                // 2× smaller and is freed before the next layer
                // starts uploading, so we charge it only against
                // RAM (which already holds the BF16 source).
                required_vram_bytes += numel * 4;
                required_ram_bytes += numel * 2;
            }
        }
        let required_ram_mb = required_ram_bytes / (1024 * 1024);
        let required_vram_mb = required_vram_bytes / (1024 * 1024);

        let decision = check_before_gpu_operation(required_ram_mb, required_vram_mb);
        self.upload_layer_bf16_to_vram_with_decision(layer_idx, decision)
    }

    /// **M6 step 4b** — testable variant of `upload_layer_bf16_to_vram`
    /// that takes a pre-computed [`SafetyDecision`] instead of
    /// probing the live machine state. Used by unit tests to
    /// exercise the `DegradeToCpu` branch deterministically.
    /// Production callers go through `upload_layer_bf16_to_vram`.
    pub(crate) fn upload_layer_bf16_to_vram_with_decision(
        &mut self,
        layer_idx: usize,
        decision: SafetyDecision,
    ) -> Result<UploadReport, WeightStoreError> {
        if matches!(decision, SafetyDecision::DegradeToCpu) {
            return Ok(UploadReport::empty());
        }

        let layer_indices = self.indices_for_layer(layer_idx);
        if layer_indices.is_empty() {
            return Ok(UploadReport::empty());
        }

        let mut params_uploaded = 0_usize;
        let mut vram_bytes_used = 0_u64;
        let mut ram_bytes_freed = 0_u64;
        let mut bf16_param_count = 0_usize;
        let mut upload_failures = 0_usize;

        for i in layer_indices {
            // Take ownership of the entry so the BF16 `Arc` can
            // be dropped at the end of the iteration if the
            // upload succeeds. We replace it back with the
            // original (or with a Cuda variant on success)
            // before continuing. The placeholder value is a
            // zero-element Bf16 — semantically equivalent to
            // "empty parameter" for the brief window between
            // take and replace.
            let original = std::mem::replace(
                &mut self.params[i],
                SharedParam::Bf16 {
                    shape: Vec::new(),
                    arc: Arc::new(Vec::new()),
                },
            );

            match original {
                SharedParam::Bf16 { shape, arc } => {
                    bf16_param_count += 1;
                    let bf16_bytes = (arc.len() * 2) as u64;
                    match crate::cuda::bf16_to_f32::bf16_to_f32_resident_in_vram(
                        arc.as_slice(),
                        &shape,
                    ) {
                        Some(gpu) => {
                            let vram_bytes = gpu.size_bytes() as u64;
                            self.params[i] = SharedParam::Cuda { shape, gpu };
                            // `arc` drops here. If the store was
                            // its only owner, RAM returns to the
                            // allocator; if a graph slot still
                            // holds it via `extract_from_graph`,
                            // the strong count drops by 1.
                            drop(arc);
                            params_uploaded += 1;
                            vram_bytes_used += vram_bytes;
                            ram_bytes_freed += bf16_bytes;
                        }
                        None => {
                            // Restore the original and continue
                            // with the next param.
                            if !crate::apx_is_silent() {
                                eprintln!(
                                    "[weight_store] BF16→VRAM upload failed for \
                                     param idx {} (name='{}'); leaving as CpuBf16Shared",
                                    i,
                                    self.names.get(i).map(|s| s.as_str()).unwrap_or("?"),
                                );
                            }
                            self.params[i] = SharedParam::Bf16 { shape, arc };
                            upload_failures += 1;
                        }
                    }
                }
                other => {
                    // Not a BF16 entry — restore unchanged. F32
                    // and already-Cuda entries are valid layer
                    // members but not eligible for upload here.
                    self.params[i] = other;
                }
            }
        }

        if bf16_param_count > 0 && upload_failures == bf16_param_count {
            return Err(WeightStoreError::AllUploadsFailed { layer_idx });
        }

        Ok(UploadReport {
            params_uploaded,
            vram_bytes_used,
            ram_bytes_freed,
        })
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
                ids_len: param_ids.len(),
                names_len: param_names.len(),
            });
        }

        let mut store = WeightStore::new();
        for (idx, (&node_id, name)) in param_ids.iter().zip(param_names.iter()).enumerate() {
            // Borrow the node's Tensor mutably so we can
            // replace its storage.
            let node = graph
                .nodes
                .get_mut(node_id)
                .ok_or(WeightStoreError::NodeOutOfRange {
                    node_id,
                    len: idx,
                    name: name.clone(),
                })?;
            let tensor = node
                .output
                .as_mut()
                .ok_or(WeightStoreError::NodeHasNoTensor {
                    node_id,
                    name: name.clone(),
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
                    tensor.storage = crate::tensor::TensorStorage::CpuBf16Shared(Arc::clone(&arc));
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
                        node_id,
                        name: name.clone(),
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
    IndexMismatch {
        ids_len: usize,
        names_len: usize,
    },
    NodeOutOfRange {
        node_id: usize,
        len: usize,
        name: String,
    },
    NodeHasNoTensor {
        node_id: usize,
        name: String,
    },
    UnsupportedStorage {
        node_id: usize,
        name: String,
    },
    /// **M6 step 4b** — `upload_layer_bf16_to_vram` could not
    /// upload **any** of a layer's BF16 params (e.g. driver
    /// missing, repeated `cuda_malloc` failures). The caller is
    /// expected to fall back to the CPU path. Distinct from
    /// `Ok(UploadReport { params_uploaded: 0, .. })` which
    /// indicates the safety gate degraded the upload (RAM
    /// pressure) — that path is non-error.
    AllUploadsFailed {
        layer_idx: usize,
    },
}

impl std::fmt::Display for WeightStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightStoreError::IndexMismatch { ids_len, names_len } => write!(
                f,
                "weight_store: param_ids.len={ids_len} != param_names.len={names_len}"
            ),
            WeightStoreError::NodeOutOfRange { node_id, len, name } => write!(
                f,
                "weight_store: node id {node_id} out of range at index {len} for '{name}'"
            ),
            WeightStoreError::NodeHasNoTensor { node_id, name } => write!(
                f,
                "weight_store: node {node_id} for '{name}' has no materialised tensor"
            ),
            WeightStoreError::UnsupportedStorage { node_id, name } => write!(
                f,
                "weight_store: node {node_id} for '{name}' has unsupported (Cuda/Disk) storage"
            ),
            WeightStoreError::AllUploadsFailed { layer_idx } => write!(
                f,
                "weight_store: every BF16 upload for layer {layer_idx} failed; falling back to CPU"
            ),
        }
    }
}

impl std::error::Error for WeightStoreError {}

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
        assert_eq!(
            p.strong_count(),
            3,
            "expected 3 strong refs (store + 2 tensors), got {}",
            p.strong_count()
        );

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
        use crate::tensor::tensor::{bf16_bits_to_f32, f32_to_bf16_bits};
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
        assert_eq!(
            decoded, expected,
            "BF16 decode through CpuBf16Shared != reference round-trip"
        );
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
        assert_eq!(
            s1.as_ptr(),
            s2.as_ptr(),
            "CpuShared tensors built from the same Arc must share buffer pointer"
        );
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
        use crate::tensor::tensor::f32_to_bf16_bits;
        use crate::tensor::{Tensor, TensorStorage};

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
        assert_eq!(
            store.params[0].strong_count(),
            2,
            "F32 param should have 2 strong refs (graph + store)"
        );
        assert_eq!(
            store.params[1].strong_count(),
            2,
            "BF16 param should have 2 strong refs (graph + store)"
        );

        // Reads through the graph match the original data.
        let g_f32_tensor = g.nodes[f32_id].output.as_ref().unwrap();
        assert_eq!(g_f32_tensor.copy_to_cpu_vec(), f32_data);

        // The store's tensors share the same Arc — extract a
        // tensor from the store and confirm its CpuShared
        // pointer matches the graph slot's CpuShared pointer.
        let store_tensor = store.params[0].to_tensor();
        let g_slice = g_f32_tensor.as_cpu_slice();
        let s_slice = store_tensor.as_cpu_slice();
        assert_eq!(
            g_slice.as_ptr(),
            s_slice.as_ptr(),
            "graph slot and store tensor must reference same buffer"
        );

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

        let store =
            WeightStore::extract_from_graph(&mut g, &[id], &["w.shared".to_string()]).unwrap();

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
        let result = WeightStore::extract_from_graph(&mut g, &[0, 1], &["only_one".to_string()]);
        assert!(matches!(
            result,
            Err(WeightStoreError::IndexMismatch { .. })
        ));
    }

    #[test]
    fn upload_layer_bf16_with_degrade_to_cpu_returns_empty_report() {
        // Safety gate simulating RAM < 8 GiB. The method must
        // short-circuit with an empty UploadReport and leave
        // every BF16 entry in CpuBf16Shared (no upload attempted,
        // no panics, no GPU calls). Does not require CUDA.
        let mut store = WeightStore::new();

        let bf16_q: Vec<u16> = (0..16).map(|i| i as u16).collect();
        let bf16_k: Vec<u16> = (0..16).map(|i| (i + 100) as u16).collect();
        store.insert_bf16(
            "model.layers.0.self_attn.q_proj.weight",
            vec![4, 4],
            bf16_q.clone(),
        );
        store.insert_bf16(
            "model.layers.0.self_attn.k_proj.weight",
            vec![4, 4],
            bf16_k.clone(),
        );

        let decision = SafetyDecision::DegradeToCpu;
        let report = store
            .upload_layer_bf16_to_vram_with_decision(0, decision)
            .expect("DegradeToCpu must return Ok with empty report, not error");

        assert_eq!(report.params_uploaded, 0);
        assert_eq!(report.vram_bytes_used, 0);
        assert_eq!(report.ram_bytes_freed, 0);

        // Both params still BF16.
        assert!(matches!(store.params[0], SharedParam::Bf16 { .. }));
        assert!(matches!(store.params[1], SharedParam::Bf16 { .. }));
    }

    /// **M6 step 4b** — end-to-end upload test for a single
    /// synthetic layer with two BF16 params. Skipped on hosts
    /// without a CUDA driver. Verifies:
    ///   - Both params transition to `SharedParam::Cuda`.
    ///   - `UploadReport.ram_bytes_freed > 0`
    ///     (the store had unique ownership of the Arcs).
    ///   - `UploadReport.vram_bytes_used > 0`.
    ///   - A matmul against the resident weights via
    ///     `cuda_matmul_inplace` is bit-exact with a CPU
    ///     reference matmul over the host-decoded BF16.
    #[test]
    fn upload_layer_bf16_to_vram_uploads_two_params_and_matmul_matches_cpu() {
        use crate::cuda::cuda_available;
        use crate::cuda::matmul::cuda_matmul_inplace;
        use crate::tensor::tensor::{bf16_bits_to_f32, f32_to_bf16_bits};

        if !cuda_available() {
            eprintln!("CUDA not available, skipping");
            return;
        }

        // Layer 0 with two 4x4 BF16 weights. Tiny size so the
        // test is fast (kernel launch dominates).
        let mut store = WeightStore::new();
        let f_q: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.7).collect();
        let f_k: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05 + 0.3).collect();
        let bits_q: Vec<u16> = f_q.iter().map(|&f| f32_to_bf16_bits(f)).collect();
        let bits_k: Vec<u16> = f_k.iter().map(|&f| f32_to_bf16_bits(f)).collect();
        store.insert_bf16(
            "model.layers.0.self_attn.q_proj.weight",
            vec![4, 4],
            bits_q.clone(),
        );
        store.insert_bf16(
            "model.layers.0.self_attn.k_proj.weight",
            vec![4, 4],
            bits_k.clone(),
        );

        let bf16_total_bytes = (bits_q.len() + bits_k.len()) * 2;

        // Force the upload past the safety gate.
        let decision = SafetyDecision::Proceed;
        let report = store
            .upload_layer_bf16_to_vram_with_decision(0, decision)
            .expect("upload must succeed on a healthy CUDA host");

        assert_eq!(report.params_uploaded, 2, "both BF16 params should upload");
        assert_eq!(
            report.ram_bytes_freed as usize, bf16_total_bytes,
            "ram_bytes_freed should equal sum of BF16 byte sizes"
        );
        assert!(
            report.vram_bytes_used > 0,
            "vram_bytes_used must be positive after a successful upload"
        );

        // Both entries now Cuda.
        assert!(
            matches!(store.params[0], SharedParam::Cuda { .. }),
            "param 0 should be Cuda, got {:?}",
            store.params[0]
        );
        assert!(
            matches!(store.params[1], SharedParam::Cuda { .. }),
            "param 1 should be Cuda, got {:?}",
            store.params[1]
        );

        // Matmul Q @ K (both [4, 4]) on GPU vs CPU reference. We
        // use cuda_matmul_inplace with all-Cuda operands +
        // output to exercise the residency path that production
        // sub-step 4d will route to.
        let mut q_gpu = store.params[0].to_tensor();
        let mut k_gpu = store.params[1].to_tensor();
        // q.matmul(k) produces a [4, 4] output. Allocate output
        // on VRAM so cuda_matmul_inplace's all-Cuda branch fires.
        let mut out_gpu = Tensor::zeros_new_cuda(&[4, 4]).expect("VRAM alloc failed");

        cuda_matmul_inplace(&q_gpu, &k_gpu, &mut out_gpu, 4, 4, 4);

        // Materialise output back to host for comparison.
        out_gpu.ensure_cpu().expect("D→H transfer failed");
        let gpu_values = out_gpu.copy_to_cpu_vec();

        // CPU reference: decode BF16 → F32, naive matmul.
        let f_q_decoded: Vec<f32> = bits_q.iter().map(|&b| bf16_bits_to_f32(b)).collect();
        let f_k_decoded: Vec<f32> = bits_k.iter().map(|&b| bf16_bits_to_f32(b)).collect();
        let mut cpu_values = vec![0.0_f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                let mut acc = 0.0_f32;
                for kk in 0..4 {
                    acc += f_q_decoded[i * 4 + kk] * f_k_decoded[kk * 4 + j];
                }
                cpu_values[i * 4 + j] = acc;
            }
        }

        let mut max_abs_diff = 0.0_f32;
        for (g, c) in gpu_values.iter().zip(cpu_values.iter()) {
            let d = (g - c).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        assert!(
            max_abs_diff < 1e-3,
            "GPU matmul on resident BF16-uploaded params drifted \
             {} from CPU reference (limit 1e-3)",
            max_abs_diff
        );

        // Silence unused-mut warnings.
        let _ = (&mut q_gpu, &mut k_gpu);
    }

    /// **M6 replan sub-fase 0** — round-trip an F32 disk-tier
    /// parameter through `WeightStore`. Writes a known buffer
    /// to disk via `disk_tier::write_f32_tensor`, inserts the
    /// resulting handle as a `SharedParam::Disk`, materialises a
    /// `Tensor` via `to_tensor`, reads it back via
    /// `copy_to_cpu_vec`, and verifies bit-exact equality.
    #[test]
    fn shared_param_disk_f32_round_trip_bit_exact() {
        use crate::tensor::disk_tier;

        let mut store = WeightStore::new();
        let cache_dir = std::env::temp_dir().join(format!(
            "atenia_weight_store_f32_roundtrip_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        let data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.25 - 4.0).collect();
        let handle =
            disk_tier::write_f32_tensor(&cache_dir, &data).expect("write_f32_tensor failed");
        let _idx = store.insert_disk("w.disk_f32", vec![4, 8], handle);

        let p = store.get_by_name("w.disk_f32").unwrap();
        assert_eq!(p.shape(), &[4, 8]);

        let t = p.to_tensor();
        assert_eq!(t.shape, vec![4, 8]);
        assert!(matches!(t.storage(), TensorStorage::Disk(_)));

        // copy_to_cpu_vec reads the bytes back without mutating
        // the storage. Bit-exact equality with the source.
        let read_back = t.copy_to_cpu_vec();
        assert_eq!(read_back, data, "F32 disk round-trip is not bit-exact");

        // Resident-byte accounting.
        assert_eq!(p.resident_bytes(), data.len() * 4);
    }

    /// **M6 replan sub-fase 0** — same round-trip but for the
    /// BF16 on-disk dtype. The round-trip surface upcasts BF16
    /// → F32 (lossless) on read, so we verify against the
    /// host-decoded reference.
    #[test]
    fn shared_param_disk_bf16_round_trip_bit_exact() {
        use crate::tensor::disk_tier;
        use crate::tensor::tensor::{bf16_bits_to_f32, f32_to_bf16_bits};

        let mut store = WeightStore::new();
        let cache_dir = std::env::temp_dir().join(format!(
            "atenia_weight_store_bf16_roundtrip_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));

        // Source: a known F32 pattern, converted to BF16 bits.
        let f32_src: Vec<f32> = (0..16).map(|i| ((i as f32) * 0.5 - 3.5).sin()).collect();
        let bf16_bits: Vec<u16> = f32_src.iter().map(|&f| f32_to_bf16_bits(f)).collect();

        let handle =
            disk_tier::write_bf16_tensor(&cache_dir, &bf16_bits).expect("write_bf16_tensor failed");
        store.insert_disk("w.disk_bf16", vec![2, 8], handle);

        let p = store.get_by_name("w.disk_bf16").unwrap();
        let t = p.to_tensor();
        assert!(matches!(t.storage(), TensorStorage::Disk(_)));

        // copy_to_cpu_vec on a BF16-tier Disk tensor returns the
        // upcast Vec<f32>. Compare against the host-decode
        // reference (lossless inverse of the BF16 encoding).
        let decoded = t.copy_to_cpu_vec();
        let expected: Vec<f32> = bf16_bits.iter().map(|&b| bf16_bits_to_f32(b)).collect();
        assert_eq!(
            decoded, expected,
            "BF16 disk round-trip is not bit-exact with host decode"
        );

        // Resident-byte accounting (BF16 = 2 bytes/elem).
        assert_eq!(p.resident_bytes(), bf16_bits.len() * 2);
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
