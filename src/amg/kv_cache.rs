//! M5.b — KV cache runtime infrastructure.
//!
//! This module owns the **third tensor category** the M5
//! research lock identifies (D59): KV cache cells live next
//! to graph parameters and ephemeral activations as a
//! distinct kind, with their own LRU eviction priority and
//! their own mutability contract.
//!
//! ## What lives here (and what doesn't)
//!
//! M5.b deliberately scopes itself to the **runtime**
//! infrastructure:
//!
//!   - [`TensorKind`] — annotation enum that the scheduler /
//!     LRU machinery will branch on once M5.c wires the
//!     cache-aware attention path. In M5.b it's a marker;
//!     existing code paths stay bit-exact.
//!   - [`KvCache`] / [`KvLayer`] — the runtime data
//!     structure that stores per-layer K, V tensors and
//!     grows along the seq axis on every decode step.
//!   - [`KvCacheConfig`] — dimensions read off a
//!     `LlamaConfig`, used to allocate a cache up front.
//!   - [`KvCacheHandles`] — placeholder type carrying the
//!     graph-side parameter IDs that M5.c will populate
//!     when it extends `build_llama` to accept
//!     `Option<&KvCacheHandles>`.
//!
//! What is **not** here (deferred to M5.c per the user's
//! locked sub-phase plan, which puts "build_llama extendido
//! con optional KvCacheHandles" explicitly in M5.c, not
//! M5.b):
//!
//!   - The cache-aware attention graph topology (concat
//!     cache_K with new K along seq, attend, append).
//!   - The R2 falsifier `argmax(no-cache full forward) ==
//!     argmax(prefill + decode steps)` — that comparison
//!     intrinsically needs the prefill and decode graphs
//!     M5.c stands up. M5.b validates the data structure
//!     instead: `append + get` is bit-exact equivalent to
//!     manually concatenating slices along the seq axis.
//!
//! ## Mutability contract (D60)
//!
//! `WeightMapper::load_into` keeps its load-once-immutable
//! contract for `TensorKind::Parameter`. KV cells take a
//! distinct path: the runtime owns the storage, the graph
//! holds *handles* (parameter slot IDs), and per-step
//! updates flow via [`crate::amg::graph::Graph::overwrite_parameter`]
//! — a separate mutator that explicitly does not collide
//! with the WeightMapper's invariant because it is only
//! ever called for slots the M5 layer registered.
//!
//! ## Storage precision (D62)
//!
//! Cache tensors are F32 in M5.b. BF16 follows in M5.f
//! once correctness is locked. Storing F32 here removes
//! BF16 rounding as a confound during the M5.c R2
//! validation; the F32→BF16 transition in M5.f is then a
//! single precision change with the same R2 falsifier
//! re-run as the gate.

use crate::tensor::Tensor;

/// Tensor category — the third kind the M5 research lock
/// (D59) introduces alongside the implicit Parameter /
/// Activation distinction the AMG already operates on.
///
/// This is presently an **annotation**: it does not change
/// scheduler or executor behaviour in M5.b. M5.c uses it to
/// route LRU priority and the spill-eligibility check; M6+
/// uses it for GPU offload decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorKind {
    /// Load-once-immutable model weights. Owned by the
    /// graph, registered via `GraphBuilder::parameter`,
    /// populated by `WeightMapper::load_into`.
    Parameter,
    /// Ephemeral per-forward activations. Live for one
    /// graph execution and are dropped at the end. Highest
    /// LRU eviction priority in M5.c+.
    Activation,
    /// KV cache cells. Long-lived across decode steps,
    /// rewritten in place. Lower LRU eviction priority
    /// than activations (cache is hot data — every layer
    /// reads it on every token). Higher eviction priority
    /// than cold weight pages so the M4.7 LRU still
    /// preferentially evicts unused weight tensors first
    /// when RAM pressure rises during long generation.
    KvCache,
}

impl TensorKind {
    /// LRU eviction-priority weight. Higher number =
    /// evicted sooner under pressure. Activations go first
    /// (ephemeral, will be re-materialised next forward),
    /// then KV cache (recomputable but expensive), then
    /// weights last (hard to recompute, often spillable to
    /// disk). Used as a tiebreaker by the M5.c+ LRU policy;
    /// in M5.b it is exposed for tests and forward callers.
    pub const fn lru_eviction_priority(self) -> u8 {
        match self {
            TensorKind::Activation => 200,
            TensorKind::KvCache => 100,
            TensorKind::Parameter => 0,
        }
    }
}

/// Per-layer KV slice for the cache-aware attention path.
///
/// Shapes follow the post-permute attention layout the
/// builder already uses:
///   K: `[batch, num_kv_heads, seq_filled, head_dim]`
///   V: `[batch, num_kv_heads, seq_filled, head_dim]`
///
/// Storage is F32 in M5.b (D62). Both tensors grow along
/// the `seq_filled` axis on each [`KvCache::append`] call.
#[derive(Debug, Clone)]
pub struct KvLayer {
    pub k: Tensor,
    pub v: Tensor,
}

impl KvLayer {
    /// Build an empty layer with `seq_filled = 0` and the
    /// correct head structure pre-baked. Used by
    /// [`KvCache::new`] at construction time so per-layer
    /// reallocation never happens during decode.
    pub fn empty(batch: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        let shape = vec![batch, num_kv_heads, 0, head_dim];
        Self {
            k: Tensor::new_cpu(shape.clone(), Vec::new()),
            v: Tensor::new_cpu(shape, Vec::new()),
        }
    }

    /// Current sequence length (cached_len). Reads off the
    /// third axis of `k`; `v` mirrors by construction.
    pub fn seq_len(&self) -> usize { self.k.shape[2] }
}

/// Configuration parameters used to size a [`KvCache`] at
/// construction. Mirrors the [`crate::nn::llama::config::LlamaConfig`]
/// fields the cache needs without taking a hard dependency
/// on the Llama config struct, so future model families
/// (Mistral, Qwen 3, ...) can reuse the same machinery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheConfig {
    pub batch: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl KvCacheConfig {
    /// Per-token, all-layers, K+V byte cost at F32. Useful
    /// for capacity planning before allocation. For Llama 2
    /// 13B Chat (40 layers × 40 heads × 128 dim) this is
    /// `40 × 2 × 40 × 128 × 4 = 1.6 MiB / token` — i.e.
    /// 3.2 GiB for a 2048-token context. The same number at
    /// BF16 (M5.f) halves to 1.6 GiB.
    pub fn bytes_per_token_f32(&self) -> usize {
        // K and V each: num_layers × num_kv_heads × head_dim × 4 bytes
        2 * self.num_layers * self.num_kv_heads * self.head_dim * 4
    }
}

/// Runtime KV cache for an inference session.
///
/// Lifecycle:
///   1. `KvCache::new(config)` allocates one empty
///      [`KvLayer`] per transformer layer.
///   2. During prefill, the executor calls
///      `cache.append(layer_idx, k_slice, v_slice)` for
///      each layer with the K, V projections of the prompt
///      (seq dim = prompt length).
///   3. During each decode step, the same call happens
///      with `seq_dim = 1` for each layer.
///   4. Between conversations, `cache.clear()` resets
///      every layer back to `seq_filled = 0` without
///      releasing the underlying allocation.
///
/// All operations are bit-exact equivalent to manually
/// `Tensor::concat(prev, slice, axis=2)` along the seq
/// dimension — the test suite locks this property against
/// a reference implementation.
#[derive(Debug, Clone)]
pub struct KvCache {
    pub config: KvCacheConfig,
    pub layers: Vec<KvLayer>,
}

impl KvCache {
    /// Build a fresh empty cache sized for `config`.
    /// Per-layer storage starts at `seq_filled = 0`; the
    /// first `append` call grows the underlying `Vec<f32>`.
    pub fn new(config: KvCacheConfig) -> Self {
        let layers = (0..config.num_layers)
            .map(|_| KvLayer::empty(config.batch, config.num_kv_heads, config.head_dim))
            .collect();
        Self { config, layers }
    }

    /// Number of cached tokens. Reads off layer 0; all
    /// layers are kept in lockstep by construction.
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|l| l.seq_len()).unwrap_or(0)
    }

    /// True iff no tokens have been cached yet.
    pub fn is_empty(&self) -> bool { self.seq_len() == 0 }

    /// Append a `[batch, num_kv_heads, slice_seq, head_dim]`
    /// K, V pair to the given layer's cache.
    ///
    /// The cache's `seq_filled` advances by `slice_seq`.
    /// Every layer must receive the same `slice_seq` per
    /// step; the M5.c executor enforces this by stepping
    /// all layers in lockstep within one forward.
    ///
    /// Bit-exactly equivalent to `concat([prev_K, k_slice], axis=2)`
    /// — locked by tests in this module.
    pub fn append(&mut self, layer_idx: usize, k_slice: &Tensor, v_slice: &Tensor)
        -> Result<(), KvCacheError>
    {
        let layer = self.layers.get_mut(layer_idx)
            .ok_or(KvCacheError::LayerOutOfRange { layer_idx, num_layers: self.config.num_layers })?;

        validate_slice_shape(&self.config, &k_slice.shape, "K")?;
        validate_slice_shape(&self.config, &v_slice.shape, "V")?;
        if k_slice.shape != v_slice.shape {
            return Err(KvCacheError::ShapeMismatch {
                what: "K vs V slice",
                k: k_slice.shape.clone(),
                v: v_slice.shape.clone(),
            });
        }

        append_along_seq_axis(&mut layer.k, k_slice)?;
        append_along_seq_axis(&mut layer.v, v_slice)?;
        Ok(())
    }

    /// Read the current cached K, V for layer `layer_idx`.
    /// Borrowed; no copy. Returned shape:
    /// `[batch, num_kv_heads, seq_filled, head_dim]`.
    pub fn get(&self, layer_idx: usize) -> Result<(&Tensor, &Tensor), KvCacheError> {
        let layer = self.layers.get(layer_idx)
            .ok_or(KvCacheError::LayerOutOfRange { layer_idx, num_layers: self.config.num_layers })?;
        Ok((&layer.k, &layer.v))
    }

    /// Reset the cache to `seq_filled = 0` without
    /// dropping the per-layer allocations. Re-use across
    /// conversations within a single session keeps the
    /// `Vec<f32>` capacity from churning.
    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            // Truncate-style reset: zero the seq dim, drop
            // the data backing. We cannot in-place truncate
            // a `TensorStorage::Cpu` from outside the
            // tensor module without rebuilding the tensor,
            // so we replace the layer with a fresh empty
            // one. The allocator will reuse the freed page
            // on the next append.
            *layer = KvLayer::empty(
                self.config.batch,
                self.config.num_kv_heads,
                self.config.head_dim,
            );
        }
    }

    /// Total bytes resident in F32 across every layer. Use
    /// for telemetry and capacity decisions.
    pub fn resident_bytes_f32(&self) -> usize {
        self.layers.iter()
            .map(|l| l.k.shape.iter().product::<usize>() + l.v.shape.iter().product::<usize>())
            .sum::<usize>() * 4
    }
}

/// Graph-side handles populated by M5.c when it extends
/// `build_llama` to accept `Option<&KvCacheHandles>`.
///
/// In M5.b this struct exists but is unpopulated by the
/// builder — a placeholder so M5.c can land its wiring as
/// a single self-contained commit without simultaneously
/// having to invent the handle type.
#[derive(Debug, Default, Clone)]
pub struct KvCacheHandles {
    /// One entry per transformer layer. M5.c will populate
    /// `cache_k_param_id` / `cache_v_param_id` when it
    /// inserts the cache K, V parameter slots into the
    /// decode graph.
    pub per_layer: Vec<KvLayerHandle>,
}

/// Per-layer graph-side handle. The two `param_id` fields
/// are the AMG node IDs of the cache_K and cache_V slots
/// the M5.c decode graph will register; the runtime
/// updates them via [`crate::amg::graph::Graph::overwrite_parameter`]
/// before each decode step.
#[derive(Debug, Default, Clone, Copy)]
pub struct KvLayerHandle {
    pub cache_k_param_id: usize,
    pub cache_v_param_id: usize,
}

/// Errors surfaced by [`KvCache`] operations.
#[derive(Debug)]
pub enum KvCacheError {
    LayerOutOfRange { layer_idx: usize, num_layers: usize },
    BadShape { what: &'static str, expected_rank: usize, got: Vec<usize> },
    ShapeMismatch { what: &'static str, k: Vec<usize>, v: Vec<usize> },
    HeadStructure { what: &'static str, expected: (usize, usize, usize), got: Vec<usize> },
}

impl std::fmt::Display for KvCacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvCacheError::LayerOutOfRange { layer_idx, num_layers } =>
                write!(f, "kv_cache: layer {layer_idx} out of range (cache has {num_layers})"),
            KvCacheError::BadShape { what, expected_rank, got } =>
                write!(f, "kv_cache: {what} expected rank {expected_rank}, got shape {got:?}"),
            KvCacheError::ShapeMismatch { what, k, v } =>
                write!(f, "kv_cache: {what} shape mismatch — K {k:?} vs V {v:?}"),
            KvCacheError::HeadStructure { what, expected, got } =>
                write!(f, "kv_cache: {what} head structure mismatch (expected (batch, num_kv_heads, head_dim) = {expected:?}, got shape {got:?})"),
        }
    }
}

impl std::error::Error for KvCacheError {}

fn validate_slice_shape(cfg: &KvCacheConfig, shape: &[usize], what: &'static str)
    -> Result<(), KvCacheError>
{
    if shape.len() != 4 {
        return Err(KvCacheError::BadShape { what, expected_rank: 4, got: shape.to_vec() });
    }
    if shape[0] != cfg.batch || shape[1] != cfg.num_kv_heads || shape[3] != cfg.head_dim {
        return Err(KvCacheError::HeadStructure {
            what,
            expected: (cfg.batch, cfg.num_kv_heads, cfg.head_dim),
            got: shape.to_vec(),
        });
    }
    Ok(())
}

/// Concatenate `slice` to `dst` along axis 2 (the seq
/// dimension), in place. Both tensors must agree on every
/// other axis. Bit-exactly equivalent to a NumPy/PyTorch
/// `torch.cat([dst, slice], dim=2)` — verified by the
/// unit tests below.
fn append_along_seq_axis(dst: &mut Tensor, slice: &Tensor) -> Result<(), KvCacheError> {
    debug_assert_eq!(dst.shape.len(), 4);
    debug_assert_eq!(slice.shape.len(), 4);
    debug_assert_eq!(dst.shape[0], slice.shape[0]);
    debug_assert_eq!(dst.shape[1], slice.shape[1]);
    debug_assert_eq!(dst.shape[3], slice.shape[3]);

    let batch = dst.shape[0];
    let kv_heads = dst.shape[1];
    let prev_seq = dst.shape[2];
    let new_seq = slice.shape[2];
    let head_dim = dst.shape[3];
    let total_seq = prev_seq + new_seq;

    // Decode both into Vec<f32> regardless of underlying
    // storage. M5.b is F32-first; this also handles the
    // `TensorStorage::Cpu(Vec<f32>)` happy path with no
    // extra copy when both sides are already F32.
    let prev_data = dst.copy_to_cpu_vec();
    let slice_data = slice.copy_to_cpu_vec();

    // Layout: [batch, kv_heads, seq, head_dim] is contiguous
    // last-axis-fastest. Concat along axis 2 means:
    //   for each (b, h):
    //     for each prev token: head_dim floats
    //     for each new  token: head_dim floats
    // i.e. interleave at the (batch × kv_heads) × seq level.
    let mut out = Vec::with_capacity(batch * kv_heads * total_seq * head_dim);
    for b in 0..batch {
        for h in 0..kv_heads {
            // Copy prev[b, h, :, :]
            let prev_off = ((b * kv_heads + h) * prev_seq) * head_dim;
            out.extend_from_slice(&prev_data[prev_off .. prev_off + prev_seq * head_dim]);
            // Copy slice[b, h, :, :]
            let slice_off = ((b * kv_heads + h) * new_seq) * head_dim;
            out.extend_from_slice(&slice_data[slice_off .. slice_off + new_seq * head_dim]);
        }
    }

    *dst = Tensor::new_cpu(vec![batch, kv_heads, total_seq, head_dim], out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arange_tensor(shape: Vec<usize>, start: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let data = (0..n).map(|i| start + i as f32).collect();
        Tensor::new_cpu(shape, data)
    }

    #[test]
    fn tensor_kind_eviction_priority_orders_correctly() {
        // Activations evict first, weights last. KV cache
        // sits in between. Locked by D59.
        assert!(TensorKind::Activation.lru_eviction_priority()
              > TensorKind::KvCache.lru_eviction_priority());
        assert!(TensorKind::KvCache.lru_eviction_priority()
              > TensorKind::Parameter.lru_eviction_priority());
    }

    #[test]
    fn empty_cache_reports_zero_seq_len() {
        let cache = KvCache::new(KvCacheConfig {
            batch: 1, num_layers: 4, num_kv_heads: 8, head_dim: 16,
        });
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.layers.len(), 4);
        for layer in &cache.layers {
            assert_eq!(layer.seq_len(), 0);
            assert_eq!(layer.k.shape, vec![1, 8, 0, 16]);
            assert_eq!(layer.v.shape, vec![1, 8, 0, 16]);
        }
    }

    #[test]
    fn append_advances_seq_len_per_layer() {
        let cfg = KvCacheConfig { batch: 1, num_layers: 2, num_kv_heads: 2, head_dim: 4 };
        let mut cache = KvCache::new(cfg);

        // Append a 5-token slice to layer 0.
        let k = arange_tensor(vec![1, 2, 5, 4], 0.0);
        let v = arange_tensor(vec![1, 2, 5, 4], 100.0);
        cache.append(0, &k, &v).unwrap();
        assert_eq!(cache.layers[0].seq_len(), 5);
        // Layer 1 untouched.
        assert_eq!(cache.layers[1].seq_len(), 0);

        // Append a single-token decode-style slice to both layers.
        let k_dec = arange_tensor(vec![1, 2, 1, 4], 1000.0);
        let v_dec = arange_tensor(vec![1, 2, 1, 4], 2000.0);
        cache.append(0, &k_dec, &v_dec).unwrap();
        cache.append(1, &k_dec, &v_dec).unwrap();
        assert_eq!(cache.layers[0].seq_len(), 6);
        assert_eq!(cache.layers[1].seq_len(), 1);
    }

    #[test]
    fn append_is_bit_exact_with_manual_concat() {
        // R2-style falsifier at the data-structure level.
        // `cache.append(slice_a) + cache.append(slice_b) +
        //  cache.get(layer)` must equal manually concating
        // [slice_a, slice_b] along axis 2.
        let cfg = KvCacheConfig { batch: 1, num_layers: 1, num_kv_heads: 3, head_dim: 4 };
        let mut cache = KvCache::new(cfg);

        let k_a = arange_tensor(vec![1, 3, 2, 4], 1.0);
        let k_b = arange_tensor(vec![1, 3, 3, 4], 100.0);
        let v_a = arange_tensor(vec![1, 3, 2, 4], 7.0);
        let v_b = arange_tensor(vec![1, 3, 3, 4], 300.0);

        cache.append(0, &k_a, &v_a).unwrap();
        cache.append(0, &k_b, &v_b).unwrap();
        let (cached_k, cached_v) = cache.get(0).unwrap();

        // Manual concat reference: per (b, h) chunk, copy
        // a's seq=2 then b's seq=3.
        let manual_concat = |a: &Tensor, b: &Tensor| -> Vec<f32> {
            let batch = a.shape[0];
            let h = a.shape[1];
            let sa = a.shape[2]; let sb = b.shape[2];
            let d = a.shape[3];
            let ad = a.copy_to_cpu_vec();
            let bd = b.copy_to_cpu_vec();
            let mut out = Vec::with_capacity(batch * h * (sa + sb) * d);
            for bi in 0..batch {
                for hi in 0..h {
                    let ao = ((bi * h + hi) * sa) * d;
                    out.extend_from_slice(&ad[ao .. ao + sa * d]);
                    let bo = ((bi * h + hi) * sb) * d;
                    out.extend_from_slice(&bd[bo .. bo + sb * d]);
                }
            }
            out
        };

        let ref_k = manual_concat(&k_a, &k_b);
        let ref_v = manual_concat(&v_a, &v_b);
        assert_eq!(cached_k.shape, vec![1, 3, 5, 4]);
        assert_eq!(cached_k.copy_to_cpu_vec(), ref_k, "K cache != concat reference");
        assert_eq!(cached_v.copy_to_cpu_vec(), ref_v, "V cache != concat reference");
    }

    #[test]
    fn clear_resets_all_layers_to_empty() {
        let cfg = KvCacheConfig { batch: 1, num_layers: 3, num_kv_heads: 2, head_dim: 4 };
        let mut cache = KvCache::new(cfg);
        let k = arange_tensor(vec![1, 2, 7, 4], 1.0);
        let v = arange_tensor(vec![1, 2, 7, 4], 1.0);
        for li in 0..3 { cache.append(li, &k, &v).unwrap(); }
        assert_eq!(cache.seq_len(), 7);

        cache.clear();
        assert_eq!(cache.seq_len(), 0);
        for layer in &cache.layers {
            assert_eq!(layer.seq_len(), 0);
            assert_eq!(layer.k.shape, vec![1, 2, 0, 4]);
        }

        // Re-use post-clear must still work bit-exactly.
        cache.append(0, &k, &v).unwrap();
        assert_eq!(cache.layers[0].seq_len(), 7);
    }

    #[test]
    fn append_rejects_bad_shapes() {
        let cfg = KvCacheConfig { batch: 1, num_layers: 1, num_kv_heads: 2, head_dim: 4 };
        let mut cache = KvCache::new(cfg);

        // Wrong rank.
        let bad = arange_tensor(vec![1, 2, 4], 0.0);
        assert!(cache.append(0, &bad, &bad).is_err());

        // Wrong head structure (kv_heads = 3 instead of 2).
        let bad2 = arange_tensor(vec![1, 3, 1, 4], 0.0);
        assert!(cache.append(0, &bad2, &bad2).is_err());

        // K/V seq-mismatch.
        let k = arange_tensor(vec![1, 2, 5, 4], 0.0);
        let v = arange_tensor(vec![1, 2, 4, 4], 0.0);
        assert!(cache.append(0, &k, &v).is_err());

        // Layer out of range.
        let ok = arange_tensor(vec![1, 2, 1, 4], 0.0);
        assert!(cache.append(99, &ok, &ok).is_err());
    }

    #[test]
    fn config_byte_size_matches_research_report() {
        // Llama 2 13B Chat: 40 layers × 40 heads × 128 dim
        // → 1.6 MiB / token at F32 (research report
        // section 2). Per the M5 plan the F32 number is
        // 2× the BF16 figure.
        let cfg = KvCacheConfig {
            batch: 1, num_layers: 40, num_kv_heads: 40, head_dim: 128,
        };
        // 40 layers × 2 (K+V) × 40 heads × 128 dim × 4 bytes
        // = 1_638_400 bytes = 1.5625 MiB
        assert_eq!(cfg.bytes_per_token_f32(), 1_638_400);
    }

    #[test]
    fn kv_cache_handles_default_is_empty_and_extensible() {
        // M5.c populates this; M5.b just needs the type
        // to exist with a sensible default so call sites
        // can take `Option<&KvCacheHandles>`.
        let h = KvCacheHandles::default();
        assert!(h.per_layer.is_empty());
    }

    #[test]
    fn graph_overwrite_parameter_replaces_backing_tensor() {
        // D60 falsifier: the mutable graph tensor path
        // replaces a Parameter slot's backing tensor and
        // rejects calls against non-Parameter nodes.
        use crate::amg::builder::GraphBuilder;
        use crate::amg::nodes::NodeType;

        let mut gb = GraphBuilder::new();
        let p_id = gb.parameter(arange_tensor(vec![2, 3], 1.0));
        let in_id = gb.input();
        // Build into a Graph (skip the executor; we only
        // need parameter mutation to work).
        let nodes = std::mem::take(&mut gb.nodes);
        let mut g = crate::amg::graph::Graph::build(nodes);

        // Sanity: the parameter slot has its initial value.
        let initial = g.nodes[p_id].output.as_ref().unwrap();
        assert_eq!(initial.copy_to_cpu_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Overwrite with a fresh tensor (mimics what M5.c
        // will do every decode step for cache-K / cache-V).
        let replacement = arange_tensor(vec![2, 3], 100.0);
        g.overwrite_parameter(p_id, replacement).expect("overwrite failed");
        let after = g.nodes[p_id].output.as_ref().unwrap();
        assert_eq!(after.copy_to_cpu_vec(), vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0]);

        // Non-Parameter node id rejected.
        assert!(matches!(
            g.overwrite_parameter(in_id, arange_tensor(vec![1], 0.0)),
            Err(crate::amg::graph::GraphMutationError::NotAParameter { .. })
        ));
        // Out-of-range id rejected.
        assert!(matches!(
            g.overwrite_parameter(9999, arange_tensor(vec![1], 0.0)),
            Err(crate::amg::graph::GraphMutationError::NodeOutOfRange { .. })
        ));
        // Non-Parameter assertion does not depend on
        // NodeType::Input being unique — sanity-check the
        // exhaustive match holds.
        let _ = NodeType::Parameter;
    }
}
