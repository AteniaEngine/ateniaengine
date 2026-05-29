//! **MOE-10** — Real expert tensor binding (experimental, CPU-only).
//!
//! MOE-9 produced a [`MoeWeightMap`]: structured *metadata* (names, shapes,
//! roles) for every MoE tensor in a checkpoint, but **no tensor data**. This
//! module is the missing bridge:
//!
//! ```text
//!   MoeWeightMap (metadata)  +  byte resolver (name → Vec<f32>)
//!     → resolve real gate/up/down bytes per expert
//!     → construct real MoeDenseExpert       (MOE-3 executor)
//!     → assemble a real MoeDenseLayer
//!     → forward_sparse over real weights    (MOE-4 sparse path)
//! ```
//!
//! ## What this is and is NOT
//!
//! * It **does** turn real checkpoint tensor data into executable experts and
//!   run the existing sparse reference forward over them, CPU-only,
//!   correctness-first.
//! * It does **not** load a full model, parse `config.json`, build a graph,
//!   wire a transformer, or lift the MOE-2 loader fail-loud guard. A real MoE
//!   checkpoint still refuses to load *as a model*. This binding is an
//!   isolated `src/moe/` path that a caller drives explicitly with a reader.
//! * No Mixtral / Qwen-MoE end-to-end support is claimed: this validates the
//!   *mechanism* (real bytes → real experts → real sparse forward) on small
//!   synthetic checkpoints that use real Mixtral / Qwen-MoE tensor names.
//!
//! ## Decoupling
//!
//! The binding never references the loader directly. It takes a **byte
//! resolver** closure `Fn(&str) -> Option<Vec<f32>>`, so `src/moe/` stays
//! free of any loader dependency. A caller wires it to a `SafetensorsReader`
//! with one line:
//! `|name| reader.get(name).and_then(|e| e.to_vec_f32().ok())`.

use super::data_plane::{ExpertTensors, MoeWeightMap};
use super::dense::{MoeDenseError, MoeDenseExpert, MoeDenseLayer};

/// Errors from resolving and binding real expert tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoeBindingError {
    /// The requested layer has no MoE weights in the map.
    LayerNotFound { layer_id: usize },
    /// The layer has no experts.
    NoExperts { layer_id: usize },
    /// An expert is missing one of its three projections in the metadata.
    IncompleteExpert { layer_id: usize, expert_id: usize },
    /// The router tensor is absent for the layer.
    MissingRouter { layer_id: usize },
    /// The byte resolver returned `None` for a tensor the metadata listed.
    UnresolvedTensor { name: String },
    /// A tensor's shape was not the expected 2-D `[rows, cols]`.
    BadRank { name: String, rank: usize },
    /// Two projections that must share a shape disagree, or a projection's
    /// shape is inconsistent with the inferred `(d_model, d_ff)`.
    ShapeInconsistency { detail: String },
    /// The resolved byte count did not match the metadata shape.
    DataLengthMismatch {
        name: String,
        expected: usize,
        actual: usize,
    },
    /// A packed expert tensor was missing (`gate_up_proj` or `down_proj`).
    MissingPackedTensor { layer_id: usize, which: &'static str },
    /// A packed tensor's shape was not the expected 3-D layout.
    PackedBadRank { name: String, rank: usize },
    /// Error bubbled up from constructing/validating the dense executor.
    Dense(MoeDenseError),
}

impl std::fmt::Display for MoeBindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MoeBindingError::LayerNotFound { layer_id } => {
                write!(f, "moe-binding: layer {layer_id} has no MoE weights")
            }
            MoeBindingError::NoExperts { layer_id } => {
                write!(f, "moe-binding: layer {layer_id} has no experts")
            }
            MoeBindingError::IncompleteExpert { layer_id, expert_id } => write!(
                f,
                "moe-binding: layer {layer_id} expert {expert_id} is missing a gate/up/down projection"
            ),
            MoeBindingError::MissingRouter { layer_id } => {
                write!(f, "moe-binding: layer {layer_id} has no router tensor")
            }
            MoeBindingError::UnresolvedTensor { name } => {
                write!(f, "moe-binding: byte resolver returned no data for '{name}'")
            }
            MoeBindingError::BadRank { name, rank } => {
                write!(f, "moe-binding: tensor '{name}' has rank {rank}, expected 2-D [rows, cols]")
            }
            MoeBindingError::ShapeInconsistency { detail } => {
                write!(f, "moe-binding: shape inconsistency: {detail}")
            }
            MoeBindingError::DataLengthMismatch { name, expected, actual } => write!(
                f,
                "moe-binding: tensor '{name}' data length {actual} != shape product {expected}"
            ),
            MoeBindingError::MissingPackedTensor { layer_id, which } => write!(
                f,
                "moe-binding: layer {layer_id} missing packed expert tensor '{which}'"
            ),
            MoeBindingError::PackedBadRank { name, rank } => write!(
                f,
                "moe-binding: packed tensor '{name}' has rank {rank}, expected 3-D"
            ),
            MoeBindingError::Dense(e) => write!(f, "moe-binding: {e}"),
        }
    }
}

impl std::error::Error for MoeBindingError {}

impl From<MoeDenseError> for MoeBindingError {
    fn from(e: MoeDenseError) -> Self {
        MoeBindingError::Dense(e)
    }
}

/// Resolved shape of a binding: the dimensions inferred from real tensor
/// metadata. `[d_ff, d_model]` for gate/up, `[d_model, d_ff]` for down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BindingShape {
    pub d_model: usize,
    pub d_ff: usize,
}

/// **MOE-10** — a real expert tensor binding: resolves one expert's three
/// projections from [`MoeWeightMap`] metadata + a byte resolver, and builds a
/// runnable [`MoeDenseExpert`].
///
/// "Binding" = the act of attaching real checkpoint bytes to the metadata
/// the data plane already holds. Construction validates ranks, cross-checks
/// that gate/up/down agree on `(d_model, d_ff)`, and that resolved byte
/// counts match the declared shapes.
#[derive(Debug, Clone)]
pub struct RealExpertTensorBinding {
    pub layer_id: usize,
    pub expert_id: usize,
    pub shape: BindingShape,
    /// The constructed, runnable expert.
    pub expert: MoeDenseExpert,
}

/// Validate a 2-D shape and return `(rows, cols)`.
fn rows_cols(name: &str, shape: &[usize]) -> Result<(usize, usize), MoeBindingError> {
    if shape.len() != 2 {
        return Err(MoeBindingError::BadRank {
            name: name.to_string(),
            rank: shape.len(),
        });
    }
    Ok((shape[0], shape[1]))
}

/// Resolve a tensor's f32 data and verify its length matches the shape.
fn resolve_checked<F>(
    name: &str,
    shape: &[usize],
    resolve: &F,
) -> Result<Vec<f32>, MoeBindingError>
where
    F: Fn(&str) -> Option<Vec<f32>>,
{
    let data = resolve(name).ok_or_else(|| MoeBindingError::UnresolvedTensor {
        name: name.to_string(),
    })?;
    let expected: usize = shape.iter().product();
    if data.len() != expected {
        return Err(MoeBindingError::DataLengthMismatch {
            name: name.to_string(),
            expected,
            actual: data.len(),
        });
    }
    Ok(data)
}

impl RealExpertTensorBinding {
    /// Resolve + bind one expert's `(gate, up, down)` projections into a
    /// runnable [`MoeDenseExpert`].
    ///
    /// Expected layouts (HuggingFace row-major): gate/up are
    /// `[d_ff, d_model]`, down is `[d_model, d_ff]` — exactly the
    /// [`MoeDenseExpert`] convention, so no transpose is needed. Mixtral's
    /// `w1/w3/w2` already map to gate/up/down via the MOE-2 classifier.
    pub fn resolve<F>(
        layer_id: usize,
        expert_id: usize,
        tensors: &ExpertTensors,
        resolve: &F,
    ) -> Result<Self, MoeBindingError>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        let (gate_m, up_m, down_m) = match (&tensors.gate, &tensors.up, &tensors.down) {
            (Some(g), Some(u), Some(d)) => (g, u, d),
            _ => {
                return Err(MoeBindingError::IncompleteExpert { layer_id, expert_id });
            }
        };

        // gate/up: [d_ff, d_model]; down: [d_model, d_ff].
        let (gate_ff, gate_dm) = rows_cols(&gate_m.name, &gate_m.shape)?;
        let (up_ff, up_dm) = rows_cols(&up_m.name, &up_m.shape)?;
        let (down_dm, down_ff) = rows_cols(&down_m.name, &down_m.shape)?;

        if gate_ff != up_ff || gate_dm != up_dm {
            return Err(MoeBindingError::ShapeInconsistency {
                detail: format!(
                    "gate {:?} and up {:?} must share shape",
                    gate_m.shape, up_m.shape
                ),
            });
        }
        if down_dm != gate_dm || down_ff != gate_ff {
            return Err(MoeBindingError::ShapeInconsistency {
                detail: format!(
                    "down {:?} inconsistent with gate/up (d_model={gate_dm}, d_ff={gate_ff})",
                    down_m.shape
                ),
            });
        }

        let d_model = gate_dm;
        let d_ff = gate_ff;

        let w_gate = resolve_checked(&gate_m.name, &gate_m.shape, resolve)?;
        let w_up = resolve_checked(&up_m.name, &up_m.shape, resolve)?;
        let w_down = resolve_checked(&down_m.name, &down_m.shape, resolve)?;

        let expert = MoeDenseExpert::new(d_model, d_ff, w_gate, w_up, w_down)?;

        Ok(Self {
            layer_id,
            expert_id,
            shape: BindingShape { d_model, d_ff },
            expert,
        })
    }
}

/// Build a runnable, real-weight [`MoeDenseLayer`] for one layer of a
/// checkpoint: resolve every expert + the router from the [`MoeWeightMap`]
/// metadata via the byte resolver, then assemble and validate the layer.
///
/// `conceptual_top_k` is carried through to the layer (the sparse forward
/// takes its own `k`); pass the checkpoint's `experts_per_token` when known.
///
/// Experts are bound in **ascending expert-id order** (the `MoeWeightMap`
/// stores them in a `BTreeMap`), and the router rows must follow that same
/// order — which is the on-disk convention for the supported families.
pub fn build_real_layer<F>(
    map: &MoeWeightMap,
    layer_id: usize,
    conceptual_top_k: usize,
    resolve: &F,
) -> Result<MoeDenseLayer, MoeBindingError>
where
    F: Fn(&str) -> Option<Vec<f32>>,
{
    let layer = map
        .layer(layer_id)
        .ok_or(MoeBindingError::LayerNotFound { layer_id })?;
    if layer.experts.is_empty() {
        return Err(MoeBindingError::NoExperts { layer_id });
    }

    // Resolve experts in ascending id order.
    let mut experts: Vec<MoeDenseExpert> = Vec::with_capacity(layer.experts.len());
    let mut d_model = 0usize;
    let mut d_ff = 0usize;
    for (&expert_id, tensors) in &layer.experts {
        let binding = RealExpertTensorBinding::resolve(layer_id, expert_id, tensors, resolve)?;
        if experts.is_empty() {
            d_model = binding.shape.d_model;
            d_ff = binding.shape.d_ff;
        } else if binding.shape.d_model != d_model || binding.shape.d_ff != d_ff {
            return Err(MoeBindingError::ShapeInconsistency {
                detail: format!(
                    "expert {expert_id} shape (d_model={}, d_ff={}) differs from layer (d_model={d_model}, d_ff={d_ff})",
                    binding.shape.d_model, binding.shape.d_ff
                ),
            });
        }
        experts.push(binding.expert);
    }

    // Resolve the router: [num_experts, d_model].
    let router_m = layer
        .router
        .as_ref()
        .ok_or(MoeBindingError::MissingRouter { layer_id })?;
    let (router_rows, router_cols) = rows_cols(&router_m.name, &router_m.shape)?;
    if router_cols != d_model {
        return Err(MoeBindingError::ShapeInconsistency {
            detail: format!(
                "router cols {router_cols} != d_model {d_model} (shape {:?})",
                router_m.shape
            ),
        });
    }
    if router_rows != experts.len() {
        return Err(MoeBindingError::ShapeInconsistency {
            detail: format!(
                "router rows {router_rows} != expert count {}",
                experts.len()
            ),
        });
    }
    let w_router = resolve_checked(&router_m.name, &router_m.shape, resolve)?;

    MoeDenseLayer::new(d_model, d_ff, w_router, experts, conceptual_top_k).map_err(Into::into)
}

// ============================================================================
// MOE-15 — packed/fused expert support
// ============================================================================

/// Dimensions inferred from a layer's packed expert tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedExpertDims {
    pub num_experts: usize,
    pub d_model: usize,
    pub d_ff: usize,
}

/// Infer `(num_experts, d_model, d_ff)` from the packed tensor shapes.
///
/// **Assumed layout** (verified against tiny Mixtral / Qwen2-MoE checkpoints):
/// - `gate_up_proj`: `[num_experts, 2*d_ff, d_model]` (gate+up fused on dim 1)
/// - `down_proj`:    `[num_experts, d_model, d_ff]`
///
/// If the real layout differs, this returns a `ShapeInconsistency` rather
/// than silently mis-slicing.
pub fn packed_dims(
    gate_up: &super::data_plane::MoeTensorEntry,
    down: &super::data_plane::MoeTensorEntry,
) -> Result<PackedExpertDims, MoeBindingError> {
    if gate_up.shape.len() != 3 {
        return Err(MoeBindingError::PackedBadRank {
            name: gate_up.name.clone(),
            rank: gate_up.shape.len(),
        });
    }
    if down.shape.len() != 3 {
        return Err(MoeBindingError::PackedBadRank {
            name: down.name.clone(),
            rank: down.shape.len(),
        });
    }
    let ne = gate_up.shape[0];
    let two_dff = gate_up.shape[1];
    let d_model = gate_up.shape[2];
    if two_dff % 2 != 0 {
        return Err(MoeBindingError::ShapeInconsistency {
            detail: format!(
                "gate_up_proj dim1 ({two_dff}) is not 2*d_ff (must be even); shape {:?}",
                gate_up.shape
            ),
        });
    }
    let d_ff = two_dff / 2;
    // Cross-check down: [num_experts, d_model, d_ff].
    if down.shape[0] != ne || down.shape[1] != d_model || down.shape[2] != d_ff {
        return Err(MoeBindingError::ShapeInconsistency {
            detail: format!(
                "down_proj {:?} inconsistent with gate_up_proj {:?} \
                 (expected [{ne}, {d_model}, {d_ff}])",
                down.shape, gate_up.shape
            ),
        });
    }
    Ok(PackedExpertDims { num_experts: ne, d_model, d_ff })
}

/// **MOE-15** — build a single `MoeDenseExpert` for `expert_id` from already
/// resolved packed tensor data.
///
/// `gate_up_data` is the full row-major `gate_up_proj`
/// (`num_experts * 2*d_ff * d_model`); `down_data` the full `down_proj`
/// (`num_experts * d_model * d_ff`). For expert `E`:
/// - gate = `gate_up[E][0 .. d_ff, :]`   (first half on the fused dim)
/// - up   = `gate_up[E][d_ff .. 2*d_ff, :]`
/// - down = `down[E]`
fn build_packed_expert_from_data(
    gate_up_data: &[f32],
    down_data: &[f32],
    dims: PackedExpertDims,
    expert_id: usize,
) -> Result<MoeDenseExpert, MoeBindingError> {
    let PackedExpertDims { num_experts, d_model, d_ff } = dims;
    let gate_up_per_expert = 2 * d_ff * d_model;
    let down_per_expert = d_model * d_ff;
    let half = d_ff * d_model;

    let gu_base = expert_id * gate_up_per_expert;
    let dn_base = expert_id * down_per_expert;
    if gu_base + gate_up_per_expert > gate_up_data.len()
        || dn_base + down_per_expert > down_data.len()
    {
        return Err(MoeBindingError::ShapeInconsistency {
            detail: format!(
                "packed expert {expert_id} slice out of range (num_experts={num_experts})"
            ),
        });
    }
    let w_gate = gate_up_data[gu_base..gu_base + half].to_vec();
    let w_up = gate_up_data[gu_base + half..gu_base + 2 * half].to_vec();
    let w_down = down_data[dn_base..dn_base + down_per_expert].to_vec();
    MoeDenseExpert::new(d_model, d_ff, w_gate, w_up, w_down).map_err(Into::into)
}

/// **MOE-15** — build a runnable `MoeDenseLayer` from a layer's packed
/// expert tensors. Resolves `gate_up_proj` + `down_proj` once, slices every
/// expert out, and attaches the router. Same output type as the classic
/// `build_real_layer`, so the rest of the pipeline is unchanged.
pub fn build_packed_layer<F>(
    map: &MoeWeightMap,
    layer_id: usize,
    conceptual_top_k: usize,
    resolve: &F,
) -> Result<MoeDenseLayer, MoeBindingError>
where
    F: Fn(&str) -> Option<Vec<f32>>,
{
    let layer = map
        .layer(layer_id)
        .ok_or(MoeBindingError::LayerNotFound { layer_id })?;
    let gate_up = layer
        .packed_gate_up
        .as_ref()
        .ok_or(MoeBindingError::MissingPackedTensor { layer_id, which: "gate_up_proj" })?;
    let down = layer
        .packed_down
        .as_ref()
        .ok_or(MoeBindingError::MissingPackedTensor { layer_id, which: "down_proj" })?;

    let dims = packed_dims(gate_up, down)?;

    let gate_up_data = resolve_checked(&gate_up.name, &gate_up.shape, resolve)?;
    let down_data = resolve_checked(&down.name, &down.shape, resolve)?;

    let mut experts = Vec::with_capacity(dims.num_experts);
    for e in 0..dims.num_experts {
        experts.push(build_packed_expert_from_data(&gate_up_data, &down_data, dims, e)?);
    }

    // Router: [num_experts, d_model].
    let router_m = layer
        .router
        .as_ref()
        .ok_or(MoeBindingError::MissingRouter { layer_id })?;
    let (router_rows, router_cols) = rows_cols(&router_m.name, &router_m.shape)?;
    if router_cols != dims.d_model || router_rows != dims.num_experts {
        return Err(MoeBindingError::ShapeInconsistency {
            detail: format!(
                "router shape {:?} != [num_experts={}, d_model={}]",
                router_m.shape, dims.num_experts, dims.d_model
            ),
        });
    }
    let w_router = resolve_checked(&router_m.name, &router_m.shape, resolve)?;

    MoeDenseLayer::new(dims.d_model, dims.d_ff, w_router, experts, conceptual_top_k)
        .map_err(Into::into)
}

// ============================================================================
// Tests (synthetic in-memory tensors — no model, no loader, no graph)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Deterministic xorshift (same generator as the dense fixture) so the
    /// real-binding tests can build reproducible weights with no dependency.
    fn seeded(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state >> 11) as u32;
            out.push((u as f32 / u32::MAX as f32) * 2.0 - 1.0);
        }
        out
    }

    /// Build a Mixtral-named metadata map + an in-memory byte store for a
    /// single layer with `n` experts of the given dims. Returns the map and
    /// a resolver closure source (HashMap name → Vec<f32>).
    fn synthetic_mixtral(
        n: usize,
        d_model: usize,
        d_ff: usize,
    ) -> (MoeWeightMap, HashMap<String, Vec<f32>>) {
        let mut names_shapes: Vec<(String, Vec<usize>)> = Vec::new();
        let mut store: HashMap<String, Vec<f32>> = HashMap::new();

        let router = "model.layers.0.block_sparse_moe.gate.weight".to_string();
        names_shapes.push((router.clone(), vec![n, d_model]));
        store.insert(router, seeded(1, n * d_model));

        for e in 0..n {
            let base = 100 + e as u64;
            // w1=gate [d_ff,d_model], w3=up [d_ff,d_model], w2=down [d_model,d_ff].
            let g = format!("model.layers.0.block_sparse_moe.experts.{e}.w1.weight");
            let u = format!("model.layers.0.block_sparse_moe.experts.{e}.w3.weight");
            let d = format!("model.layers.0.block_sparse_moe.experts.{e}.w2.weight");
            names_shapes.push((g.clone(), vec![d_ff, d_model]));
            names_shapes.push((u.clone(), vec![d_ff, d_model]));
            names_shapes.push((d.clone(), vec![d_model, d_ff]));
            store.insert(g, seeded(base * 10 + 1, d_ff * d_model));
            store.insert(u, seeded(base * 10 + 2, d_ff * d_model));
            store.insert(d, seeded(base * 10 + 3, d_model * d_ff));
        }

        let map = MoeWeightMap::from_tensors(
            names_shapes.iter().map(|(n, s)| (n.as_str(), s.clone())),
        );
        (map, store)
    }

    fn resolver(store: &HashMap<String, Vec<f32>>) -> impl Fn(&str) -> Option<Vec<f32>> + '_ {
        move |name: &str| store.get(name).cloned()
    }

    #[test]
    fn resolves_a_single_real_expert() {
        let (map, store) = synthetic_mixtral(2, 8, 16);
        let resolve = resolver(&store);
        let tensors = map.expert(0, 1).unwrap();
        let binding = RealExpertTensorBinding::resolve(0, 1, tensors, &resolve).unwrap();
        assert_eq!(binding.shape.d_model, 8);
        assert_eq!(binding.shape.d_ff, 16);
        assert_eq!(binding.expert.w_gate.len(), 16 * 8);
        assert_eq!(binding.expert.w_down.len(), 8 * 16);
    }

    #[test]
    fn builds_a_real_layer_and_runs_sparse_forward() {
        let (map, store) = synthetic_mixtral(4, 8, 16);
        let resolve = resolver(&store);
        let layer = build_real_layer(&map, 0, 2, &resolve).unwrap();
        assert_eq!(layer.num_experts(), 4);
        let x = seeded(7, layer.d_model);
        let sparse = layer.forward_sparse(&x, 2).unwrap();
        assert_eq!(sparse.selected_experts.len(), 2);
        assert!(sparse.output.iter().all(|v| v.is_finite()));
        // Sparse over real weights matches the dense-restricted oracle.
        let oracle = layer
            .forward_dense_restricted(&x, &sparse.selected_experts)
            .unwrap();
        for d in 0..layer.d_model {
            assert!((sparse.output[d] - oracle[d]).abs() < 1e-5);
        }
    }

    #[test]
    fn missing_tensor_data_is_reported() {
        let (map, mut store) = synthetic_mixtral(2, 8, 16);
        // Drop one expert tensor's bytes; metadata still lists it.
        store.remove("model.layers.0.block_sparse_moe.experts.1.w3.weight");
        let resolve = resolver(&store);
        let tensors = map.expert(0, 1).unwrap();
        let err = RealExpertTensorBinding::resolve(0, 1, tensors, &resolve).unwrap_err();
        assert!(matches!(err, MoeBindingError::UnresolvedTensor { .. }));
    }

    #[test]
    fn incomplete_expert_metadata_is_rejected() {
        // Only a gate tensor — no up/down.
        let map = MoeWeightMap::from_tensors(vec![(
            "model.layers.0.block_sparse_moe.experts.0.w1.weight",
            vec![16, 8],
        )]);
        let store: HashMap<String, Vec<f32>> = HashMap::new();
        let resolve = resolver(&store);
        let tensors = map.expert(0, 0).unwrap();
        let err = RealExpertTensorBinding::resolve(0, 0, tensors, &resolve).unwrap_err();
        assert!(matches!(err, MoeBindingError::IncompleteExpert { .. }));
    }

    #[test]
    fn data_length_mismatch_is_caught() {
        let (map, mut store) = synthetic_mixtral(2, 8, 16);
        // Corrupt a tensor's length.
        store.insert(
            "model.layers.0.block_sparse_moe.experts.0.w1.weight".to_string(),
            vec![0.0; 3],
        );
        let resolve = resolver(&store);
        let tensors = map.expert(0, 0).unwrap();
        let err = RealExpertTensorBinding::resolve(0, 0, tensors, &resolve).unwrap_err();
        assert!(matches!(err, MoeBindingError::DataLengthMismatch { .. }));
    }

    #[test]
    fn missing_layer_is_reported() {
        let (map, store) = synthetic_mixtral(2, 8, 16);
        let resolve = resolver(&store);
        let err = build_real_layer(&map, 5, 2, &resolve).unwrap_err();
        assert!(matches!(err, MoeBindingError::LayerNotFound { layer_id: 5 }));
    }

    #[test]
    fn real_layer_with_identical_experts_reproduces_single_expert() {
        // If every expert shares the same weights, the renormalised sparse
        // combine must equal a single expert's output — proving the real
        // bound weights flow through the math correctly.
        let d_model = 4;
        let d_ff = 8;
        let n = 3;
        let wg = seeded(21, d_ff * d_model);
        let wu = seeded(22, d_ff * d_model);
        let wd = seeded(23, d_model * d_ff);
        let mut names_shapes: Vec<(String, Vec<usize>)> = Vec::new();
        let mut store: HashMap<String, Vec<f32>> = HashMap::new();
        let router = "model.layers.0.block_sparse_moe.gate.weight".to_string();
        names_shapes.push((router.clone(), vec![n, d_model]));
        store.insert(router, seeded(24, n * d_model));
        for e in 0..n {
            let g = format!("model.layers.0.block_sparse_moe.experts.{e}.w1.weight");
            let u = format!("model.layers.0.block_sparse_moe.experts.{e}.w3.weight");
            let d = format!("model.layers.0.block_sparse_moe.experts.{e}.w2.weight");
            names_shapes.push((g.clone(), vec![d_ff, d_model]));
            names_shapes.push((u.clone(), vec![d_ff, d_model]));
            names_shapes.push((d.clone(), vec![d_model, d_ff]));
            store.insert(g, wg.clone());
            store.insert(u, wu.clone());
            store.insert(d, wd.clone());
        }
        let map = MoeWeightMap::from_tensors(
            names_shapes.iter().map(|(n, s)| (n.as_str(), s.clone())),
        );
        let resolve = resolver(&store);
        let layer = build_real_layer(&map, 0, 2, &resolve).unwrap();
        let x = seeded(25, d_model);
        let single = layer.experts[0].forward(&x).unwrap();
        let sparse = layer.forward_sparse(&x, n).unwrap();
        for d in 0..d_model {
            assert!((sparse.output[d] - single[d]).abs() < 1e-5);
        }
    }
}
