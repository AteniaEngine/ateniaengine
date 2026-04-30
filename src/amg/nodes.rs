//! Graph node definitions for Atenia Model Graph.

use crate::tensor::Tensor;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionTarget {
    CPU,
    GPU,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActType {
    ReLU,
    SiLU,
    GELU,
}

/// Configuration for a Conv2D node (APX v20 M1).
///
/// Only the hyperparameters that cannot be derived from tensor shapes
/// live here. Kernel spatial size is inferred at execution time from
/// the weight tensor shape (OIHW layout: `kernel_h = weight.shape[2]`,
/// `kernel_w = weight.shape[3]`). Weights and bias enter the node via
/// the graph's input edges, not via this config.
///
/// Convention: tuples are `(h, w)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv2DConfig {
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl Conv2DConfig {
    pub const fn new(stride: (usize, usize), padding: (usize, usize)) -> Self {
        Self { stride, padding }
    }
}

impl Default for Conv2DConfig {
    fn default() -> Self {
        Self {
            stride: (1, 1),
            padding: (0, 0),
        }
    }
}

/// Configuration for a MaxPool2D node (APX v20 M1).
///
/// MaxPool2D has no learnable weights; kernel spatial size lives in
/// the config because there is no weight tensor to derive it from.
///
/// Convention: tuples are `(h, w)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaxPool2DConfig {
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub padding: (usize, usize),
}

impl MaxPool2DConfig {
    pub const fn new(
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self { kernel, stride, padding }
    }

    /// Convenience constructor for the common non-overlapping case:
    /// `stride == kernel` and `padding == (0, 0)`.
    pub const fn non_overlapping(kernel: (usize, usize)) -> Self {
        Self {
            kernel,
            stride: kernel,
            padding: (0, 0),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeType {
    Input,
    Parameter,
    Add,
    Sub,
    Mul,
    MatMul,
    Transpose2D,
    /// Permute (general transpose) — reorders tensor dimensions
    /// according to `perm`.
    ///
    /// Output shape: `[input.shape[perm[0]], input.shape[perm[1]], ...]`.
    ///
    /// `perm` must:
    /// - have length equal to input rank
    /// - contain every index in `[0..rank)` exactly once
    ///   (no duplicates, no out-of-range)
    ///
    /// ## Examples
    /// - `perm=[1,0]` on shape `[3,4]` → shape `[4,3]` (2D transpose)
    /// - `perm=[0,2,1,3]` on shape `[b,s,h,d]` → `[b,h,s,d]`
    ///   (multi-head attention layout swap)
    ///
    /// ## Implementation
    /// Performs a data copy (not stride manipulation). Output is
    /// always `Layout::Contiguous`, consistent with the rest of the
    /// codebase which assumes contiguous tensors.
    ///
    /// ## Backward
    /// Uses the inverse permutation: `inv[perm[i]] = i`. NOT
    /// self-inverse like `Transpose2D`.
    Permute { perm: Vec<usize> },
    IndexSelect,
    Reshape { target: Vec<isize> },
    TransposeLastTwo,
    BatchMatMul,
    BroadcastAdd,
    /// Element-wise multiplication with broadcasting — analogous to
    /// [`NodeType::BroadcastAdd`] but with `*` in place of `+`.
    ///
    /// Convention: both inputs must have the same rank. For each
    /// dimension `d`, if `b.shape[d] == 1` then `b` is broadcast
    /// over that dimension; otherwise the dim sizes must agree.
    /// Output shape equals `a.shape`. Use `Reshape` upstream if
    /// you need to align ranks (e.g. expand a `[hidden]` gamma to
    /// `[1, 1, hidden]` before broadcasting against `[b, s, hidden]`).
    ///
    /// ## Primary use case
    /// Apply a learnable per-feature scale (RMSNorm γ, LayerNorm γ,
    /// per-channel bias, etc.) without expanding the parameter into a
    /// full-shape tensor.
    ///
    /// ## Backward
    /// `grad_a[i] = out_grad[i] * b[broadcast(i)]` (same shape as `a`).
    /// `grad_b[j] = Σ_{i mapped to j} out_grad[i] * a[i]` (reduced over
    /// the broadcast dims of `b`).
    BroadcastMul,
    LogSoftmax,
    Gather,
    CrossEntropyLoss,
    Linear,
    /// RMSNorm over the last dimension.
    ///
    /// ## Parameters
    /// - `eps_bits`: small constant added to the mean of squares
    ///   before sqrt, to avoid division by zero. Per-model value
    ///   (1e-5 for Llama/TinyLlama/SmolLM2, 1e-6 for Qwen 2.5).
    ///   Stored as `u32` (raw `f32::to_bits` representation) so
    ///   the variant remains `Eq`-derivable, mirroring the RoPE
    ///   `base_freq` convention. Use [`Self::rms_norm_eps`] to
    ///   recover the `f32` value.
    RmsNorm { eps_bits: u32 },
    SiLU,
    Softmax,
    /// Rotary Positional Embedding (half-split layout).
    ///
    /// ## Layout convention
    /// Half-split (HuggingFace), NOT interleaved (original
    /// paper). Required for numerical equivalence with PyTorch
    /// reference implementations.
    ///
    /// ## Position assumption
    /// Positions are implicit `[0..seq_len)` for inference
    /// without KV cache. KV cache support (planned M5+)
    /// requires extending this NodeType to accept explicit
    /// position offset.
    ///
    /// ## Parameters
    /// - `head_dim`: dimension of each attention head
    ///   (must be even)
    /// - `base_freq`: base frequency for theta computation
    ///   (10000 for Llama 1/2/TinyLlama, 500000 for Llama 3+).
    ///   `u32` is used (not `f32`) so the variant remains
    ///   `Eq`-derivable; sub-integer base frequencies are not
    ///   used by any model in scope.
    /// - `scaling`: optional Llama 3 piecewise frequency
    ///   scaling. `None` reproduces plain RoPE bit-for-bit
    ///   (TinyLlama, SmolLM2, Qwen 2.5). `Some(RopeScalingLlama3)`
    ///   activates the long-context scaling used by Llama 3.x.
    ///   See [`RopeScalingLlama3`] and
    ///   [`crate::nn::rope::compute_inv_freqs_llama3`].
    RoPE {
        head_dim: usize,
        base_freq: u32,
        scaling: Option<RopeScalingLlama3>,
    },
    /// Generic activation wrapper for APX 4.8+ (keeps SiLU variant for compat).
    Activation(ActType),
    /// Fused Linear + Activation node.
    FusedLinearActivation(ActType),
    /// Fused multi-node chain: Linear -> [Activations...] -> Linear.
    FusedLinearActivationChain(Vec<ActType>),
    /// 2D convolution (APX v20 M1). Inputs: `[input, weight]` or
    /// `[input, weight, bias]`. Weight layout: OIHW. Input layout: NCHW.
    Conv2D(Conv2DConfig),
    /// 2D max pooling (APX v20 M1). Inputs: `[input]`. Input layout: NCHW.
    MaxPool2D(MaxPool2DConfig),
    /// Concatenate two tensors along the given axis (M5.c).
    ///
    /// Inputs: `[a, b]`. Both tensors must have the same rank
    /// and agree on every dimension *except* `axis`. Output
    /// shape equals `a.shape` with `out.shape[axis] =
    /// a.shape[axis] + b.shape[axis]`.
    ///
    /// ## Primary use case
    /// Cache-aware attention path (M5.c+): `K_full =
    /// concat(cache_K, new_K, axis=2)` produces the
    /// `[batch, n_kv_heads, cached_len + new_len, head_dim]`
    /// tensor the attention kernels read at decode time.
    ///
    /// ## Implementation
    /// Forward performs a contiguous data copy. Inputs are
    /// materialised to CPU via `Tensor::ensure_cpu` before
    /// concatenation; output is `Layout::Contiguous`. Backward
    /// is a TODO (forward-only is sufficient for inference;
    /// training-time gradient split lives in the autograd
    /// tape extension that lands with the next training-mode
    /// milestone).
    Concat { axis: usize },
    /// No-op placeholder used for structurally removed nodes.
    NoOp,
    Output,
}

/// Llama 3 piecewise inverse-frequency scaling parameters carried by
/// [`NodeType::RoPE`].
///
/// Scalars are stored as raw `f32::to_bits` (`u32`) so the enclosing
/// `NodeType` can stay `Eq + Hash`-derivable. Mirrors the same trick
/// already used for `NodeType::RmsNorm { eps_bits }` and the legacy
/// `RoPE { base_freq: u32 }` representation. Use the named accessors
/// to recover the `f32` values.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RopeScalingLlama3 {
    factor_bits: u32,
    low_freq_factor_bits: u32,
    high_freq_factor_bits: u32,
    /// Pre-training context length used as the wavelength reference.
    /// Stored as `u32` directly because it is integer-valued.
    pub original_max_position_embeddings: u32,
}

impl RopeScalingLlama3 {
    /// Build from the `f32` parameters directly read off the config.
    pub fn new(
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_position_embeddings: u32,
    ) -> Self {
        Self {
            factor_bits: factor.to_bits(),
            low_freq_factor_bits: low_freq_factor.to_bits(),
            high_freq_factor_bits: high_freq_factor.to_bits(),
            original_max_position_embeddings,
        }
    }

    /// Long-context expansion factor (32.0 for Llama 3.2).
    pub fn factor(&self) -> f32 {
        f32::from_bits(self.factor_bits)
    }

    /// Lower-bound frequency multiplier (1.0 for Llama 3.2).
    pub fn low_freq_factor(&self) -> f32 {
        f32::from_bits(self.low_freq_factor_bits)
    }

    /// Upper-bound frequency multiplier (4.0 for Llama 3.2).
    pub fn high_freq_factor(&self) -> f32 {
        f32::from_bits(self.high_freq_factor_bits)
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: usize,
    pub node_type: NodeType,
    pub inputs: Vec<usize>,
    pub output: Option<Tensor>,
    pub target: ExecutionTarget,
}

impl Node {
    pub fn new(id: usize, node_type: NodeType, inputs: Vec<usize>) -> Self {
        Self {
            id,
            node_type,
            inputs,
            output: None,
            target: ExecutionTarget::CPU,
        }
    }

    pub fn set_output(&mut self, tensor: Tensor) {
        self.output = Some(tensor);
    }
}
