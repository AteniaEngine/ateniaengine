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
    /// Positions are `[position_offset..position_offset + seq_len)`.
    /// For prefill (no cache, M4.6 / M4.7 / M4.8 / M4.9 path) the
    /// offset is 0 and positions are `[0..seq_len)` — bit-exact
    /// equivalent to the pre-M5 behaviour.
    ///
    /// **M5.c.2.c**: `position_offset` is the *cached_len* at
    /// build time of a decode-step graph. Q at seq=1 with
    /// offset=cached_len rotates at the correct absolute
    /// position in the running conversation, which is what
    /// the cached K already encodes (cache K was rotated at
    /// its own original position when first projected).
    ///
    /// ## Parameters
    /// - `head_dim`: dimension of each attention head
    ///   (must be even)
    /// - `base_freq`: base frequency for theta computation
    ///   (10000 for Llama 1/2/TinyLlama, 500000 for Llama 3+).
    ///   `u32` is used (not `f32`) so the variant remains
    ///   `Eq`-derivable; sub-integer base frequencies are not
    ///   used by any model in scope.
    /// - `scaling`: optional RoPE frequency scaling. `None`
    ///   reproduces plain RoPE bit-for-bit (TinyLlama, SmolLM2,
    ///   Qwen 2.5). `Some(NodeRopeScaling::Llama3(_))` activates
    ///   the Llama 3.x piecewise scaling. `Some(NodeRopeScaling::
    ///   LongRope(_))` activates the Phi-3 / Phi-3.5 per-dimension
    ///   factor vectors. See [`NodeRopeScaling`],
    ///   [`crate::nn::rope::compute_inv_freqs_llama3`], and
    ///   [`crate::nn::rope::compute_inv_freqs_longrope`].
    /// - `position_offset`: M5.c.2.c addition. The first sequence
    ///   position to rotate against; positions are
    ///   `[position_offset..position_offset + seq_len)`. Default
    ///   `0` reproduces pre-M5 behaviour bit-exactly across the
    ///   M4.6 four-model fixtures and the M4.7 13B forward.
    RoPE {
        head_dim: usize,
        base_freq: u32,
        scaling: Option<NodeRopeScaling>,
        position_offset: u32,
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
    /// **M11.B step 3.5** — take a contiguous slice of the input
    /// tensor's **last** axis. Inputs: `[x]`. The output keeps
    /// every leading dimension intact and replaces the last
    /// dimension with `end - start`.
    ///
    /// ## Shape
    ///
    /// Given input `[d0, d1, ..., d_{N-1}]`, output is
    /// `[d0, d1, ..., end - start]`. `start` is inclusive,
    /// `end` is exclusive. The contract `0 <= start < end
    /// <= d_{N-1}` is enforced at executor time; out-of-range
    /// values panic.
    ///
    /// ## Primary use case
    ///
    /// Runtime split of fused weight matmul outputs for
    /// architectures that ship `qkv_proj` and `gate_up_proj`
    /// fused (Phi-3 / Phi-3.5 in M11.B; expected to apply to
    /// Gemma 2 in M11.C). The fused weight is loaded as a
    /// single parameter; after the matmul that produces a
    /// `[..., (n_q + 2 n_kv) * head_dim]` activation, three
    /// `SliceLastDim` nodes split it into Q / K / V activations.
    /// Same pattern halves a `gate_up` matmul output for the
    /// SwiGLU FFN.
    ///
    /// ## Implementation
    ///
    /// Forward materialises the input to CPU (defensive
    /// `ensure_cpu` to handle BF16 / Cuda / Disk-resident
    /// inputs) and copies the selected last-axis range row-by-
    /// row. Output is `Layout::Contiguous` and a fresh `Vec`
    /// — a slice on a `Vec<f32>` does not yield a separate
    /// owned buffer in Rust without `Cow`-shaped storage,
    /// which Atenia's Tensor does not currently expose, so the
    /// implementation copies. For Phi-3.5 Mini's split the
    /// per-call cost is `~B * 3072 * 8` bytes (24 KB at seq=2)
    /// — negligible vs the matmul that produces the input.
    ///
    /// Backward is unimplemented for now (forward-only is
    /// sufficient for inference; training-time gradient
    /// scatter into the parent slice lives in the autograd
    /// extension that lands with the next training-mode
    /// milestone). Mirrors `NodeType::Concat`'s status.
    SliceLastDim {
        start: usize,
        /// Exclusive upper bound. Must satisfy `start < end <=
        /// last_dim_of_input`.
        end: usize,
    },
    /// **M11.C step 2** — soft-cap (a.k.a. logit-cap, attention
    /// soft-cap) primitive: `out[i] = cap * tanh(in[i] / cap)`.
    ///
    /// ## Numerical behaviour
    ///
    /// Saturates smoothly toward `±cap` as `|in[i]|` grows. For
    /// `|in[i] / cap| > ~20`, `f32::tanh` returns exactly `±1.0`
    /// (no NaN, no overflow), so the worst-case output is
    /// `±cap` exactly. Around the origin the function is
    /// approximately the identity (`tanh(x) ≈ x` for small `x`,
    /// so `cap * tanh(x/cap) ≈ x`), preserving the dynamic
    /// range of small logits.
    ///
    /// ## Storage
    ///
    /// `cap_bits` is `f32::to_bits(cap)`; recovered via
    /// `f32::from_bits` at executor time. Same trick used by
    /// `RmsNorm { eps_bits }` so `NodeType` can stay
    /// `Eq + Hash`-derivable.
    ///
    /// ## Primary use case
    ///
    /// Gemma 2's two soft-caps:
    /// - `cap = 50.0` on attention scores pre-softmax (every layer).
    /// - `cap = 30.0` on the LM-head logits pre-sampling.
    ///
    /// No other certified-set architecture uses soft-cap, so the
    /// node fires only on Gemma 2 paths.
    ///
    /// ## Implementation
    ///
    /// Forward materialises the input to CPU and applies the
    /// scalar function elementwise. Output shape == input shape;
    /// layout is fresh `Layout::Contiguous`. Backward is
    /// unimplemented (forward-only is sufficient for inference;
    /// derivative is `1 - tanh(x/cap)^2` if a future training
    /// milestone needs it).
    SoftCap {
        /// `f32::to_bits(cap)`. Reading code must use
        /// `f32::from_bits(cap_bits)`. The builder enforces
        /// `cap > 0.0` so the `cap_bits = 0` (i.e. `+0.0`) and
        /// `cap_bits = 0x80000000` (`-0.0`) bit patterns are
        /// unreachable through the public API.
        cap_bits: u32,
    },
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

/// **M11.B step 2** — Microsoft Phi-3 / Phi-3.5 LongRope scaling
/// parameters carried by [`NodeType::RoPE`].
///
/// The factor vectors are stored as `Vec<u32>` (raw `f32::to_bits`)
/// for the same reason `RopeScalingLlama3` stores its scalars as
/// `u32`: `Vec<u32>` derives `Eq + Hash`, so the enclosing
/// [`NodeType`] keeps both traits without manual impls. The two
/// vectors must have length `head_dim / 2` (validated by the
/// runtime via [`crate::nn::rope::compute_inv_freqs_longrope`]).
///
/// `original_max_position_embeddings` and `max_position_embeddings`
/// are integer-valued; the latter is needed to derive the runtime
/// `attention_factor` (see
/// [`crate::nn::rope::compute_attention_factor_longrope`]).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RopeScalingLongRope {
    short_factor_bits: Vec<u32>,
    long_factor_bits: Vec<u32>,
    /// Pre-training context length (4096 for Phi-3.5 Mini).
    pub original_max_position_embeddings: u32,
    /// Configured max-context length (131072 for Phi-3.5 Mini).
    pub max_position_embeddings: u32,
}

impl RopeScalingLongRope {
    /// Build from the per-dimension `f32` factor vectors directly
    /// read off the config. Both vectors must have length
    /// `head_dim / 2`; the constructor stores them as `to_bits`
    /// for the `Eq + Hash` derive on the enclosing `NodeType`.
    pub fn new(
        short_factor: &[f32],
        long_factor: &[f32],
        original_max_position_embeddings: u32,
        max_position_embeddings: u32,
    ) -> Self {
        Self {
            short_factor_bits: short_factor.iter().map(|f| f.to_bits()).collect(),
            long_factor_bits: long_factor.iter().map(|f| f.to_bits()).collect(),
            original_max_position_embeddings,
            max_position_embeddings,
        }
    }

    /// Recover the short-path per-dimension factors as `Vec<f32>`.
    /// Allocates on each call; the call sites either run once per
    /// forward pass (executor) or once per build (validation), so
    /// the allocation cost is negligible.
    pub fn short_factor(&self) -> Vec<f32> {
        self.short_factor_bits.iter().map(|b| f32::from_bits(*b)).collect()
    }

    /// Recover the long-path per-dimension factors as `Vec<f32>`.
    pub fn long_factor(&self) -> Vec<f32> {
        self.long_factor_bits.iter().map(|b| f32::from_bits(*b)).collect()
    }
}

/// **M11.B step 2** — discriminated union of the RoPE scaling
/// schemes that AMG nodes can carry. Wraps either Llama 3
/// piecewise scaling or Phi-3 LongRope scaling. Plain RoPE
/// (TinyLlama / SmolLM2 / Qwen 2.5) is encoded by the absence of
/// this enum (`scaling: None` on the `RoPE` node).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NodeRopeScaling {
    /// Llama 3.x piecewise inverse-frequency scaling.
    Llama3(RopeScalingLlama3),
    /// Phi-3 / Phi-3.5 LongRope (per-dimension factor vectors +
    /// optional attention-factor compensation).
    LongRope(RopeScalingLongRope),
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
