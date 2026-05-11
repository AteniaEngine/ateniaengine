//! Graph builder utilities for assembling Atenia Model Graphs.

use super::nodes::{ActType, Node, NodeType};
use super::scheduler::build_execution_plan;
use crate::apx4_7::{FusionPlan, PersistentPlan};
use crate::apx4_8::pattern::detect_and_fuse_linear_activation;
use crate::apx4_9::patterns::fuse_linear_activation_linear;
use crate::tensor::Tensor;

pub struct GraphBuilder {
    pub nodes: Vec<Node>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn add_node(&mut self, node_type: NodeType, inputs: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node::new(id, node_type, inputs));
        id
    }

    pub fn add_node_of_type(&mut self, node_type: NodeType, inputs: Vec<usize>) -> usize {
        self.add_node(node_type, inputs)
    }

    pub fn input(&mut self) -> usize {
        self.add_node(NodeType::Input, vec![])
    }

    pub fn parameter(&mut self, value: Tensor) -> usize {
        let id = self.add_node(NodeType::Parameter, vec![]);
        self.nodes[id].output = Some(value);
        id
    }

    pub fn add(&mut self, a: usize, b: usize) -> usize {
        self.add_node(NodeType::Add, vec![a, b])
    }

    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        self.add_node(NodeType::Sub, vec![a, b])
    }

    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        self.add_node(NodeType::Mul, vec![a, b])
    }

    pub fn matmul(&mut self, a: usize, b: usize) -> usize {
        self.add_node(NodeType::MatMul, vec![a, b])
    }

    /// Matrix multiply against the transpose of `b` without building a
    /// `Transpose2D` tensor first: `[m, k] x [n, k]^T -> [m, n]`.
    pub fn matmul_rhs_transposed(&mut self, a: usize, b: usize) -> usize {
        self.add_node(NodeType::MatMulRhsTransposed, vec![a, b])
    }

    /// Gather rows from a parameter tensor using index tensor.
    pub fn index_select(&mut self, table_id: usize, indices_id: usize) -> usize {
        self.add_node(NodeType::IndexSelect, vec![table_id, indices_id])
    }

    pub fn reshape(&mut self, src: usize, target: Vec<isize>) -> usize {
        self.add_node(NodeType::Reshape { target }, vec![src])
    }

    pub fn transpose_last_two(&mut self, src: usize) -> usize {
        self.add_node(NodeType::TransposeLastTwo, vec![src])
    }

    /// 2D matrix transpose: `[a, b] -> [b, a]`. Input must be rank 2.
    pub fn transpose_2d(&mut self, src: usize) -> usize {
        self.add_node(NodeType::Transpose2D, vec![src])
    }

    /// Permute a tensor's dimensions according to `perm` (general
    /// transpose). See [`NodeType::Permute`] for details and shape
    /// constraints on `perm`.
    pub fn permute(&mut self, x_id: usize, perm: Vec<usize>) -> usize {
        self.add_node(NodeType::Permute { perm }, vec![x_id])
    }

    pub fn batch_matmul(&mut self, a: usize, b: usize) -> usize {
        self.add_node(NodeType::BatchMatMul, vec![a, b])
    }

    pub fn broadcast_add(&mut self, a: usize, b: usize) -> usize {
        self.add_node(NodeType::BroadcastAdd, vec![a, b])
    }

    /// Element-wise broadcast multiply. Both inputs must have the
    /// same rank; dims where `b.shape[d] == 1` are broadcast.
    /// Use case: per-feature learnable scale (e.g. RMSNorm γ).
    pub fn broadcast_mul(&mut self, a: usize, b: usize) -> usize {
        self.add_node(NodeType::BroadcastMul, vec![a, b])
    }

    pub fn log_softmax(&mut self, src: usize) -> usize {
        self.add_node(NodeType::LogSoftmax, vec![src])
    }

    pub fn gather(&mut self, data: usize, indices: usize) -> usize {
        self.add_node(NodeType::Gather, vec![data, indices])
    }

    pub fn cross_entropy_loss(&mut self, log_probs: usize, targets: usize) -> usize {
        self.add_node(NodeType::CrossEntropyLoss, vec![log_probs, targets])
    }

    pub fn output(&mut self, src: usize) -> usize {
        self.add_node(NodeType::Output, vec![src])
    }

    /// M5.c — concatenate two tensors along `axis`. See
    /// [`NodeType::Concat`] for shape and semantics. Used by
    /// the cache-aware attention path to splice the resident
    /// KV cache to the current step's K, V projections.
    pub fn concat(&mut self, a: usize, b: usize, axis: usize) -> usize {
        self.add_node(NodeType::Concat { axis }, vec![a, b])
    }

    /// Create a Linear node with required inputs [x, w] and optional bias.
    pub fn linear(&mut self, x_id: usize, w_id: usize, b_id: Option<usize>) -> usize {
        let mut inputs = vec![x_id, w_id];
        if let Some(b) = b_id {
            inputs.push(b);
        }
        self.add_node(NodeType::Linear, inputs)
    }

    /// RMSNorm over last dimension.
    ///
    /// `eps` is the per-model stabilizer (1e-5 for Llama family,
    /// 1e-6 for Qwen 2.5).
    pub fn rms_norm(&mut self, x_id: usize, eps: f32) -> usize {
        self.add_node(
            NodeType::RmsNorm {
                eps_bits: eps.to_bits(),
            },
            vec![x_id],
        )
    }

    /// ReLU activation.
    pub fn relu(&mut self, x_id: usize) -> usize {
        self.add_node(NodeType::Activation(ActType::ReLU), vec![x_id])
    }

    /// SiLU activation.
    pub fn silu(&mut self, x_id: usize) -> usize {
        self.add_node(NodeType::Activation(ActType::SiLU), vec![x_id])
    }

    /// GELU activation.
    pub fn gelu(&mut self, x_id: usize) -> usize {
        self.add_node(NodeType::Activation(ActType::GELU), vec![x_id])
    }

    /// Softmax along last dimension.
    pub fn softmax(&mut self, x_id: usize) -> usize {
        self.add_node(NodeType::Softmax, vec![x_id])
    }

    /// Add a RoPE node (Rotary Positional Embedding, half-split layout).
    ///
    /// Input shape: `[batch, seq_len, n_heads, head_dim]`. Positions are
    /// implicit `[0..seq_len)`. See [`NodeType::RoPE`] for details.
    pub fn rope(&mut self, x_id: usize, head_dim: usize, base_freq: u32) -> usize {
        self.add_node(
            NodeType::RoPE {
                head_dim,
                base_freq,
                scaling: None,
                position_offset: 0,
            },
            vec![x_id],
        )
    }

    /// **M5.c.2.c** — RoPE at a non-zero starting position.
    /// Used by decode-step graphs where Q at seq=1 must
    /// rotate at the absolute position `cached_len`, not
    /// position 0. `offset = 0` is bit-exact equivalent to
    /// [`Self::rope`].
    pub fn rope_with_offset(
        &mut self,
        x_id: usize,
        head_dim: usize,
        base_freq: u32,
        position_offset: u32,
    ) -> usize {
        self.add_node(
            NodeType::RoPE {
                head_dim,
                base_freq,
                scaling: None,
                position_offset,
            },
            vec![x_id],
        )
    }

    /// Add a RoPE node with Llama 3 piecewise frequency scaling.
    ///
    /// Equivalent to [`Self::rope`] but threads the long-context
    /// scaling parameters through the variant. Used by `build_llama`
    /// when the parsed config carries a `RopeScaling::Llama3` block
    /// (Llama 3.1, 3.2, 3.3).
    pub fn rope_scaled(
        &mut self,
        x_id: usize,
        head_dim: usize,
        base_freq: u32,
        scaling: super::nodes::RopeScalingLlama3,
    ) -> usize {
        self.add_node(
            NodeType::RoPE {
                head_dim,
                base_freq,
                scaling: Some(super::nodes::NodeRopeScaling::Llama3(scaling)),
                position_offset: 0,
            },
            vec![x_id],
        )
    }

    /// **M5.c.2.c** — Llama 3-scaled RoPE at a non-zero
    /// starting position. Decode-step counterpart of
    /// [`Self::rope_scaled`].
    pub fn rope_scaled_with_offset(
        &mut self,
        x_id: usize,
        head_dim: usize,
        base_freq: u32,
        scaling: super::nodes::RopeScalingLlama3,
        position_offset: u32,
    ) -> usize {
        self.add_node(
            NodeType::RoPE {
                head_dim,
                base_freq,
                scaling: Some(super::nodes::NodeRopeScaling::Llama3(scaling)),
                position_offset,
            },
            vec![x_id],
        )
    }

    /// **M11.B step 2** — Phi-3 / Phi-3.5 LongRope-scaled RoPE.
    /// Mirrors [`Self::rope_scaled`] for the LongRope scaling
    /// scheme: the per-dimension `short_factor` / `long_factor`
    /// vectors plus the original / configured max-position
    /// values needed to derive the runtime
    /// `attention_factor`. Plain RoPE keeps `Self::rope`; Llama
    /// 3.x scaling keeps `Self::rope_scaled`. Position offset 0
    /// (prefill semantics).
    pub fn rope_longrope(
        &mut self,
        x_id: usize,
        head_dim: usize,
        base_freq: u32,
        scaling: super::nodes::RopeScalingLongRope,
    ) -> usize {
        self.add_node(
            NodeType::RoPE {
                head_dim,
                base_freq,
                scaling: Some(super::nodes::NodeRopeScaling::LongRope(scaling)),
                position_offset: 0,
            },
            vec![x_id],
        )
    }

    /// **M11.B step 2** — LongRope-scaled RoPE at a non-zero
    /// starting position. Decode-step counterpart of
    /// [`Self::rope_longrope`].
    pub fn rope_longrope_with_offset(
        &mut self,
        x_id: usize,
        head_dim: usize,
        base_freq: u32,
        scaling: super::nodes::RopeScalingLongRope,
        position_offset: u32,
    ) -> usize {
        self.add_node(
            NodeType::RoPE {
                head_dim,
                base_freq,
                scaling: Some(super::nodes::NodeRopeScaling::LongRope(scaling)),
                position_offset,
            },
            vec![x_id],
        )
    }

    /// **M11.B step 3.5** — construct a `NodeType::SliceLastDim`
    /// node that takes the contiguous range `[start..end)` of the
    /// input's last axis. Validation of the range against the
    /// actual input shape happens at executor time (the input
    /// shape is not always known at build time when the input is
    /// a non-parameter node). The build-time contract enforced
    /// here is `start < end`; the rest is the executor's job.
    ///
    /// See [`NodeType::SliceLastDim`] for the layout contract
    /// and intended use case (runtime split of fused weight
    /// matmul outputs for Phi-3 / Phi-3.5 / Gemma 2).
    pub fn slice_last_dim(&mut self, x_id: usize, start: usize, end: usize) -> usize {
        assert!(
            start < end,
            "GraphBuilder::slice_last_dim: start ({start}) must be < end ({end})"
        );
        self.add_node(NodeType::SliceLastDim { start, end }, vec![x_id])
    }

    /// **M11.C step 2** — construct a `NodeType::SoftCap` node:
    /// `out[i] = cap * tanh(in[i] / cap)`.
    ///
    /// Build-time contract: `cap` must be finite and strictly
    /// positive. The executor re-validates defensively, but this
    /// assertion gives a clear failure at graph-construction time
    /// when a builder pass passes a bad scalar.
    ///
    /// `cap` is stored as `f32::to_bits(cap)` inside the node so
    /// `NodeType` keeps its `Eq + Hash` derive (mirrors
    /// `RmsNorm { eps_bits }`).
    ///
    /// See [`NodeType::SoftCap`] for the saturation contract,
    /// numerical behaviour at extreme inputs, and intended use
    /// case (Gemma 2 attention soft-cap at 50.0 and final-logit
    /// soft-cap at 30.0).
    pub fn soft_cap(&mut self, x_id: usize, cap: f32) -> usize {
        assert!(
            cap.is_finite() && cap > 0.0,
            "GraphBuilder::soft_cap: cap must be a finite positive f32, got {cap}"
        );
        self.add_node(
            NodeType::SoftCap {
                cap_bits: cap.to_bits(),
            },
            vec![x_id],
        )
    }

    pub fn build(self) -> super::graph::Graph {
        let mut graph = super::graph::Graph::new(self.nodes);

        // Apply APX 4.8 and 4.9 structural fusions on the already-built graph.
        let _ = detect_and_fuse_linear_activation(&mut graph);
        let _ = fuse_linear_activation_linear(&mut graph);

        // Rebuild execution plan so it matches the fused graph structure.
        let (plan, fused_ops) = build_execution_plan(&graph.nodes);
        graph.plan = plan;
        graph.fused_ops = fused_ops;

        // Recompute APX 4.7 persistent and fusion plans so that any Linear→Linear
        // fusions are consistent with the transformed graph (and do not intercept
        // newly fused APX 4.9 nodes incorrectly).
        graph.persistent_plan = Some(PersistentPlan::analyze(&graph));
        graph.fusion_plan = Some(FusionPlan::analyze(&graph));

        graph
    }
}

#[cfg(test)]
mod m11_b_step2_tests {
    //! **M11.B step 2** — AMG NodeType + executor wire-up tests
    //! for LongRope. Covers: (a) the builder produces a node with
    //! the correct enum variant, (b) the per-dimension factor
    //! vectors round-trip without precision loss, (c) the
    //! executor branch produces a finite output equivalent to
    //! the manual `compute_inv_freqs_longrope` reference, and
    //! (d) the existing Llama 3 / plain RoPE paths are byte-
    //! stable across this commit.
    use super::*;
    use crate::amg::nodes::{NodeRopeScaling, NodeType, RopeScalingLongRope};
    use crate::tensor::Tensor;

    #[test]
    fn matmul_rhs_transposed_matches_explicit_tied_head_math() {
        let mut gb = GraphBuilder::new();
        let x = gb.input();
        let embedding = gb.parameter(Tensor::new_cpu(
            vec![3, 2],
            vec![
                1.0, 2.0, //
                3.0, 4.0, //
                5.0, 6.0,
            ],
        ));
        let logits = gb.matmul_rhs_transposed(x, embedding);
        gb.output(logits);

        let mut graph = gb.build();
        let outs = graph.execute(vec![Tensor::new_cpu(vec![2, 2], vec![2.0, 10.0, 1.0, 1.0])]);
        let out = &outs[0];
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.as_cpu_slice(), &[22.0, 46.0, 70.0, 3.0, 7.0, 11.0]);
    }

    #[test]
    fn index_select_decodes_only_selected_bf16_rows() {
        let mut gb = GraphBuilder::new();
        let indices = gb.input();
        let bits: Vec<u16> = [
            1.0_f32, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ]
        .iter()
        .map(|v| (v.to_bits() >> 16) as u16)
        .collect();
        let table = gb.parameter(Tensor::new_cpu_bf16(vec![3, 3], bits));
        let selected = gb.index_select(table, indices);
        gb.output(selected);

        let mut graph = gb.build();
        let outs = graph.execute(vec![Tensor::new_cpu(vec![2], vec![2.0, 0.0])]);
        let out = &outs[0];
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(out.as_cpu_slice(), &[7.0, 8.0, 9.0, 1.0, 2.0, 3.0]);
    }

    /// LongRope-bearing node carries the right variant. Round-trip
    /// the factor vectors through `to_bits` / `from_bits` and
    /// confirm bit-exact recovery.
    #[test]
    fn longrope_node_round_trips_factor_vectors() {
        let head_dim = 64;
        let half = head_dim / 2;
        let short: Vec<f32> = (0..half).map(|i| 1.0 + i as f32 * 0.0125).collect();
        let long: Vec<f32> = (0..half).map(|i| 2.0 + i as f32 * 0.05).collect();
        let scaling = RopeScalingLongRope::new(&short, &long, 4096, 131_072);

        let mut gb = GraphBuilder::new();
        let x = gb.input();
        let _rope_id = gb.rope_longrope(x, head_dim, 10_000, scaling);

        // Locate the node and confirm the variant + payload.
        let nodes = &gb.nodes;
        let rope_node = nodes
            .iter()
            .find(|n| {
                matches!(
                    n.node_type,
                    NodeType::RoPE {
                        scaling: Some(_),
                        ..
                    }
                )
            })
            .expect("RoPE node with scaling must be present");
        match &rope_node.node_type {
            NodeType::RoPE {
                scaling: Some(NodeRopeScaling::LongRope(s)),
                ..
            } => {
                let recovered_short = s.short_factor();
                let recovered_long = s.long_factor();
                for (i, (a, b)) in short.iter().zip(recovered_short.iter()).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "short[{i}] round-trip mismatch");
                }
                for (i, (a, b)) in long.iter().zip(recovered_long.iter()).enumerate() {
                    assert_eq!(a.to_bits(), b.to_bits(), "long[{i}] round-trip mismatch");
                }
                assert_eq!(s.original_max_position_embeddings, 4096);
                assert_eq!(s.max_position_embeddings, 131_072);
            }
            other => panic!("expected NodeType::RoPE with LongRope scaling, got {other:?}"),
        }
    }

    /// Plain RoPE (no scaling) and Llama3 RoPE paths must be
    /// byte-stable across this commit. Verified by constructing
    /// nodes with both APIs and asserting the variant on the
    /// resulting node matches what each builder method declared.
    #[test]
    fn llama3_and_plain_rope_variants_unchanged() {
        use crate::amg::nodes::RopeScalingLlama3;
        let mut gb = GraphBuilder::new();
        let x = gb.input();
        let _plain = gb.rope(x, 64, 10_000);
        let scaling = RopeScalingLlama3::new(32.0, 1.0, 4.0, 8192);
        let _llama3 = gb.rope_scaled(x, 64, 500_000, scaling);

        let plain_node = &gb.nodes[1]; // node 0 is the Input
        match plain_node.node_type {
            NodeType::RoPE { scaling: None, .. } => {}
            ref other => panic!("plain RoPE must carry scaling: None, got {other:?}"),
        }

        let llama3_node = &gb.nodes[2];
        match &llama3_node.node_type {
            NodeType::RoPE {
                scaling: Some(NodeRopeScaling::Llama3(_)),
                ..
            } => {}
            other => panic!("rope_scaled must carry NodeRopeScaling::Llama3, got {other:?}"),
        }
    }

    /// End-to-end: build a tiny graph that applies LongRope to a
    /// known input and confirm the executor produces a finite
    /// output of the correct shape. Numerical equivalence to the
    /// reference `compute_inv_freqs_longrope` (already covered by
    /// the rope.rs unit tests) is sufficient correctness — this
    /// test just locks down that the executor branch is wired.
    #[test]
    fn longrope_executor_runs_to_completion_on_small_input() {
        let head_dim = 8;
        let half = head_dim / 2;
        let short: Vec<f32> = vec![1.0; half];
        let long: Vec<f32> = vec![2.0; half];
        let scaling = RopeScalingLongRope::new(&short, &long, 16, 64);

        let mut gb = GraphBuilder::new();
        let x_id = gb.input();
        let rope_id = gb.rope_longrope(x_id, head_dim, 10_000, scaling);
        let _out = gb.output(rope_id);

        let mut graph = gb.build();
        // Synthetic input: arange-like.
        let input = Tensor::new_cpu(
            vec![1, 2, 1, head_dim],
            (0..1 * 2 * 1 * head_dim).map(|i| i as f32 * 0.1).collect(),
        );
        let outs = graph.execute(vec![input]);
        assert_eq!(outs.len(), 1, "graph should produce one output");
        let out = outs[0].copy_to_cpu_vec();
        assert_eq!(out.len(), 1 * 2 * 1 * head_dim);
        for (i, v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "executor output[{i}] is not finite: {v} — LongRope branch is broken"
            );
        }
    }
}

#[cfg(test)]
mod m11_b_step3_5_tests {
    //! **M11.B step 3.5** — `NodeType::SliceLastDim` end-to-end
    //! tests. Covers: (a) prefix slice, (b) suffix slice, (c)
    //! Phi-3.5 Q-projection-shape slice, (d) a 3-rank slice
    //! preserves leading dims, (e) backstop on degenerate
    //! ranges (start >= end, end > last_dim).
    use super::*;
    use crate::tensor::Tensor;

    /// Helper: build `[input → slice_last_dim → output]` and run
    /// once with the given input tensor. Returns the resulting
    /// shape and flat data for comparison.
    fn slice_once(
        input_shape: Vec<usize>,
        input_data: Vec<f32>,
        start: usize,
        end: usize,
    ) -> (Vec<usize>, Vec<f32>) {
        let mut gb = GraphBuilder::new();
        let x = gb.input();
        let s = gb.slice_last_dim(x, start, end);
        let _ = gb.output(s);
        let mut graph = gb.build();
        let outs = graph.execute(vec![Tensor::new_cpu(input_shape, input_data)]);
        assert_eq!(outs.len(), 1, "graph should produce exactly one output");
        let out = &outs[0];
        (out.shape.clone(), out.copy_to_cpu_vec())
    }

    /// Prefix slice: `[4, 8]` with `start=0, end=3` → `[4, 3]`.
    /// Each row is the first three elements of the input row.
    #[test]
    fn slice_last_dim_prefix_4x8() {
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let (shape, out) = slice_once(vec![4, 8], data, 0, 3);
        assert_eq!(shape, vec![4, 3]);
        assert_eq!(out.len(), 12);
        // Row 0 starts at index 0: [0, 1, 2]
        // Row 1 starts at index 8: [8, 9, 10]
        // Row 2 starts at index 16: [16, 17, 18]
        // Row 3 starts at index 24: [24, 25, 26]
        let expected: Vec<f32> = vec![
            0.0, 1.0, 2.0, 8.0, 9.0, 10.0, 16.0, 17.0, 18.0, 24.0, 25.0, 26.0,
        ];
        assert_eq!(out, expected);
    }

    /// Suffix slice: `[4, 8]` with `start=3, end=8` → `[4, 5]`.
    /// Each row is the last five elements of the input row.
    #[test]
    fn slice_last_dim_suffix_4x8() {
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let (shape, out) = slice_once(vec![4, 8], data, 3, 8);
        assert_eq!(shape, vec![4, 5]);
        // Row 0: indices 3..8 of input row 0 = [3, 4, 5, 6, 7]
        // Row 3: indices 3..8 of input row 3 = [27, 28, 29, 30, 31]
        let expected: Vec<f32> = vec![
            3.0, 4.0, 5.0, 6.0, 7.0, 11.0, 12.0, 13.0, 14.0, 15.0, 19.0, 20.0, 21.0, 22.0, 23.0,
            27.0, 28.0, 29.0, 30.0, 31.0,
        ];
        assert_eq!(out, expected);
    }

    /// Phi-3.5 Mini Q-slice shape: `[2, 3, 9216]` with
    /// `start=0, end=3072` → `[2, 3, 3072]`. The fused QKV matmul
    /// output has the layout `[batch, seq, n_heads_q*head_dim +
    /// 2*n_heads_kv*head_dim]`; slicing the first
    /// `n_heads_q*head_dim` (= 32*96 = 3072) values along the
    /// last axis yields the Q activation.
    #[test]
    fn slice_last_dim_phi35_q_shape() {
        let total: usize = 2 * 3 * 9216;
        let data: Vec<f32> = (0..total).map(|i| i as f32 * 1e-4).collect();
        let (shape, out) = slice_once(vec![2, 3, 9216], data.clone(), 0, 3072);
        assert_eq!(shape, vec![2, 3, 3072]);
        assert_eq!(out.len(), 2 * 3 * 3072);
        // Spot-check first three values of each (batch, seq) row
        // match the input's [0..3] of the same row.
        for b in 0..2 {
            for s in 0..3 {
                let in_row_off = (b * 3 + s) * 9216;
                let out_row_off = (b * 3 + s) * 3072;
                for k in 0..3 {
                    assert_eq!(
                        out[out_row_off + k],
                        data[in_row_off + k],
                        "Q slice mismatch at (b={b}, s={s}, k={k})"
                    );
                }
            }
        }
    }

    /// 3-rank slice preserves both leading dims. Catches a bug
    /// where the executor accidentally collapses leading dims.
    #[test]
    fn slice_last_dim_preserves_leading_dims() {
        let total = 2 * 4 * 6;
        let data: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let (shape, out) = slice_once(vec![2, 4, 6], data, 1, 4);
        assert_eq!(shape, vec![2, 4, 3]);
        assert_eq!(out.len(), 24);
        // Sample (b=1, s=2): input row offset = (1 * 4 + 2) * 6 = 36.
        // Slice [1..4] → indices 37, 38, 39 of the flat input.
        let in_row_off = (1 * 4 + 2) * 6;
        let out_row_off = (1 * 4 + 2) * 3;
        for k in 0..3 {
            assert_eq!(
                out[out_row_off + k],
                (in_row_off + 1 + k) as f32,
                "row (b=1, s=2) mismatch at k={k}"
            );
        }
    }

    /// `start >= end` must panic at builder time (caught by the
    /// `slice_last_dim` helper's assert before any node is added).
    #[test]
    #[should_panic(expected = "start (5) must be < end (5)")]
    fn slice_last_dim_builder_rejects_start_eq_end() {
        let mut gb = GraphBuilder::new();
        let x = gb.input();
        let _ = gb.slice_last_dim(x, 5, 5);
    }

    /// `end > last_dim` is caught at executor time (the input
    /// shape is not statically known at build time). Forward
    /// the request and confirm it panics with a clear message.
    #[test]
    #[should_panic(expected = "exceeds last-axis length")]
    fn slice_last_dim_executor_rejects_end_past_last_dim() {
        let mut gb = GraphBuilder::new();
        let x = gb.input();
        let s = gb.slice_last_dim(x, 0, 100); // last-dim is 8
        let _ = gb.output(s);
        let mut graph = gb.build();
        let _ = graph.execute(vec![Tensor::new_cpu(vec![2, 8], vec![0.0; 16])]);
    }
}

#[cfg(test)]
mod m11_c_step2_tests {
    //! **M11.C step 2** — `NodeType::SoftCap` end-to-end tests.
    //! Covers: identity at zero, exact saturation point at `±cap`,
    //! large-magnitude saturation behaviour (no NaN), shape
    //! preservation, builder rejection of non-positive caps, and
    //! a Gemma 2 attention-shape case for the 50.0 cap.
    use super::*;
    use crate::tensor::Tensor;

    /// Helper: build `[input → soft_cap → output]` and execute
    /// once with the given input tensor. Returns shape and
    /// flat output for assertions.
    fn soft_cap_once(
        input_shape: Vec<usize>,
        input_data: Vec<f32>,
        cap: f32,
    ) -> (Vec<usize>, Vec<f32>) {
        let mut gb = GraphBuilder::new();
        let x = gb.input();
        let s = gb.soft_cap(x, cap);
        let _ = gb.output(s);
        let mut graph = gb.build();
        let outs = graph.execute(vec![Tensor::new_cpu(input_shape, input_data)]);
        assert_eq!(outs.len(), 1, "graph should produce exactly one output");
        let out = &outs[0];
        (out.shape.clone(), out.copy_to_cpu_vec())
    }

    /// `soft_cap(0.0, cap=50.0) == 0.0` exactly. `tanh(0) = 0`,
    /// so `cap * 0 = 0`. Round-trip through the executor preserves
    /// the bit pattern.
    #[test]
    fn soft_cap_identity_at_zero() {
        let (shape, out) = soft_cap_once(vec![1, 1], vec![0.0], 50.0);
        assert_eq!(shape, vec![1, 1]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], 0.0_f32, "soft_cap(0) must be exactly 0");
    }

    /// At the exact saturation point `x = cap`, the output is
    /// `cap * tanh(1)`. For cap=50 that's `50 * 0.7615941559…`.
    /// Verify within a tight tolerance — `f32::tanh` precision
    /// is ~1 ULP at unity argument.
    #[test]
    fn soft_cap_at_cap_equals_cap_times_tanh_one() {
        let (_shape, out) = soft_cap_once(vec![2], vec![50.0, -50.0], 50.0);
        let expected = 50.0_f32 * 1.0_f32.tanh();
        assert!(
            (out[0] - expected).abs() < 1e-5,
            "soft_cap(cap) ≈ cap*tanh(1): got {}, expected {expected}",
            out[0]
        );
        assert!(
            (out[1] - (-expected)).abs() < 1e-5,
            "soft_cap(-cap) ≈ -cap*tanh(1): got {}, expected {}",
            out[1],
            -expected
        );
    }

    /// Saturation regime: `|x| >> cap` produces output very close
    /// to `±cap`, and crucially is finite (no NaN, no Inf). This
    /// is the regression check for the M11.C step 2 PARAR rule —
    /// `f32::tanh(x/cap)` for `|x/cap| > ~9` returns exactly
    /// `±1.0` so the output is exactly `±cap`.
    #[test]
    fn soft_cap_saturates_finite_no_nan() {
        let extreme: Vec<f32> = vec![
            1_000.0, // 20× cap
            -1_000.0,
            1.0e6, // 20000× cap
            -1.0e6,
            f32::MAX / 2.0, // ~1.7e38, still finite when divided
            -f32::MAX / 2.0,
        ];
        let cap = 50.0_f32;
        let (_shape, out) = soft_cap_once(vec![extreme.len()], extreme.clone(), cap);
        for (i, v) in out.iter().enumerate() {
            assert!(
                v.is_finite(),
                "soft_cap({}) produced non-finite output {v} — \
                 PARAR: f32::tanh path leaks NaN/Inf",
                extreme[i]
            );
            assert!(
                v.abs() <= cap + 1e-4,
                "soft_cap({}) overshot cap=50: got {v}",
                extreme[i]
            );
        }
        // Direction-of-saturation check: extreme positive → near +cap,
        // extreme negative → near -cap.
        assert!((out[0] - cap).abs() < 1e-4, "+1000 → +cap got {}", out[0]);
        assert!((out[1] + cap).abs() < 1e-4, "-1000 → -cap got {}", out[1]);
    }

    /// Shape is preserved across the soft-cap. `[4, 8]` input
    /// must yield `[4, 8]` output with element-wise application
    /// of the scalar function.
    #[test]
    fn soft_cap_preserves_shape_4x8() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 5.0).collect();
        let (shape, out) = soft_cap_once(vec![4, 8], data.clone(), 30.0);
        assert_eq!(shape, vec![4, 8]);
        assert_eq!(out.len(), 32);
        let cap = 30.0_f32;
        for (i, &x) in data.iter().enumerate() {
            let expected = cap * (x / cap).tanh();
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "elementwise mismatch at index {i}: got {} expected {}",
                out[i],
                expected
            );
        }
    }

    /// Gemma 2-shaped attention-score case: a `[1, num_heads,
    /// seq, seq]` tensor with cap=50.0. Verifies the executor
    /// handles the rank-4 layout the production graph will hand
    /// it. Synthetic data spans values that are both well below
    /// and well above the cap so both linear-regime and
    /// saturation-regime branches of `f32::tanh` execute.
    #[test]
    fn soft_cap_gemma2_attention_shape() {
        let shape = vec![1, 8, 4, 4];
        let total: usize = shape.iter().product();
        let data: Vec<f32> = (0..total)
            .map(|i| (i as f32 - total as f32 / 2.0) * 2.0)
            .collect();
        let (out_shape, out) = soft_cap_once(shape.clone(), data.clone(), 50.0);
        assert_eq!(out_shape, shape);
        assert_eq!(out.len(), total);
        let cap = 50.0_f32;
        for (i, &x) in data.iter().enumerate() {
            let expected = cap * (x / cap).tanh();
            assert!(
                (out[i] - expected).abs() < 1e-5,
                "attn-shape mismatch at {i}"
            );
            assert!(out[i].is_finite(), "non-finite at {i}");
        }
    }

    /// Builder rejects `cap == 0.0` (the saturation function is
    /// undefined when `cap == 0` because `1/cap` is `+inf`).
    #[test]
    #[should_panic(expected = "cap must be a finite positive f32")]
    fn soft_cap_builder_rejects_zero_cap() {
        let mut gb = GraphBuilder::new();
        let x = gb.input();
        let _ = gb.soft_cap(x, 0.0);
    }

    /// Builder rejects negative caps. `out = cap * tanh(x/cap)`
    /// is mathematically defined for cap < 0 (an odd-symmetric
    /// reflection), but every real use-case wants positive caps,
    /// and a negative cap usually indicates an upstream bug.
    #[test]
    #[should_panic(expected = "cap must be a finite positive f32")]
    fn soft_cap_builder_rejects_negative_cap() {
        let mut gb = GraphBuilder::new();
        let x = gb.input();
        let _ = gb.soft_cap(x, -1.0);
    }

    /// Builder rejects non-finite caps (NaN / ±Inf).
    #[test]
    #[should_panic(expected = "cap must be a finite positive f32")]
    fn soft_cap_builder_rejects_nan_cap() {
        let mut gb = GraphBuilder::new();
        let x = gb.input();
        let _ = gb.soft_cap(x, f32::NAN);
    }
}
