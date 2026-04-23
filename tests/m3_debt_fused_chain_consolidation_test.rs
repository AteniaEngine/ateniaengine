//! APX v20 M3 debt cleanup — `FusedLinearActivationChain`
//! consolidation tests.
//!
//! Accompanies the graph.rs refactor that:
//! - Extracted the two duplicate dispatch paths for
//!   `NodeType::FusedLinearActivationChain` into a single helper
//!   `exec_fused_linear_activation_chain`.
//! - Removed the dead-code match-arm that had drifted on input
//!   validation (accepted only 4/5 inputs where the pre-match
//!   handler correctly accepted 3/4/5).
//! - Relaxed `Graph::validate()`'s check for this NodeType from
//!   `in_len == 4 || 5` to `in_len >= 3 && in_len <= 5`, matching
//!   the APX 4.9 fusion detector's actual output.
//!
//! Two tests:
//!
//! 1. `test_validator_accepts_3_input_fused_chain`: constructs a
//!    synthetic graph with a 3-input FusedLinearActivationChain
//!    (no biases), calls `validate()`, asserts `Ok(_)`. Locks the
//!    contract that the validator now accepts 3-input chains.
//!
//! 2. `test_fused_chain_4_input_single_bias_equivalence`: checks
//!    numerical equivalence between the reference
//!    Linear→SiLU→Linear pipeline and the APX 4.9 fused form
//!    when exactly ONE bias is present. The 3-input case is
//!    covered by `apx_2_5_fused_kernels_test` (which runs
//!    implicitly with no biases). The 5-input case is covered
//!    by `apx_4_9_fusion_test::test_apx_4_9_equivalencia_numerica`.
//!    The 4-input case (single bias) had no dedicated test
//!    until this cleanup.

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::nodes::{ActType, Node, NodeType};
use atenia_engine::nn::activations::silu;
use atenia_engine::nn::linear::linear as linear_op;
use atenia_engine::tensor::{Device, Tensor};

// ---------------------------------------------------------------------
// Test 1: validator accepts 3-input FusedLinearActivationChain
// ---------------------------------------------------------------------

#[test]
fn test_validator_accepts_3_input_fused_chain() {
    // Build a hand-crafted graph that contains a
    // FusedLinearActivationChain with exactly 3 inputs. Goes
    // through Node::new instead of GraphBuilder because the
    // builder would either fuse the pattern itself (producing
    // the chain from the back end of Linear+SiLU+Linear) or
    // reject an unreachable configuration.
    //
    // Graph layout: [input, w1, w2, chain, output]
    // The chain's inputs are [input_id, w1_id, w2_id] — 3 inputs,
    // no biases.
    let nodes = vec![
        Node::new(0, NodeType::Input, vec![]),
        Node::new(1, NodeType::Parameter, vec![]),
        Node::new(2, NodeType::Parameter, vec![]),
        Node::new(
            3,
            NodeType::FusedLinearActivationChain(vec![ActType::SiLU]),
            vec![0, 1, 2],
        ),
        Node::new(4, NodeType::Output, vec![3]),
    ];
    let g = atenia_engine::amg::graph::Graph::build(nodes);

    // Pre-cleanup, this would error with:
    //   "Node 3 (FusedLinearActivationChain(...)) has 3 inputs,
    //    but a different number was expected"
    // Post-cleanup, validator accepts 3 ∨ 4 ∨ 5.
    let result = g.validate();
    assert!(
        result.is_ok(),
        "validate() must accept 3-input FusedLinearActivationChain post-cleanup, got {:?}",
        result
    );
}

// ---------------------------------------------------------------------
// Test 2: 4-input single-bias forward equivalence
// ---------------------------------------------------------------------

#[test]
fn test_fused_chain_4_input_single_bias_equivalence() {
    // Reference computation (no graph, just nn ops): run the
    // Linear→SiLU→Linear pipeline manually with a bias on the
    // FIRST linear only. We also add a second variant with the
    // bias on the SECOND linear so the test covers both 4-input
    // permutations the APX 4.9 detector can produce.
    //
    // The APX 4.9 fusion detector constructs fused_inputs as:
    //   [x, w1, b1?, w2, b2?]
    // so a 4-input chain is either [x, w1, b1, w2] (bias on first
    // linear) or [x, w1, w2, b2] (bias on second linear). This
    // test exercises both.
    let in_dim = 8usize;
    let hidden = 16usize;
    let out_dim = 4usize;

    let x = Tensor::randn(&[1, in_dim], Device::CPU);
    let w1 = Tensor::randn(&[in_dim, hidden], Device::CPU);
    let w2 = Tensor::randn(&[hidden, out_dim], Device::CPU);
    let b1 = Tensor::randn(&[hidden], Device::CPU);
    let b2 = Tensor::randn(&[out_dim], Device::CPU);

    // --- Variant A: bias on first linear only ---
    {
        // Reference.
        let h1 = linear_op(&x, &w1, Some(&b1));
        let a1 = silu(&h1);
        let ref_out = linear_op(&a1, &w2, None);

        // Fused graph.
        let mut gb = GraphBuilder::new();
        let x_id = gb.input();
        let w1_id = gb.parameter(w1.clone());
        let b1_id = gb.parameter(b1.clone());
        let w2_id = gb.parameter(w2.clone());
        let l1 = gb.linear(x_id, w1_id, Some(b1_id));
        let a_node = gb.silu(l1);
        let l2 = gb.linear(a_node, w2_id, None);
        gb.output(l2);
        let mut g = gb.build();

        // Verify the fusion produced a chain with exactly 4 inputs
        // (the 4-input case: [x, w1, b1, w2]).
        let mut chain_arity: Option<usize> = None;
        for n in &g.nodes {
            if let NodeType::FusedLinearActivationChain(_) = &n.node_type {
                chain_arity = Some(n.inputs.len());
                break;
            }
        }
        assert_eq!(
            chain_arity,
            Some(4),
            "variant A: expected a 4-input fused chain; got {:?}",
            chain_arity
        );

        let out = g.execute(vec![x.clone()]);
        assert_eq!(out.len(), 1);

        let mut max_diff = 0.0f32;
        for i in 0..ref_out.numel() {
            let d = (ref_out.as_cpu_slice()[i] - out[0].as_cpu_slice()[i]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff < 1e-3,
            "variant A: max_diff={} exceeds tolerance",
            max_diff
        );
    }

    // --- Variant B: bias on second linear only ---
    {
        let h1 = linear_op(&x, &w1, None);
        let a1 = silu(&h1);
        let ref_out = linear_op(&a1, &w2, Some(&b2));

        let mut gb = GraphBuilder::new();
        let x_id = gb.input();
        let w1_id = gb.parameter(w1.clone());
        let w2_id = gb.parameter(w2.clone());
        let b2_id = gb.parameter(b2.clone());
        let l1 = gb.linear(x_id, w1_id, None);
        let a_node = gb.silu(l1);
        let l2 = gb.linear(a_node, w2_id, Some(b2_id));
        gb.output(l2);
        let mut g = gb.build();

        // This time the chain should be [x, w1, w2, b2] — still
        // 4 inputs, but with the bias in a different slot. The
        // helper's parsing logic handles both via the `idx <
        // inputs.len()` trailing-bias check.
        let mut chain_arity: Option<usize> = None;
        for n in &g.nodes {
            if let NodeType::FusedLinearActivationChain(_) = &n.node_type {
                chain_arity = Some(n.inputs.len());
                break;
            }
        }
        assert_eq!(
            chain_arity,
            Some(4),
            "variant B: expected a 4-input fused chain; got {:?}",
            chain_arity
        );

        let out = g.execute(vec![x.clone()]);
        let mut max_diff = 0.0f32;
        for i in 0..ref_out.numel() {
            let d = (ref_out.as_cpu_slice()[i] - out[0].as_cpu_slice()[i]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff < 1e-3,
            "variant B: max_diff={} exceeds tolerance",
            max_diff
        );
    }
}
