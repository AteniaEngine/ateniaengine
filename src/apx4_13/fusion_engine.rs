use crate::amg::nodes::{Node, NodeType};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum FusedOp {
    LinearSilu { x: usize, w: usize, b: Option<usize> },
    FusedQKV {
        x: usize,
        wq: usize,
        wk: usize,
        wv: usize,
        bq: Option<usize>,
        bk: Option<usize>,
        bv: Option<usize>,
        q_id: usize,
        k_id: usize,
        v_id: usize,
    },
    /// APX 4.17 / 4.18: fusion of the Self-Attention core.
    /// Includes references to X, weights and biases for Q/K/V so
    /// we can perform fused backward in APX 4.18.
    FusedSelfAttention {
        x: usize,
        wq: usize,
        wk: usize,
        wv: usize,
        bq: Option<usize>,
        bk: Option<usize>,
        bv: Option<usize>,
        q: usize,
        k: usize,
        v: usize,
        /// Representative node of the pattern (typically the MatMul A·V).
        out_id: usize,
    },
}

pub struct FusionEngine;

impl FusionEngine {
    /// Detect fusible patterns over the static node list and return a vector
    /// of (node_id, FusedOp) where node_id is the pattern “representative”
    /// node (e.g., the initial Linear).
    pub fn detect_patterns(nodes: &[Node]) -> Vec<(usize, FusedOp)> {
        let mut fused: Vec<(usize, FusedOp)> = Vec::new();
        let mode = crate::apx_mode();
        // In 4.17 and 4.18 we disable classic FusedQKV to avoid interfering
        // with the experimental Self-Attention flow; in 4.19 we want both
        // features enabled.
        let disable_qkv = mode.starts_with("4.17") || mode.starts_with("4.18");
        let enable_sa = mode.starts_with("4.17") || mode.starts_with("4.18") || mode.starts_with("4.19");

        // --- APX 4.13: Linear → SiLU ---
        for (id, node) in nodes.iter().enumerate() {
            if let NodeType::Linear = node.node_type {
                if let Some(next) = nodes.get(id + 1) {
                    if matches!(next.node_type, NodeType::SiLU) {
                        if node.inputs.len() >= 2 {
                            if crate::apx_debug_enabled() {
                                eprintln!(
                                    "[APX 4.13] Fusion detected at node {}: LinearSilu {{ x: {}, w: {}, b: {:?} }}",
                                    id,
                                    node.inputs[0],
                                    node.inputs[1],
                                    node.inputs.get(2),
                                );
                            }
                            fused.push((
                                id,
                                FusedOp::LinearSilu {
                                    x: node.inputs[0],
                                    w: node.inputs[1],
                                    b: node.inputs.get(2).cloned(),
                                },
                            ));
                        }
                    }
                }
            }
        }

        // --- APX 4.14 / 4.17: QKV fusion (3 Linear that share X) ---
        let mut by_input: HashMap<usize, Vec<(usize, &Node)>> = HashMap::new();

        for (id, node) in nodes.iter().enumerate() {
            if let NodeType::Linear = node.node_type {
                if node.inputs.len() >= 2 {
                    let x = node.inputs[0];
                    by_input.entry(x).or_default().push((id, node));
                }
            }
        }

        for (x, group) in by_input {
            if group.len() != 3 {
                continue;
            }

            // For now, use appearance order as (Q, K, V).
            let mut g = group.clone();
            g.sort_by_key(|(id, _)| *id);

            let (id_q, q) = g[0];
            let (id_k, k_) = g[1];
            let (id_v, v_) = g[2];

            if q.inputs.len() < 2 || k_.inputs.len() < 2 || v_.inputs.len() < 2 {
                continue;
            }

            let wq = q.inputs[1];
            let wk = k_.inputs[1];
            let wv = v_.inputs[1];

            let bq = if q.inputs.len() == 3 { Some(q.inputs[2]) } else { None };
            let bk = if k_.inputs.len() == 3 { Some(k_.inputs[2]) } else { None };
            let bv = if v_.inputs.len() == 3 { Some(v_.inputs[2]) } else { None };

            // Standard QKV fusion (4.14 / 4.16 / 4.19) remains active when we
            // are not in the experimental 4.17/4.18 modes.
            if !disable_qkv {
                fused.push((
                    id_q,
                    FusedOp::FusedQKV {
                        x,
                        wq,
                        wk,
                        wv,
                        bq,
                        bk,
                        bv,
                        q_id: id_q,
                        k_id: id_k,
                        v_id: id_v,
                    },
                ));

                if crate::apx_debug_enabled() {
                    eprintln!(
                        "[APX 4.14] QKV fusion detected at nodes [{}, {}, {}]",
                        id_q, id_k, id_v
                    );
                }
            }

            // APX 4.17 / 4.18 / 4.19: detect Self-Attention core
            if enable_sa {
                // 1) MatMul(Q, K^T) where RHS is a Transpose(K).
                let mut qk_matmul: Option<usize> = None;
                for n in nodes {
                    if let NodeType::MatMul = n.node_type {
                        if n.inputs.len() == 2 && n.inputs[0] == id_q {
                            let rhs = n.inputs[1];
                            let rhs_node = &nodes[rhs];
                            match rhs_node.node_type {
                                NodeType::TransposeLastTwo | NodeType::Transpose2D => {
                                    if rhs_node.inputs.len() == 1 && rhs_node.inputs[0] == id_k {
                                        qk_matmul = Some(n.id);
                                        break;
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }

                let qk_id = match qk_matmul {
                    Some(id) => id,
                    None => continue,
                };

                // 2) Softmax(QK)
                let mut softmax_id: Option<usize> = None;
                for n in nodes {
                    if let NodeType::Softmax = n.node_type {
                        if n.inputs.len() == 1 && n.inputs[0] == qk_id {
                            softmax_id = Some(n.id);
                            break;
                        }
                    }
                }

                let att_id = match softmax_id {
                    Some(id) => id,
                    None => continue,
                };

                // 3) MatMul(A, V)
                let mut av_matmul: Option<usize> = None;
                for n in nodes {
                    if let NodeType::MatMul = n.node_type {
                        if n.inputs.len() == 2 && n.inputs[0] == att_id && n.inputs[1] == id_v {
                            av_matmul = Some(n.id);
                            break;
                        }
                    }
                }

                let out_id = match av_matmul {
                    Some(id) => id,
                    None => continue,
                };

                fused.push((
                    out_id,
                    FusedOp::FusedSelfAttention {
                        x,
                        wq,
                        wk,
                        wv,
                        bq,
                        bk,
                        bv,
                        q: id_q,
                        k: id_k,
                        v: id_v,
                        out_id,
                    },
                ));

                if crate::apx_debug_enabled() {
                    eprintln!(
                        "[APX 4.17] SelfAttention fusion detected at nodes [q={}, k={}, v={}, out={}]",
                        id_q, id_k, id_v, out_id
                    );
                }
            }
        }

        fused
    }
}