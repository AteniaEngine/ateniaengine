use crate::amg::nodes::{Node, NodeType};
use std::collections::HashMap;
use crate::apx4_13::fusion_engine::{FusionEngine, FusedOp};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecStep {
    /// Execute a single node by ID.
    Single(usize),
    /// Fused pattern: (Add → Mul) where Mul takes Add's output and another input.
    FusedAddMul {
        add_node: usize,
        mul_node: usize,
    },
}

#[derive(Clone, Debug)]
pub struct ExecutionPlan {
    pub steps: Vec<ExecStep>,
}

impl ExecutionPlan {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }
}

pub fn build_execution_plan(nodes: &Vec<Node>) -> (ExecutionPlan, HashMap<usize, FusedOp>) {
    // Debug: dump initial node list before building the plan.
    if crate::apx_debug_enabled() {
        eprintln!("[SCHED] Initial nodes:");
        for (id, node) in nodes.iter().enumerate() {
            eprintln!("  id={}: {:?}", id, node.node_type);
        }
    }

    let mut plan = ExecutionPlan::new();
    let mut fused_map: HashMap<usize, FusedOp> = HashMap::new();

    // APX 4.13 / 4.14 / 4.16 / 4.17 / 4.18 / 4.19: detectar patrones fusionables en el grafo original.
    let mode = crate::apx_mode();
    if mode.starts_with("4.13")
        || mode.starts_with("4.14")
        || mode.starts_with("4.16")
        || mode.starts_with("4.17")
        || mode.starts_with("4.18")
        || mode.starts_with("4.19")
    {
        let fused = FusionEngine::detect_patterns(nodes);
        for (id, fop) in fused {
            eprintln!("[APX 4.13] Fusion detected at node {id}: {:?}", fop);
            fused_map.insert(id, fop);
        }
    }
    let mut i = 0;

    while i < nodes.len() {
        let node = &nodes[i];

        match node.node_type {
            NodeType::Add => {
                // Try to fuse this Add with a following Mul.
                let add_id = node.id;

                if i + 1 < nodes.len() {
                    let next = &nodes[i + 1];
                    if let NodeType::Mul = next.node_type {
                        if !next.inputs.is_empty() && next.inputs[0] == add_id {
                            // Pattern: Add (id = add_id), then Mul(add_output, something)
                            plan.steps.push(ExecStep::FusedAddMul {
                                add_node: add_id,
                                mul_node: next.id,
                            });
                            i += 2;
                            continue;
                        }
                    }
                }

                // If no fusion possible, keep as single.
                plan.steps.push(ExecStep::Single(node.id));
                i += 1;
            }
            _ => {
                plan.steps.push(ExecStep::Single(node.id));
                i += 1;
            }
        }
    }

    if crate::apx_debug_enabled() {
        eprintln!("[SCHED] Final plan steps:");
        for step in &plan.steps {
            eprintln!("  {:?}", step);
        }
    }

    (plan, fused_map)
}
