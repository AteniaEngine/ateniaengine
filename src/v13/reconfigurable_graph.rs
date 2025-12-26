use super::execution_planner::{ExecutionTarget, HybridExecutionPlanner};
use super::hybrid_memory::HybridMemoryManager;
use super::kernel_model::KernelProfile;
use super::memory_types::{MemorySnapshot, MemoryTier};

pub type NodeId = u64;

#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: NodeId,
    pub kernel: KernelProfile,
    pub tensor_ids: Vec<String>,
    pub tensor_tiers: Vec<MemoryTier>,
}

#[derive(Debug, Default)]
pub struct ReconfigurableGraph {
    nodes: Vec<GraphNode>,
    next_id: NodeId,
}

impl ReconfigurableGraph {
    pub fn new() -> Self {
        ReconfigurableGraph {
            nodes: Vec::new(),
            next_id: 0,
        }
    }

    pub fn add_node(&mut self, kernel: KernelProfile, tensor_tiers: Vec<MemoryTier>) -> NodeId {
        let id = self.next_id;
        let generated_ids: Vec<String> = (0..tensor_tiers.len())
            .map(|idx| format!("t{}_{}", id, idx))
            .collect();
        self.add_node_with_tensors(kernel, generated_ids, tensor_tiers)
    }

    pub fn add_node_with_tensors(
        &mut self,
        kernel: KernelProfile,
        tensor_ids: Vec<String>,
        tensor_tiers: Vec<MemoryTier>,
    ) -> NodeId {
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);

        // Ensure we do not go out of bounds if lengths differ.
        let pair_len = std::cmp::min(tensor_ids.len(), tensor_tiers.len());
        let tensor_ids = tensor_ids.into_iter().take(pair_len).collect();
        let tensor_tiers = tensor_tiers.into_iter().take(pair_len).collect();

        let node = GraphNode {
            id,
            kernel,
            tensor_ids,
            tensor_tiers,
        };
        self.nodes.push(node);
        id
    }

    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    pub fn plan_for_snapshot(
        &self,
        snapshot: &MemorySnapshot,
        gpu_available: bool,
    ) -> GraphPlacementPlan {
        let vram_p = snapshot.vram.pressure.unwrap_or(0.0);
        let ram_p = snapshot.ram.pressure.unwrap_or(0.0);
        let snapshot_summary = format!(
            "Snapshot pressures: vram={:.4}, ram={:.4}",
            vram_p, ram_p
        );

        let mut placements = Vec::with_capacity(self.nodes.len());

        for node in &self.nodes {
            let plan = HybridExecutionPlanner::plan(
                &node.kernel,
                &node.tensor_tiers,
                snapshot,
                gpu_available,
            );

            let node_placement = NodePlacement {
                node_id: node.id,
                target: plan.target,
                reason: plan.reason,
            };
            placements.push(node_placement);
        }

        GraphPlacementPlan {
            placements,
            snapshot_summary,
        }
    }

    pub fn plan_for_snapshot_with_mem(
        &self,
        mem: &HybridMemoryManager,
        snapshot: &MemorySnapshot,
        gpu_available: bool,
    ) -> GraphPlacementPlan {
        let vram_p = snapshot.vram.pressure.unwrap_or(0.0);
        let ram_p = snapshot.ram.pressure.unwrap_or(0.0);
        let snapshot_summary = format!(
            "Snapshot pressures: vram={:.4}, ram={:.4}",
            vram_p, ram_p
        );

        let mut placements = Vec::with_capacity(self.nodes.len());

        for node in &self.nodes {
            let mut tiers: Vec<MemoryTier> = Vec::with_capacity(node.tensor_ids.len());
            for id in &node.tensor_ids {
                let tier = match mem.get_tier(id) {
                    Some(t) => t,
                    None => MemoryTier::Ram,
                };
                tiers.push(tier);
            }

            let plan = HybridExecutionPlanner::plan(&node.kernel, &tiers, snapshot, gpu_available);

            let node_placement = NodePlacement {
                node_id: node.id,
                target: plan.target,
                reason: plan.reason,
            };
            placements.push(node_placement);
        }

        GraphPlacementPlan {
            placements,
            snapshot_summary,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NodePlacement {
    pub node_id: NodeId,
    pub target: ExecutionTarget,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct GraphPlacementPlan {
    pub placements: Vec<NodePlacement>,
    pub snapshot_summary: String,
}
