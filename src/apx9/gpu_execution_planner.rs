// APX 9.7 — GPU Execution Planner v0 (GXP)
// Fully simulated GPU execution planner. Does not execute kernels nor use real VRAM.

use crate::amg::graph::Graph;
use crate::amg::nodes::NodeType;
use crate::tensor::Tensor;
use crate::apx8::device_planner::plan_for_ir;
use crate::apx8::gpu_partition::suggest_partition;
use crate::apx9::memory_planner::{GPUMemoryPlanner, MemoryPlan};

#[derive(Debug, Clone)]
pub struct GPUPlanStep {
    pub node_id: usize,
    pub device: String,       // "cpu" / "cuda0" / "hip0" / etc
    pub partitions: Vec<(usize, usize)>, // simulated partitions (start,end)
    pub kernel_name: String,  // mock kernel name (from IR/Codegen)
    pub estimated_time_ms: f32,
    pub spill_to_cpu: bool,
}

#[derive(Debug, Clone)]
pub struct GPUExecutionPlan {
    pub steps: Vec<GPUPlanStep>,
    pub total_vram_needed: usize,
    pub spills: Vec<usize>,
}

pub struct GPUExecutionPlanner {
    pub total_vram_sim: usize,
}

impl GPUExecutionPlanner {
    pub fn new(total_vram_sim: usize) -> Self {
        Self { total_vram_sim }
    }

    /// Build a symbolic GPU execution plan for a given graph.
    /// Does not modify the graph nor touch real tensors.
    pub fn build_plan(&self, graph: &Graph) -> GPUExecutionPlan {
        if graph.nodes.is_empty() {
            return GPUExecutionPlan { steps: Vec::new(), total_vram_needed: 0, spills: Vec::new() };
        }

        // 1) Symbolic memory plan (GMPv0)
        let mut mem_planner = GPUMemoryPlanner::new(self.total_vram_sim);
        let mem_plan: MemoryPlan = mem_planner.plan_for_graph(graph);

        let mut steps = Vec::new();
        let mut spills_all = mem_plan.spills.clone();

        // 2) Walk nodes and decide device/partitions/kernel/time
        for (idx, node) in graph.nodes.iter().enumerate() {
            // Only some types are GPU candidates; the rest are marked CPU-only.
            let eligible = matches!(
                node.node_type,
                NodeType::Add
                    | NodeType::Mul
                    | NodeType::MatMul
                    | NodeType::BatchMatMul
                    | NodeType::Linear
                    | NodeType::RmsNorm
                    | NodeType::SiLU
                    | NodeType::Softmax
            );

            // Determine size (if there is an output tensor).
            let tensor_opt: Option<&Tensor> = node.output.as_ref();
            let tensor_size_bytes: usize = tensor_opt.map(|t| t.estimated_bytes()).unwrap_or(0);

            // Device planner 8.18 (used only to select a mock device string).
            let dp = plan_for_ir(&format!("{:?}", node.node_type));
            let mut device = if dp.target_gpu.is_some() && eligible {
                "cuda0".to_string()
            } else {
                "cpu".to_string()
            };

            let mut spill_to_cpu = false;

            // If MemoryPlanner marked spill, force CPU-only.
            if mem_plan.spills.contains(&idx) {
                device = "cpu".to_string();
                spill_to_cpu = true;
                if !spills_all.contains(&idx) {
                    spills_all.push(idx);
                }
            }

            // Partitions (using size heuristic + 8.19 planner symbolically).
            let partitions = if tensor_size_bytes > 256 * 1024 * 1024 {
                // Huge tensor: split into 4 uniform (symbolic) partitions.
                let chunk = tensor_size_bytes / 4;
                vec![
                    (0, chunk),
                    (chunk, 2 * chunk),
                    (2 * chunk, 3 * chunk),
                    (3 * chunk, tensor_size_bytes),
                ]
            } else if let Some(t) = tensor_opt {
                let part_plan = suggest_partition(&t.shape);
                match part_plan.policy {
                    crate::apx8::gpu_partition::PartitionPolicy::Split2D { rows, cols } => {
                        let tiles = (rows * cols) as usize;
                        let chunk = (tensor_size_bytes / tiles.max(1)).max(1);
                        let mut v = Vec::new();
                        let mut start = 0usize;
                        for _ in 0..tiles {
                            let end = (start + chunk).min(tensor_size_bytes);
                            v.push((start, end));
                            start = end;
                        }
                        if v.is_empty() {
                            v.push((0, tensor_size_bytes));
                        }
                        v
                    }
                    crate::apx8::gpu_partition::PartitionPolicy::Split1D { chunks } => {
                        let tiles = chunks as usize;
                        let chunk = (tensor_size_bytes / tiles.max(1)).max(1);
                        let mut v = Vec::new();
                        let mut start = 0usize;
                        for _ in 0..tiles {
                            let end = (start + chunk).min(tensor_size_bytes);
                            v.push((start, end));
                            start = end;
                        }
                        if v.is_empty() {
                            v.push((0, tensor_size_bytes));
                        }
                        v
                    }
                    _ => vec![(0, tensor_size_bytes)],
                }
            } else {
                vec![(0, tensor_size_bytes)]
            };

            // Synthetic kernel name based on node type.
            let kernel_name = match node.node_type {
                NodeType::MatMul | NodeType::BatchMatMul | NodeType::Linear => "kernel_matmul_v0".to_string(),
                NodeType::Add | NodeType::BroadcastAdd => "kernel_add_v0".to_string(),
                NodeType::RmsNorm => "kernel_rmsnorm_v0".to_string(),
                NodeType::SiLU | NodeType::Activation(_) => "kernel_activation_v0".to_string(),
                NodeType::Softmax | NodeType::LogSoftmax => "kernel_softmax_v0".to_string(),
                _ => "kernel_cpu_fallback_v0".to_string(),
            };

            // Symbolic estimated time.
            let estimated_time_ms = (tensor_size_bytes as f32) / 50_000_000.0;

            steps.push(GPUPlanStep {
                node_id: idx,
                device,
                partitions,
                kernel_name,
                estimated_time_ms,
                spill_to_cpu,
            });
        }

        GPUExecutionPlan {
            steps,
            total_vram_needed: mem_plan.total_required,
            spills: spills_all,
        }
    }
}
