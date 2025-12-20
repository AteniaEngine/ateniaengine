// APX 9.6 — GPU Memory Planner v0 (GMPv0)
// GPU memory planning simulator. Does not use real VRAM nor execute kernels.

use crate::amg::graph::Graph;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct MemAssign {
    pub node_id: usize,
    pub offset: usize,
    pub size: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryPlan {
    pub total_required: usize,
    pub temp_peak: usize,
    pub assignments: Vec<MemAssign>,
    pub spills: Vec<usize>, // node ids that should spill to CPU
}

pub struct GPUMemoryPlanner {
    pub total_vram_sim: usize,
    pub heap: Vec<u8>,
}

impl GPUMemoryPlanner {
    pub fn new(total_vram_sim: usize) -> Self {
        // Simulated heap: empty vector with optional capacity.
        let heap = Vec::with_capacity(total_vram_sim);
        Self { total_vram_sim, heap }
    }

    /// Estimate bytes required by a tensor (including dtype).
    pub fn estimate_tensor_size(t: &Tensor) -> usize {
        t.estimated_bytes()
    }

    /// Generate a symbolic memory plan for a graph.
    /// Does not modify the graph nor touch real data.
    pub fn plan_for_graph(&mut self, graph: &Graph) -> MemoryPlan {
        let mut plan = MemoryPlan {
            total_required: 0,
            temp_peak: 0,
            assignments: Vec::new(),
            spills: Vec::new(),
        };

        // Free list simple: (offset, size)
        let mut free_list: Vec<(usize, usize)> = Vec::new();
        let mut next_offset: usize = 0;
        let mut current_usage: usize = 0;

        // Store previous sizes to simulate reuse in an A->B->C chain.
        let mut sizes: Vec<usize> = vec![0; graph.nodes.len()];
        let mut offsets: Vec<Option<usize>> = vec![None; graph.nodes.len()];

        const SPILL_THRESHOLD_BYTES: usize = 200 * 1024 * 1024; // ~200MB
        const LARGE_BLOCK_THRESHOLD_BYTES: usize = 50 * 1024 * 1024; // 50MB

        for (idx, node) in graph.nodes.iter().enumerate() {
            // Only consider nodes with outputs (Parameter/Input/Output or intermediates).
            let t_opt: Option<&Tensor> = node.output.as_ref();
            if t_opt.is_none() {
                continue;
            }
            let t = t_opt.unwrap();

            // Basic size-based policy.
            let sz = Self::estimate_tensor_size(t);
            sizes[idx] = sz;

            if sz > SPILL_THRESHOLD_BYTES {
                // Too large: mark spill to CPU.
                plan.spills.push(idx);
                continue;
            }

            // Block assignment (normal or symbolic "large").
            let alloc_size = if sz >= LARGE_BLOCK_THRESHOLD_BYTES { sz } else { sz };

            let (offset, assigned) = allocate_from_free_list(&mut free_list, &mut next_offset, alloc_size);
            current_usage += assigned;

            if current_usage > plan.temp_peak {
                plan.temp_peak = current_usage;
            }
            if next_offset > plan.total_required {
                plan.total_required = next_offset;
            }

            plan.assignments.push(MemAssign { node_id: idx, offset, size: assigned });
            offsets[idx] = Some(offset);

            // Very simple reuse heuristic: in an A->B->C chain, free the
            // previous node before the current one so the next can reuse its
            // region. Does not modify real data.
            if idx > 0 {
                let prev_id = idx - 1;
                if !plan.spills.contains(&prev_id) {
                    let prev_size = sizes[prev_id];
                    if let Some(prev_off) = offsets[prev_id] {
                        if prev_size > 0 {
                            current_usage = current_usage.saturating_sub(prev_size);
                            free_list.push((prev_off, prev_size));
                        }
                    }
                }
            }
        }

        plan
    }
}

/// Very simple allocation: first try to reuse a hole from the free list;
/// otherwise, allocate at the end of the simulated heap.
fn allocate_from_free_list(
    free_list: &mut Vec<(usize, usize)>,
    next_offset: &mut usize,
    size: usize,
) -> (usize, usize) {
    // First-fit: find a hole large enough.
    if let Some((idx, (off, _sz))) = free_list
        .iter()
        .enumerate()
        .find(|(_, (_, sz))| *sz >= size)
    {
        let off = *off;
        free_list.swap_remove(idx);
        return (off, size);
    }

    let off = *next_offset;
    *next_offset += size;
    (off, size)
}
