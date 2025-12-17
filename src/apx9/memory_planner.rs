// APX 9.6 — GPU Memory Planner v0 (GMPv0)
// Simulador de planificación de memoria GPU. No usa VRAM real ni ejecuta kernels.

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
    pub spills: Vec<usize>, // ids de nodo que deberían caer a CPU
}

pub struct GPUMemoryPlanner {
    pub total_vram_sim: usize,
    pub heap: Vec<u8>,
}

impl GPUMemoryPlanner {
    pub fn new(total_vram_sim: usize) -> Self {
        // Heap simulado: vector vacío con capacidad opcional.
        let heap = Vec::with_capacity(total_vram_sim);
        Self { total_vram_sim, heap }
    }

    /// Estima los bytes requeridos por un tensor (incluyendo dtype).
    pub fn estimate_tensor_size(t: &Tensor) -> usize {
        t.estimated_bytes()
    }

    /// Genera un plan de memoria simbólico para un grafo.
    /// No modifica el grafo ni toca datos reales.
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

        // Guardar tamaños previos para simular reutilización en cadena A->B->C.
        let mut sizes: Vec<usize> = vec![0; graph.nodes.len()];
        let mut offsets: Vec<Option<usize>> = vec![None; graph.nodes.len()];

        const SPILL_THRESHOLD_BYTES: usize = 200 * 1024 * 1024; // ~200MB
        const LARGE_BLOCK_THRESHOLD_BYTES: usize = 50 * 1024 * 1024; // 50MB

        for (idx, node) in graph.nodes.iter().enumerate() {
            // Sólo consideramos nodos con salida (Parameter/Input/Output o intermedios).
            let t_opt: Option<&Tensor> = node.output.as_ref();
            if t_opt.is_none() {
                continue;
            }
            let t = t_opt.unwrap();

            // Política básica por tamaño.
            let sz = Self::estimate_tensor_size(t);
            sizes[idx] = sz;

            if sz > SPILL_THRESHOLD_BYTES {
                // Demasiado grande: marcar spill a CPU.
                plan.spills.push(idx);
                continue;
            }

            // Asignación de bloque (normal o "grande" simbólico).
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

            // Heurística de reutilización muy simple: en una cadena A->B->C,
            // liberamos el nodo anterior al actual para permitir que el
            // siguiente reutilice su región. No modifica datos reales.
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

/// Asignación muy sencilla: primero intenta reutilizar un hueco de la free list,
/// si no, reserva al final del heap simulado.
fn allocate_from_free_list(
    free_list: &mut Vec<(usize, usize)>,
    next_offset: &mut usize,
    size: usize,
) -> (usize, usize) {
    // Primer ajuste: buscar hueco suficientemente grande.
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
