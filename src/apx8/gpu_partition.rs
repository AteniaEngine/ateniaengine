// APX 8.19 — GPU Partitioning Simulator (GPS)
// Planificador de partición simulado, sin particionar datos reales ni tocar ejecución.

#[derive(Debug, Clone)]
pub enum PartitionPolicy {
    None,
    Split1D { chunks: u32 },
    Split2D { rows: u32, cols: u32 },
    Auto,
}

#[derive(Debug, Clone)]
pub struct PartitionPlan {
    pub policy: PartitionPolicy,
    pub estimated_speedup: f32, // purely symbolic
}

pub fn suggest_partition(shape: &[usize]) -> PartitionPlan {
    if shape.len() == 1 {
        // Vector
        if shape[0] < 2048 {
            return PartitionPlan {
                policy: PartitionPolicy::None,
                estimated_speedup: 1.0,
            };
        }
        return PartitionPlan {
            policy: PartitionPolicy::Split1D { chunks: 2 },
            estimated_speedup: 1.2,
        };
    }

    if shape.len() == 2 {
        let (m, n) = (shape[0], shape[1]);

        if m * n < 200_000 {
            return PartitionPlan {
                policy: PartitionPolicy::None,
                estimated_speedup: 1.0,
            };
        }

        if m > 1024 && n > 1024 {
            return PartitionPlan {
                policy: PartitionPolicy::Split2D { rows: 2, cols: 2 },
                estimated_speedup: 1.4,
            };
        }

        return PartitionPlan {
            policy: PartitionPolicy::Split1D { chunks: 2 },
            estimated_speedup: 1.15,
        };
    }

    // Default
    PartitionPlan {
        policy: PartitionPolicy::Auto,
        estimated_speedup: 1.0,
    }
}
