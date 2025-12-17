// APX 8.12 — GPU MetaLayer (GPU IR Optimizer v0)
// Opera sólo sobre IR sintético; no ejecuta kernels reales ni cambia la matemática.

use crate::apx8::kernel_generator::{KernelIR, KernelOp};
use std::collections::HashMap;

/// Resultado optimizado del metalayer.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizedIR {
    pub ops: Vec<KernelOp>,
    pub meta: HashMap<String, String>, // info extra (tiling, fusion, vector width)
}

/// MetaLayer v0 — puramente sintético.
/// - Elimina NOPs
/// - Mantiene Load/Compute/Store triviales
/// - Añade metadatos de tiling/vectorización (texto)
/// No cambia la matemática.
pub fn optimize_ir(ir: &KernelIR) -> OptimizedIR {
    let mut ops = Vec::new();

    for op in &ir.ops {
        match op {
            KernelOp::Nop => {
                // eliminar operación inútil
            }
            KernelOp::LoadTensor(_) | KernelOp::StoreTensor(_) | KernelOp::Compute(_) => {
                ops.push(op.clone());
            }
        }
    }

    // “Simulación” de reglas de optimización
    let mut meta = HashMap::new();
    meta.insert("fusion".into(), "trivial-load-compute-store".into());
    meta.insert("tiling".into(), "8x8".into());
    meta.insert("vectorization".into(), "v4-synthetic".into());

    OptimizedIR { ops, meta }
}
