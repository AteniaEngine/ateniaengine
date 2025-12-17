// APX 8.9 — GPU Kernel Generators v0
// Plantillas abstractas de kernels GPU. No ejecutan nada ni producen código real.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GpuKernelOp {
    VecAdd,
    MatMul,
    Linear,
    SiLU,
    BatchMatMul,
    FusedLinearSiLU,
}

#[derive(Debug, Clone)]
pub struct GpuKernelTemplate {
    pub op: GpuKernelOp,
    pub block: (usize, usize),
    pub tile_k: usize,
    pub use_shared: bool,
    pub vectorize: bool,
    pub unroll: usize,
}

#[derive(Debug, Clone)]
pub struct GpuKernelIR {
    pub op: GpuKernelOp,
    pub params: HashMap<String, String>,
}

impl GpuKernelTemplate {
    pub fn to_ir(&self) -> GpuKernelIR {
        let mut params = HashMap::new();
        params.insert("block_x".into(), self.block.0.to_string());
        params.insert("block_y".into(), self.block.1.to_string());
        params.insert("tile_k".into(), self.tile_k.to_string());
        params.insert("use_shared".into(), self.use_shared.to_string());
        params.insert("vectorize".into(), self.vectorize.to_string());
        params.insert("unroll".into(), self.unroll.to_string());

        GpuKernelIR { op: self.op.clone(), params }
    }
}

// === APX 8.12: IR basado en lista de operaciones para el MetaLayer ===

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KernelOp {
    Nop,
    LoadTensor(String),
    StoreTensor(String),
    Compute(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct KernelIR {
    pub ops: Vec<KernelOp>,
    pub name: String,
    pub params: Vec<String>,
}

impl KernelIR {
    /// Construye un IR de prueba para una suma vectorial.
    pub fn mock_add() -> Self {
        KernelIR {
            ops: vec![
                KernelOp::LoadTensor("A".into()),
                KernelOp::Compute("Add".into()),
                KernelOp::StoreTensor("A".into()),
            ],
            name: "mock_add".into(),
            params: vec![],
        }
    }

    /// Construye un IR mínimo para pruebas de codegen con un nombre dado.
    pub fn new_mock(name: &str) -> Self {
        KernelIR {
            ops: Vec::new(),
            name: name.into(),
            params: Vec::new(),
        }
    }

    /// Hash estable simple basado en las operaciones y el nombre.
    pub fn hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut s = std::collections::hash_map::DefaultHasher::new();
        self.name.hash(&mut s);
        for op in &self.ops {
            op.hash(&mut s);
        }
        s.finish()
    }

    /// Firma textual estable del IR para usar como clave de cache.
    pub fn signature(&self) -> String {
        format!("{}::{}", self.name, self.hash())
    }
}
