// APX 8.9 — GPU Kernel Generators v0
// Abstract GPU kernel templates. They do not execute anything nor produce real code.

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

// === APX 8.12: operation-list-based IR for the MetaLayer ===

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
    /// Build a test IR for a vector addition.
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

    /// Build a minimal IR for codegen tests with a given name.
    pub fn new_mock(name: &str) -> Self {
        KernelIR {
            ops: Vec::new(),
            name: name.into(),
            params: Vec::new(),
        }
    }

    /// Simple stable hash based on ops and name.
    pub fn hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut s = std::collections::hash_map::DefaultHasher::new();
        self.name.hash(&mut s);
        for op in &self.ops {
            op.hash(&mut s);
        }
        s.finish()
    }

    /// Stable textual IR signature to use as cache key.
    pub fn signature(&self) -> String {
        format!("{}::{}", self.name, self.hash())
    }
}
