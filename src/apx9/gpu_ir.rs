// APX 9.1 — GPU IR v1
// Minimal and safe IR for GPU kernels, 100% CPU-only and without CUDA/HIP dependencies.

#[derive(Debug, Clone)]
pub struct GpuIrKernel {
    pub name: String,
    pub params: Vec<GpuIrParam>,
    pub body: Vec<GpuIrStmt>,
}

#[derive(Debug, Clone)]
pub struct GpuIrParam {
    pub name: String,
    pub ty: GpuIrType,
}

#[derive(Debug, Clone)]
pub enum GpuIrType {
    F32,
    I32,
    Ptr,
}

#[derive(Debug, Clone)]
pub enum GpuIrStmt {
    Comment(String),
    Load { dst: String, src: String },
    Store { dst: String, src: String },
    AddF32 { dst: String, a: String, b: String },
    MulF32 { dst: String, a: String, b: String },
    LocalVar { name: String, ty: GpuIrType },

    // Thread control (abstract)
    ThreadIdxX(String),
    BlockIdxX(String),
    BlockDimX(String),

    // Simple loops
    For {
        var: String,
        start: i32,
        end: i32,
        body: Vec<GpuIrStmt>,
    },
}

impl GpuIrKernel {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), params: vec![], body: vec![] }
    }
}

// APX 9.1 / 9.2 — Simplified kernel IR for PTX emitter v0.
// Fully independent of real CUDA; only describes high-level ops.

#[derive(Debug, Clone)]
pub struct GpuKernelIR {
    pub name: String,
    pub threads: u32,
    pub ops: Vec<GpuOp>,
}

#[derive(Debug, Clone)]
pub enum GpuOp {
    Load { dst: String, src: String },
    Add { dst: String, a: String, b: String },
    Store { dst: String, src: String },
    // APX 9.16 — symbolic synchronization barrier between threads in a block.
    // In the current simulation it is treated as a sequential no-op, but preserves ordering.
    Sync,
    // APX 9.17 — symbolic SIMT predication for warp divergence.
    // At this stage it is a logical marker, with no effect on math.
    Predicate { lane_mod: usize, value: usize },
}
