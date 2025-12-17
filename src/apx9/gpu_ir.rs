// APX 9.1 — GPU IR v1
// IR minimalista y seguro para kernels GPU, 100% CPU-only y sin dependencias de CUDA/HIP.

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

    // Control de hilo (abstracto)
    ThreadIdxX(String),
    BlockIdxX(String),
    BlockDimX(String),

    // Bucles simples
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

// APX 9.1 / 9.2 — Kernel IR simplificado para el PTX emitter v0.
// Totalmente independiente de CUDA real; sólo describe ops de alto nivel.

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
    // APX 9.16 — barrera de sincronización simbólica entre threads de un bloque.
    // En la simulación actual se interpreta como no-op secuencial, pero preserva el orden.
    Sync,
    // APX 9.17 — predicación SIMT simbólica para divergencia de warps.
    // En esta fase se trata de un marcador lógico, sin efecto en la matemática.
    Predicate { lane_mod: usize, value: usize },
}
