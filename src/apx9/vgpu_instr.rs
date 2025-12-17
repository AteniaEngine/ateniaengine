// APX 9.20 — Instrucciones virtuales SIMT para el pipeline VGPU.
// Totalmente simbólicas, no tocan memoria real ni GPU.

#[derive(Debug, Clone)]
pub enum VGPUInstr {
    Noop,
    Add { dst: usize, a: usize, b: usize },
    If {
        pred: Vec<bool>,
        then_pc: usize,
        else_pc: usize,
        join_pc: usize,
    },
    Reconverge,
    // APX 9.24 — operación HMMA/MMA simbólica sobre tiles pequeños en memoria global.
    HMMA {
        a_ptr: usize,
        b_ptr: usize,
        c_ptr: usize,
        m: usize,
        k: usize,
        n: usize,
    },
}

impl VGPUInstr {
    /// Registros fuente que esta instrucción lee (para RAW/WAR hazards simbólicos).
    pub fn read_regs(&self) -> Vec<usize> {
        match self {
            VGPUInstr::Noop => Vec::new(),
            VGPUInstr::Add { a, b, .. } => vec![*a, *b],
            VGPUInstr::If { .. } => Vec::new(),
            VGPUInstr::Reconverge => Vec::new(),
            VGPUInstr::HMMA { .. } => Vec::new(), // usa memoria, no registros
        }
    }

    /// Registros destino que esta instrucción escribe (para WAW hazards simbólicos).
    pub fn write_regs(&self) -> Vec<usize> {
        match self {
            VGPUInstr::Noop => Vec::new(),
            VGPUInstr::Add { dst, .. } => vec![*dst],
            VGPUInstr::If { .. } => Vec::new(),
            VGPUInstr::Reconverge => Vec::new(),
            VGPUInstr::HMMA { .. } => Vec::new(),
        }
    }
}
