// APX 9.20 — SIMT virtual instructions for the VGPU pipeline.
// Fully symbolic; do not touch real memory nor GPU.

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
    // APX 9.24 — symbolic HMMA/MMA op over small tiles in global memory.
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
    /// Source registers read by this instruction (for symbolic RAW/WAR hazards).
    pub fn read_regs(&self) -> Vec<usize> {
        match self {
            VGPUInstr::Noop => Vec::new(),
            VGPUInstr::Add { a, b, .. } => vec![*a, *b],
            VGPUInstr::If { .. } => Vec::new(),
            VGPUInstr::Reconverge => Vec::new(),
            VGPUInstr::HMMA { .. } => Vec::new(), // uses memory, not registers
        }
    }

    /// Destination registers written by this instruction (for symbolic WAW hazards).
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
