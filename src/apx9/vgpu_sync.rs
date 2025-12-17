// APX 9.16 — VGPU Synchronization Layer (Sync & Barriers)
// DISCLAIMER: sólo modela semántica de sincronización GPU de forma simulada.
// No ejecuta hilos reales, no usa GPU ni VRAM, no modifica backward ni kernels CPU.

#[derive(Debug, Clone)]
pub struct VGPUBarrier {
    pub threads: usize,
    pub arrived: usize,
}

impl VGPUBarrier {
    pub fn new(threads: usize) -> Self {
        Self { threads, arrived: 0 }
    }

    pub fn arrive(&mut self) {
        self.arrived += 1;
        if self.arrived == self.threads {
            self.arrived = 0; // reset cuando todos "llegan"
        }
    }

    pub fn is_complete(&self) -> bool {
        self.arrived == 0
    }
}

pub struct VGPUThreadContext {
    pub tid: usize,
    pub bid: usize,
    pub local_mem: Vec<f32>,
}

pub struct VGPUBlockContext {
    pub threads: Vec<VGPUThreadContext>,
    pub barrier: VGPUBarrier,
}

impl VGPUBlockContext {
    /// Sincroniza todos los threads simulados del bloque.
    pub fn sync(&mut self) {
        for _t in 0..self.threads.len() {
            self.barrier.arrive();
        }
    }
}
