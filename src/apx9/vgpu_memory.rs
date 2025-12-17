// APX 9.13 — Virtual GPU Memory Model
// Simulación segura: sin acceso real a VRAM ni CUDA.

/// Memoria global simulada: arreglo continuo de f32.
#[derive(Debug)]
pub struct VGlobalMemory {
    pub data: Vec<f32>,
}

impl VGlobalMemory {
    pub fn new(size: usize) -> Self {
        Self { data: vec![0.0; size] }
    }
}

/// Shared memory por bloque.
#[derive(Debug, Clone)]
pub struct VSharedMemory {
    pub data: Vec<f32>,
}

impl VSharedMemory {
    pub fn new(size: usize) -> Self {
        Self { data: vec![0.0; size] }
    }
}

/// Local memory por thread (como registers extendidos).
#[derive(Debug, Clone)]
pub struct VLocalMemory {
    pub data: Vec<f32>,
}

impl VLocalMemory {
    pub fn new(size: usize) -> Self {
        Self { data: vec![0.0; size] }
    }
}

/// Estado completo de memoria para un kernel.
pub struct VGpuMemory {
    pub global: VGlobalMemory,
    pub shared_per_block: Vec<VSharedMemory>,
    pub locals_per_thread: Vec<VLocalMemory>,
}

impl VGpuMemory {
    pub fn new(global_size: usize, shared_size: usize, blocks: usize, threads: usize) -> Self {
        Self {
            global: VGlobalMemory::new(global_size),
            shared_per_block: (0..blocks).map(|_| VSharedMemory::new(shared_size)).collect(),
            locals_per_thread: (0..(blocks * threads))
                .map(|_| VLocalMemory::new(32))
                .collect(),
        }
    }

    /// Acceso seguro a memoria global.
    pub fn load_global(&self, idx: usize) -> f32 {
        self.global.data[idx]
    }

    pub fn store_global(&mut self, idx: usize, val: f32) {
        self.global.data[idx] = val;
    }

    /// Alias de conveniencia para operaciones tipo HMMA en memoria global plana.
    pub fn load_f32(&self, idx: usize) -> f32 {
        self.load_global(idx)
    }

    pub fn store_f32(&mut self, idx: usize, val: f32) {
        self.store_global(idx, val);
    }

    /// Shared memory por bloque.
    pub fn load_shared(&self, block: usize, idx: usize) -> f32 {
        self.shared_per_block[block].data[idx]
    }

    pub fn store_shared(&mut self, block: usize, idx: usize, val: f32) {
        self.shared_per_block[block].data[idx] = val;
    }

    /// Local memory por thread.
    pub fn load_local(&self, thread: usize, idx: usize) -> f32 {
        self.locals_per_thread[thread].data[idx]
    }

    pub fn store_local(&mut self, thread: usize, idx: usize, val: f32) {
        self.locals_per_thread[thread].data[idx] = val;
    }
}
