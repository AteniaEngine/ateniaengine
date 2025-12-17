// APX 8.4 — GPU Mirroring Layer
// No ejecuta GPU real ni mueve datos. GPU es sólo un cache opcional.
// CPU sigue siendo la fuente de verdad.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MirrorState {
    Synced,      // CPU = GPU
    CleanCPU,    // GPU está vacío o desinicializado, CPU tiene la verdad
    DirtyCPU,    // GPU está limpio, CPU cambió
    CleanGPU,    // CPU está limpio, GPU cambió
    DirtyGPU,    // CPU antiguo, GPU tiene cambios
    None,        // No existe buffer GPU todavía
}

#[derive(Clone, Debug, PartialEq)]
pub struct GPUMirror {
    // Representamos el puntero de dispositivo sólo como tamaño en bytes; en 8.4
    // no necesitamos un puntero real y evitamos problemas de Send/Sync.
    pub bytes: usize,
    pub state: MirrorState,
}

impl GPUMirror {
    pub fn new_empty() -> Self {
        GPUMirror {
            bytes: 0,
            state: MirrorState::None,
        }
    }

    pub fn allocate(&mut self, bytes: usize) {
        // Stub: no reservamos memoria real en GPU todavía.
        self.bytes = bytes;
        if self.state == MirrorState::None {
            self.state = MirrorState::CleanCPU;
        }
    }

    pub fn upload_from_cpu(&mut self, _cpu_ptr: *const f32, bytes: usize) {
        // Stub seguro: asumimos que los datos de CPU se copiarían a GPU.
        self.bytes = bytes;
        if self.state == MirrorState::None {
            self.state = MirrorState::CleanCPU;
        } else {
            self.state = MirrorState::Synced;
        }
    }

    pub fn download_to_cpu(&self, _cpu_ptr: *mut f32, _bytes: usize) {
        // Stub seguro: en 8.4 no tocamos realmente los datos.
        // Se mantiene la semántica de que CPU es la verdad.
    }

    pub fn mark_dirty_cpu(&mut self) {
        match self.state {
            MirrorState::None => {
                // Sin buffer GPU, nada que marcar.
                self.state = MirrorState::None;
            }
            _ => {
                self.state = MirrorState::DirtyCPU;
            }
        }
    }

    pub fn mark_dirty_gpu(&mut self) {
        match self.state {
            MirrorState::None => {
                // Creamos el concepto de buffer GPU pero aún sin datos fiables.
                self.state = MirrorState::DirtyGPU;
            }
            _ => {
                self.state = MirrorState::DirtyGPU;
            }
        }
    }

    pub fn mark_synced(&mut self) {
        if self.state != MirrorState::None {
            self.state = MirrorState::Synced;
        }
    }
}
