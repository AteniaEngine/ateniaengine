// APX 8.4 — GPU Mirroring Layer
// Does not execute real GPU nor move data. GPU is only an optional cache.
// CPU remains the source of truth.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MirrorState {
    Synced,      // CPU = GPU
    CleanCPU,    // GPU is empty or uninitialized; CPU has the truth
    DirtyCPU,    // GPU is clean; CPU changed
    CleanGPU,    // CPU is clean; GPU changed
    DirtyGPU,    // CPU is stale; GPU has changes
    None,        // GPU buffer does not exist yet
}

#[derive(Clone, Debug, PartialEq)]
pub struct GPUMirror {
    // We represent the device pointer only as size in bytes; in 8.4 we do not
    // need a real pointer and avoid Send/Sync issues.
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
        // Stub: we do not allocate real GPU memory yet.
        self.bytes = bytes;
        if self.state == MirrorState::None {
            self.state = MirrorState::CleanCPU;
        }
    }

    pub fn upload_from_cpu(&mut self, _cpu_ptr: *const f32, bytes: usize) {
        // Safe stub: assume CPU data would be copied to GPU.
        self.bytes = bytes;
        if self.state == MirrorState::None {
            self.state = MirrorState::CleanCPU;
        } else {
            self.state = MirrorState::Synced;
        }
    }

    pub fn download_to_cpu(&self, _cpu_ptr: *mut f32, _bytes: usize) {
        // Safe stub: in 8.4 we do not touch real data.
        // Keep the semantics that CPU is the truth.
    }

    pub fn mark_dirty_cpu(&mut self) {
        match self.state {
            MirrorState::None => {
                // No GPU buffer: nothing to mark.
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
                // Create the concept of a GPU buffer, but still without reliable data.
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
