pub struct GpuMemoryPool {
    pub blocks: Vec<*mut std::ffi::c_void>,
    pub block_size: usize,
}

impl GpuMemoryPool {
    pub fn new(block_size: usize, initial_blocks: usize) -> Self {
        unsafe {
            let mut blocks = Vec::with_capacity(initial_blocks);
            for _ in 0..initial_blocks {
                let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
                cuda_malloc(&mut ptr, block_size);
                blocks.push(ptr);
            }
            Self { blocks, block_size }
        }
    }

    pub fn alloc(&mut self) -> *mut std::ffi::c_void {
        if let Some(ptr) = self.blocks.pop() {
            ptr
        } else {
            unsafe {
                let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
                cuda_malloc(&mut ptr, self.block_size);
                ptr
            }
        }
    }

    pub fn free(&mut self, ptr: *mut std::ffi::c_void) {
        self.blocks.push(ptr);
    }

    /// Generic kernel variant: try to serve any size less than or equal to the
    /// block size from the pool.
    ///
    /// The request is rounded up to a full block (the pool does not
    /// sub-allocate), so any request larger than `block_size` cannot
    /// be served and would silently return a too-small buffer. The
    /// assertion catches that case early rather than letting the
    /// caller overwrite VRAM past the end of the block.
    pub unsafe fn alloc_device(&mut self, bytes: usize) -> *mut u8 {
        assert!(
            bytes <= self.block_size,
            "pool alloc_device request of {} bytes exceeds block size {}; \
             the pool does not sub-allocate, so larger requests cannot be served",
            bytes,
            self.block_size
        );
        self.alloc() as *mut u8
    }

    pub unsafe fn free_device(&mut self, ptr: *mut u8, _bytes: usize) {
        self.free(ptr as *mut std::ffi::c_void);
    }

    /// Simple heuristic: we can serve up to block_size bytes per single allocation.
    pub fn can_alloc(&self, bytes: usize) -> bool {
        bytes <= self.block_size
    }
}

#[cfg(atenia_cuda)]
#[allow(dead_code)]
unsafe extern "C" {
    fn cuda_malloc(ptr: *mut *mut std::ffi::c_void, bytes: usize);
    fn cuda_free(ptr: *mut std::ffi::c_void);
}

// **CPU-2 C2a** — CUDA-less build. The GPU memory pool is only
// constructed lazily via `apx4_12::get_pool()` on the GPU path, so
// these are unreachable in a CPU-only build; the identical-signature
// stubs exist purely so `GpuMemoryPool`'s methods still type-check
// without a CUDA symbol to link.
#[cfg(not(atenia_cuda))]
#[allow(dead_code, unused_variables)]
unsafe fn cuda_malloc(ptr: *mut *mut std::ffi::c_void, bytes: usize) {
    unreachable!(
        "CUDA symbol cuda_malloc called in CPU-only build (atenia_cuda not enabled)"
    )
}

#[cfg(not(atenia_cuda))]
#[allow(dead_code, unused_variables)]
unsafe fn cuda_free(ptr: *mut std::ffi::c_void) {
    unreachable!(
        "CUDA symbol cuda_free called in CPU-only build (atenia_cuda not enabled)"
    )
}

// Safety: GpuMemoryPool only stores opaque device pointers and is always
// accessed behind a Mutex in a global OnceLock, so we treat it as Send.
unsafe impl Send for GpuMemoryPool {}
