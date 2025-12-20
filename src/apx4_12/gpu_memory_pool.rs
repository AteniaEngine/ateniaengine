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
    pub unsafe fn alloc_device(&mut self, _bytes: usize) -> *mut u8 {
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

#[allow(dead_code)]
unsafe extern "C" {
    fn cuda_malloc(ptr: *mut *mut std::ffi::c_void, bytes: usize);
    fn cuda_free(ptr: *mut std::ffi::c_void);
}

// Safety: GpuMemoryPool only stores opaque device pointers and is always
// accessed behind a Mutex in a global OnceLock, so we treat it as Send.
unsafe impl Send for GpuMemoryPool {}
