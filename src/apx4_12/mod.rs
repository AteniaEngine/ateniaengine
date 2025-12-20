use std::sync::{Mutex, OnceLock};

mod gpu_memory_pool;
pub mod pool_dispatcher;
use self::gpu_memory_pool::GpuMemoryPool;

pub static GPU_MEMORY_POOL: OnceLock<Mutex<GpuMemoryPool>> = OnceLock::new();

// Default configuration used for lazy initialization.
const DEFAULT_BLOCK_SIZE: usize = 64 * 1024 * 1024; // 64 MB
const DEFAULT_BLOCKS: usize = 8;

fn get_pool() -> &'static Mutex<GpuMemoryPool> {
    GPU_MEMORY_POOL.get_or_init(|| {
        Mutex::new(GpuMemoryPool::new(DEFAULT_BLOCK_SIZE, DEFAULT_BLOCKS))
    })
}

pub fn init_pool(block_size: usize, blocks: usize) {
    // Explicit initialization: if someone already initialized it lazily, this
    // set will fail silently and we will keep the existing instance.
    let _ = GPU_MEMORY_POOL.set(Mutex::new(GpuMemoryPool::new(block_size, blocks)));
}

pub fn pool_alloc() -> *mut std::ffi::c_void {
    get_pool().lock().unwrap().alloc()
}

pub fn pool_free(ptr: *mut std::ffi::c_void) {
    // Even if it was not initialized, get_pool() will create it with the
    // default configuration before freeing the pointer.
    get_pool().lock().unwrap().free(ptr);
}

/// Check whether the pool can serve at least `bytes` in a single allocation
/// (heuristic based on block size).
pub fn pool_can_alloc(bytes: usize) -> bool {
    get_pool().lock().unwrap().can_alloc(bytes)
}

/// Global C API so CUDA kernels can request memory from the pool.
#[unsafe(no_mangle)]
pub extern "C" fn atenia_pool_alloc(bytes: usize) -> *mut u8 {
    let pool = get_pool();
    let mut guard = pool.lock().unwrap();
    unsafe { guard.alloc_device(bytes) }
}

#[unsafe(no_mangle)]
pub extern "C" fn atenia_pool_free(ptr: *mut u8, bytes: usize) {
    let pool = get_pool();
    let mut guard = pool.lock().unwrap();
    unsafe { guard.free_device(ptr, bytes) }
}
