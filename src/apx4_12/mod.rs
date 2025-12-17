use std::sync::{Mutex, OnceLock};

mod gpu_memory_pool;
pub mod pool_dispatcher;
use self::gpu_memory_pool::GpuMemoryPool;

pub static GPU_MEMORY_POOL: OnceLock<Mutex<GpuMemoryPool>> = OnceLock::new();

// Configuración por defecto usada para inicialización perezosa.
const DEFAULT_BLOCK_SIZE: usize = 64 * 1024 * 1024; // 64 MB
const DEFAULT_BLOCKS: usize = 8;

fn get_pool() -> &'static Mutex<GpuMemoryPool> {
    GPU_MEMORY_POOL.get_or_init(|| {
        Mutex::new(GpuMemoryPool::new(DEFAULT_BLOCK_SIZE, DEFAULT_BLOCKS))
    })
}

pub fn init_pool(block_size: usize, blocks: usize) {
    // Inicialización explícita: si alguien ya lo inicializó perezosamente,
    // este set fallará en silencio y conservaremos la instancia existente.
    let _ = GPU_MEMORY_POOL.set(Mutex::new(GpuMemoryPool::new(block_size, blocks)));
}

pub fn pool_alloc() -> *mut std::ffi::c_void {
    get_pool().lock().unwrap().alloc()
}

pub fn pool_free(ptr: *mut std::ffi::c_void) {
    // Incluso si no estaba inicializado, get_pool() lo creará con la
    // configuración por defecto antes de liberar el puntero.
    get_pool().lock().unwrap().free(ptr);
}

/// Consulta si el pool puede servir al menos `bytes` en una asignación
/// individual (heurística basada en el tamaño de bloque).
pub fn pool_can_alloc(bytes: usize) -> bool {
    get_pool().lock().unwrap().can_alloc(bytes)
}

/// API C global para que los kernels CUDA pidan memoria al pool.
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
