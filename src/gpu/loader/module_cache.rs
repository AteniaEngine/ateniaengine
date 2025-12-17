use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

// Para evitar requerir que CudaModule (que envuelve un puntero CUmodule) sea Send/Sync,
// el cache global sólo almacena el valor numérico del handle como u64.

fn module_cache() -> &'static Mutex<HashMap<u64, u64>> {
    static MODULE_CACHE: OnceLock<Mutex<HashMap<u64, u64>>> = OnceLock::new();
    MODULE_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub fn get_cached_module(hash: u64) -> Option<u64> {
    module_cache().lock().unwrap().get(&hash).cloned()
}

pub fn insert_cached_module(hash: u64, handle: u64) {
    module_cache().lock().unwrap().insert(hash, handle);
}

pub fn hash_ptx(ptx: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ptx.hash(&mut h);
    h.finish()
}
