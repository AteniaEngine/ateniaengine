use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

pub type KernelFn = Arc<dyn Fn(&[f32], &[f32], &mut [f32], usize, usize, usize) + Send + Sync>;

#[derive(Default)]
pub struct KernelRegistry {
    kernels: RwLock<HashMap<&'static str, KernelFn>>,
}

impl KernelRegistry {
    pub fn global() -> &'static KernelRegistry {
        static REGISTRY: OnceLock<KernelRegistry> = OnceLock::new();
        REGISTRY.get_or_init(KernelRegistry::default)
    }

    pub fn register(&self, name: &'static str, f: KernelFn) {
        let mut map = self.kernels.write().unwrap();
        map.insert(name, f);
    }

    pub fn get(&self, name: &'static str) -> Option<KernelFn> {
        let map = self.kernels.read().unwrap();
        map.get(name).cloned()
    }
}
