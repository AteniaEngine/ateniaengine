// APX 8.15 — GPU Pre-Compilation Cache v0
// Cache sintético de kernels "precompilados" basado en KernelIR.

use std::collections::HashMap;
use crate::apx8::kernel_generator::KernelIR;

#[derive(Debug, Default)]
pub struct PrecompileCache {
    store: HashMap<String, String>, // key: signature, value: compiled_kernel (string)
}

impl PrecompileCache {
    pub fn new() -> Self {
        Self { store: HashMap::new() }
    }

    pub fn compile_if_missing(&mut self, ir: &KernelIR) -> String {
        let key = ir.signature();
        if let Some(k) = self.store.get(&key) {
            return k.clone();
        }

        // Simulación determinística
        let compiled = format!("compiled::<{}>", key);
        self.store.insert(key.clone(), compiled.clone());
        compiled
    }

    pub fn contains(&self, ir: &KernelIR) -> bool {
        self.store.contains_key(&ir.signature())
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }
}
