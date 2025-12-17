// APX 8.11 — GPU Compiler Stub
// Simula compilación de kernels GPU sin tocar GPU real ni ejecutar nada.

use std::collections::HashMap;

use crate::apx8::kernel_generator::KernelIR;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GpuTarget {
    NvidiaPTX,
    NvidiaSASS,
    AMDHSACO,
    IntelLevelZero,
}

#[derive(Debug, Clone)]
pub struct CompiledKernelStub {
    pub ir_hash: u64,
    pub target: GpuTarget,
    pub binary_stub: String,
}

pub struct GpuCompilerStub {
    cache: HashMap<(u64, GpuTarget), CompiledKernelStub>,
}

impl GpuCompilerStub {
    pub fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    pub fn compile(&mut self, ir: &KernelIR, target: GpuTarget) -> CompiledKernelStub {
        let key = (ir.hash(), target.clone());

        if let Some(cached) = self.cache.get(&key) {
            return cached.clone();
        }

        let bin = CompiledKernelStub {
            ir_hash: ir.hash(),
            target: target.clone(),
            binary_stub: format!("/* STUB {:?} IR:{} */", target, ir.hash()),
        };

        self.cache.insert(key, bin.clone());
        bin
    }

    pub fn has_cache(&self, ir: &KernelIR, target: GpuTarget) -> bool {
        self.cache.contains_key(&(ir.hash(), target))
    }
}
