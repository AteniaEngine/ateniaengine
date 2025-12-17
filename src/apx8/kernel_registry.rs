// APX 8.7 — GPU Kernel Registry v1
// Registry seguro y extensible para mini-kernels GPU (simulados o reales).
// No ejecuta GPU real en 8.7; sólo registra funciones y permite consultarlas.

use std::collections::HashMap;
use std::sync::RwLock;

use once_cell::sync::Lazy;

use crate::tensor::Tensor;
use crate::apx8::kernel_generator::{GpuKernelOp, GpuKernelTemplate};
use crate::apx9::gpu_codegen_real::RealKernel;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum KernelKey {
    VecAdd,
    MatMulSmall,
    MatMulLarge,
    Custom(&'static str),
}

/// Firma uniforme de los kernels GPU v0.
pub type KernelFn = fn(&mut Tensor, &Tensor);

#[derive(Clone)]
pub struct RegisteredKernel {
    pub func_cpu: Option<KernelFn>,
    pub func_gpu_stub: Option<String>,
}

pub struct KernelRegistry {
    pub map: RwLock<HashMap<KernelKey, KernelFn>>,
    pub templates: RwLock<HashMap<GpuKernelOp, GpuKernelTemplate>>,
    pub compiled: RwLock<HashMap<KernelKey, RegisteredKernel>>,
}

pub static KERNEL_REGISTRY: Lazy<KernelRegistry> = Lazy::new(|| KernelRegistry {
    map: RwLock::new(HashMap::new()),
    templates: RwLock::new(HashMap::new()),
    compiled: RwLock::new(HashMap::new()),
});

impl KernelRegistry {
    pub fn register(&self, key: KernelKey, func: KernelFn) {
        let mut map = self.map.write().unwrap();
        map.insert(key, func);
    }

    pub fn get(&self, key: &KernelKey) -> Option<KernelFn> {
        let map = self.map.read().unwrap();
        map.get(key).cloned()
    }

    // APX 8.9: soporte opcional para registrar plantillas de kernels GPU.
    pub fn register_template(&self, op: GpuKernelOp, tpl: GpuKernelTemplate) {
        let mut tpls = self.templates.write().unwrap();
        tpls.insert(op, tpl);
    }

    pub fn get_template(&self, op: &GpuKernelOp) -> Option<GpuKernelTemplate> {
        let tpls = self.templates.read().unwrap();
        tpls.get(op).cloned()
    }

    /// APX 8.11: registrar/metaconsultar un stub de kernel GPU compilado.
    pub fn set_gpu_stub(&self, key: KernelKey, stub: String) {
        let mut map = self.compiled.write().unwrap();
        let entry = map.entry(key).or_insert(RegisteredKernel {
            func_cpu: None,
            func_gpu_stub: None,
        });
        entry.func_gpu_stub = Some(stub);
    }

    pub fn get_registered(&self, key: &KernelKey) -> Option<RegisteredKernel> {
        let map = self.compiled.read().unwrap();
        map.get(key).cloned()
    }
}

/// APX 9.10: registrar un kernel "real" (PTX/OpenCL) como stub GPU en el registry.
/// Esto sólo almacena el string de código; no ejecuta ni compila nada.
pub fn register_real_kernel(kernel: RealKernel) {
    // Usamos una clave Custom basada en el nombre del kernel. Como requiere
    // &'static str, fugamos el String de forma controlada. Es aceptable para
    // un registro global de larga vida.
    let name_static: &'static str = Box::leak(kernel.signature.name.clone().into_boxed_str());
    let key = KernelKey::Custom(name_static);

    let mut compiled = KERNEL_REGISTRY.compiled.write().unwrap();
    let entry = compiled.entry(key).or_insert(RegisteredKernel {
        func_cpu: None,
        func_gpu_stub: None,
    });
    entry.func_gpu_stub = Some(kernel.code.clone());
}
