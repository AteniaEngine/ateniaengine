// APX 8.8 — GPU Kernel Signatures v0
// Infraestructura de firmas para kernels GPU reales.
// No ejecuta GPU, no modifica backward ni resultados numéricos.

use std::collections::HashMap;
use std::sync::RwLock;

use once_cell::sync::Lazy;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GpuKernelType {
    VecAdd,
    MatMul,
    MatMulTiled32,
    LayerNorm,
    Custom(&'static str),
}

#[derive(Debug, Clone)]
pub struct GpuKernelSignature {
    pub key: GpuKernelType,
    pub min_dims: (usize, usize, usize),
    pub max_dims: (usize, usize, usize),
    pub workspace_bytes: usize,
    pub launcher_name: &'static str,
}

pub static GPU_KERNEL_SIGNATURES: Lazy<RwLock<HashMap<GpuKernelType, GpuKernelSignature>>> =
    Lazy::new(|| RwLock::new(HashMap::new()));

pub fn register_signature(sig: GpuKernelSignature) {
    let mut map = GPU_KERNEL_SIGNATURES.write().unwrap();
    map.insert(sig.key.clone(), sig);
}

pub fn get_signature(key: &GpuKernelType) -> Option<GpuKernelSignature> {
    let map = GPU_KERNEL_SIGNATURES.read().unwrap();
    map.get(key).cloned()
}

/// Stub de debug para integración con dispatcher; no hace nada en 8.8.
pub fn record_kernel_signature(_sig: &GpuKernelSignature) {
    // En futuras versiones se podría loguear o acumular estadísticas.
}
