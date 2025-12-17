//! APX 12.2.5  GPU Loader Compatibility Layer
//! Esta capa intenta TODAS las rutas posibles para garantizar que
//! siempre se puede cargar un kernel CUDA, incluso si NVRTC, nvJitLink
//! o el driver fallan.

use crate::gpu::loader::{CudaLoader, CudaModule, CudaLoaderError};
use crate::gpu::linker::NvJitLinker;
use std::sync::atomic::{AtomicBool, Ordering};

static FORCED_FALLBACK: AtomicBool = AtomicBool::new(false);

pub struct CompatLoader;

impl CompatLoader {
    pub fn is_forced_fallback() -> bool {
        FORCED_FALLBACK.load(Ordering::Relaxed)
    }

    fn mark_fallback() {
        FORCED_FALLBACK.store(true, Ordering::Relaxed);
    }

    pub fn try_all_paths(loader: &CudaLoader, ptx: &str) -> Result<CudaModule, CudaLoaderError> {

        // 1. Try nvJitLink (PTX -> CUBIN)
        if let Ok(linker) = NvJitLinker::new() {
            if let Ok(cubin) = linker.link_ptx_to_cubin(ptx) {
                if let Ok(module) = loader.load_module_from_cubin(&cubin) {
                    println!("[COMPAT] Loaded via nvJitLink CUBIN");
                    return Ok(module);
                }
            }
        }

        // 2. Try PTX direct
        if let Ok(module) = loader.load_module_from_ptx_direct(ptx) {
            println!("[COMPAT] Loaded via PTX direct");
            return Ok(module);
        }

        // 3. Try normalized PTX
        let normalized = ptx.replace(".target sm_89", ".target compute_89");
        if let Ok(module) = loader.load_module_from_ptx_direct(&normalized) {
            println!("[COMPAT] Loaded via normalized PTX");
            return Ok(module);
        }

        // 4. Try downgraded version (8.5  8.0)
        let downgraded = ptx.replace(".version 8.5", ".version 8.0");
        if let Ok(module) = loader.load_module_from_ptx_direct(&downgraded) {
            println!("[COMPAT] Loaded via downgraded PTX");
            return Ok(module);
        }

        // 5. FINAL FALLBACK → CPU fallback (no GPU necesaria)
        println!("[COMPAT] GPU failing → using CPU fallback module");
        CompatLoader::mark_fallback();
        Err(CudaLoaderError::CpuFallback)
    }
}
