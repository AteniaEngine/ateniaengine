//! APX 12.2.5  GPU Loader Compatibility Layer
//! Esta capa intenta TODAS las rutas posibles para garantizar que
//! siempre se puede cargar un kernel CUDA, incluso si NVRTC, nvJitLink
//! o el driver fallan.

use crate::gpu::linker::NvJitLinker;
use crate::gpu::loader::{CudaLoader, CudaLoaderError, CudaModule};
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
        // **M12.4 H3** — every attempt below used to swallow its
        // `Err` arm, so a `CpuFallback` return told the operator
        // *that* the GPU path failed but never *why*. Accumulate the
        // per-attempt reasons; the success path stays unchanged (only
        // the existing single `[COMPAT] Loaded via …` line) and the
        // reasons are flushed once on the terminal CPU fallback so a
        // working early path is exactly as quiet as before.
        let mut reasons: Vec<String> = Vec::new();

        // 1. Try nvJitLink (PTX -> CUBIN). `NvJitLinkError` is
        //    `Debug`-only, so its reasons use `{:?}`.
        match NvJitLinker::new() {
            Ok(linker) => match linker.link_ptx_to_cubin(ptx) {
                Ok(cubin) => match loader.load_module_from_cubin(&cubin) {
                    Ok(module) => {
                        println!("[COMPAT] Loaded via nvJitLink CUBIN");
                        return Ok(module);
                    }
                    Err(e) => reasons.push(format!("nvJitLink CUBIN module-load failed: {e}")),
                },
                Err(e) => reasons.push(format!("nvJitLink PTX->CUBIN link failed: {e:?}")),
            },
            Err(e) => reasons.push(format!("nvJitLink unavailable: {e:?}")),
        }

        // 2. Try PTX direct
        match loader.load_module_from_ptx_direct(ptx) {
            Ok(module) => {
                println!("[COMPAT] Loaded via PTX direct");
                return Ok(module);
            }
            Err(e) => reasons.push(format!("PTX direct load failed: {e}")),
        }

        // 3. Try normalized PTX
        let normalized = ptx.replace(".target sm_89", ".target compute_89");
        match loader.load_module_from_ptx_direct(&normalized) {
            Ok(module) => {
                println!("[COMPAT] Loaded via normalized PTX");
                return Ok(module);
            }
            Err(e) => reasons.push(format!("normalized PTX load failed: {e}")),
        }

        // 4. Try downgraded version (8.5  8.0)
        let downgraded = ptx.replace(".version 8.5", ".version 8.0");
        match loader.load_module_from_ptx_direct(&downgraded) {
            Ok(module) => {
                println!("[COMPAT] Loaded via downgraded PTX");
                return Ok(module);
            }
            Err(e) => reasons.push(format!("downgraded PTX load failed: {e}")),
        }

        // Terminal CPU fallback ahead. Flush the accumulated reasons
        // to stderr (stdout is reserved for generation output)
        // before the existing control-flow, which stays byte-
        // identical to pre-M12.4.
        for line in render_compat_fallback_reasons(&reasons) {
            eprintln!("{line}");
        }

        // 5. FINAL FALLBACK → CPU fallback (no GPU necesaria)
        println!("[COMPAT] GPU failing → using CPU fallback module");
        CompatLoader::mark_fallback();
        Err(CudaLoaderError::CpuFallback)
    }
}

/// **M12.4 H3** — format the accumulated GPU module-load failure
/// reasons into the lines emitted just before the terminal CPU
/// fallback. Pure (no I/O) so the diagnostic shape is unit-testable
/// without a live CUDA driver.
fn render_compat_fallback_reasons(reasons: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(reasons.len() + 1);
    out.push(
        "[COMPAT][warn] all GPU module-load paths failed; falling back to \
         CPU. Reasons (in attempt order):"
            .to_string(),
    );
    for (i, r) in reasons.iter().enumerate() {
        out.push(format!("[COMPAT][warn]   {}. {}", i + 1, r));
    }
    out
}

#[cfg(test)]
mod m12_4 {
    use super::render_compat_fallback_reasons;

    #[test]
    fn renders_header_plus_numbered_reasons() {
        let reasons = vec![
            "nvJitLink unavailable: LoadError(\"no dll\")".to_string(),
            "PTX direct load failed: cuModuleLoadData failed".to_string(),
        ];
        let lines = render_compat_fallback_reasons(&reasons);
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("[COMPAT][warn] all GPU module-load paths failed"));
        assert_eq!(
            lines[1],
            "[COMPAT][warn]   1. nvJitLink unavailable: LoadError(\"no dll\")"
        );
        assert_eq!(
            lines[2],
            "[COMPAT][warn]   2. PTX direct load failed: cuModuleLoadData failed"
        );
    }

    /// Defensive: an empty reason list (cannot happen in practice —
    /// at least one attempt must fail to reach the terminal
    /// fallback) still yields just the header.
    #[test]
    fn empty_reasons_yields_header_only() {
        let lines = render_compat_fallback_reasons(&[]);
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("falling back to"));
    }
}
