//! **M12.3** — consolidated environment / hardware diagnostics.
//!
//! Emitted once per CLI entry point (`atenia generate` /
//! `atenia run`), right after argument validation and before the
//! model load. This is *read-and-echo only*: no environment
//! variable is interpreted differently, no default is changed,
//! no tiering / numeric behaviour is affected. The scope is the
//! M12.3 variable set; other `ATENIA_*` variables are
//! intentionally not surfaced here to avoid output spam.
//!
//! The renderer is a pure function (testable without touching the
//! environment or a GPU); the logger builds the snapshot from the
//! live process state and prints it, suppressed under
//! [`crate::apx_is_silent`] like the other `[ATENIA]` operator
//! logs.

use crate::gpu::safety::resource_check::{
    probe_free_ram_bytes, probe_free_vram_bytes_detailed, probe_total_ram_bytes,
};

/// Immutable capture of the diagnostics-relevant process state.
#[derive(Debug, Clone)]
pub struct EnvDiagSnapshot {
    pub apx_mode: String,
    pub apx_mode_from_env: bool,
    /// Raw `ATENIA_DEBUG` (or `APX_DEBUG`) value, if either is set.
    pub debug_raw: Option<String>,
    /// Whether `debug_raw` is a recognised on/off token.
    pub debug_recognized: bool,
    pub m8_bf16_kernel: bool,
    pub m8_bf16_kernel_set: bool,
    pub gpu_residency: bool,
    pub gpu_residency_set: bool,
    pub force_cuda: bool,
    /// Free VRAM bytes, or the probe-failure reason.
    pub free_vram: Result<u64, String>,
    pub free_ram_bytes: u64,
    pub total_ram_bytes: u64,
    pub cpu_threads: usize,
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// `1` / `0` / `true` / `false` (any case) or empty are
/// recognised on/off tokens for `ATENIA_DEBUG`. Anything else is
/// a malformed value that silently behaves as "off" — M12.3
/// surfaces it as a one-line WARN.
fn debug_value_recognized(v: &str) -> bool {
    matches!(
        v.trim().to_lowercase().as_str(),
        "1" | "0" | "true" | "false" | ""
    )
}

/// Pure renderer — no env / GPU access. Each returned line is
/// already `[ATENIA]`-prefixed.
pub fn render_env_diagnostics(s: &EnvDiagSnapshot) -> Vec<String> {
    let mut out = Vec::new();
    out.push("[ATENIA] env/hardware diagnostics:".to_string());
    out.push(format!(
        "[ATENIA]   apx_mode = {} ({})",
        s.apx_mode,
        if s.apx_mode_from_env {
            "from ATENIA_APX_MODE"
        } else {
            "default"
        }
    ));
    out.push(format!(
        "[ATENIA]   ATENIA_M8_BF16_KERNEL = {} ({})",
        if s.m8_bf16_kernel { "on" } else { "off" },
        if s.m8_bf16_kernel_set { "set" } else { "default" }
    ));
    out.push(format!(
        "[ATENIA]   ATENIA_GPU_RESIDENCY = {} ({}; tier-aware loader supersedes it)",
        if s.gpu_residency { "on" } else { "off" },
        if s.gpu_residency_set { "set" } else { "default" }
    ));
    out.push(format!(
        "[ATENIA]   ATENIA_FORCE_CUDA = {}",
        if s.force_cuda {
            "set (CUDA detection forced true)"
        } else {
            "unset"
        }
    ));
    match &s.free_vram {
        Ok(b) => out.push(format!("[ATENIA]   VRAM free = {:.2} GiB", gib(*b))),
        Err(e) => out.push(format!(
            "[ATENIA]   VRAM free = unavailable (probe failed: {e}); weights → RAM/Disk"
        )),
    }
    out.push(format!(
        "[ATENIA]   RAM free/total = {:.2} / {:.2} GiB",
        gib(s.free_ram_bytes),
        gib(s.total_ram_bytes)
    ));
    out.push(format!(
        "[ATENIA]   CPU = {} threads, AVX2={} AVX512={} FMA={}",
        s.cpu_threads, s.avx2, s.avx512f, s.fma
    ));
    if let Some(raw) = &s.debug_raw {
        if !s.debug_recognized {
            out.push(format!(
                "[ATENIA] WARN: ATENIA_DEBUG/APX_DEBUG={raw:?} not recognized \
                 (use 1/0/true/false); treating as off"
            ));
        }
    }
    out
}

/// Build the snapshot from the live environment + probes and emit
/// it once. Suppressed under [`crate::apx_is_silent`]. Called by
/// the `atenia generate` / `atenia run` entry points after arg
/// validation and before the load.
pub fn log_env_and_hardware_diagnostics() {
    if crate::apx_is_silent() {
        return;
    }
    let debug_raw = std::env::var("ATENIA_DEBUG")
        .ok()
        .or_else(|| std::env::var("APX_DEBUG").ok());
    let debug_recognized = debug_raw
        .as_deref()
        .map(debug_value_recognized)
        .unwrap_or(true);
    let feats = crate::cpu_features::cpu_features();
    let snap = EnvDiagSnapshot {
        apx_mode: crate::apx_mode(),
        apx_mode_from_env: std::env::var("ATENIA_APX_MODE").is_ok(),
        debug_raw,
        debug_recognized,
        m8_bf16_kernel: std::env::var("ATENIA_M8_BF16_KERNEL").as_deref() == Ok("1"),
        m8_bf16_kernel_set: std::env::var("ATENIA_M8_BF16_KERNEL").is_ok(),
        gpu_residency: std::env::var("ATENIA_GPU_RESIDENCY").as_deref() == Ok("1"),
        gpu_residency_set: std::env::var("ATENIA_GPU_RESIDENCY").is_ok(),
        force_cuda: std::env::var_os("ATENIA_FORCE_CUDA").is_some(),
        free_vram: probe_free_vram_bytes_detailed().map_err(|e| e.to_string()),
        free_ram_bytes: probe_free_ram_bytes(),
        total_ram_bytes: probe_total_ram_bytes(),
        cpu_threads: feats.threads,
        avx2: feats.avx2,
        avx512f: feats.avx512f,
        fma: feats.fma,
    };
    for line in render_env_diagnostics(&snap) {
        eprintln!("{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> EnvDiagSnapshot {
        EnvDiagSnapshot {
            apx_mode: "7.2".into(),
            apx_mode_from_env: false,
            debug_raw: None,
            debug_recognized: true,
            m8_bf16_kernel: false,
            m8_bf16_kernel_set: false,
            gpu_residency: false,
            gpu_residency_set: false,
            force_cuda: false,
            free_vram: Ok(8 * 1024 * 1024 * 1024),
            free_ram_bytes: 16 * 1024 * 1024 * 1024,
            total_ram_bytes: 32 * 1024 * 1024 * 1024,
            cpu_threads: 24,
            avx2: true,
            avx512f: false,
            fma: true,
        }
    }

    fn joined(s: &EnvDiagSnapshot) -> String {
        render_env_diagnostics(s).join("\n")
    }

    #[test]
    fn every_line_is_atenia_prefixed_and_default_is_marked() {
        let lines = render_env_diagnostics(&base());
        assert!(lines.iter().all(|l| l.starts_with("[ATENIA]")));
        let j = lines.join("\n");
        assert!(j.contains("apx_mode = 7.2 (default)"), "{j}");
        assert!(j.contains("CPU = 24 threads, AVX2=true AVX512=false FMA=true"));
        assert!(j.contains("VRAM free = 8.00 GiB"));
        assert!(j.contains("RAM free/total = 16.00 / 32.00 GiB"));
    }

    #[test]
    fn apx_mode_from_env_is_marked() {
        let mut s = base();
        s.apx_mode = "4.19".into();
        s.apx_mode_from_env = true;
        assert!(joined(&s).contains("apx_mode = 4.19 (from ATENIA_APX_MODE)"));
    }

    #[test]
    fn malformed_debug_emits_warn_recognized_does_not() {
        let mut s = base();
        s.debug_raw = Some("yes".into());
        s.debug_recognized = false;
        let j = joined(&s);
        assert!(j.contains("WARN: ATENIA_DEBUG/APX_DEBUG=\"yes\" not recognized"), "{j}");

        let mut ok = base();
        ok.debug_raw = Some("true".into());
        ok.debug_recognized = true;
        assert!(!joined(&ok).contains("WARN"));
    }

    #[test]
    fn vram_probe_failure_is_surfaced() {
        let mut s = base();
        s.free_vram = Err("nvidia-smi not found".into());
        let j = joined(&s);
        assert!(j.contains("VRAM free = unavailable (probe failed: nvidia-smi not found)"), "{j}");
        assert!(j.contains("weights → RAM/Disk"));
    }

    #[test]
    fn force_cuda_and_flags_are_surfaced() {
        let mut s = base();
        s.force_cuda = true;
        s.m8_bf16_kernel = true;
        s.m8_bf16_kernel_set = true;
        s.gpu_residency = true;
        s.gpu_residency_set = true;
        let j = joined(&s);
        assert!(j.contains("ATENIA_FORCE_CUDA = set (CUDA detection forced true)"));
        assert!(j.contains("ATENIA_M8_BF16_KERNEL = on (set)"));
        assert!(j.contains("ATENIA_GPU_RESIDENCY = on (set"));
    }

    #[test]
    fn debug_value_recognition() {
        for v in ["1", "0", "true", "FALSE", " True ", ""] {
            assert!(debug_value_recognized(v), "{v:?} should be recognized");
        }
        for v in ["yes", "on", "2", "enabled"] {
            assert!(!debug_value_recognized(v), "{v:?} should NOT be recognized");
        }
    }
}
