use std::os::raw::c_int;

pub mod batch_matmul;
pub mod bf16_to_f32;
pub mod disk_prefetch;
pub mod fused_linear_silu;
pub mod int8_to_bf16;
pub mod linear;
pub mod matmul;
pub(crate) mod pool_helpers;

#[cfg(atenia_cuda)]
#[link(name = "atenia_kernels", kind = "static")]
unsafe extern "C" {
    pub fn vec_add_cuda(a: *const f32, b: *const f32, out: *mut f32, n: c_int);
}

// **CPU-2 C2a** — CUDA-less build: there is no `atenia_kernels`
// static library to link. Identical-signature stub so call sites
// still type-check; unreachable because `vec_add_gpu` is only
// invoked on the GPU path. (`vec_add_gpu` itself is left for C2c
// per the CPU-2 plan.)
#[cfg(not(atenia_cuda))]
#[allow(unused_variables)]
pub unsafe fn vec_add_cuda(a: *const f32, b: *const f32, out: *mut f32, n: c_int) {
    unreachable!(
        "CUDA symbol vec_add_cuda called in CPU-only build (atenia_cuda not enabled)"
    )
}

/// **M6 step 1** — cached CUDA-driver probe.
///
/// Previously this function spawned `nvidia-smi` on every call. On
/// Windows the spawn + driver-enumeration cost is ~50–300 ms. With
/// the gate G5 (pool 64 MiB ceiling) blocking the 13B path today,
/// the spawn never happens on the Llama hot path — but any future
/// step that lifts G5 (or routes more traffic through
/// `gpu_can_run_matmul`) would call this per-matmul (~360
/// calls/decode-step), adding tens of seconds/token of pure
/// orchestration overhead. Caching is a strict prerequisite of any
/// further GPU-path activation.
///
/// The cache is `OnceLock<bool>` — first call spawns `nvidia-smi`
/// exactly once, subsequent calls read a single atomic bool. The
/// driver state can in principle change between calls (driver crash,
/// GPU eject) but losing CUDA dispatch mid-session was never a
/// behaviour the engine gracefully recovered from anyway; if the
/// driver dies, the kernel call panics regardless of what this
/// function returns. Caching is the right call.
///
/// Behaviour contract:
///   - First call: spawns `nvidia-smi`, returns `output().is_ok()`.
///   - Subsequent calls: return the cached value, no spawn.
///   - Result is identical to the pre-cache version on a stable
///     driver.
pub fn cuda_available() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::process::Command::new("nvidia-smi").output().is_ok())
}

#[cfg(test)]
mod cuda_available_tests {
    use super::cuda_available;

    /// Two consecutive calls must return the same value. With the
    /// `OnceLock` cache, the second call must not spawn a process —
    /// we assert behavioural equivalence (the operator can verify
    /// no-spawn manually with Process Explorer if needed).
    #[test]
    fn cuda_available_is_stable_across_calls() {
        let first = cuda_available();
        let second = cuda_available();
        assert_eq!(
            first, second,
            "cuda_available() must return the same value across calls"
        );
    }

    /// Sanity: the cached value matches a fresh `nvidia-smi` probe at
    /// test time. If the test host has CUDA, `cuda_available()`
    /// returns true; if not, false. We compare against a direct
    /// spawn so the test is self-validating on either kind of host.
    #[test]
    fn cuda_available_matches_direct_probe() {
        let direct = std::process::Command::new("nvidia-smi").output().is_ok();
        assert_eq!(
            cuda_available(),
            direct,
            "cuda_available() result must match a fresh nvidia-smi probe"
        );
    }
}

pub fn vec_add_gpu(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());

    let n = a.len() as c_int;
    let mut out = vec![0.0f32; a.len()];

    unsafe {
        vec_add_cuda(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n);
    }

    out
}

/// Returns the raw device pointer (read-only) backing a
/// `TensorStorage::Cuda`. Panics with `unreachable!` on `Cpu` storage:
/// callers must guard the invocation with a prior check that the
/// storage variant is `Cuda` (typically an `all_cuda` match at the op
/// entry point).
pub(crate) fn cuda_device_ptr(storage: &crate::tensor::TensorStorage) -> *const f32 {
    match storage {
        crate::tensor::TensorStorage::Cuda(g) => g.device_ptr() as *const f32,
        crate::tensor::TensorStorage::Cpu(_) => {
            unreachable!("cuda_device_ptr called on Cpu storage")
        }
        crate::tensor::TensorStorage::Disk(_) => {
            unreachable!(
                "cuda_device_ptr called on Disk storage — the caller must \
                 verify via matches!(_, TensorStorage::Cuda(_)) before \
                 extracting the device pointer"
            )
        }
        crate::tensor::TensorStorage::CpuBf16(_) => {
            // M4.7.2 panic-stub: the GPU path will land in M4.7.3
            // (residency-aware kernels, BF16 host→device transfers).
            // Until then, callers that want to run a CpuBf16 param
            // through CUDA must transition the variant via
            // `Tensor::ensure_cpu` first (eager F32 upcast) and then
            // `ensure_gpu`. The Llama hot path in `graph.rs` does
            // not reach this function for CpuBf16 because the
            // executor decode-on-access pattern materialises a
            // transient F32 vec before any CUDA call.
            unreachable!(
                "cuda_device_ptr called on CpuBf16 storage — M4.7.3 \
                 (residency-aware GPU kernels with BF16 host transfers) \
                 has not landed yet; transition the variant via \
                 ensure_cpu then ensure_gpu first."
            )
        }
        crate::tensor::TensorStorage::CpuShared(_)
        | crate::tensor::TensorStorage::CpuBf16Shared(_) => {
            // M5.c.2.a — Arc-shared host storage is intentionally
            // host-only. GPU dispatch on shared parameters is M6+
            // territory (paired with cuda_matmul non-pooled).
            // The current Llama hot path runs entirely on CPU for
            // 13B per the M4.7.6.c gating, so this arm is
            // unreachable from the production graph.
            unreachable!(
                "cuda_device_ptr called on Arc-shared host storage. \
                 Shared parameters are CPU-only in M5; GPU dispatch \
                 lands with M6 (cuda_matmul non-pooled + offload)."
            )
        }
        crate::tensor::TensorStorage::CpuInt8 { .. } => {
            // M9.1 — INT8 weights have no F32 device pointer.
            // The dedicated GPU path is `int8_to_bf16_in_vram`
            // which uploads INT8 + scales and dequantises to a
            // BF16 VRAM buffer; that buffer is then consumed via
            // the existing BF16 dispatch. M9.2 wires this into
            // `try_gpu_matmul`; until then, callers that reach
            // this function with CpuInt8 storage are bugs.
            unreachable!(
                "cuda_device_ptr called on CpuInt8 storage — INT8 \
                 weights flow through the dedicated \
                 `int8_to_bf16_in_vram` upload path (M9.1+), not via \
                 this F32 device pointer accessor."
            )
        }
    }
}

/// Mutable counterpart of [`cuda_device_ptr`]. Same precondition.
pub(crate) fn cuda_device_ptr_mut(storage: &crate::tensor::TensorStorage) -> *mut f32 {
    match storage {
        crate::tensor::TensorStorage::Cuda(g) => g.device_ptr() as *mut f32,
        crate::tensor::TensorStorage::Cpu(_) => {
            unreachable!("cuda_device_ptr_mut called on Cpu storage")
        }
        crate::tensor::TensorStorage::Disk(_) => {
            unreachable!(
                "cuda_device_ptr_mut called on Disk storage — the caller must \
                 verify via matches!(_, TensorStorage::Cuda(_)) before \
                 extracting the device pointer"
            )
        }
        crate::tensor::TensorStorage::CpuBf16(_) => {
            // M4.7.2 panic-stub: see `cuda_device_ptr` for the
            // milestone breakdown. Mutating CpuBf16 in place would
            // also lose the BF16 precision contract; this call site
            // is unreachable on the Llama forward path.
            unreachable!(
                "cuda_device_ptr_mut called on CpuBf16 storage — M4.7.3 \
                 (residency-aware GPU kernels with BF16 host transfers) \
                 has not landed yet."
            )
        }
        crate::tensor::TensorStorage::CpuShared(_)
        | crate::tensor::TensorStorage::CpuBf16Shared(_) => {
            // M5.c.2.a — Arc-shared storage is read-only by
            // construction; mutating it would race with siblings.
            unreachable!(
                "cuda_device_ptr_mut called on Arc-shared host storage. \
                 Shared parameters are read-only by construction \
                 (M5.c.2.a) — call ensure_owned() first if mutation \
                 is needed."
            )
        }
        crate::tensor::TensorStorage::CpuInt8 { .. } => {
            // M9.1 — INT8 weights are read-only after quantisation
            // (mutating in place would invalidate the per-channel
            // scales). Mutable GPU access on an INT8 tensor is a bug
            // by construction.
            unreachable!(
                "cuda_device_ptr_mut called on CpuInt8 storage — INT8 \
                 weights are read-only after quantisation (M9.1)."
            )
        }
    }
}
