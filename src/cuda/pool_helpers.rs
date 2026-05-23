//! Shared Rust-side helper for the CPU-path of the CUDA ops.
//!
//! Centralizes the allocate-buffers → copy-H→D → launch-kernel →
//! copy-D→H → free cycle that `cuda_matmul`, `cuda_linear_raw`,
//! `cuda_batch_matmul_raw`, and `cuda_fused_linear_silu_raw` used to
//! duplicate inline. Before this helper existed, each of the 3 legacy
//! `.cu` host wrappers (`launch_linear_f32`, `launch_batch_matmul_f32`,
//! `launch_fused_linear_silu_f32`) contained the same cookie-cutter
//! C boilerplate with `goto cleanup` error handling, and the Rust-side
//! `cuda_matmul` re-implemented the same pattern directly in Rust —
//! ~210 LOC of C and ~70 LOC of Rust duplication total.
//!
//! Moving the logic here produces a single testable implementation,
//! unifies error propagation through [`StorageTransferError`], and
//! lets the `.cu` translation units shrink to just the kernel + the
//! `_device_ptrs` launcher.
//!
//! # Vendor-neutrality boundary
//!
//! This file lives under `src/cuda/` and is allowed to import
//! `cudart` FFI symbols directly (invariant #9 in the handoff). The
//! logic it implements is vendor-specific to CUDA; the helper is
//! wired into Rust-side dispatchers that remain vendor-neutral above
//! this boundary.

// FFI bindings below mirror CUDA C symbol names verbatim (`cudaMemcpy`).
// Renaming them would break linker resolution; silence the snake_case
// lint module-wide.
#![allow(non_snake_case)]

use std::ffi::c_void;
use std::mem;
use std::os::raw::c_int;

use crate::apx4_12::{pool_alloc, pool_free};
use crate::tensor::StorageTransferError;

// Direct FFI to cudart for host↔device transfers.
#[cfg(atenia_cuda)]
#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
}

// **CPU-2 C2b** — CUDA-less build: no cudart to link. Identical
// signature; unreachable because `with_pooled_device_buffers` has a
// `#[cfg(not(atenia_cuda))]` sibling that returns
// `Err(EngineUnavailable)` before any pooled transfer is attempted.
#[cfg(not(atenia_cuda))]
#[allow(dead_code, unused_variables)]
unsafe fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int {
    unreachable!("CUDA symbol cudaMemcpy called in CPU-only build (atenia_cuda not enabled)")
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

/// RAII guard that releases every pooled device buffer registered into
/// it when dropped. Guarantees cleanup across every exit path of
/// [`with_pooled_device_buffers`] — success, `PoolExhausted` from a
/// later alloc, memcpy failure, or kernel-launch failure.
#[cfg(atenia_cuda)]
struct PoolGuard {
    ptrs: Vec<*mut c_void>,
}

#[cfg(atenia_cuda)]
impl PoolGuard {
    fn new(capacity: usize) -> Self {
        Self {
            ptrs: Vec::with_capacity(capacity),
        }
    }

    /// Register a pool-owned pointer; it will be freed on drop.
    fn track(&mut self, p: *mut c_void) {
        self.ptrs.push(p);
    }
}

#[cfg(atenia_cuda)]
impl Drop for PoolGuard {
    fn drop(&mut self) {
        for p in self.ptrs.drain(..) {
            // `pool_free` is a safe Rust function; every pointer
            // tracked here was returned by `pool_alloc` above and is
            // non-null (null ptrs short-circuit with `PoolExhausted`
            // before reaching the guard).
            pool_free(p);
        }
    }
}

/// Allocate one pooled device buffer per input + one for the output,
/// stage the inputs H→D, run the kernel closure, copy the output D→H,
/// and free every buffer on the way out.
///
/// # Closure contract
///
/// `kernel_call` receives:
/// - `d_inputs`: slice of `*const f32` with one device pointer per
///   entry in `inputs`, in the same order.
/// - `d_output`: `*mut f32` pointing at the output device buffer.
///
/// Must return `0` on success. Non-zero values are surfaced as
/// [`StorageTransferError::TransferFailed`] (the helper does not
/// preserve the numeric launch/sync code — callers that need finer
/// resolution must invoke the `_device_ptrs` launcher directly).
///
/// # Errors
///
/// - [`StorageTransferError::PoolExhausted`] if any `pool_alloc`
///   returns null; any buffers successfully allocated before the
///   failure are freed before returning.
/// - [`StorageTransferError::TransferFailed`] on `cudaMemcpy` failure
///   (H→D or D→H) or non-zero closure return.
///
/// # Safety
///
/// The caller must ensure:
/// - Every `inputs[i]` slice remains valid for reads of
///   `inputs[i].len() * sizeof(f32)` bytes during the entire call.
/// - `output` remains valid for writes of `output.len() * sizeof(f32)`
///   bytes during the entire call.
/// - The closure does not store the received device pointers beyond
///   its own scope (they are freed immediately after it returns).
/// - A working CUDA driver is available (the helper does no
///   driver-availability probing of its own).
// **CPU-2 C2b** — CUDA-less build. The pooled device path is
// entirely VRAM-backed (`pool_alloc` → `cuda_malloc`, `cudaMemcpy`),
// so there is no CPU-only behaviour to emulate: report the engine
// as unavailable. Callers (`cuda_*_raw` in linear/batch/fused) are
// themselves CUDA-gated, so this is reached only if a future caller
// invokes it directly on a CUDA-less build.
#[cfg(not(atenia_cuda))]
#[allow(unused_variables)]
pub(crate) unsafe fn with_pooled_device_buffers<F>(
    inputs: &[&[f32]],
    output: &mut [f32],
    kernel_call: F,
) -> Result<(), StorageTransferError>
where
    F: FnOnce(&[*const f32], *mut f32) -> i32,
{
    Err(StorageTransferError::EngineUnavailable)
}

#[cfg(atenia_cuda)]
pub(crate) unsafe fn with_pooled_device_buffers<F>(
    inputs: &[&[f32]],
    output: &mut [f32],
    kernel_call: F,
) -> Result<(), StorageTransferError>
where
    F: FnOnce(&[*const f32], *mut f32) -> i32,
{
    let mut guard = PoolGuard::new(inputs.len() + 1);

    // 1. Allocate device buffers for inputs.
    let mut d_inputs: Vec<*const f32> = Vec::with_capacity(inputs.len());
    for input in inputs {
        let bytes = input.len() * mem::size_of::<f32>();
        let p = pool_alloc();
        if p.is_null() {
            return Err(StorageTransferError::PoolExhausted { size_bytes: bytes });
        }
        guard.track(p);
        d_inputs.push(p as *const f32);
    }

    // 2. Allocate device buffer for output.
    let out_bytes = output.len() * mem::size_of::<f32>();
    let p_out = pool_alloc();
    if p_out.is_null() {
        return Err(StorageTransferError::PoolExhausted {
            size_bytes: out_bytes,
        });
    }
    guard.track(p_out);
    let d_output = p_out as *mut f32;

    // 3. Stage inputs host → device.
    for (i, input) in inputs.iter().enumerate() {
        let bytes = input.len() * mem::size_of::<f32>();
        let err = unsafe {
            cudaMemcpy(
                d_inputs[i] as *mut c_void,
                input.as_ptr() as *const c_void,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if err != 0 {
            return Err(StorageTransferError::TransferFailed);
        }
    }

    // 4. Run the kernel. The `_device_ptrs` launchers (linear, batch_matmul,
    //    fused_linear_silu) internally call `cudaDeviceSynchronize`, so no
    //    extra sync is needed here. `matmul_f32_launch_device` also syncs
    //    internally (it just returns void instead of an error code).
    let rc = kernel_call(&d_inputs, d_output);
    if rc != 0 {
        return Err(StorageTransferError::TransferFailed);
    }

    // 5. Copy output device → host.
    let err = unsafe {
        cudaMemcpy(
            output.as_mut_ptr() as *mut c_void,
            d_output as *const c_void,
            out_bytes,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    };
    if err != 0 {
        return Err(StorageTransferError::TransferFailed);
    }

    // 6. Guard drops here → frees every tracked pointer.
    Ok(())
}
