//! APX v20 M3-d.2.C — tests for `TensorStorage::Cuda` and the real
//! host⇄device transfer methods (`ensure_gpu`, `ensure_cpu`).
//!
//! These tests validate the post-M3-d.2 invariants:
//! - `ensure_gpu` on a CPU tensor allocates VRAM, performs H→D, and
//!   switches the storage variant to `Cuda`.
//! - `ensure_cpu` on a GPU tensor performs D→H and switches the
//!   storage variant back to `Cpu` with bit-identical data.
//! - Repeated calls on the already-resident side are no-ops (no
//!   realloc, no unnecessary transfer).
//! - `as_cpu_slice` panics on a GPU-resident tensor with a guiding
//!   message.
//! - `copy_to_cpu_vec` on a GPU tensor reads data without migrating
//!   the storage variant.
//!
//! Every test graceful-skips if the singleton engine is unavailable.

use atenia_engine::gpu::gpu_engine;
use atenia_engine::tensor::{GpuTransferError, Tensor, TensorStorage};

/// Skip the rest of a test if no GPU is available in this environment.
/// Returns `true` when the caller should proceed.
fn require_gpu(test_name: &str) -> bool {
    if gpu_engine().is_some() {
        true
    } else {
        println!(
            "[TEST:{}] no GPU available (gpu_engine() = None) -> graceful skip",
            test_name
        );
        false
    }
}

#[test]
fn test_ensure_gpu_then_ensure_cpu_roundtrip() {
    if !require_gpu("test_ensure_gpu_then_ensure_cpu_roundtrip") {
        return;
    }

    let data = vec![1.5f32, 2.5, 3.5, 4.5, 5.5, 6.5];
    let mut t = Tensor::new_cpu(vec![2, 3], data.clone());

    t.ensure_gpu().expect("ensure_gpu must succeed when engine is available");
    assert!(
        matches!(t.storage(), TensorStorage::Cuda(_)),
        "after ensure_gpu the storage variant must be Cuda"
    );

    t.ensure_cpu().expect("ensure_cpu must succeed on a just-uploaded tensor");
    assert!(
        matches!(t.storage(), TensorStorage::Cpu(_)),
        "after ensure_cpu the storage variant must be Cpu"
    );

    assert_eq!(
        t.as_cpu_slice(),
        data.as_slice(),
        "roundtrip must preserve data bit-for-bit"
    );
    assert_eq!(t.shape, vec![2, 3], "shape must survive the roundtrip");
}

#[test]
fn test_ensure_gpu_on_gpu_is_noop() {
    if !require_gpu("test_ensure_gpu_on_gpu_is_noop") {
        return;
    }

    let mut t = Tensor::new_cpu(vec![4], vec![10.0f32, 20.0, 30.0, 40.0]);
    t.ensure_gpu().expect("first ensure_gpu must succeed");

    // Snapshot the underlying device pointer before the second call.
    let ptr_before = match t.storage() {
        TensorStorage::Cuda(g) => g.device_ptr(),
        TensorStorage::Cpu(_) => unreachable!("storage should be Cuda here"),
    };

    t.ensure_gpu()
        .expect("second ensure_gpu on an already-GPU tensor must succeed");

    let ptr_after = match t.storage() {
        TensorStorage::Cuda(g) => g.device_ptr(),
        TensorStorage::Cpu(_) => unreachable!("storage should still be Cuda"),
    };

    assert_eq!(
        ptr_before, ptr_after,
        "ensure_gpu on an already-GPU tensor must not realloc VRAM"
    );
}

#[test]
fn test_ensure_cpu_on_cpu_is_noop() {
    let mut t = Tensor::new_cpu(vec![3], vec![7.0f32, 8.0, 9.0]);

    // Does not need a GPU: purely a CPU-side no-op path.
    t.ensure_cpu().expect("ensure_cpu on CPU tensor must be a trivial Ok");

    assert!(matches!(t.storage(), TensorStorage::Cpu(_)));
    assert_eq!(t.as_cpu_slice(), &[7.0f32, 8.0, 9.0]);
}

#[test]
fn test_as_cpu_slice_panics_on_gpu_tensor() {
    if !require_gpu("test_as_cpu_slice_panics_on_gpu_tensor") {
        return;
    }

    let mut t = Tensor::new_cpu(vec![2], vec![1.0f32, 2.0]);
    t.ensure_gpu().expect("ensure_gpu must succeed");

    // Catch the panic instead of using `#[should_panic]` so the test
    // can skip honestly when no GPU is available (the skip path above)
    // rather than falsely satisfying `should_panic` with a fake panic.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = t.as_cpu_slice();
    }));

    assert!(
        result.is_err(),
        "as_cpu_slice on a GPU-resident tensor must panic"
    );
}

#[test]
fn test_copy_to_cpu_vec_from_gpu_tensor() {
    if !require_gpu("test_copy_to_cpu_vec_from_gpu_tensor") {
        return;
    }

    let data = vec![0.125f32, 0.25, 0.5, 1.0, 2.0, 4.0];
    let mut t = Tensor::new_cpu(vec![6], data.clone());
    t.ensure_gpu().expect("ensure_gpu must succeed");

    let copied = t.copy_to_cpu_vec();
    assert_eq!(
        copied, data,
        "copy_to_cpu_vec must return the uploaded data unchanged"
    );

    // Critical invariant: copy_to_cpu_vec is a read, not a migration.
    // The tensor's storage must remain Cuda after the call.
    assert!(
        matches!(t.storage(), TensorStorage::Cuda(_)),
        "copy_to_cpu_vec must not migrate storage off the device"
    );
}

#[test]
fn test_storage_variant_is_cuda_after_ensure_gpu() {
    if !require_gpu("test_storage_variant_is_cuda_after_ensure_gpu") {
        return;
    }

    let mut t = Tensor::new_cpu(vec![2, 2], vec![1.0f32, 2.0, 3.0, 4.0]);
    assert!(
        matches!(t.storage(), TensorStorage::Cpu(_)),
        "new_cpu must start on Cpu storage"
    );

    t.ensure_gpu().expect("ensure_gpu must succeed");
    assert!(
        matches!(t.storage(), TensorStorage::Cuda(_)),
        "ensure_gpu must switch the storage variant to Cuda"
    );
}

#[test]
fn test_graceful_skip_no_gpu() {
    if gpu_engine().is_some() {
        println!(
            "[TEST:test_graceful_skip_no_gpu] GPU available; \
             the Err(EngineUnavailable) path is not exercised in this run."
        );
        return;
    }

    // Without GPU, ensure_gpu must fail cleanly with EngineUnavailable
    // rather than panicking, so callers can fall back to CPU execution.
    let mut t = Tensor::new_cpu(vec![2], vec![1.0f32, 2.0]);
    match t.ensure_gpu() {
        Err(GpuTransferError::EngineUnavailable) => {}
        Err(other) => panic!(
            "expected EngineUnavailable, got {:?}",
            other
        ),
        Ok(_) => panic!("ensure_gpu should not succeed when no GPU is available"),
    }
}
