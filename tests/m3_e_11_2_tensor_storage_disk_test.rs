//! APX v20 M3-e.11.2 — integration tests for `TensorStorage::Disk`.
//!
//! These tests exercise the new variant end-to-end through the
//! `Tensor` API: construction, `ensure_cpu` bring-back (including the
//! `Arc<InnerDiskFile>` drop cleanup), `ensure_gpu` two-hop, size
//! mismatch detection, and the panic-behavior of `as_cpu_slice` /
//! `copy_to_cpu_vec` on a `Disk`-resident tensor.
//!
//! M3-e.11.2 does NOT produce `TensorStorage::Disk` from any engine
//! path — no `migrate_all_cpu_to_disk` exists yet (lands in M3-e.11.4).
//! The tests construct the variant directly by calling
//! `disk_tier::write_f32_tensor` + replacing the `storage` field on a
//! mutable `Tensor`. This is the idiomatic way to exercise lazy-read
//! behavior without the production migration path.

use std::path::PathBuf;
use uuid::Uuid;

use atenia_engine::tensor::disk_tier;
use atenia_engine::tensor::{DType, Device, Layout, StorageTransferError, Tensor, TensorStorage};

/// Throwaway cache directory unique per test. Cleanup is best-effort.
fn test_cache_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("atenia_m3_e_11_2_{}_{}", label, Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("create test cache dir");
    dir
}

fn cleanup(dir: &PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

/// Build a fresh `Tensor` with the given `shape` whose storage is
/// `TensorStorage::Disk`, backed by a file written into `cache_dir`
/// containing `data` as raw f32 bytes.
fn build_disk_tensor(cache_dir: &PathBuf, shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    let handle = disk_tier::write_f32_tensor(cache_dir, &data).expect("write tensor to disk");
    let mut t = Tensor::with_layout(shape, 0.0, Device::CPU, Layout::Contiguous, DType::F32);
    t.storage = TensorStorage::Disk(handle);
    t
}

#[test]
fn test_tensor_storage_disk_variant_exists() {
    // Trivial: the type constructs and matches. The test exists so a
    // refactor that accidentally removes or renames the variant
    // fails here first rather than deep in integration.
    let dir = test_cache_dir("variant_exists");
    let data: Vec<f32> = vec![1.0, 2.0];
    let handle = disk_tier::write_f32_tensor(&dir, &data).expect("write ok");
    let storage: TensorStorage = TensorStorage::Disk(handle);
    assert!(matches!(storage, TensorStorage::Disk(_)));
    drop(storage);
    cleanup(&dir);
}

#[test]
fn test_ensure_cpu_from_disk_reads_data() {
    // Write a tensor to disk, build a Disk-storage Tensor, call
    // ensure_cpu, verify bit-exact bytes in the resulting Cpu
    // storage.
    let dir = test_cache_dir("ensure_cpu_reads");
    let expected: Vec<f32> = vec![1.5, -2.5, 3.75, 0.0, f32::INFINITY];
    let mut t = build_disk_tensor(&dir, vec![expected.len()], expected.clone());

    assert!(matches!(t.storage, TensorStorage::Disk(_)));

    t.ensure_cpu().expect("ensure_cpu from Disk must succeed");

    assert!(
        matches!(t.storage, TensorStorage::Cpu(_)),
        "storage must be Cpu after ensure_cpu"
    );

    let got = t.as_cpu_slice();
    assert_eq!(got.len(), expected.len());
    for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "mismatch at index {}", i);
    }

    cleanup(&dir);
}

#[test]
fn test_ensure_cpu_from_disk_cleans_up_file() {
    // The Arc<InnerDiskFile> inside the old TensorStorage::Disk
    // drops when ensure_cpu replaces `storage` with Cpu. If this
    // was the last clone (as it is in this test), the file is
    // removed from disk via InnerDiskFile::Drop.
    let dir = test_cache_dir("ensure_cpu_cleans_up");
    let data: Vec<f32> = vec![42.0; 4];
    let mut t = build_disk_tensor(&dir, vec![4], data);

    // Capture the path BEFORE ensure_cpu so we can inspect after.
    let path = match &t.storage {
        TensorStorage::Disk(h) => h.path().to_path_buf(),
        _ => unreachable!("constructed as Disk"),
    };
    assert!(
        path.exists(),
        "pre-condition: file exists while Disk handle is alive"
    );

    t.ensure_cpu().expect("ensure_cpu ok");

    assert!(
        !path.exists(),
        "file must be removed after Disk handle drops (path {:?})",
        path
    );

    cleanup(&dir);
}

#[test]
fn test_ensure_cpu_from_disk_size_mismatch() {
    // Write a file with 4 floats, construct a Tensor whose shape
    // claims 10 elements, call ensure_cpu → DiskSizeMismatch. The
    // check catches the case where a corrupted / truncated file
    // from a previous crashed process would otherwise silently
    // return wrong-size data.
    let dir = test_cache_dir("size_mismatch");
    let file_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // 4 floats on disk
    let handle = disk_tier::write_f32_tensor(&dir, &file_data).expect("write ok");

    // The handle itself caches the *actual* numel (4). To force a
    // mismatch against the tensor shape, we synthesize a handle
    // that LIES about its numel: same path, wrong numel. We cannot
    // do this through the public API because `DiskTensorHandle`
    // has no constructor that accepts an explicit numel — but the
    // semantics we want to test are "what if ensure_cpu's final
    // size check catches a byte-count discrepancy".
    //
    // Skipping the synthetic-handle path: the production system
    // ensures the handle.numel() matches whatever was written.
    // The size-mismatch case in ensure_cpu triggers when the
    // tensor's *shape.numel()* disagrees with the data length.
    // That is the relevant test.

    let mut t = Tensor::with_layout(
        vec![10], // claims 10 elements
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    t.storage = TensorStorage::Disk(handle);
    assert_eq!(t.numel(), 10, "shape product is 10");

    let result = t.ensure_cpu();
    assert!(result.is_err(), "ensure_cpu must fail on size mismatch");
    match result {
        Err(StorageTransferError::DiskSizeMismatch { expected, got }) => {
            assert_eq!(expected, 10, "expected = tensor.numel()");
            assert_eq!(got, 4, "got = actual file element count");
        }
        Err(other) => panic!("expected DiskSizeMismatch, got {:?}", other),
        Ok(_) => panic!("ensure_cpu succeeded when it should have failed"),
    }

    cleanup(&dir);
}

#[test]
fn test_ensure_gpu_from_disk_two_hop() {
    // Disk -> Cpu -> Cuda. If no CUDA engine is available, the
    // second hop returns EngineUnavailable — but the first hop
    // (disk read) still ran. In both cases the test validates
    // the two-hop structural behavior without requiring a GPU.
    let dir = test_cache_dir("two_hop");
    let data: Vec<f32> = vec![7.0, 8.0, 9.0];
    let mut t = build_disk_tensor(&dir, vec![data.len()], data.clone());
    assert!(matches!(t.storage, TensorStorage::Disk(_)));

    let result = t.ensure_gpu();
    match result {
        Ok(_) => {
            // Host has CUDA — Disk -> Cpu -> Cuda completed.
            assert!(
                matches!(t.storage, TensorStorage::Cuda(_)),
                "storage must be Cuda after successful two-hop"
            );
        }
        Err(StorageTransferError::EngineUnavailable) => {
            // Host has no CUDA. The first hop (Disk -> Cpu) MUST
            // still have completed — that's the guarantee of
            // ensure_gpu's internal sequence. Verify that we
            // landed on Cpu rather than staying on Disk.
            assert!(
                matches!(t.storage, TensorStorage::Cpu(_)),
                "first hop (Disk -> Cpu) must run even if the second hop fails; \
                 storage is {:?}",
                std::mem::discriminant(&t.storage)
            );
            // Values should still be intact.
            assert_eq!(t.as_cpu_slice(), data.as_slice());
            println!(
                "[TEST:test_ensure_gpu_from_disk_two_hop] \
                 CUDA unavailable -> verified Disk->Cpu hop still ran"
            );
        }
        Err(other) => panic!("unexpected error from ensure_gpu: {:?}", other),
    }

    cleanup(&dir);
}

#[test]
#[should_panic(expected = "Disk-resident tensor")]
fn test_as_cpu_slice_panics_on_disk() {
    let dir = test_cache_dir("as_cpu_slice_panics");
    let t = build_disk_tensor(&dir, vec![3], vec![1.0, 2.0, 3.0]);
    // Triggers the panic branch for Disk storage. The panic
    // message must mention "Disk-resident" for clarity.
    let _ = t.as_cpu_slice();
    // If we somehow returned (we shouldn't), the cleanup below
    // would never run under `should_panic`, which is fine.
    cleanup(&dir);
}

#[test]
fn test_copy_to_cpu_vec_from_disk() {
    // copy_to_cpu_vec is a non-mutating accessor — it reads the
    // file and returns a fresh Vec<f32> without transitioning the
    // storage variant. The file stays on disk; the handle is not
    // dropped.
    let dir = test_cache_dir("copy_to_cpu_vec");
    let data: Vec<f32> = vec![100.0, 200.0, 300.0];
    let t = build_disk_tensor(&dir, vec![data.len()], data.clone());
    assert!(matches!(t.storage, TensorStorage::Disk(_)));

    let path = match &t.storage {
        TensorStorage::Disk(h) => h.path().to_path_buf(),
        _ => unreachable!(),
    };
    assert!(path.exists(), "pre-condition: file exists on disk");

    let copied = t.copy_to_cpu_vec();
    assert_eq!(copied, data, "copied bytes must match source");

    // storage is unchanged.
    assert!(
        matches!(t.storage, TensorStorage::Disk(_)),
        "copy_to_cpu_vec must not mutate storage"
    );
    // File is still on disk.
    assert!(path.exists(), "file must survive a non-mutating read");

    // Dropping the tensor cleans up.
    drop(t);
    assert!(
        !path.exists(),
        "file must be removed when the tensor (holding the last Arc) drops"
    );

    cleanup(&dir);
}
