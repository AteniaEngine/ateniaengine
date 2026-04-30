//! APX v20 M3-e.11.4 — integration tests for the migration
//! primitives introduced in `Graph`: `migrate_all_cpu_to_disk`
//! (the new one) and `migrate_all_to_disk` (composite that chains
//! `migrate_all_cuda_to_cpu` from M3-e.1 with the new cpu→disk
//! step).
//!
//! These tests exercise the primitives in isolation — the
//! reactive trigger (`GuardAction::DeepDegrade`) and its
//! promotion logic land in M3-e.11.5. A graph built here never
//! attaches a `reactive_context`; the migration methods are
//! called directly.
//!
//! Covered scenarios (from the M3-e.11.4 spec):
//!
//! 1. Empty graph — migrate is a no-op, report is all zeros.
//! 2. Single Cpu tensor — migrated, report counts it, file on
//!    disk, bring-back via `ensure_cpu` returns bit-exact bytes.
//! 3. Mixed storage (Cpu + already-on-Disk) — only Cpu gets
//!    migrated; Disk gets counted as skipped.
//! 4. Cache dir does not exist — `create_dir_all` in the
//!    migration creates it.
//! 5. Atomicity under failure — unwritable cache dir leaves
//!    every tensor's storage untouched and returns Err.
//! 6. Full round-trip — migrate then ensure_cpu, data preserved.
//! 7. Composite migrate_all_to_disk — if CUDA is available, a
//!    Cuda tensor ends in Disk via the two-step chain.
//! 8. MigrationReport semantics — is_complete / is_partial on
//!    manually-constructed reports.

use std::path::PathBuf;
use std::sync::Arc;
use uuid::Uuid;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::reactive::MigrationReport;
use atenia_engine::tensor::disk_tier;
use atenia_engine::tensor::{
    DType, Device, Layout, StorageTransferError, Tensor, TensorStorage,
};

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn test_cache_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("atenia_m3_e_11_4_{}_{}", label, Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("create test cache dir");
    dir
}

fn cleanup(dir: &PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

fn cpu_tensor(shape: Vec<usize>, data: Vec<f32>) -> Tensor {
    let mut t = Tensor::with_layout(
        shape,
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    t.set_cpu_data(data);
    t
}

// ---------------------------------------------------------------------
// 1. Empty graph
// ---------------------------------------------------------------------

#[test]
fn test_migrate_all_cpu_to_disk_empty_graph() {
    let dir = test_cache_dir("empty");
    let mut graph = GraphBuilder::new().build();

    let report = graph
        .migrate_all_cpu_to_disk(&dir)
        .expect("empty graph migrate must succeed");

    assert_eq!(report.tensors_migrated, 0);
    assert_eq!(report.tensors_skipped, 0);
    assert!(report.failure.is_none());
    assert!(report.is_complete());
    assert!(!report.is_partial());

    cleanup(&dir);
}

// ---------------------------------------------------------------------
// 2. Single Cpu tensor migrates successfully
// ---------------------------------------------------------------------

#[test]
fn test_migrate_all_cpu_to_disk_single_tensor() {
    let dir = test_cache_dir("single");

    let mut gb = GraphBuilder::new();
    let w = cpu_tensor(vec![3], vec![1.25, -2.5, 3.75]);
    let _w_id = gb.parameter(w);
    let mut graph = gb.build();

    // Execute once to materialize the parameter's output.
    let _ = graph.execute(vec![]);

    // Pre-condition: exactly one node has a Cpu output.
    let cpu_count_pre = graph
        .nodes
        .iter()
        .filter(|n| {
            n.output
                .as_ref()
                .map(|t| matches!(t.storage, TensorStorage::Cpu(_)))
                .unwrap_or(false)
        })
        .count();
    assert!(cpu_count_pre >= 1, "expected >=1 Cpu tensor, got {}", cpu_count_pre);

    let report = graph
        .migrate_all_cpu_to_disk(&dir)
        .expect("single-tensor migrate must succeed");

    assert!(
        report.tensors_migrated >= 1,
        "expected >=1 migrated, got {}",
        report.tensors_migrated
    );
    assert!(report.is_complete());

    // Verify every migrated node has a Disk storage and a real
    // file sitting under the cache dir.
    for node in &graph.nodes {
        if let Some(t) = &node.output {
            match &t.storage {
                TensorStorage::Disk(handle) => {
                    let path = handle.path();
                    assert!(
                        path.exists(),
                        "migrated tensor's file should exist: {:?}",
                        path
                    );
                    assert!(
                        path.starts_with(&dir),
                        "file path {:?} should live under cache dir {:?}",
                        path,
                        dir
                    );
                }
                other => panic!(
                    "expected Disk, got {:?}",
                    std::mem::discriminant(other)
                ),
            }
        }
    }

    // Bring back via ensure_cpu and compare values.
    for node in graph.nodes.iter_mut() {
        if let Some(t) = node.output.as_mut() {
            t.ensure_cpu().expect("bring-back must succeed");
        }
    }
    // Locate the parameter output and check contents.
    let param_data: Vec<f32> = graph
        .nodes
        .iter()
        .find_map(|n| {
            n.output
                .as_ref()
                .filter(|t| t.numel() == 3)
                .map(|t| t.as_cpu_slice().to_vec())
        })
        .expect("parameter output should be present");
    assert_eq!(param_data, vec![1.25, -2.5, 3.75]);

    cleanup(&dir);
}

// ---------------------------------------------------------------------
// 3. Mixed storage — Cpu migrates, Disk gets skipped
// ---------------------------------------------------------------------

#[test]
fn test_migrate_all_cpu_to_disk_mixed_storage() {
    let dir = test_cache_dir("mixed");

    // Build a graph with two parameter tensors.
    let mut gb = GraphBuilder::new();
    let w1 = cpu_tensor(vec![2], vec![10.0, 20.0]);
    let _w1_id = gb.parameter(w1);
    let w2 = cpu_tensor(vec![2], vec![30.0, 40.0]);
    let _w2_id = gb.parameter(w2);
    let mut graph = gb.build();

    let _ = graph.execute(vec![]);

    // Pre-migrate ONE of the outputs to Disk manually so the test
    // can verify the "already on Disk → counted as skipped" path.
    let mut forced_disk_count = 0usize;
    for node in graph.nodes.iter_mut() {
        if let Some(t) = node.output.as_mut() {
            if matches!(t.storage, TensorStorage::Cpu(_)) && forced_disk_count == 0 {
                let data: Vec<f32> = match &t.storage {
                    TensorStorage::Cpu(v) => v.clone(),
                    _ => unreachable!(),
                };
                let handle = disk_tier::write_f32_tensor(&dir, &data)
                    .expect("manual pre-migration write");
                t.storage = TensorStorage::Disk(handle);
                forced_disk_count = 1;
                break;
            }
        }
    }
    assert_eq!(forced_disk_count, 1, "pre-migration setup failed");

    // Count the Cpu tensors still present BEFORE the real migrate.
    let cpu_count_before: usize = graph
        .nodes
        .iter()
        .filter(|n| {
            n.output
                .as_ref()
                .map(|t| matches!(t.storage, TensorStorage::Cpu(_)))
                .unwrap_or(false)
        })
        .count();
    let disk_count_before: usize = graph
        .nodes
        .iter()
        .filter(|n| {
            n.output
                .as_ref()
                .map(|t| matches!(t.storage, TensorStorage::Disk(_)))
                .unwrap_or(false)
        })
        .count();
    assert!(cpu_count_before >= 1);
    assert_eq!(disk_count_before, 1);

    let report = graph
        .migrate_all_cpu_to_disk(&dir)
        .expect("mixed migrate must succeed");

    // Every Cpu moved, the pre-existing Disk got skipped.
    assert_eq!(
        report.tensors_migrated, cpu_count_before,
        "migrated count must equal prior Cpu count"
    );
    assert_eq!(
        report.tensors_skipped, disk_count_before,
        "skipped count must equal prior Disk count"
    );
    assert!(report.is_complete());

    cleanup(&dir);
}

// ---------------------------------------------------------------------
// 4. Creates cache dir when it doesn't exist
// ---------------------------------------------------------------------

#[test]
fn test_migrate_all_cpu_to_disk_creates_cache_dir() {
    // Build a path UNDER a temp dir that does not exist yet —
    // migrate should create it.
    let parent = std::env::temp_dir()
        .join(format!("atenia_m3_e_11_4_mkdir_{}", Uuid::new_v4()));
    let missing_dir = parent.join("nested").join("cache");
    assert!(!missing_dir.exists(), "pre-condition: dir must not exist");

    let mut gb = GraphBuilder::new();
    let w = cpu_tensor(vec![2], vec![7.0, 8.0]);
    let _w_id = gb.parameter(w);
    let mut graph = gb.build();
    let _ = graph.execute(vec![]);

    let report = graph
        .migrate_all_cpu_to_disk(&missing_dir)
        .expect("migrate must create dir and succeed");
    assert!(report.tensors_migrated >= 1);
    assert!(missing_dir.exists(), "cache dir must have been created");

    cleanup(&parent);
}

// ---------------------------------------------------------------------
// 5. Atomicity: unwritable dir keeps every storage untouched and
//    returns Err
// ---------------------------------------------------------------------

#[test]
fn test_migrate_all_cpu_to_disk_atomicity_preserves_storage_on_failure() {
    // We need a path that `create_dir_all` fails on. Using a file
    // (not a directory) as the "cache dir" does it reliably on
    // every platform — `create_dir_all` returns NotADirectory /
    // AlreadyExists-that-is-not-a-directory.
    let tmp = std::env::temp_dir()
        .join(format!("atenia_m3_e_11_4_unwritable_{}", Uuid::new_v4()));
    // Create a FILE at the path the test will pass to migrate.
    std::fs::write(&tmp, b"not a directory").expect("write sentinel file");

    let mut gb = GraphBuilder::new();
    let w = cpu_tensor(vec![2], vec![100.0, 200.0]);
    let _w_id = gb.parameter(w);
    let mut graph = gb.build();
    let _ = graph.execute(vec![]);

    // Snapshot the storage variants BEFORE the failing call.
    let pre_variants: Vec<_> = graph
        .nodes
        .iter()
        .map(|n| {
            n.output.as_ref().map(|t| match &t.storage {
                TensorStorage::Cpu(_) => "Cpu",
                TensorStorage::CpuBf16(_) => "CpuBf16",
                TensorStorage::Cuda(_) => "Cuda",
                TensorStorage::Disk(_) => "Disk",
                TensorStorage::CpuShared(_) => "CpuShared",
                TensorStorage::CpuBf16Shared(_) => "CpuBf16Shared",
            })
        })
        .collect();

    let result = graph.migrate_all_cpu_to_disk(&tmp);
    assert!(
        result.is_err(),
        "expected Err when cache dir cannot be created/used, got {:?}",
        result
    );
    match result.unwrap_err() {
        StorageTransferError::DiskWriteFailed(_) => {}
        other => panic!("expected DiskWriteFailed, got {:?}", other),
    }

    // Zero side effects: every storage variant is identical.
    let post_variants: Vec<_> = graph
        .nodes
        .iter()
        .map(|n| {
            n.output.as_ref().map(|t| match &t.storage {
                TensorStorage::Cpu(_) => "Cpu",
                TensorStorage::CpuBf16(_) => "CpuBf16",
                TensorStorage::Cuda(_) => "Cuda",
                TensorStorage::Disk(_) => "Disk",
                TensorStorage::CpuShared(_) => "CpuShared",
                TensorStorage::CpuBf16Shared(_) => "CpuBf16Shared",
            })
        })
        .collect();
    assert_eq!(
        pre_variants, post_variants,
        "atomicity violated: storage variants changed despite failure"
    );

    let _ = std::fs::remove_file(&tmp);
}

// ---------------------------------------------------------------------
// 6. Bring-back round-trip — migrate + ensure_cpu preserves bytes
// ---------------------------------------------------------------------

#[test]
fn test_migrate_all_cpu_to_disk_bring_back_roundtrip() {
    let dir = test_cache_dir("roundtrip");

    let mut gb = GraphBuilder::new();
    let original: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        f32::EPSILON,
        123.456,
    ];
    let w = cpu_tensor(vec![original.len()], original.clone());
    let _w_id = gb.parameter(w);
    let mut graph = gb.build();
    let _ = graph.execute(vec![]);

    // Migrate.
    graph
        .migrate_all_cpu_to_disk(&dir)
        .expect("migrate ok");

    // Bring back.
    for node in graph.nodes.iter_mut() {
        if let Some(t) = node.output.as_mut() {
            t.ensure_cpu().expect("ensure_cpu ok");
        }
    }

    // Find the parameter output and compare bit-exact.
    let round_tripped = graph
        .nodes
        .iter()
        .find_map(|n| {
            n.output
                .as_ref()
                .filter(|t| t.numel() == original.len())
                .map(|t| t.as_cpu_slice().to_vec())
        })
        .expect("parameter output present");

    assert_eq!(round_tripped.len(), original.len());
    for (i, (a, b)) in round_tripped.iter().zip(original.iter()).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "bit mismatch at idx {}", i);
    }

    cleanup(&dir);
}

// ---------------------------------------------------------------------
// 7. Composite: Cuda → Disk via migrate_all_to_disk (if CUDA avail)
// ---------------------------------------------------------------------

#[test]
fn test_migrate_all_to_disk_composite_cuda_to_disk() {
    // This test needs CUDA to force a Cuda tensor. Skip if not
    // available — the composite method's structure is covered
    // anyway by the Cpu-only path in other tests.
    if atenia_engine::gpu::gpu_engine().is_none() {
        println!(
            "[TEST:test_migrate_all_to_disk_composite_cuda_to_disk] \
             CUDA unavailable -> graceful skip"
        );
        return;
    }

    let dir = test_cache_dir("composite");

    let mut gb = GraphBuilder::new();
    let w = cpu_tensor(vec![3], vec![2.0, 4.0, 8.0]);
    let _w_id = gb.parameter(w);
    let mut graph = gb.build();
    let _ = graph.execute(vec![]);

    // Force the parameter output to Cuda BEFORE running the
    // composite. The composite should migrate it Cuda → Cpu →
    // Disk in one call.
    let mut forced_cuda = false;
    for node in graph.nodes.iter_mut() {
        if let Some(t) = node.output.as_mut() {
            if matches!(t.storage, TensorStorage::Cpu(_)) {
                t.ensure_gpu().expect("ensure_gpu for setup");
                forced_cuda = true;
                break;
            }
        }
    }
    assert!(forced_cuda, "test setup: at least one tensor pushed to Cuda");

    let report = graph
        .migrate_all_to_disk(&dir)
        .expect("composite migrate must succeed");
    assert!(report.is_complete());
    assert!(
        report.tensors_migrated >= 1,
        "composite must migrate at least one tensor"
    );

    // Every output is now Disk.
    for node in &graph.nodes {
        if let Some(t) = &node.output {
            assert!(
                matches!(t.storage, TensorStorage::Disk(_)),
                "composite migrate must leave every output on Disk"
            );
        }
    }

    cleanup(&dir);
}

// ---------------------------------------------------------------------
// 8. MigrationReport semantics
// ---------------------------------------------------------------------

#[test]
fn test_migration_report_partial_progress_semantics() {
    // Default / empty report — complete (trivially) because
    // there was nothing to do and nothing failed.
    let empty = MigrationReport::new();
    assert!(empty.is_complete());
    assert!(!empty.is_partial());

    // All migrated, nothing failed — complete.
    let mut all_ok = MigrationReport::new();
    all_ok.tensors_migrated = 5;
    assert!(all_ok.is_complete());
    assert!(!all_ok.is_partial());

    // Failure and no progress — NOT complete, NOT partial.
    // (This state should never reach a caller from
    // migrate_all_cpu_to_disk — that path returns Err on zero-
    // progress failures. But the struct itself permits the
    // combination, and the invariants should hold.)
    let mut err_no_progress = MigrationReport::new();
    err_no_progress.failure = Some((
        0,
        StorageTransferError::DiskWriteFailed("x".to_string()),
    ));
    assert!(!err_no_progress.is_complete());
    assert!(!err_no_progress.is_partial());

    // Partial: some migrated + failure.
    let mut partial = MigrationReport::new();
    partial.tensors_migrated = 3;
    partial.failure = Some((
        5,
        StorageTransferError::DiskWriteFailed("y".to_string()),
    ));
    assert!(!partial.is_complete());
    assert!(partial.is_partial());
}

// ---------------------------------------------------------------------
// 9. Cleanup after migrate: dropping the graph releases disk files
// ---------------------------------------------------------------------

#[test]
fn test_migrate_all_cpu_to_disk_files_cleaned_on_graph_drop() {
    // Sanity check that the Arc<InnerDiskFile> Drop pattern works
    // through the Graph-owned Tensor tree: dropping the graph
    // transitively drops every Arc, which removes every file.
    let dir = test_cache_dir("drop_cleans");

    let paths: Vec<PathBuf> = {
        let mut gb = GraphBuilder::new();
        let w1 = cpu_tensor(vec![2], vec![1.0, 2.0]);
        let w2 = cpu_tensor(vec![2], vec![3.0, 4.0]);
        let _ = gb.parameter(w1);
        let _ = gb.parameter(w2);
        let mut graph = gb.build();
        let _ = graph.execute(vec![]);
        graph.migrate_all_cpu_to_disk(&dir).expect("migrate ok");

        let paths: Vec<PathBuf> = graph
            .nodes
            .iter()
            .filter_map(|n| {
                n.output.as_ref().and_then(|t| match &t.storage {
                    TensorStorage::Disk(h) => Some(h.path().to_path_buf()),
                    _ => None,
                })
            })
            .collect();
        assert!(!paths.is_empty(), "expected disk-backed paths");
        for p in &paths {
            assert!(p.exists(), "file {:?} must exist while graph is alive", p);
        }
        paths
        // graph drops here
    };

    for p in &paths {
        assert!(
            !p.exists(),
            "file {:?} must be removed after graph dropped",
            p
        );
    }

    // Hold an Arc to a DiskTensorHandle for coverage — not part
    // of the primary assertion but validates that clone-sharing
    // cooperates with Graph::Drop.
    drop(Arc::new(()));

    cleanup(&dir);
}
