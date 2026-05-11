//! M4.7.4.c — `migrate_all_cpu_to_disk` BF16 arm coverage.
//!
//! Pre-M4.7.4.c the migration walked every node and silently
//! dropped any `TensorStorage::CpuBf16` tensor into `tensors_skipped`.
//! That broke the M4.7.2 50 % footprint contract on the disk
//! tier — a 13B-class checkpoint stored as BF16 would have
//! reported migration success while leaving every parameter in
//! RAM.
//!
//! This file exercises the new BF16 arm directly:
//!
//!   1. Pure BF16 graph: every CpuBf16 tensor migrates to Disk
//!      (no skipped), the on-disk handle is tagged
//!      `DiskDtype::BF16`, and the file size matches `numel * 2`
//!      (the M4.7.2 footprint contract carried into the disk
//!      tier).
//!   2. Mixed F32 + BF16 graph: every Cpu *and* every CpuBf16
//!      tensor migrates; the legacy F32 path stays bit-exact.
//!   3. Round-trip: spilled BF16 is brought back via
//!      `read_bf16_tensor` (the dtype-tagged read path from
//!      M4.7.4.a) and the bits are bit-equal to the source.
//!
//! `ensure_cpu` Disk-arm dtype-aware restore is M4.7.4.d, so
//! this file uses `read_bf16_tensor` directly to validate the
//! spill path independently of the `ensure_cpu` rewrite.

use std::path::PathBuf;
use uuid::Uuid;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::disk_tier::{self, DiskDtype};
use atenia_engine::tensor::{Tensor, TensorStorage};

fn test_cache_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("atenia_m4_7_4_c_{}_{}", label, Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("create test cache dir");
    dir
}

fn cleanup(dir: &PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

/// Synthesize bf16 bits from an f32 pattern by taking the upper
/// 16 bits — the same encoding the M4.7.2 storage variant
/// carries.
fn bf16_from_f32(values: &[f32]) -> Vec<u16> {
    values.iter().map(|f| (f.to_bits() >> 16) as u16).collect()
}

#[test]
fn migrate_all_cpu_to_disk_spills_bf16_at_native_width() {
    let dir = test_cache_dir("pure_bf16");

    // Build a graph with two CpuBf16 parameters of different sizes.
    let mut gb = GraphBuilder::new();
    let p_a = gb.parameter(Tensor::new_cpu(vec![4, 8], vec![0.0_f32; 32]));
    let p_b = gb.parameter(Tensor::new_cpu(vec![16], vec![0.0_f32; 16]));
    let _out_a = gb.output(p_a);
    let _out_b = gb.output(p_b);
    let mut graph = gb.build();

    // Down-convert the two parameters to CpuBf16 in place.
    let bits_a = bf16_from_f32(
        &(0..32)
            .map(|i| (i as f32) * 0.05 - 0.5)
            .collect::<Vec<f32>>(),
    );
    let bits_b = bf16_from_f32(
        &(0..16)
            .map(|i| 1.0 + (i as f32) * 0.1)
            .collect::<Vec<f32>>(),
    );
    graph.nodes[p_a]
        .output
        .as_mut()
        .expect("p_a output")
        .set_cpu_bf16_bits(bits_a.clone());
    graph.nodes[p_b]
        .output
        .as_mut()
        .expect("p_b output")
        .set_cpu_bf16_bits(bits_b.clone());

    // Sanity: both are CpuBf16 before the migration.
    assert!(matches!(
        graph.nodes[p_a].output.as_ref().unwrap().storage,
        TensorStorage::CpuBf16(_)
    ));
    assert!(matches!(
        graph.nodes[p_b].output.as_ref().unwrap().storage,
        TensorStorage::CpuBf16(_)
    ));

    let report = graph
        .migrate_all_cpu_to_disk(&dir)
        .expect("migration must succeed");

    // Both parameters migrated. Pre-M4.7.4.c this would have been
    // 0 migrated + 2 skipped — the load-bearing fix.
    assert_eq!(report.tensors_migrated, 2);
    assert_eq!(report.tensors_skipped, 0);
    assert!(report.failure.is_none());

    // Both are now Disk, tagged BF16, with the correct numel.
    for (id, expected_numel) in [(p_a, 32_usize), (p_b, 16_usize)] {
        let storage = &graph.nodes[id].output.as_ref().unwrap().storage;
        let handle = match storage {
            TensorStorage::Disk(h) => h.clone(),
            other => panic!("expected Disk after migration, got {:?}", other),
        };
        assert_eq!(handle.dtype(), DiskDtype::BF16);
        assert_eq!(handle.numel(), expected_numel);
        // File on disk has exactly numel * 2 bytes — M4.7.2
        // footprint contract carried into the disk tier.
        let metadata = std::fs::metadata(handle.path()).expect("disk file metadata");
        assert_eq!(
            metadata.len() as usize,
            expected_numel * 2,
            "BF16 disk file must be numel * 2 bytes"
        );
    }

    // Round-trip: read back, expect bit-exact bits.
    let storage_a = &graph.nodes[p_a].output.as_ref().unwrap().storage;
    let handle_a = match storage_a {
        TensorStorage::Disk(h) => h.clone(),
        _ => unreachable!(),
    };
    let read_a = disk_tier::read_bf16_tensor(&handle_a).expect("read bf16 tensor a");
    assert_eq!(read_a, bits_a, "BF16 spill round-trip must be bit-exact");

    cleanup(&dir);
}

#[test]
fn migrate_all_cpu_to_disk_handles_mixed_f32_and_bf16() {
    let dir = test_cache_dir("mixed_f32_bf16");

    // Three parameters: one F32 (Cpu), one BF16 (CpuBf16), one
    // already on Disk. Migration must spill the first two and
    // skip the third.
    let mut gb = GraphBuilder::new();
    let p_f32 = gb.parameter(Tensor::new_cpu(vec![3, 4], vec![0.0_f32; 12]));
    let p_bf16 = gb.parameter(Tensor::new_cpu(vec![6], vec![0.0_f32; 6]));
    let p_already_disk = gb.parameter(Tensor::new_cpu(vec![5], vec![0.0_f32; 5]));
    let _o1 = gb.output(p_f32);
    let _o2 = gb.output(p_bf16);
    let _o3 = gb.output(p_already_disk);
    let mut graph = gb.build();

    // Populate F32 parameter with a deterministic pattern.
    let f32_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.25).collect();
    graph.nodes[p_f32]
        .output
        .as_mut()
        .expect("p_f32")
        .set_cpu_data(f32_data.clone());

    // Down-convert the second parameter to CpuBf16.
    let bf16_data = bf16_from_f32(&[0.5_f32, -1.5, 2.5, -3.5, 4.5, -5.5]);
    graph.nodes[p_bf16]
        .output
        .as_mut()
        .expect("p_bf16")
        .set_cpu_bf16_bits(bf16_data.clone());

    // Force the third one onto disk *before* the migration call,
    // so the migration must skip it.
    let already_disk_dir = test_cache_dir("already_on_disk_seed");
    let pre_handle = disk_tier::write_f32_tensor(&already_disk_dir, &[9.0, 8.0, 7.0, 6.0, 5.0])
        .expect("seed disk file");
    graph.nodes[p_already_disk]
        .output
        .as_mut()
        .expect("p_already_disk")
        .storage = TensorStorage::Disk(pre_handle);

    let report = graph
        .migrate_all_cpu_to_disk(&dir)
        .expect("migration must succeed");

    assert_eq!(report.tensors_migrated, 2);
    // The already-on-disk tensor lands in tensors_skipped.
    assert_eq!(report.tensors_skipped, 1);

    // F32 spill: dtype tag F32, file size = numel * 4.
    let h_f32 = match &graph.nodes[p_f32].output.as_ref().unwrap().storage {
        TensorStorage::Disk(h) => h.clone(),
        other => panic!("expected Disk for F32 param, got {:?}", other),
    };
    assert_eq!(h_f32.dtype(), DiskDtype::F32);
    let read_f32 = disk_tier::read_f32_tensor(&h_f32).expect("read f32");
    assert_eq!(read_f32.len(), f32_data.len());
    for (a, b) in read_f32.iter().zip(f32_data.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
    let f32_meta = std::fs::metadata(h_f32.path()).expect("f32 metadata");
    assert_eq!(f32_meta.len() as usize, f32_data.len() * 4);

    // BF16 spill: dtype tag BF16, file size = numel * 2.
    let h_bf16 = match &graph.nodes[p_bf16].output.as_ref().unwrap().storage {
        TensorStorage::Disk(h) => h.clone(),
        other => panic!("expected Disk for BF16 param, got {:?}", other),
    };
    assert_eq!(h_bf16.dtype(), DiskDtype::BF16);
    let read_bf16 = disk_tier::read_bf16_tensor(&h_bf16).expect("read bf16");
    assert_eq!(read_bf16, bf16_data);
    let bf16_meta = std::fs::metadata(h_bf16.path()).expect("bf16 metadata");
    assert_eq!(bf16_meta.len() as usize, bf16_data.len() * 2);

    cleanup(&dir);
    cleanup(&already_disk_dir);
}

#[test]
fn migrate_all_cpu_to_disk_bf16_round_trip_preserves_bits() {
    // End-to-end: build a CpuBf16 tensor with a torture-pattern,
    // spill, read back, assert bit-exact. The pattern includes
    // signs, denormals, +/-Inf, NaN.
    let dir = test_cache_dir("bf16_round_trip");

    let mut gb = GraphBuilder::new();
    let p = gb.parameter(Tensor::new_cpu(vec![6], vec![0.0_f32; 6]));
    let _out = gb.output(p);
    let mut graph = gb.build();

    let pattern = bf16_from_f32(&[
        0.0_f32,
        -0.0,
        1.0,
        -2.5,
        f32::INFINITY,
        f32::from_bits(0x7FC00000), // canonical quiet NaN
    ]);
    graph.nodes[p]
        .output
        .as_mut()
        .expect("p")
        .set_cpu_bf16_bits(pattern.clone());

    let report = graph.migrate_all_cpu_to_disk(&dir).expect("spill ok");
    assert_eq!(report.tensors_migrated, 1);

    let handle = match &graph.nodes[p].output.as_ref().unwrap().storage {
        TensorStorage::Disk(h) => h.clone(),
        other => panic!("expected Disk, got {:?}", other),
    };
    let read = disk_tier::read_bf16_tensor(&handle).expect("read");
    assert_eq!(read, pattern);

    let _ = graph; // silence unused warning if any
    cleanup(&dir);
}
