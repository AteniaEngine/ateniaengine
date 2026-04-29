//! M4.7.4.d — `ensure_cpu` Disk-arm dtype dispatch.
//!
//! After M4.7.4.c, a `TensorStorage::CpuBf16` tensor can be
//! spilled to disk via `migrate_all_cpu_to_disk` and the on-disk
//! handle is tagged `DiskDtype::BF16`. Pre-M4.7.4.d the
//! `ensure_cpu` Disk arm only knew how to read f32 files: a
//! BF16-tagged spill restored as garbage floats (4-byte
//! re-interpretation of a 2-byte file) — or, more likely, surfaced
//! as a `DiskSizeMismatch` because the file was half the expected
//! byte count.
//!
//! This file exercises the new dispatch:
//!
//!   1. Spill CpuBf16 → Disk (the M4.7.4.c primitive),
//!      `ensure_cpu`, assert the resulting `Vec<f32>` is the
//!      element-wise BF16 → F32 upcast of the original bits.
//!   2. Spill Cpu → Disk, `ensure_cpu`, assert bit-exact F32
//!      round-trip (the legacy F32 path stays untouched).
//!   3. `copy_to_cpu_vec` on a BF16-tagged Disk handle (which is
//!      called by parts of the executor) also upcasts.
//!   4. After the BF16 round-trip, `tensor.dtype` is flipped to
//!      `DType::F32` to match the new storage — a downstream op
//!      reading `dtype` to dispatch must see a coherent
//!      (storage, dtype) pair, not a BF16-tagged tensor whose
//!      bytes are F32. Same lock the M4.7.2 CpuBf16 → Cpu arm
//!      already enforces.

use std::path::PathBuf;
use uuid::Uuid;

use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::{DType, Tensor, TensorStorage};

fn test_cache_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("atenia_m4_7_4_d_{}_{}", label, Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("create test cache dir");
    dir
}

fn cleanup(dir: &PathBuf) {
    let _ = std::fs::remove_dir_all(dir);
}

/// Inline bf16 ↔ f32 helpers — match what
/// `crate::tensor::tensor::bf16_bits_to_f32` and the BF16 storage
/// path do, but kept private to this test so we are not asserting
/// circularly.
fn bf16_from_f32(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[test]
fn ensure_cpu_after_bf16_disk_spill_upcasts_to_f32() {
    let dir = test_cache_dir("bf16_ensure_cpu");

    let mut gb = GraphBuilder::new();
    let p = gb.parameter(Tensor::new_cpu(vec![8], vec![0.0_f32; 8]));
    let _out = gb.output(p);
    let mut graph = gb.build();

    // Source f32 pattern → bf16 bits → install on the parameter.
    let source: Vec<f32> = vec![0.0, -1.5, 2.5, -3.25, 4.0, -8.5, 16.0, -32.0];
    let bf16_bits: Vec<u16> = source.iter().map(|&f| bf16_from_f32(f)).collect();
    graph.nodes[p]
        .output
        .as_mut()
        .expect("p")
        .set_cpu_bf16_bits(bf16_bits.clone());

    // Spill (M4.7.4.c).
    let report = graph.migrate_all_cpu_to_disk(&dir).expect("spill ok");
    assert_eq!(report.tensors_migrated, 1);

    // Sanity: storage is now Disk with the BF16 dtype tag.
    {
        let t = graph.nodes[p].output.as_ref().unwrap();
        match &t.storage {
            TensorStorage::Disk(h) => assert_eq!(
                h.dtype(),
                atenia_engine::tensor::disk_tier::DiskDtype::BF16
            ),
            other => panic!("expected Disk, got {:?}", other),
        }
        // dtype is still BF16 at this point (the storage is the
        // disk-spilled bits; the logical dtype matches the bits).
        assert_eq!(t.dtype, DType::BF16);
    }

    // Restore. M4.7.4.d makes this dispatch on DiskDtype::BF16
    // and upcast to f32.
    graph.nodes[p]
        .output
        .as_mut()
        .expect("p")
        .ensure_cpu()
        .expect("ensure_cpu after bf16 spill must succeed");

    // Now: storage is Cpu(Vec<f32>), dtype is F32, values are the
    // bf16 → f32 upcast of the source.
    let t = graph.nodes[p].output.as_ref().unwrap();
    let restored = match &t.storage {
        TensorStorage::Cpu(v) => v.clone(),
        other => panic!("expected Cpu after ensure_cpu, got {:?}", other),
    };
    assert_eq!(t.dtype, DType::F32, "dtype must flip to F32 after BF16 disk restore");

    let expected: Vec<f32> = bf16_bits.iter().map(|&b| bf16_to_f32(b)).collect();
    assert_eq!(restored.len(), expected.len());
    for (i, (a, b)) in restored.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "bf16 → disk → f32 mismatch at index {} (src={}, expected={}, got={})",
            i,
            source[i],
            b,
            a
        );
    }

    cleanup(&dir);
}

#[test]
fn ensure_cpu_after_f32_disk_spill_stays_bit_exact() {
    // The F32 path must remain bit-exact under M4.7.4.d.
    let dir = test_cache_dir("f32_ensure_cpu");

    let mut gb = GraphBuilder::new();
    let p = gb.parameter(Tensor::new_cpu(vec![5], vec![0.0_f32; 5]));
    let _out = gb.output(p);
    let mut graph = gb.build();

    let source = vec![0.0_f32, 1.5, -2.5, 3.14159, -7.25];
    graph.nodes[p]
        .output
        .as_mut()
        .expect("p")
        .set_cpu_data(source.clone());

    let report = graph.migrate_all_cpu_to_disk(&dir).expect("spill ok");
    assert_eq!(report.tensors_migrated, 1);

    // Storage is Disk + F32 tag.
    match &graph.nodes[p].output.as_ref().unwrap().storage {
        TensorStorage::Disk(h) => assert_eq!(
            h.dtype(),
            atenia_engine::tensor::disk_tier::DiskDtype::F32
        ),
        other => panic!("expected Disk, got {:?}", other),
    }

    graph.nodes[p]
        .output
        .as_mut()
        .expect("p")
        .ensure_cpu()
        .expect("ensure_cpu after f32 spill must succeed");

    let t = graph.nodes[p].output.as_ref().unwrap();
    let restored = match &t.storage {
        TensorStorage::Cpu(v) => v.clone(),
        other => panic!("expected Cpu after ensure_cpu, got {:?}", other),
    };
    assert_eq!(t.dtype, DType::F32);

    for (a, b) in restored.iter().zip(source.iter()) {
        assert_eq!(a.to_bits(), b.to_bits(), "F32 disk round-trip must be bit-exact");
    }

    cleanup(&dir);
}

#[test]
fn copy_to_cpu_vec_dispatches_on_disk_dtype() {
    // Non-mutating accessor `copy_to_cpu_vec` was also extended
    // in M4.7.4.d to handle BF16-tagged Disk handles. Sanity-check
    // the dispatch on a tensor whose storage is Disk(BF16).
    let dir = test_cache_dir("copy_to_cpu_vec_bf16");

    let mut gb = GraphBuilder::new();
    let p = gb.parameter(Tensor::new_cpu(vec![4], vec![0.0_f32; 4]));
    let _out = gb.output(p);
    let mut graph = gb.build();

    let source: Vec<f32> = vec![0.5, -1.25, 2.0, -4.0];
    let bf16_bits: Vec<u16> = source.iter().map(|&f| bf16_from_f32(f)).collect();
    graph.nodes[p]
        .output
        .as_mut()
        .expect("p")
        .set_cpu_bf16_bits(bf16_bits);

    graph.migrate_all_cpu_to_disk(&dir).expect("spill ok");

    // Direct call to copy_to_cpu_vec — no storage mutation.
    let t = graph.nodes[p].output.as_ref().unwrap();
    let copy = t.copy_to_cpu_vec();
    assert_eq!(copy.len(), source.len());
    for (a, b) in copy.iter().zip(source.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // Storage was not mutated by the copy.
    match &t.storage {
        TensorStorage::Disk(_) => {} // good
        other => panic!("copy_to_cpu_vec must not mutate storage, got {:?}", other),
    }

    cleanup(&dir);
}
