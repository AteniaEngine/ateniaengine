//! **STREAMING-LOADER-1** — memory-mapped safetensors `open` must be
//! byte-for-byte equivalent to the owned-`from_bytes` path. mmap only changes
//! the RAM profile of loading, never the bytes the reader yields.

use std::collections::BTreeMap;
use std::path::PathBuf;

use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

fn fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pytorch_bin/tiny_reference.safetensors")
}

/// Collect tensors keyed by NAME (the safetensors crate's `tensors()` iteration
/// order is not stable across `deserialize` calls; the mapper looks tensors up
/// by name, so the by-name view is the meaningful equivalence).
fn dump(r: &SafetensorsReader) -> BTreeMap<String, (Vec<usize>, String, Vec<f32>)> {
    r.iter()
        .map(|e| {
            (
                e.name.to_string(),
                (e.shape.to_vec(), format!("{:?}", e.dtype), e.to_vec_f32().expect("decode")),
            )
        })
        .collect()
}

#[test]
fn mmap_open_matches_owned_from_bytes() {
    let path = fixture();
    let mmapped = SafetensorsReader::open(&path).expect("mmap open");
    let owned = SafetensorsReader::from_bytes(std::fs::read(&path).expect("read"))
        .expect("owned from_bytes");
    let dm = dump(&mmapped);
    let dox = dump(&owned);
    assert_eq!(dm, dox, "mmap and owned reads must be identical (by name)");
    assert!(!dm.is_empty());
}

#[test]
fn disable_mmap_env_reads_identically() {
    let path = fixture();
    let mmapped = dump(&SafetensorsReader::open(&path).expect("mmap"));
    // SAFETY: single-threaded test process; we set and clear the override.
    unsafe { std::env::set_var("ATENIA_DISABLE_MMAP", "1") };
    let owned_via_env = SafetensorsReader::open(&path).expect("owned via env");
    let dumped = dump(&owned_via_env);
    unsafe { std::env::remove_var("ATENIA_DISABLE_MMAP") };
    assert_eq!(dumped, mmapped, "ATENIA_DISABLE_MMAP path must read identically");
}

#[test]
fn open_missing_file_errors() {
    let r = SafetensorsReader::open(&PathBuf::from("definitely/missing/model.safetensors"));
    assert!(r.is_err(), "opening a missing file must error, not panic");
}
