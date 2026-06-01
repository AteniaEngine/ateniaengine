//! **MOE-PROD-1** — sharded MoE loader validation.
//!
//! Real MoE checkpoints (Mixtral-8x7B, Qwen1.5-MoE-A2.7B, DeepSeek-V2) all ship
//! **sharded** (`model-NNNNN-of-NNNNN.safetensors` + `model.safetensors.index.json`).
//! Before MOE-PROD-1 the MoE runtime opened only the first `*.safetensors` in a
//! directory, so a sharded checkpoint silently lost 7/8 of its tensors and
//! failed to assemble. This suite proves the new `MoeWeightSource` abstraction:
//!
//!   1. single-file load is unchanged,
//!   2. a directory with a shard index loads across shards,
//!   3. **sharded == single-file bit-for-bit** (same logits, same generation),
//!   4. missing shard / missing tensor / corrupt index fail with clear errors.
//!
//! Fixtures are built **at test time** by splitting the committed tiny
//! `full_mixtral.safetensors` (MOE-FULL-6) into two shards — no model is
//! downloaded, no large fixture is committed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use atenia_engine::moe::runtime::{MixtralRuntime, MoeRuntimeError};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn enable_opt_in() {
    unsafe {
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
    }
}

/// Unique scratch directory under the OS temp dir (no `tempfile` dep).
fn scratch(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("atenia_moe_shard_{label}_{}_{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Read every tensor from the committed single-file fixture as
/// `(name, shape, f32 data)`.
fn read_full_mixtral() -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let path = fixture_dir().join("full_mixtral.safetensors");
    let reader = SafetensorsReader::open(&path).unwrap();
    reader
        .iter()
        .map(|e| {
            let data = e.to_vec_f32().unwrap();
            (e.name.to_string(), e.shape.to_vec(), data)
        })
        .collect()
}

/// Serialise a subset of tensors to `path` as an F32 safetensors file.
fn write_shard(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    // Keep byte buffers alive for the lifetime of the TensorViews.
    let byte_bufs: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, _, data)| {
            let mut b = Vec::with_capacity(data.len() * 4);
            for &f in data {
                b.extend_from_slice(&f.to_le_bytes());
            }
            b
        })
        .collect();
    let mut views: BTreeMap<String, TensorView> = BTreeMap::new();
    for (i, (name, shape, _)) in tensors.iter().enumerate() {
        views.insert(name.clone(), TensorView::new(StDtype::F32, shape.clone(), &byte_bufs[i]).unwrap());
    }
    let bytes = safetensors::serialize(&views, &None).unwrap();
    std::fs::write(path, bytes).unwrap();
}

/// Build a 2-shard checkpoint dir from `tensors`, optionally dropping a tensor
/// name and optionally writing a broken index. Returns the directory.
fn build_sharded_dir(
    label: &str,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
    drop_tensor: Option<&str>,
    reference_missing_shard: bool,
    corrupt_index: bool,
) -> PathBuf {
    let dir = scratch(label);
    std::fs::copy(fixture_dir().join("mixtral_tiny_config.json"), dir.join("config.json")).unwrap();

    let kept: Vec<_> = tensors
        .iter()
        .filter(|(n, _, _)| drop_tensor != Some(n.as_str()))
        .cloned()
        .collect();

    // Split tensors across two shards, round-robin.
    let shard1_name = "model-00001-of-00002.safetensors";
    let shard2_name = "model-00002-of-00002.safetensors";
    let mut s1 = Vec::new();
    let mut s2 = Vec::new();
    for (i, t) in kept.iter().enumerate() {
        if i % 2 == 0 {
            s1.push(t.clone());
        } else {
            s2.push(t.clone());
        }
    }
    write_shard(&dir.join(shard1_name), &s1);
    write_shard(&dir.join(shard2_name), &s2);

    // Build the weight_map.
    let mut weight_map = serde_json::Map::new();
    for t in &s1 {
        weight_map.insert(t.0.clone(), serde_json::Value::String(shard1_name.to_string()));
    }
    for t in &s2 {
        let shard = if reference_missing_shard {
            "model-00003-of-00002.safetensors".to_string() // does not exist
        } else {
            shard2_name.to_string()
        };
        weight_map.insert(t.0.clone(), serde_json::Value::String(shard));
    }

    let index_path = dir.join("model.safetensors.index.json");
    if corrupt_index {
        std::fs::write(&index_path, b"{ this is not valid json ]").unwrap();
    } else {
        let index = serde_json::json!({
            "metadata": { "total_size": 0 },
            "weight_map": weight_map,
        });
        std::fs::write(&index_path, serde_json::to_vec_pretty(&index).unwrap()).unwrap();
    }
    dir
}

const PROMPT: &[u32] = &[22, 25, 29];

/// **The core correctness proof:** loading the same weights as 8-way-split
/// shards produces **bit-for-bit identical** logits to the single-file load.
#[test]
fn sharded_equals_single_file_logits_and_generation() {
    enable_opt_in();
    let tensors = read_full_mixtral();

    // Single-file reference.
    let single = MixtralRuntime::load_from_files(
        &fixture_dir().join("mixtral_tiny_config.json"),
        &fixture_dir().join("full_mixtral.safetensors"),
    )
    .expect("single-file load");
    let single_logits = single.forward_logits(PROMPT);
    let single_gen = single.generate(PROMPT, 8);

    // Sharded load from a directory with an index.
    let dir = build_sharded_dir("ok", &tensors, None, false, false);
    let sharded = MixtralRuntime::load_from_dir(&dir).expect("sharded load");
    let sharded_logits = sharded.forward_logits(PROMPT);
    let sharded_gen = sharded.generate(PROMPT, 8);

    assert_eq!(
        single_logits.len(),
        sharded_logits.len(),
        "logit vector length must match"
    );
    let max_abs = single_logits
        .iter()
        .zip(sharded_logits.iter())
        .fold(0.0_f32, |a, (x, y)| a.max((x - y).abs()));
    assert_eq!(max_abs, 0.0, "sharded logits must be bit-identical to single-file (got {max_abs:.3e})");
    assert_eq!(single_gen, sharded_gen, "sharded generation must match single-file");

    std::fs::remove_dir_all(&dir).ok();
}

/// A directory with a single `.safetensors` and **no** index still loads
/// (back-compat with the MOE-FULL-14 single-file directory path).
#[test]
fn single_file_directory_without_index_still_loads() {
    enable_opt_in();
    let dir = scratch("singledir");
    std::fs::copy(fixture_dir().join("mixtral_tiny_config.json"), dir.join("config.json")).unwrap();
    std::fs::copy(
        fixture_dir().join("full_mixtral.safetensors"),
        dir.join("model.safetensors"),
    )
    .unwrap();

    let rt = MixtralRuntime::load_from_dir(&dir).expect("single-file dir load");
    let out = rt.generate(PROMPT, 8);
    assert_eq!(out, vec![17, 20], "single-file dir must generate then stop at EOS");
    std::fs::remove_dir_all(&dir).ok();
}

/// A `weight_map` that points a tensor at a shard file that does not exist
/// fails with a clear error (not a panic / silent wrong answer).
#[test]
fn missing_shard_file_errors_clearly() {
    enable_opt_in();
    let tensors = read_full_mixtral();
    let dir = build_sharded_dir("missingshard", &tensors, None, true, false);
    let err = MixtralRuntime::load_from_dir(&dir).expect_err("must fail on missing shard");
    match &err {
        MoeRuntimeError::Load(m) => {
            assert!(m.contains("shard") || m.contains("open"), "error should mention the shard: {m}");
        }
        other => panic!("expected Load error, got {other:?}"),
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// A tensor that is absent from every shard fails with a clear "missing
/// tensor" error during assembly.
#[test]
fn missing_tensor_errors_clearly() {
    enable_opt_in();
    let tensors = read_full_mixtral();
    // Drop a tensor that assembly requires (layer-0 input layernorm).
    let dir = build_sharded_dir(
        "missingtensor",
        &tensors,
        Some("model.layers.0.input_layernorm.weight"),
        false,
        false,
    );
    let err = MixtralRuntime::load_from_dir(&dir).expect_err("must fail on missing tensor");
    match &err {
        // A dropped required tensor surfaces either as an assembly "missing
        // tensor" Load error or as a config/tensor consistency failure — both
        // are clear, non-panicking failures.
        MoeRuntimeError::Load(m) => {
            assert!(
                m.to_lowercase().contains("missing") || m.contains("layer") || m.contains("tensor"),
                "error should be clear about the missing tensor: {m}"
            );
        }
        MoeRuntimeError::ConfigInconsistent(_) => { /* also an acceptable clear failure */ }
        other => panic!("expected a clear load/config error, got {other:?}"),
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// A corrupt `model.safetensors.index.json` fails at open time with a clear
/// error.
#[test]
fn corrupt_index_errors_clearly() {
    enable_opt_in();
    let tensors = read_full_mixtral();
    let dir = build_sharded_dir("corruptindex", &tensors, None, false, true);
    let err = MixtralRuntime::load_from_dir(&dir).expect_err("must fail on corrupt index");
    match &err {
        MoeRuntimeError::Load(m) => {
            assert!(m.contains("index"), "error should mention the index: {m}");
        }
        other => panic!("expected Load error mentioning the index, got {other:?}"),
    }
    std::fs::remove_dir_all(&dir).ok();
}
