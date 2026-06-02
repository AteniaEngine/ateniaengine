//! **NUMERIC-POLICY-2** — per-row int8 quantized expert tier.
//!
//! On a tiny Qwen-MoE checkpoint: a cold load with `ATENIA_MOE_TIER_QUANT=int8`
//! must (a) persist the **routed + shared experts** as `qint8` (manifest dtype +
//! `rows*4 + numel` byte size — smaller than bf16), keeping the router /
//! shared-gate / backend non-int8, and (b) warm-reconstruct **from the int8 tier
//! alone** (shards deleted) and generate **identically to the cold load** (the
//! warm read dequantises the exact same bytes). Single sequential test (the
//! env vars are process-global).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use atenia_engine::moe::runtime::MixtralRuntime;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}
fn scratch(label: &str) -> PathBuf {
    let n = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("atenia_q8_{label}_{}_{n}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}
fn write_shard(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let bufs: Vec<Vec<u8>> =
        tensors.iter().map(|(_, _, d)| d.iter().flat_map(|f| f.to_le_bytes()).collect()).collect();
    let mut views: BTreeMap<String, TensorView> = BTreeMap::new();
    for (i, (n, s, _)) in tensors.iter().enumerate() {
        views.insert(n.clone(), TensorView::new(StDtype::F32, s.clone(), &bufs[i]).unwrap());
    }
    std::fs::write(path, safetensors::serialize(&views, &None).unwrap()).unwrap();
}
fn read_manifest(tier_base: &Path) -> serde_json::Value {
    let sub = std::fs::read_dir(tier_base.join("moe_tier"))
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .unwrap();
    serde_json::from_str(&std::fs::read_to_string(sub.join("tier_manifest.json")).unwrap()).unwrap()
}

#[test]
fn qint8_expert_tier_reconstructs_and_is_smaller() {
    let reader = SafetensorsReader::open(&fixture_dir().join("qwen_moe_tiny.safetensors")).unwrap();
    let tensors: Vec<(String, Vec<usize>, Vec<f32>)> = reader
        .iter()
        .map(|e| (e.name.to_string(), e.shape.to_vec(), e.to_vec_f32().unwrap()))
        .collect();

    let model_dir = scratch("model");
    std::fs::copy(fixture_dir().join("qwen_moe_tiny_config.json"), model_dir.join("config.json"))
        .unwrap();
    let (s1n, s2n) = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors");
    let (mut s1, mut s2) = (Vec::new(), Vec::new());
    let mut weight_map = serde_json::Map::new();
    for (i, t) in tensors.iter().enumerate() {
        let shard = if i % 2 == 0 { s1n } else { s2n };
        if i % 2 == 0 { s1.push(t.clone()) } else { s2.push(t.clone()) }
        weight_map.insert(t.0.clone(), serde_json::Value::String(shard.to_string()));
    }
    write_shard(&model_dir.join(s1n), &s1);
    write_shard(&model_dir.join(s2n), &s2);
    std::fs::write(
        model_dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"metadata":{"total_size":0},"weight_map":weight_map}),
        )
        .unwrap(),
    )
    .unwrap();

    let tier_base = scratch("tierbase");
    unsafe {
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
        std::env::set_var("ATENIA_MOE_EXPERT_TIER", "disk");
        std::env::set_var("ATENIA_MOE_TIER_PERSIST", "1");
        std::env::set_var("ATENIA_DISK_TIER_DIR", &tier_base);
        std::env::set_var("ATENIA_MOE_TIER_QUANT", "int8");
    }

    let cold = MixtralRuntime::load_from_dir(&model_dir).expect("cold int8 load");
    let gen_cold = cold.generate(&[22, 25, 29], 8);
    drop(cold);

    // Experts (keys containing ".e" or ".shared.") must be qint8; the router /
    // shared_gate / backend must NOT be qint8.
    let m = read_manifest(&tier_base);
    let entries = m["entries"].as_array().unwrap();
    let n_q8_experts = entries
        .iter()
        .filter(|e| {
            let k = e["key"].as_str().unwrap();
            (k.contains(".e") || k.contains(".shared.")) && e["dtype"] == "qint8"
        })
        .count();
    assert!(n_q8_experts > 0, "routed/shared experts must be qint8");
    // The router must not be quantised.
    let router_q8 = entries
        .iter()
        .any(|e| e["key"].as_str().unwrap().ends_with(".router") && e["dtype"] == "qint8");
    assert!(!router_q8, "router must not be qint8");
    // A qint8 expert file is `rows*4 + numel` bytes < the `numel*2` bf16 size
    // (for the realistic widths in the fixture).
    let e = entries
        .iter()
        .find(|e| e["key"].as_str().unwrap().contains(".e") && e["dtype"] == "qint8")
        .unwrap();
    let (numel, bytes) = (e["numel"].as_u64().unwrap(), e["bytes"].as_u64().unwrap());
    assert!(bytes < numel * 2, "qint8 expert {bytes}B must beat bf16 {}B", numel * 2);
    assert!(bytes >= numel, "qint8 is ~1 byte/element + scales");

    // Warm reconstruction from the int8 tier alone (delete the shards).
    std::fs::remove_file(model_dir.join(s1n)).unwrap();
    std::fs::remove_file(model_dir.join(s2n)).unwrap();
    let warm = MixtralRuntime::load_from_dir(&model_dir).expect("warm int8 reconstruct");
    let gen_warm = warm.generate(&[22, 25, 29], 8);
    drop(warm);
    assert_eq!(gen_warm, gen_cold, "int8 warm reconstruction must match cold (same bytes)");

    unsafe {
        std::env::remove_var("ATENIA_MOE_TIER_QUANT");
        std::env::remove_var("ATENIA_MOE_TIER_PERSIST");
        std::env::remove_var("ATENIA_MOE_EXPERT_TIER");
        std::env::remove_var("ATENIA_DISK_TIER_DIR");
        std::env::remove_var("ATENIA_EXPERIMENTAL_MOE");
    }
    std::fs::remove_dir_all(&model_dir).ok();
    std::fs::remove_dir_all(&tier_base).ok();
}
