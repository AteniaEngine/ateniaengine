//! **MOE-PROD-5** — warm backend reconstruction **without the shards**.
//!
//! Scope C: after a cold load persists the whole backend to the tier (experts +
//! attention + embed + lm_head + router + gate), a warm load must rebuild the
//! runtime **from the tier alone**, never reading the safetensors shards. This
//! test proves it the hard way: it builds a 2-shard checkpoint, cold-loads it
//! (persist on) to populate the tier, then **deletes the shard files** (keeping
//! only `config.json` + `model.safetensors.index.json` + the tier) and warm-
//! loads again — which must still succeed and generate **bit-identically**.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

/// The two reconstruction tests both mutate process-global env vars
/// (`ATENIA_DISK_TIER_DIR`, `ATENIA_MOE_TIER_PERSIST`, …). Cargo runs tests in
/// a file on parallel threads, so without serialization they race on those
/// vars and one load reads the other's tier dir. Hold this lock for the whole
/// body of each test.
static ENV_LOCK: Mutex<()> = Mutex::new(());

use atenia_engine::moe::runtime::MixtralRuntime;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn scratch(label: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("atenia_recon_{label}_{}_{nanos}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_shard(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let bufs: Vec<Vec<u8>> = tensors
        .iter()
        .map(|(_, _, d)| d.iter().flat_map(|f| f.to_le_bytes()).collect())
        .collect();
    let mut views: BTreeMap<String, TensorView> = BTreeMap::new();
    for (i, (n, s, _)) in tensors.iter().enumerate() {
        views.insert(n.clone(), TensorView::new(StDtype::F32, s.clone(), &bufs[i]).unwrap());
    }
    std::fs::write(path, safetensors::serialize(&views, &None).unwrap()).unwrap();
}

#[test]
fn warm_reconstructs_mixtral_from_tier_without_shards() {
    reconstruct_without_shards("mixtral", "mixtral_tiny_config.json", "full_mixtral.safetensors");
}

/// **Regression for the shared-expert FFN-width bug:** Qwen-MoE's shared expert
/// uses `shared_expert_intermediate_size` (≠ the routed `moe_intermediate_size`),
/// so the warm reconstruction must size the shared expert from its tier file,
/// not the routed d_ff. The tiny Mixtral fixture has no shared expert and so
/// could not catch this.
#[test]
fn warm_reconstructs_qwen_with_shared_expert_without_shards() {
    reconstruct_without_shards("qwen", "qwen_moe_tiny_config.json", "qwen_moe_tiny.safetensors");
}

fn reconstruct_without_shards(label: &str, config_name: &str, weights_name: &str) {
    let _env = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    // Build a 2-shard checkpoint dir from a committed tiny fixture.
    let reader = SafetensorsReader::open(&fixture_dir().join(weights_name)).unwrap();
    let tensors: Vec<(String, Vec<usize>, Vec<f32>)> =
        reader.iter().map(|e| (e.name.to_string(), e.shape.to_vec(), e.to_vec_f32().unwrap())).collect();

    let model_dir = scratch(label);
    std::fs::copy(fixture_dir().join(config_name), model_dir.join("config.json")).unwrap();
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
        serde_json::to_vec_pretty(&serde_json::json!({"metadata":{"total_size":0},"weight_map":weight_map})).unwrap(),
    )
    .unwrap();

    let tier_base = scratch(&format!("{label}_tierbase"));
    unsafe {
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
        std::env::set_var("ATENIA_MOE_EXPERT_TIER", "disk");
        std::env::set_var("ATENIA_MOE_TIER_PERSIST", "1");
        std::env::set_var("ATENIA_DISK_TIER_DIR", &tier_base);
    }

    // Cold load — populates the tier.
    let cold = MixtralRuntime::load_from_dir(&model_dir).expect("cold sharded load");
    let gen_cold = cold.generate(&[22, 25, 29], 8);
    drop(cold);

    // Delete the shard files — the warm load must NOT need them.
    std::fs::remove_file(model_dir.join(s1n)).unwrap();
    std::fs::remove_file(model_dir.join(s2n)).unwrap();
    assert!(!model_dir.join(s1n).exists() && !model_dir.join(s2n).exists());

    // Warm load — reconstruct entirely from the tier (index + config remain so
    // the model_id signature still resolves, but no shard bytes are read).
    let warm = MixtralRuntime::load_from_dir(&model_dir).expect("warm load without shards");
    let gen_warm = warm.generate(&[22, 25, 29], 8);
    assert_eq!(gen_warm, gen_cold, "warm reconstruction must match cold bit-for-bit");

    unsafe {
        std::env::remove_var("ATENIA_MOE_TIER_PERSIST");
        std::env::remove_var("ATENIA_MOE_EXPERT_TIER");
        std::env::remove_var("ATENIA_DISK_TIER_DIR");
    }
    std::fs::remove_dir_all(&model_dir).ok();
    std::fs::remove_dir_all(&tier_base).ok();
}
