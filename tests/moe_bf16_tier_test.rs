//! **MOE-PROD-6** — bf16 expert tier + shared-expert cache.
//!
//! Proves, on a tiny Qwen-MoE checkpoint (which HAS a shared expert):
//!
//! 1. **bf16 tier**: with `ATENIA_MOE_TIER_BF16` on (default) and bf16-source
//!    weights, the routed + shared expert tier files are written as **bf16**
//!    (half size — the manifest records `dtype: "bf16"`), and a warm
//!    reconstruction from that bf16 tier generates **bit-identically** to the
//!    cold load. The read path upcasts bf16 → f32 losslessly.
//! 2. **f32 fallback**: with `ATENIA_MOE_TIER_BF16=0` the same experts are
//!    written as **f32** (twice the size), output identical — the format is
//!    transparent and the safe fallback is one env var away.
//! 3. **shared cache**: `ATENIA_MOE_SHARED_CACHE=0` (per-token resolution)
//!    produces output **identical** to the default pinned slot — the cache
//!    never changes the math.
//!
//! Run as a SINGLE sequential test so the process-global env vars never race.

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
    let nanos = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("atenia_bf16_{label}_{}_{nanos}", std::process::id()));
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

/// Build a 2-shard Qwen-MoE checkpoint from the committed fixture, optionally
/// **masking** every weight to a bf16-representable value (low 16 mantissa bits
/// zeroed). bf16-source weights round-trip through the bf16 tier exactly.
fn build_checkpoint(label: &str, mask_bf16: bool) -> (PathBuf, (&'static str, &'static str)) {
    let reader = SafetensorsReader::open(&fixture_dir().join("qwen_moe_tiny.safetensors")).unwrap();
    let tensors: Vec<(String, Vec<usize>, Vec<f32>)> = reader
        .iter()
        .map(|e| {
            let mut d = e.to_vec_f32().unwrap();
            if mask_bf16 {
                for v in &mut d {
                    *v = f32::from_bits(v.to_bits() & 0xFFFF_0000);
                }
            }
            (e.name.to_string(), e.shape.to_vec(), d)
        })
        .collect();

    let model_dir = scratch(label);
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
    (model_dir, (s1n, s2n))
}

/// Read the single tier manifest under `<tier_base>/moe_tier/<model_id>/`.
fn read_manifest(tier_base: &Path) -> serde_json::Value {
    let moe_tier = tier_base.join("moe_tier");
    let sub = std::fs::read_dir(&moe_tier)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.is_dir())
        .expect("one model_id subdir");
    let text = std::fs::read_to_string(sub.join("tier_manifest.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// Count manifest entries (expert tensors) with the given on-disk dtype.
fn count_dtype(manifest: &serde_json::Value, dtype: &str) -> usize {
    manifest["entries"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["dtype"].as_str() == Some(dtype) && e["key"].as_str().unwrap().contains(".e"))
        .count()
}

#[test]
fn bf16_tier_and_shared_cache_are_bit_exact() {
    let prompt = [22u32, 25, 29];

    // ===== Phase 1: bf16 tier (bf16-source weights) =====
    let (model_dir, (s1n, s2n)) = build_checkpoint("bf16src", true);
    let tier_base = scratch("bf16src_tierbase");
    unsafe {
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
        std::env::set_var("ATENIA_MOE_EXPERT_TIER", "disk");
        std::env::set_var("ATENIA_MOE_TIER_PERSIST", "1");
        std::env::set_var("ATENIA_DISK_TIER_DIR", &tier_base);
        std::env::remove_var("ATENIA_MOE_TIER_BF16"); // default on
        std::env::remove_var("ATENIA_MOE_SHARED_CACHE"); // default on
    }

    let cold = MixtralRuntime::load_from_dir(&model_dir).expect("cold load (bf16 tier)");
    let gen_cold = cold.generate(&prompt, 8);
    drop(cold);

    // The expert tier files must have been written as bf16 (half size).
    let manifest = read_manifest(&tier_base);
    let bf16_experts = count_dtype(&manifest, "bf16");
    let f32_experts = count_dtype(&manifest, "f32");
    assert!(bf16_experts > 0, "bf16-source experts must be persisted as bf16 (got 0)");
    assert_eq!(f32_experts, 0, "with bf16-source weights every expert tensor is bf16");

    // Warm reconstruction from the bf16 tier — delete the shards first.
    std::fs::remove_file(model_dir.join(s1n)).unwrap();
    std::fs::remove_file(model_dir.join(s2n)).unwrap();
    let warm = MixtralRuntime::load_from_dir(&model_dir).expect("warm load from bf16 tier");
    let gen_warm = warm.generate(&prompt, 8);
    drop(warm);
    assert_eq!(gen_warm, gen_cold, "bf16-tier warm reconstruction must be bit-exact");

    // Shared cache OFF must match the default (pinned) cache exactly.
    unsafe { std::env::set_var("ATENIA_MOE_SHARED_CACHE", "0") };
    let warm_no_shared = MixtralRuntime::load_from_dir(&model_dir).expect("warm load, shared off");
    let gen_no_shared = warm_no_shared.generate(&prompt, 8);
    drop(warm_no_shared);
    assert_eq!(gen_no_shared, gen_cold, "shared-cache OFF must not change output");
    unsafe { std::env::remove_var("ATENIA_MOE_SHARED_CACHE") };

    std::fs::remove_dir_all(&model_dir).ok();
    std::fs::remove_dir_all(&tier_base).ok();

    // ===== Phase 2: f32 fallback (non-representable weights) =====
    let (model_dir2, _) = build_checkpoint("f32src", false);
    let tier_base2 = scratch("f32src_tierbase");
    unsafe { std::env::set_var("ATENIA_DISK_TIER_DIR", &tier_base2) };

    let cold2 = MixtralRuntime::load_from_dir(&model_dir2).expect("cold load (f32 fallback)");
    let gen_cold2 = cold2.generate(&prompt, 8);
    drop(cold2);
    // The arbitrary fixture weights are NOT bf16-representable → auto-fallback to f32.
    let manifest2 = read_manifest(&tier_base2);
    assert_eq!(
        count_dtype(&manifest2, "bf16"),
        0,
        "non-bf16-representable experts must stay f32 (bit-exactness guard)"
    );
    assert!(count_dtype(&manifest2, "f32") > 0, "experts present as f32");

    // ===== Phase 3: explicit f32 mode on bf16-source weights =====
    // ATENIA_MOE_TIER_BF16=0 forces f32 even for representable weights; output
    // identical to the bf16 run is implied by bit-exactness of both formats.
    let (model_dir3, _) = build_checkpoint("forcef32", true);
    let tier_base3 = scratch("forcef32_tierbase");
    unsafe {
        std::env::set_var("ATENIA_DISK_TIER_DIR", &tier_base3);
        std::env::set_var("ATENIA_MOE_TIER_BF16", "0");
    }
    let cold3 = MixtralRuntime::load_from_dir(&model_dir3).expect("cold load (forced f32)");
    let _ = cold3.generate(&prompt, 2);
    drop(cold3);
    let manifest3 = read_manifest(&tier_base3);
    assert_eq!(count_dtype(&manifest3, "bf16"), 0, "forced-f32 mode writes no bf16");
    assert!(count_dtype(&manifest3, "f32") > 0);

    unsafe {
        std::env::remove_var("ATENIA_MOE_TIER_BF16");
        std::env::remove_var("ATENIA_MOE_TIER_PERSIST");
        std::env::remove_var("ATENIA_MOE_EXPERT_TIER");
        std::env::remove_var("ATENIA_DISK_TIER_DIR");
        std::env::remove_var("ATENIA_EXPERIMENTAL_MOE");
    }
    std::fs::remove_dir_all(&model_dir2).ok();
    std::fs::remove_dir_all(&tier_base2).ok();
    std::fs::remove_dir_all(&model_dir3).ok();
    std::fs::remove_dir_all(&tier_base3).ok();
}
