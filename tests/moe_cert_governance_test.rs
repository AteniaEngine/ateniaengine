//! **NUMERIC-POLICY-3** — certificate governance, end to end.
//!
//! On a tiny Qwen-MoE checkpoint: (a) `certify_model` generates + persists a
//! certificate; (b) under `ATENIA_NUMERIC_REQUIRE_CERT=1` the runtime **refuses**
//! a quantised (int8) tier with no valid certificate; (c) it **allows** it once a
//! valid passing certificate is present. Single sequential test (env is global).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use atenia_engine::moe::cert::{
    certificate_path, load_certificate, save_certificate, VALIDATION_SET_ID,
};
use atenia_engine::moe::numeric_policy::{
    clear_numeric_policy_override, set_numeric_policy, NumericPolicy,
};
use atenia_engine::moe::runtime::{certify_model, MixtralRuntime};
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;
use safetensors::tensor::TensorView;
use safetensors::Dtype as StDtype;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}
fn scratch(label: &str) -> PathBuf {
    let n = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("atenia_certgov_{label}_{}_{n}", std::process::id()));
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

fn build_checkpoint() -> PathBuf {
    let reader = SafetensorsReader::open(&fixture_dir().join("qwen_moe_tiny.safetensors")).unwrap();
    let tensors: Vec<(String, Vec<usize>, Vec<f32>)> = reader
        .iter()
        .map(|e| (e.name.to_string(), e.shape.to_vec(), e.to_vec_f32().unwrap()))
        .collect();
    let dir = scratch("model");
    std::fs::copy(fixture_dir().join("qwen_moe_tiny_config.json"), dir.join("config.json")).unwrap();
    let (s1n, s2n) = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors");
    let (mut s1, mut s2) = (Vec::new(), Vec::new());
    let mut wm = serde_json::Map::new();
    for (i, t) in tensors.iter().enumerate() {
        let shard = if i % 2 == 0 { s1n } else { s2n };
        if i % 2 == 0 { s1.push(t.clone()) } else { s2.push(t.clone()) }
        wm.insert(t.0.clone(), serde_json::Value::String(shard.to_string()));
    }
    write_shard(&dir.join(s1n), &s1);
    write_shard(&dir.join(s2n), &s2);
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"metadata":{"total_size":0},"weight_map":wm}))
            .unwrap(),
    )
    .unwrap();
    dir
}

#[test]
fn certificate_governance_refuses_then_allows_int8() {
    let model_dir = build_checkpoint();
    let tier_base = scratch("tier");
    unsafe {
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
        std::env::set_var("ATENIA_MOE_EXPERT_TIER", "disk");
        std::env::set_var("ATENIA_MOE_TIER_PERSIST", "1");
        std::env::set_var("ATENIA_DISK_TIER_DIR", &tier_base);
        std::env::remove_var("ATENIA_MOE_TIER_QUANT");
        std::env::remove_var("ATENIA_NUMERIC_REQUIRE_CERT");
        std::env::remove_var("ATENIA_NUMERIC_POLICY");
    }

    // (a) certify_model generates + persists a certificate for (Strict, qint8).
    let (cert, cert_path) = certify_model(&model_dir, NumericPolicy::Strict, "qint8")
        .expect("certify_model should run");
    assert!(cert_path.exists(), "certificate file must be written");
    assert_eq!(cert.numeric_policy, "strict");
    assert_eq!(cert.tier_dtype, "qint8");
    assert_eq!(cert.cases.len(), 6, "validation set has 6 cases");
    let model_id_dir = cert_path.parent().unwrap().to_path_buf();

    // Force the certificate to PASS so the allow-path is deterministic on the
    // tiny fixture (the mechanism is what we test, not whether int8 happens to
    // certify on synthetic weights). Re-save with the real manifest version.
    let mut c = load_certificate(&cert_path).unwrap();
    c.pass = true;
    for case in &mut c.cases {
        case.tokens_match = true;
        case.argmax_match_rate = 1.0;
        case.max_abs_diff = 0.0;
    }
    save_certificate(&c, &cert_path).unwrap();
    assert!(c.is_valid_for(&c.model_id, NumericPolicy::Strict, "qint8", c.manifest_version, VALIDATION_SET_ID));

    // (b) REQUIRE_CERT + int8 + a *valid* cert → load is ALLOWED. The requested
    // compute policy is driven via the in-process override (the env is cached
    // per-process; the override is the dynamic equivalent the cert runner uses).
    set_numeric_policy(NumericPolicy::Strict);
    unsafe {
        std::env::set_var("ATENIA_MOE_TIER_QUANT", "int8");
        std::env::set_var("ATENIA_NUMERIC_REQUIRE_CERT", "1");
    }
    let allowed = MixtralRuntime::load_from_dir(&model_dir);
    assert!(allowed.is_ok(), "load must be allowed with a valid passing certificate: {allowed:?}");
    drop(allowed);

    // (c) Remove the certificate → REQUIRE_CERT + int8 → load is REFUSED.
    std::fs::remove_file(certificate_path(&model_id_dir, NumericPolicy::Strict, "qint8")).ok();
    let refused = MixtralRuntime::load_from_dir(&model_dir);
    assert!(refused.is_err(), "int8 tier without a certificate must be refused under REQUIRE_CERT");

    // Default Certified is never refused (lossless): unset quant, policy Certified.
    set_numeric_policy(NumericPolicy::Certified);
    unsafe {
        std::env::remove_var("ATENIA_MOE_TIER_QUANT");
    }
    let certified = MixtralRuntime::load_from_dir(&model_dir);
    assert!(certified.is_ok(), "Certified + lossless tier is always allowed: {certified:?}");
    drop(certified);

    clear_numeric_policy_override();
    unsafe {
        std::env::remove_var("ATENIA_NUMERIC_REQUIRE_CERT");
        std::env::remove_var("ATENIA_MOE_TIER_PERSIST");
        std::env::remove_var("ATENIA_MOE_EXPERT_TIER");
        std::env::remove_var("ATENIA_DISK_TIER_DIR");
        std::env::remove_var("ATENIA_EXPERIMENTAL_MOE");
    }
    std::fs::remove_dir_all(&model_dir).ok();
    std::fs::remove_dir_all(&tier_base).ok();
}
