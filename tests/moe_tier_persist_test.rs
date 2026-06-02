//! **MOE-PROD-4** — persistent expert tier (validation + reuse + integrity).
//!
//! With `ATENIA_MOE_TIER_PERSIST=1` the disk tier writes deterministically
//! named files under `<base>/moe_tier/<model_id>/` and **reuses** them on a
//! later load instead of rewriting the (multi-GB, on a real model) tier. This
//! test proves, on the committed tiny Mixtral fixture:
//!
//!   1. first load writes the tier + a `tier_manifest.json`, and the files
//!      **survive** the runtime being dropped (persistent handles don't delete);
//!   2. a second load **reuses** them — the `.bin` files are not rewritten
//!      (mtimes unchanged) — and produces **identical** generation (bit-exact);
//!   3. integrity: deleting a tier file makes the next load **regenerate** it,
//!      still producing identical output.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use atenia_engine::moe::runtime::MixtralRuntime;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn unique_base(label: &str) -> PathBuf {
    let nanos = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos();
    let d = std::env::temp_dir().join(format!("atenia_tier_{label}_{}_{nanos}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The single per-model tier dir under `<base>/moe_tier/`.
fn tier_model_dir(base: &Path) -> PathBuf {
    let moe = base.join("moe_tier");
    std::fs::read_dir(&moe)
        .expect("moe_tier dir must exist after a persistent load")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| p.is_dir())
        .expect("one model_id tier dir")
}

fn bin_mtimes(dir: &Path) -> BTreeMap<String, SystemTime> {
    let mut m = BTreeMap::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let p = e.path();
        if p.extension().and_then(|x| x.to_str()) == Some("bin") {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            m.insert(name, e.metadata().unwrap().modified().unwrap());
        }
    }
    m
}

fn generate_once(base: &Path) -> Vec<u32> {
    unsafe {
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
        std::env::set_var("ATENIA_MOE_EXPERT_TIER", "disk");
        std::env::set_var("ATENIA_MOE_TIER_PERSIST", "1");
        std::env::set_var("ATENIA_DISK_TIER_DIR", base);
    }
    let rt = MixtralRuntime::load_from_files(
        &fixture_dir().join("mixtral_tiny_config.json"),
        &fixture_dir().join("full_mixtral.safetensors"),
    )
    .expect("persistent-tier load");
    let out = rt.generate(&[22, 25, 29], 8);
    // Drop the runtime: persistent tier files must NOT be deleted.
    drop(rt);
    out
}

#[test]
fn persistent_tier_reuses_without_rewrite_and_regenerates_on_loss() {
    let base = unique_base("persist");

    // 1) First load — writes the tier.
    let gen1 = generate_once(&base);
    assert_eq!(gen1, vec![17, 20], "fixture greedy path");
    let tdir = tier_model_dir(&base);
    assert!(tdir.join("tier_manifest.json").exists(), "manifest written");
    let m1 = bin_mtimes(&tdir);
    assert!(!m1.is_empty(), "tier .bin files written + survive runtime drop");

    // 2) Second load — must reuse (no rewrite) and match bit-for-bit.
    let gen2 = generate_once(&base);
    assert_eq!(gen2, gen1, "reused-tier generation must equal first load");
    let m2 = bin_mtimes(&tdir);
    assert_eq!(m1, m2, "tier files must be reused, not rewritten (mtimes unchanged)");

    // 3) Integrity — delete one tier file; next load regenerates it.
    let victim = tdir.join(m1.keys().next().unwrap());
    std::fs::remove_file(&victim).unwrap();
    assert!(!victim.exists());
    let gen3 = generate_once(&base);
    assert_eq!(gen3, gen1, "regenerated-tier generation must equal first load");
    assert!(victim.exists(), "deleted tier file must be regenerated");

    // Cleanup (persistent files won't self-delete).
    unsafe {
        std::env::remove_var("ATENIA_MOE_TIER_PERSIST");
        std::env::remove_var("ATENIA_MOE_EXPERT_TIER");
        std::env::remove_var("ATENIA_DISK_TIER_DIR");
    }
    std::fs::remove_dir_all(&base).ok();
}
