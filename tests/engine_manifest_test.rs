#![allow(dead_code)]

use atenia_engine::v17;

use v17::manifest::engine_manifest::EngineManifest;
use v17::manifest::manifest_errors::ManifestError;
use v17::manifest::version_seal::VersionSeal;

#[test]
fn engine_manifest_is_constructed_correctly() {
    let m = EngineManifest::apx17_default();
    assert_eq!(m.engine_version, "17.x");
    assert!(m.enabled_backends.contains(&"cpu".to_string()));
    assert!(m.enabled_backends.contains(&"gpu".to_string()));
    assert!(m.snapshot_support);
    assert!(m.consistency_guard_support);
    assert!(!m.learning_enabled);
}

#[test]
fn manifest_reflects_enabled_backends_and_features() {
    let m = EngineManifest::apx17_default();
    let json = m.to_json();
    assert!(json.contains("\"engine_version\":\"17.x\""));
    assert!(json.contains("cpu"));
    assert!(json.contains("gpu"));
}

#[test]
fn version_seal_is_deterministic() {
    let m = EngineManifest::apx17_default();
    let s1 = VersionSeal::from_manifest(&m).expect("seal1");
    let s2 = VersionSeal::from_manifest(&m).expect("seal2");
    assert_eq!(s1, s2);
}

#[test]
fn identical_manifests_yield_identical_seals() {
    let m1 = EngineManifest::apx17_default();
    let m2 = EngineManifest::apx17_default();
    let s1 = VersionSeal::from_manifest(&m1).expect("s1");
    let s2 = VersionSeal::from_manifest(&m2).expect("s2");
    assert_eq!(s1.manifest_hash, s2.manifest_hash);
}

#[test]
fn invalid_manifest_yields_explicit_error() {
    let mut m = EngineManifest::apx17_default();
    m.engine_version = "".to_string();
    let res = VersionSeal::from_manifest(&m);
    assert!(matches!(res, Err(ManifestError::InvalidVersion(_))));

    let mut m = EngineManifest::apx17_default();
    m.learning_enabled = true;
    let res = VersionSeal::from_manifest(&m);
    assert!(matches!(
        res,
        Err(ManifestError::InconsistentCapabilities(_))
    ));
}
