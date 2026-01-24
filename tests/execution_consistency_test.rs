#![allow(dead_code)]

use atenia_engine::v17;

use v17::snapshot::execution_snapshot::ExecutionSnapshot;
use v17::snapshot::snapshot_hash::hash_str;
use v17::consistency::consistency_checker::ConsistencyChecker;
use v17::consistency::consistency_errors::ConsistencyError;
use v17::consistency::drift_report::DriftSeverity;
use v17::consistency::execution_baseline::ExecutionBaseline;

fn make_snapshot(backend: &str, output: &str, profile: &str) -> ExecutionSnapshot {
    ExecutionSnapshot {
        model_id: "model-a".to_string(),
        contract_fingerprint: hash_str("contract"),
        plan_fingerprint: hash_str("plan"),
        backend_usage: backend.to_string(),
        profile_hash: hash_str(profile),
        output_signature: hash_str(output),
        explanation_signature: hash_str("explain"),
        snapshot_hash: hash_str(&format!("{}|{}|{}", backend, output, profile)),
    }
}

#[test]
fn identical_snapshots_yield_compatible_consistency() {
    let snap = make_snapshot("cpu", "out", "profile");
    let baseline = ExecutionBaseline {
        reference_snapshot: snap.clone(),
        allow_backend_change: false,
    };

    let report = ConsistencyChecker::compare(&baseline, &snap).expect("report");
    assert_eq!(report.severity, DriftSeverity::Compatible);
    assert!(report.differences.is_empty());
}

#[test]
fn allowed_backend_change_yields_non_critical_drift() {
    let base = make_snapshot("cpu", "out", "profile");
    let current = make_snapshot("gpu", "out", "profile");
    let baseline = ExecutionBaseline {
        reference_snapshot: base,
        allow_backend_change: true,
    };

    let report = ConsistencyChecker::compare(&baseline, &current).expect("report");
    assert!(matches!(report.severity, DriftSeverity::MinorDrift));
    assert!(report
        .differences
        .iter()
        .any(|d| d.contains("backend changed")));
}

#[test]
fn unexpected_execution_change_yields_critical_drift() {
    let base = make_snapshot("cpu", "out", "profile");
    let current = make_snapshot("cpu", "out2", "profile");
    let baseline = ExecutionBaseline {
        reference_snapshot: base,
        allow_backend_change: false,
    };

    let report = ConsistencyChecker::compare(&baseline, &current).expect("report");
    assert!(matches!(report.severity, DriftSeverity::CriticalDrift));
    assert!(report
        .differences
        .iter()
        .any(|d| d.contains("output signature differs")));
}

#[test]
fn consistency_checker_is_deterministic() {
    let base = make_snapshot("cpu", "out", "profile");
    let current = make_snapshot("gpu", "out", "profile2");
    let baseline = ExecutionBaseline {
        reference_snapshot: base,
        allow_backend_change: true,
    };

    let r1 = ConsistencyChecker::compare(&baseline, &current).expect("r1");
    let r2 = ConsistencyChecker::compare(&baseline, &current).expect("r2");

    assert_eq!(r1, r2);
    assert_eq!(r1.to_json(), r2.to_json());
}

#[test]
fn drift_report_is_stable_and_serializable() {
    let base = make_snapshot("cpu", "out", "profile");
    let current = make_snapshot("gpu", "out2", "profile2");
    let baseline = ExecutionBaseline {
        reference_snapshot: base,
        allow_backend_change: false,
    };

    let report = ConsistencyChecker::compare(&baseline, &current).expect("report");
    let json1 = report.to_json();
    let json2 = report.to_json();
    assert_eq!(json1, json2);
}

#[test]
fn invalid_baseline_yields_error() {
    let snap = ExecutionSnapshot {
        model_id: "".to_string(),
        contract_fingerprint: hash_str("c"),
        plan_fingerprint: hash_str("p"),
        backend_usage: "cpu".to_string(),
        profile_hash: hash_str("prof"),
        output_signature: hash_str("out"),
        explanation_signature: hash_str("exp"),
        snapshot_hash: hash_str("s"),
    };

    let baseline = ExecutionBaseline {
        reference_snapshot: snap,
        allow_backend_change: false,
    };
    let current = make_snapshot("cpu", "out", "profile");

    let res = ConsistencyChecker::compare(&baseline, &current);
    assert!(matches!(res, Err(ConsistencyError::InvalidBaseline(_))));
}
