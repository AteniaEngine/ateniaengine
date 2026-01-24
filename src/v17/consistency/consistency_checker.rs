#![allow(dead_code)]

use crate::v17::snapshot::execution_snapshot::ExecutionSnapshot;
use crate::v17::snapshot::snapshot_hash::hash_str;

use super::consistency_errors::ConsistencyError;
use super::drift_report::{DriftReport, DriftSeverity};
use super::execution_baseline::ExecutionBaseline;

pub struct ConsistencyChecker;

impl ConsistencyChecker {
    pub fn compare(
        baseline: &ExecutionBaseline,
        current: &ExecutionSnapshot,
    ) -> Result<DriftReport, ConsistencyError> {
        if baseline.reference_snapshot.model_id.is_empty() {
            return Err(ConsistencyError::InvalidBaseline(
                "baseline snapshot has empty model_id".to_string(),
            ));
        }

        let mut differences = Vec::new();
        let mut severity = DriftSeverity::Compatible;

        // If snapshots are identical, behavior is fully compatible.
        if baseline.reference_snapshot.snapshot_hash == current.snapshot_hash {
            let fingerprint = hash_str(&format!("noop:{}", current.snapshot_hash));
            return Ok(DriftReport {
                severity,
                differences,
                change_fingerprint: fingerprint,
            });
        }

        // Compare backend usage.
        if baseline.reference_snapshot.backend_usage != current.backend_usage {
            if baseline.allow_backend_change {
                differences.push(format!(
                    "backend changed from {} to {}",
                    baseline.reference_snapshot.backend_usage, current.backend_usage
                ));
                severity = DriftSeverity::MinorDrift;
            } else {
                differences.push(format!(
                    "unexpected backend change from {} to {}",
                    baseline.reference_snapshot.backend_usage, current.backend_usage
                ));
                severity = DriftSeverity::CriticalDrift;
            }
        }

        // Compare output signatures.
        if baseline.reference_snapshot.output_signature != current.output_signature {
            differences.push("output signature differs".to_string());
            severity = DriftSeverity::CriticalDrift;
        }

        // Compare profile hashes.
        if baseline.reference_snapshot.profile_hash != current.profile_hash {
            differences.push("profiling hash differs".to_string());
            if severity == DriftSeverity::Compatible {
                severity = DriftSeverity::MinorDrift;
            }
        }

        // Compare contract/plan fingerprints.
        if baseline.reference_snapshot.contract_fingerprint != current.contract_fingerprint {
            differences.push("contract fingerprint differs".to_string());
            severity = DriftSeverity::CriticalDrift;
        }
        if baseline.reference_snapshot.plan_fingerprint != current.plan_fingerprint {
            differences.push("plan fingerprint differs".to_string());
            severity = DriftSeverity::CriticalDrift;
        }

        if differences.is_empty() {
            // No differences recorded but hashes differ: treat as incompatible.
            return Err(ConsistencyError::IncompatibleSnapshots(
                "snapshots differ but no structured differences were recorded".to_string(),
            ));
        }

        let fingerprint = hash_str(&differences.join("|"));
        Ok(DriftReport {
            severity,
            differences,
            change_fingerprint: fingerprint,
        })
    }
}
