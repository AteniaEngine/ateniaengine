#![allow(dead_code)]

use crate::v16::contract::execution_contract::ExecutionContract;
use crate::v16::planner::execution_plan::ExecutionPlan;
use crate::v16::feedback::execution_outcome::ExecutionOutcomeKind;
use crate::v17::inference::inference_result::InferenceResult;
use crate::v17::profiling::backend_metrics::ExecutionProfile;

use super::execution_snapshot::ExecutionSnapshot;
use super::snapshot_errors::SnapshotError;
use super::snapshot_hash::hash_str;

pub struct SnapshotBuilder;

impl SnapshotBuilder {
    pub fn build(
        result: &InferenceResult,
        contract: &ExecutionContract,
        plan: &ExecutionPlan,
    ) -> Result<ExecutionSnapshot, SnapshotError> {
        if result.profile.is_none() {
            return Err(SnapshotError::MissingProfile(
                "inference result is missing profiling information".to_string(),
            ));
        }
        if result.explanation_text.trim().is_empty() || result.explanation_json.trim().is_empty() {
            return Err(SnapshotError::MissingExplanation(
                "inference result is missing explanation".to_string(),
            ));
        }
        if result.executed_steps.is_empty()
            || matches!(result.outcome.kind, ExecutionOutcomeKind::Failed)
        {
            return Err(SnapshotError::IncompleteExecution(
                "execution was incomplete or failed".to_string(),
            ));
        }

        let profile: &ExecutionProfile = result.profile.as_ref().unwrap();

        let model_id = contract.bias.risk_weight.to_string(); // placeholder fingerprint source
        let contract_fingerprint = hash_str(&format!("{:?}", contract));
        let plan_fingerprint = hash_str(&format!("{:?}", plan));
        let backend_usage = if profile
            .backends
            .iter()
            .any(|b| matches!(b.backend, crate::v17::profiling::backend_metrics::BackendKind::Gpu))
        {
            "gpu".to_string()
        } else {
            "cpu".to_string()
        };

        let profile_hash = hash_str(&profile.to_json());
        let output_signature = hash_str(&format!("{:?}:{:?}", result.output.shape, result.output.data));
        let explanation_signature = hash_str(&format!(
            "{}:{}",
            result.explanation_text,
            result.explanation_json
        ));

        let snapshot_concat = format!(
            "{}|{}|{}|{}|{}|{}|{}",
            model_id,
            contract_fingerprint,
            plan_fingerprint,
            backend_usage,
            profile_hash,
            output_signature,
            explanation_signature,
        );
        let snapshot_hash = hash_str(&snapshot_concat);

        Ok(ExecutionSnapshot {
            model_id,
            contract_fingerprint,
            plan_fingerprint,
            backend_usage,
            profile_hash,
            output_signature,
            explanation_signature,
            snapshot_hash,
        })
    }
}
