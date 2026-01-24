#![allow(dead_code)]

/// Immutable, hashable snapshot of an execution.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionSnapshot {
    pub model_id: String,
    pub contract_fingerprint: String,
    pub plan_fingerprint: String,
    pub backend_usage: String,
    pub profile_hash: String,
    pub output_signature: String,
    pub explanation_signature: String,
    pub snapshot_hash: String,
}

impl ExecutionSnapshot {
    /// Stable JSON-like serialization without external dependencies.
    pub fn to_json(&self) -> String {
        format!(
            "{{\"model_id\":\"{}\",\"contract\":\"{}\",\"plan\":\"{}\",\"backend\":\"{}\",\"profile\":\"{}\",\"output\":\"{}\",\"explanation\":\"{}\",\"hash\":\"{}\"}}",
            self.model_id,
            self.contract_fingerprint,
            self.plan_fingerprint,
            self.backend_usage,
            self.profile_hash,
            self.output_signature,
            self.explanation_signature,
            self.snapshot_hash
        )
    }
}
