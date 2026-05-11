#![allow(dead_code)]

use crate::v15::policy::evidence::signals::PolicySignalKind;
use crate::v15::policy::evidence::snapshot::PolicyEvidenceSnapshot;
use crate::v15::policy::policy::ExecutionPolicy;
use crate::v15::policy::types::{DecisionBias, PolicyInput};

pub struct StabilityFirstPolicy;

impl ExecutionPolicy for StabilityFirstPolicy {
    fn name(&self) -> &'static str {
        "stability_first"
    }

    fn evaluate(&self, _input: &PolicyInput) -> DecisionBias {
        DecisionBias {
            risk_weight: 0.1,
            latency_weight: 0.4,
            stability_weight: 1.0,
            memory_pressure_weight: 0.7,
            offload_cost_weight: 0.6,
        }
    }

    fn evaluate_with_evidence(
        &self,
        input: &PolicyInput,
        evidence: Option<&PolicyEvidenceSnapshot>,
    ) -> DecisionBias {
        let mut bias = self.evaluate(input);

        if let Some(snapshot) = evidence {
            let mut max_pre_oom_score = 0.0f32;

            for signal in snapshot.all_signals() {
                if matches!(signal.kind, PolicySignalKind::PreOomSignal) {
                    if signal.score > max_pre_oom_score {
                        max_pre_oom_score = signal.score;
                    }
                }
            }

            if max_pre_oom_score > 0.0 {
                let adjust = 0.2 * max_pre_oom_score;

                // Increase stability and reduce risk in the presence of
                // stronger pre-OOM evidence, clamping to [0.0, 1.0].
                bias.stability_weight = (bias.stability_weight + adjust).min(1.0);
                bias.risk_weight = (bias.risk_weight - adjust).max(0.0);
            }
        }

        bias
    }
}
