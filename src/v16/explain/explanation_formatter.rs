#![allow(dead_code)]

use super::execution_explanation::ExecutionExplanation;

pub fn format_explanation_text(expl: &ExecutionExplanation) -> String {
    let mut out = String::new();
    out.push_str(&format!("Summary: {}\n", expl.summary));
    out.push_str(&format!("Outcome: {:?}\n", expl.outcome.kind));
    out.push_str(&format!("Policy bias: {:?}\n", expl.decision_bias));
    out.push_str(&format!("Plan: {}\n", expl.plan_summary));
    out.push_str("Steps:\n");
    for step in &expl.steps {
        out.push_str(&format!(
            "  - step {}: {} (guard: {:?}, speculative: {})\n",
            step.step_index, step.description, step.guard_action, step.speculative
        ));
    }
    out.push_str("Events:\n");
    for e in &expl.events {
        out.push_str(&format!(
            "  - ts {}: {:?} (step: {:?})\n",
            e.logical_timestamp, e.kind, e.step_index
        ));
    }
    out
}

pub fn format_explanation_json(expl: &ExecutionExplanation) -> String {
    // Simple, stable JSON-like formatting without external dependencies.
    let mut out = String::new();
    out.push_str("{");
    out.push_str(&format!("\"summary\":\"{}\",", escape(&expl.summary)));
    out.push_str(&format!("\"outcome\":\"{:?}\",", expl.outcome.kind));
    out.push_str(&format!("\"steps\":["));
    for (i, step) in expl.steps.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"index\":{},\"speculative\":{},\"guard\":\"{:?}\"}}",
            step.step_index, step.speculative, step.guard_action
        ));
    }
    out.push(']');
    out.push('}');
    out
}

fn escape(s: &str) -> String {
    s.replace('"', "\\\"")
}
