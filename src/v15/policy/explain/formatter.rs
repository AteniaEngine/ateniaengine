#![allow(dead_code)]

use crate::v15::policy::explain::explanation::{PolicyExplanation, PreferenceStatus};

/// Renders a PolicyExplanation as a stable, human-readable multiline string.
pub fn format_explanation_text(expl: &PolicyExplanation) -> String {
    let mut out = String::new();

    out.push_str(&format!("policy_name: {}\n", expl.policy_name));

    out.push_str(&format!(
        "final_bias: risk={:.3}, latency={:.3}, stability={:.3}, memory_pressure={:.3}, offload_cost={:.3}\n",
        expl.final_bias.risk_weight,
        expl.final_bias.latency_weight,
        expl.final_bias.stability_weight,
        expl.final_bias.memory_pressure_weight,
        expl.final_bias.offload_cost_weight,
    ));

    out.push_str("signals:\n");
    for s in &expl.considered_signals {
        out.push_str(&format!("  - kind={:?}, score={:.3}\n", s.kind, s.score));
    }

    out.push_str("preferences:\n");
    for p in &expl.preference_explanations {
        let status_str = match p.status {
            PreferenceStatus::Applied => "applied",
            PreferenceStatus::IgnoredDueToRisk => "ignored_due_to_risk",
            PreferenceStatus::Inactive => "inactive",
        };
        out.push_str(&format!("  - {}: {}\n", p.name, status_str));
    }

    out.push_str("notes:\n");
    for n in &expl.notes {
        out.push_str(&format!("  - {}\n", n));
    }

    out
}

/// Renders a PolicyExplanation as a stable JSON object string.
///
/// This is a manual serializer to keep full control over field order
/// and formatting.
pub fn format_explanation_json(expl: &PolicyExplanation) -> String {
    let mut out = String::new();
    out.push('{');

    // policy_name
    out.push_str("\"policy_name\":\"");
    out.push_str(&escape_str(&expl.policy_name));
    out.push_str("\",");

    // final_bias
    out.push_str("\"final_bias\":{");
    out.push_str(&format!(
        "\"risk_weight\":{:.6},\"latency_weight\":{:.6},\"stability_weight\":{:.6},\"memory_pressure_weight\":{:.6},\"offload_cost_weight\":{:.6}",
        expl.final_bias.risk_weight,
        expl.final_bias.latency_weight,
        expl.final_bias.stability_weight,
        expl.final_bias.memory_pressure_weight,
        expl.final_bias.offload_cost_weight,
    ));
    out.push('}');
    out.push(',');

    // signals
    out.push_str("\"considered_signals\":[");
    for (i, s) in expl.considered_signals.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('{');
        out.push_str("\"kind\":\"");
        out.push_str(&format!("{:?}", s.kind));
        out.push_str("\",\"score\":");
        out.push_str(&format!("{:.6}", s.score));
        out.push('}');
    }
    out.push(']');
    out.push(',');

    // preferences
    out.push_str("\"preferences\":[");
    for (i, p) in expl.preference_explanations.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('{');
        out.push_str("\"name\":\"");
        out.push_str(&escape_str(p.name));
        out.push_str("\",\"status\":\"");
        let status_str = match p.status {
            PreferenceStatus::Applied => "applied",
            PreferenceStatus::IgnoredDueToRisk => "ignored_due_to_risk",
            PreferenceStatus::Inactive => "inactive",
        };
        out.push_str(status_str);
        out.push_str("\"");
        out.push('}');
    }
    out.push(']');
    out.push(',');

    // notes
    out.push_str("\"notes\":[");
    for (i, n) in expl.notes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('"');
        out.push_str(&escape_str(n));
        out.push('"');
    }
    out.push(']');

    out.push('}');
    out
}

fn escape_str(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}
