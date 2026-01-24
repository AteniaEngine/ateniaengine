#![allow(dead_code)]

/// Severity of observed behavioral drift.
#[derive(Debug, Clone, PartialEq)]
pub enum DriftSeverity {
    Compatible,
    MinorDrift,
    CriticalDrift,
}

/// Structured report describing differences between baseline and current
/// behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct DriftReport {
    pub severity: DriftSeverity,
    pub differences: Vec<String>,
    pub change_fingerprint: String,
}

impl DriftReport {
    /// Stable JSON-like representation for audit and CI usage.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push('{');
        let sev = match self.severity {
            DriftSeverity::Compatible => "compatible",
            DriftSeverity::MinorDrift => "minor",
            DriftSeverity::CriticalDrift => "critical",
        };
        out.push_str(&format!("\"severity\":\"{}\",\"differences\":[", sev));
        for (i, d) in self.differences.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push('"');
            for ch in d.chars() {
                if ch == '"' {
                    out.push_str("\\\"");
                } else {
                    out.push(ch);
                }
            }
            out.push('"');
        }
        out.push(']');
        out.push_str(&format!(",\"fingerprint\":\"{}\"", self.change_fingerprint));
        out.push('}');
        out
    }
}
