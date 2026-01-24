#![allow(dead_code)]

use super::capability_descriptor::CapabilityDescriptor;

/// Stable declaration of engine capabilities for APX 17.x.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineManifest {
    pub engine_version: String,
    pub enabled_backends: Vec<String>,
    pub supported_execution_modes: Vec<String>,
    pub profiling_level: String,
    pub snapshot_support: bool,
    pub consistency_guard_support: bool,
    pub learning_enabled: bool,
    pub capabilities: CapabilityDescriptor,
}

impl EngineManifest {
    /// Return a canonical APX 17 manifest describing the current engine.
    pub fn apx17_default() -> Self {
        Self {
            engine_version: "17.x".to_string(),
            enabled_backends: vec!["cpu".to_string(), "gpu".to_string()],
            supported_execution_modes: vec!["contracted".to_string(), "speculative".to_string()],
            profiling_level: "logical".to_string(),
            snapshot_support: true,
            consistency_guard_support: true,
            learning_enabled: false,
            capabilities: CapabilityDescriptor {
                supports_gpu_execution: true,
                supports_replay: true,
                supports_abortability: true,
                supports_determinism: true,
                supports_snapshot_sealing: true,
            },
        }
    }

    /// Stable JSON-like serialization without IO.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push('{');
        out.push_str(&format!("\"engine_version\":\"{}\",", self.engine_version));
        out.push_str("\"enabled_backends\":[");
        for (i, b) in self.enabled_backends.iter().enumerate() {
            if i > 0 { out.push(','); }
            out.push('"'); out.push_str(b); out.push('"');
        }
        out.push(']');
        out.push_str(",\"profiling_level\":\"");
        out.push_str(&self.profiling_level);
        out.push_str("\",\"snapshot_support\":");
        out.push_str(if self.snapshot_support {"true"} else {"false"});
        out.push_str(",\"consistency_guard_support\":");
        out.push_str(if self.consistency_guard_support {"true"} else {"false"});
        out.push_str(",\"learning_enabled\":");
        out.push_str(if self.learning_enabled {"true"} else {"false"});
        out.push('}');
        out
    }
}
