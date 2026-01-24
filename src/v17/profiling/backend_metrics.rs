#![allow(dead_code)]

/// Backend kind used during execution.
#[derive(Debug, Clone, PartialEq)]
pub enum BackendKind {
    Cpu,
    Gpu,
}

/// Aggregated metrics per backend.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendMetrics {
    pub backend: BackendKind,
    pub matmul_count: usize,
    pub add_count: usize,
    pub relu_count: usize,
    pub elements_processed: usize,
}

impl BackendMetrics {
    pub fn new(backend: BackendKind) -> Self {
        Self {
            backend,
            matmul_count: 0,
            add_count: 0,
            relu_count: 0,
            elements_processed: 0,
        }
    }
}

/// Stable, serializable execution profile.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionProfile {
    pub steps: Vec<super::step_metrics::StepMetrics>,
    pub backends: Vec<BackendMetrics>,
}

impl ExecutionProfile {
    /// Produce a stable JSON-like representation without external dependencies.
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push('{');
        out.push_str("\"steps\":[");
        for (i, s) in self.steps.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"index\":{},\"kind\":\"{}\",\"backend\":\"{}\"}}",
                s.step_index, s.kind, s.backend
            ));
        }
        out.push(']');
        out.push_str(",\"backends\":[");
        for (i, b) in self.backends.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let bk = match b.backend {
                BackendKind::Cpu => "cpu",
                BackendKind::Gpu => "gpu",
            };
            out.push_str(&format!(
                "{{\"backend\":\"{}\",\"matmul\":{},\"add\":{},\"relu\":{}}}",
                bk, b.matmul_count, b.add_count, b.relu_count
            ));
        }
        out.push(']');
        out.push('}');
        out
    }
}
