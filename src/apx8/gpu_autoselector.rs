// APX 8.14 — GPU Auto-Selector v0
// Symbolic GPU backend selection based on KernelIR name.

use crate::apx8::kernel_generator::KernelIR;

#[derive(Debug, Clone)]
pub struct GPUAutoSelector {
    pub vendors: Vec<String>,   // ["nvidia", "amd", "intel"]
    pub preferred: String,      // final pick
}

impl GPUAutoSelector {
    /// Simulated detection without touching real hardware.
    pub fn detect() -> Self {
        let vendors = vec!["nvidia".into(), "amd".into(), "intel".into()];
        let preferred = "nvidia".into(); // deterministic stub
        Self { vendors, preferred }
    }

    /// Purely textual backend selection based on IR name.
    pub fn choose_backend(&self, ir: &KernelIR) -> String {
        if ir.name.contains("matmul") {
            "cuda".into()
        } else if ir.name.contains("vec") {
            "hip".into()
        } else {
            self.preferred.clone()
        }
    }
}
