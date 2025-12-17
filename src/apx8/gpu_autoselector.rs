// APX 8.14 — GPU Auto-Selector v0
// Selección simbólica de backend GPU basada en el nombre del KernelIR.

use crate::apx8::kernel_generator::KernelIR;

#[derive(Debug, Clone)]
pub struct GPUAutoSelector {
    pub vendors: Vec<String>,   // ["nvidia", "amd", "intel"]
    pub preferred: String,      // elegido final
}

impl GPUAutoSelector {
    /// Detección simulada, sin tocar hardware real.
    pub fn detect() -> Self {
        let vendors = vec!["nvidia".into(), "amd".into(), "intel".into()];
        let preferred = "nvidia".into(); // stub determinístico
        Self { vendors, preferred }
    }

    /// Selección de backend puramente textual basada en el nombre del IR.
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
