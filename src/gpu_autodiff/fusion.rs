use crate::gpu_autodiff::ir_backward::BackwardKernelSpec;

#[derive(Clone, Debug)]
pub struct FusedKernelSpec {
    pub name: String,
    pub code: String,
    pub parts: Vec<BackwardKernelSpec>,
}

impl FusedKernelSpec {
    pub fn new(name: impl Into<String>, parts: Vec<BackwardKernelSpec>) -> Self {
        let mut code = String::new();
        for p in &parts {
            code.push_str("// ---- part ----\n");
            code.push_str(&p.code);
            code.push('\n');
        }

        Self {
            name: name.into(),
            code,
            parts,
        }
    }
}
