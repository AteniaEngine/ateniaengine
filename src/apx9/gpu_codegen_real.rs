// APX 9.10 — Real GPU Codegen v0 (estructura únicamente)
// Genera firmas y código PTX/OpenCL como strings inertes.

#[derive(Debug, Clone)]
pub struct RealKernelSignature {
    pub name: String,
    pub params: Vec<String>,
    pub target: KernelTarget,
}

#[derive(Debug, Clone)]
pub enum KernelTarget {
    PTX,
    OpenCL,
}

#[derive(Debug, Clone)]
pub struct RealKernel {
    pub signature: RealKernelSignature,
    pub code: String,
}

pub struct RealKernelBuilder;

impl RealKernelBuilder {
    pub fn new_ptx(name: &str, params: &[&str]) -> RealKernel {
        let header = ".version 7.0\n.target sm_75\n.address_size 64\n\n";
        let mut body = format!(".entry {}(\n", name);
        for (i, p) in params.iter().enumerate() {
            body.push_str(&format!("    .param .u64 param{}{},\n", i, p));
        }
        body.push_str(") {\n    // body to be filled by APX 9.11–9.13\n    ret;\n}\n");

        RealKernel {
            signature: RealKernelSignature {
                name: name.to_string(),
                params: params.iter().map(|s| s.to_string()).collect(),
                target: KernelTarget::PTX,
            },
            code: format!("{}{}", header, body),
        }
    }

    pub fn new_cl(name: &str, params: &[&str]) -> RealKernel {
        let mut code = format!("__kernel void {}(", name);
        let mut ps = vec![];
        for p in params {
            ps.push(format!("__global float* {}", p));
        }
        code.push_str(&ps.join(", "));
        code.push_str(") {\n    // body to be filled later\n}\n");

        RealKernel {
            signature: RealKernelSignature {
                name: name.to_string(),
                params: params.iter().map(|s| s.to_string()).collect(),
                target: KernelTarget::OpenCL,
            },
            code,
        }
    }
}
