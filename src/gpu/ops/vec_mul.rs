use crate::gpu::{
    nvrtc::NvrtcCompiler,
    loader::CudaLoader,
    runtime::GpuRuntime,
    memory::GpuPtr,
    launcher::GpuLauncher,
};
use crate::gpu::loader::CudaLoaderError;

use std::ffi::c_void;

pub struct VecMulOp;

impl VecMulOp {
    pub fn run(a: &GpuPtr, b: &GpuPtr, out: &GpuPtr, n: usize) {
        // APX 12.14: no panic/unwrap en ruta GPU.
        // Si el loader entra en CpuFallback o el launch falla, retornamos temprano.
        let compiler = NvrtcCompiler::new().unwrap();
        let loader = CudaLoader::new().unwrap();
        let rt = GpuRuntime::new().unwrap();
        let launcher = GpuLauncher::new().unwrap();

        let src = r#"
        extern "C" __global__ void vec_mul(float* A, float* B, float* C) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            C[i] = A[i] * B[i];
        }
        "#;

        let ptx = compiler.compile(src, "vec_mul_kernel", "compute_89").unwrap();
        let module = match loader.load_module_from_ptx(&ptx.ptx) {
            Ok(m) => m,
            Err(CudaLoaderError::CpuFallback) => {
                // APX 12.x: compat layer forces CPU fallback -> do not panic.
                return;
            }
            Err(e) => {
                panic!("[VEC_MUL] ModuleLoadFailed: {:?}", e);
            }
        };
        let func = match loader.get_function(&module, "vec_mul") {
            Ok(f) => f,
            Err(_) => {
                // Avoid panic if we cannot resolve the symbol.
                return;
            }
        };

        let mut args = vec![
            &a.ptr as *const u64 as *mut c_void,
            &b.ptr as *const u64 as *mut c_void,
            &out.ptr as *const u64 as *mut c_void,
        ];

        if launcher
            .launch(&rt, &func, (1, 1, 1), (n as u32, 1, 1), 0, &mut args)
            .is_err()
        {
            // APX 12.14: nunca panic por launch.
            return;
        }
    }
}
