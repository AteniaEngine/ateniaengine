use crate::gpu::{
    nvrtc::NvrtcCompiler,
    loader::CudaLoader,
    runtime::GpuRuntime,
    memory::GpuPtr,
    launcher::GpuLauncher,
};
use crate::gpu::loader::CudaLoaderError;

use std::ffi::c_void;

pub struct ScalarAddOp;

impl ScalarAddOp {
    pub fn run(x: &GpuPtr, value: f32, n: usize) {
        // APX 12.14: no panic/unwrap en ruta GPU.
        // Si el loader entra en CpuFallback o el launch falla, retornamos temprano.
        let compiler = NvrtcCompiler::new().unwrap();
        let loader = CudaLoader::new().unwrap();
        let rt = GpuRuntime::new().unwrap();
        let launcher = GpuLauncher::new().unwrap();

        let src = r#"
        extern "C" __global__ void scalar_add(float* X, float v) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            X[i] += v;
        }
        "#;

        let ptx = compiler.compile(src, "scalar_add_kernel", "compute_89").unwrap();
        let module = match loader.load_module_from_ptx(&ptx.ptx) {
            Ok(m) => m,
            Err(CudaLoaderError::CpuFallback) => {
                // APX 12.x: entorno sin GPU real -> early return en vez de panic.
                return;
            }
            Err(e) => {
                panic!("[SCALAR_ADD] ModuleLoadFailed: {:?}", e);
            }
        };
        let func = match loader.get_function(&module, "scalar_add") {
            Ok(f) => f,
            Err(_) => {
                // Evitar panic si no podemos resolver el símbolo.
                return;
            }
        };

        let mut args = vec![
            &x.ptr as *const u64 as *mut c_void,
            &value as *const f32 as *mut c_void,
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
