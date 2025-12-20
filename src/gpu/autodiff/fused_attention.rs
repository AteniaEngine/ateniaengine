use crate::gpu::tensor::{TensorGPU, GpuTensorManager};
use crate::gpu::nvrtc::compiler::NvrtcCompiler;
use crate::gpu::loader::CudaLoader;
use crate::gpu::launcher::GpuLauncher;
use crate::gpu::runtime::GpuRuntime;

pub struct FusedAttentionGPU;

impl FusedAttentionGPU {
    /// APX 11.10 — Full fused attention kernel (minimal version):
    /// out = softmax(Q*K^T / sqrt(d)) * V
    pub fn run(
        mgr: &GpuTensorManager,
        q: &TensorGPU,
        k: &TensorGPU,
        v: &TensorGPU,
    ) -> Result<TensorGPU, ()> {
        let m = q.rows as i32;
        let d = q.cols as i32;

        // output: [M, D]
        let out = TensorGPU::empty(&mgr.mem, q.rows, q.cols)?;

        let kernel = r#"
        extern \"C\" __global__
        void fused_attention(
            const float* __restrict__ Q, // [M,D]
            const float* __restrict__ K, // [M,D]
            const float* __restrict__ V, // [M,D]
            float* __restrict__ OUT,     // [M,D]
            int M, int D
        ) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = threadIdx.x;

            if (row >= M) return;

            // Find max for numerical stability
            float max_val = -1e30f;
            for (int j = 0; j < M; ++j) {
                float dot = 0.0f;
                for (int i = 0; i < D; ++i)
                    dot += Q[row*D + i] * K[j*D + i];

                dot /= sqrtf((float)D);
                if (dot > max_val) max_val = dot;
            }

            // Compute sum of exp
            float sum_exp = 0.0f;
            for (int j = 0; j < M; ++j) {
                float dot = 0.0f;
                for (int i = 0; i < D; ++i)
                    dot += Q[row*D + i] * K[j*D + i];

                dot /= sqrtf((float)D);
                sum_exp += __expf(dot - max_val);
            }

            // softmax and output accumulation
            if (col < D) {
                float out_val = 0.0f;

                for (int j = 0; j < M; ++j) {
                    float dot = 0.0f;
                    for (int i = 0; i < D; ++i)
                        dot += Q[row*D + i] * K[j*D + i];

                    dot /= sqrtf((float)D);
                    float w = __expf(dot - max_val) / sum_exp;

                    out_val += w * V[j*D + col];
                }

                OUT[row*D + col] = out_val;
            }
        }
        "#;

        // NVRTC compile
        let compiler = NvrtcCompiler::new().map_err(|_| ())?;
        let prog = compiler
            .compile(kernel, "fused_attention", "auto")
            .map_err(|_| ())?;

        // Load
        let loader = CudaLoader::new().map_err(|_| ())?;
        let module = loader
            .load_module_from_ptx(&prog.ptx)
            .map_err(|_| ())?;
        let func = loader
            .get_function(&module, "fused_attention")
            .map_err(|_| ())?;

        // Runtime + launcher
        let rt = GpuRuntime::new().map_err(|_| ())?;
        let launcher = GpuLauncher::new().map_err(|_| ())?;

        // Launch config: 1 bloque por fila, D hilos en x (capado a 32)
        let block = (32u32, 1u32, 1u32);
        let grid = (1u32, m as u32, 1u32);

        let mut args: [*mut core::ffi::c_void; 6] = [
            q.ptr.ptr as *mut core::ffi::c_void,
            k.ptr.ptr as *mut core::ffi::c_void,
            v.ptr.ptr as *mut core::ffi::c_void,
            out.ptr.ptr as *mut core::ffi::c_void,
            &m as *const _ as *mut core::ffi::c_void,
            &d as *const _ as *mut core::ffi::c_void,
        ];

        launcher
            .launch(&rt, &func, grid, block, 0, &mut args)
            .map_err(|_| ())?;

        Ok(out)
    }
}
