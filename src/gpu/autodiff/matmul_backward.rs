use crate::gpu::tensor::{TensorGPU, GpuTensorManager};
use crate::gpu::nvrtc::compiler::NvrtcCompiler;
use crate::gpu::loader::CudaLoader;
use crate::gpu::launcher::GpuLauncher;
use crate::gpu::runtime::GpuRuntime;

pub struct MatMulBackwardGPU;

impl MatMulBackwardGPU {
    pub fn run(
        mgr: &GpuTensorManager,
        a: &TensorGPU,        // [M, K]
        b: &TensorGPU,        // [K, N]
        dout: &TensorGPU,     // [M, N]
    ) -> Result<(TensorGPU, TensorGPU), ()> {
        // shapes
        let m = a.rows as i32;
        let k = a.cols as i32;
        let n = b.cols as i32;

        // allocate outputs: dA [M, K] and dB [K, N]
        let d_a = TensorGPU::empty(&mgr.mem, a.rows, a.cols)?;
        let d_b = TensorGPU::empty(&mgr.mem, b.rows, b.cols)?;

        // CUDA kernel real (simple, no tiling for now)
        let kernel_code = r#"
        extern \"C\" __global__
        void matmul_backward_real(
            const float* __restrict__ A,
            const float* __restrict__ B,
            const float* __restrict__ dOut,
            float* __restrict__ dA,
            float* __restrict__ dB,
            int M, int K, int N
        ) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            // dA = dOut * B^T
            if (row < M && col < K) {
                float val = 0.0f;
                for (int j = 0; j < N; ++j) {
                    val += dOut[row * N + j] * B[col * N + j];
                }
                dA[row * K + col] = val;
            }

            // dB = A^T * dOut
            if (row < K && col < N) {
                float val = 0.0f;
                for (int i = 0; i < M; ++i) {
                    val += A[i * K + row] * dOut[i * N + col];
                }
                dB[row * N + col] = val;
            }
        }
        "#;

        // NVRTC compile with arch="auto"
        let compiler = NvrtcCompiler::new().map_err(|_| ())?;
        let program = compiler
            .compile(kernel_code, "matmul_backward_real", "auto")
            .map_err(|_| ())?;

        // load module + kernel symbol
        let loader = CudaLoader::new().map_err(|_| ())?;
        let module = loader.load_module_from_ptx(&program.ptx).map_err(|_| ())?;
        let func = loader.get_function(&module, "matmul_backward_real").map_err(|_| ())?;

        // runtime + launcher
        let rt = GpuRuntime::new().map_err(|_| ())?;
        let launcher = GpuLauncher::new().map_err(|_| ())?;

        // launch config (square over max(M,K,N))
        let block = (16u32, 16u32, 1u32);
        let max_dim = m.max(k).max(n) as u32;
        let grid_xy = (max_dim + block.0 - 1) / block.0;
        let grid = (grid_xy, grid_xy, 1u32);

        // build args as *mut c_void (8 parameters)
        let mut args: [*mut core::ffi::c_void; 8] = [
            a.ptr.ptr as *mut core::ffi::c_void,
            b.ptr.ptr as *mut core::ffi::c_void,
            dout.ptr.ptr as *mut core::ffi::c_void,
            d_a.ptr.ptr as *mut core::ffi::c_void,
            d_b.ptr.ptr as *mut core::ffi::c_void,
            &m as *const _ as *mut core::ffi::c_void,
            &k as *const _ as *mut core::ffi::c_void,
            &n as *const _ as *mut core::ffi::c_void,
        ];

        launcher
            .launch(&rt, &func, grid, block, 0, &mut args)
            .map_err(|_| ())?;

        Ok((d_a, d_b))
    }
}
