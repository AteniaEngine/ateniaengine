use crate::gpu::tensor::{TensorGPU, GpuTensorManager};
use crate::gpu::nvrtc::compiler::NvrtcCompiler;
use crate::gpu::loader::CudaLoader;
use crate::gpu::launcher::GpuLauncher;
use crate::gpu::runtime::GpuRuntime;

pub struct LinearBackwardGPU;

impl LinearBackwardGPU {
    /// APX 11.8 — backward real de Linear en GPU:
    /// dX = dOut * W^T
    /// dW = X^T * dOut
    /// dB = sum_i dOut[i, :]
    pub fn run(
        mgr: &GpuTensorManager,
        x: &TensorGPU,      // [M, K]
        w: &TensorGPU,      // [K, N]
        dout: &TensorGPU,   // [M, N]
    ) -> Result<(TensorGPU, TensorGPU, TensorGPU), ()> {
        let m = x.rows as i32;
        let k = x.cols as i32;
        let n = w.cols as i32;

        // dX [M, K], dW [K, N], dB [1, N] (bias como fila)
        let d_x = TensorGPU::empty(&mgr.mem, x.rows, x.cols)?;
        let d_w = TensorGPU::empty(&mgr.mem, w.rows, w.cols)?;
        let d_b = TensorGPU::empty(&mgr.mem, 1, w.cols)?;

        let kernel_code = r#"
        extern \"C\" __global__
        void linear_backward_real(
            const float* __restrict__ X,    // [M,K]
            const float* __restrict__ W,    // [K,N]
            const float* __restrict__ dOut, // [M,N]
            float* __restrict__ dX,         // [M,K]
            float* __restrict__ dW,         // [K,N]
            float* __restrict__ dB,         // [N]
            int M, int K, int N
        ) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            // dX = dOut * W^T   => [M,K]
            if (row < M && col < K) {
                float val = 0.0f;
                for (int j = 0; j < N; ++j) {
                    val += dOut[row * N + j] * W[col * N + j];
                }
                dX[row * K + col] = val;
            }

            // dW = X^T * dOut   => [K,N]
            if (row < K && col < N) {
                float val = 0.0f;
                for (int i = 0; i < M; ++i) {
                    val += X[i * K + row] * dOut[i * N + col];
                }
                dW[row * N + col] = val;
            }

            // dB = sum_i dOut[i, col] => [N]
            if (row == 0 && col < N) {
                float val = 0.0f;
                for (int i = 0; i < M; ++i) {
                    val += dOut[i * N + col];
                }
                dB[col] = val;
            }
        }
        "#;

        // NVRTC compile (same pattern as MatMulBackwardGPU)
        let compiler = NvrtcCompiler::new().map_err(|_| ())?;
        let program = compiler
            .compile(kernel_code, "linear_backward_real", "auto")
            .map_err(|_| ())?;

        let loader = CudaLoader::new().map_err(|_| ())?;
        let module = loader.load_module_from_ptx(&program.ptx).map_err(|_| ())?;
        let func = loader
            .get_function(&module, "linear_backward_real")
            .map_err(|_| ())?;

        let rt = GpuRuntime::new().map_err(|_| ())?;
        let launcher = GpuLauncher::new().map_err(|_| ())?;

        let block = (16u32, 16u32, 1u32);
        let max_dim = m.max(k).max(n) as u32;
        let grid_xy = (max_dim + block.0 - 1) / block.0;
        let grid = (grid_xy, grid_xy, 1u32);

        let mut args: [*mut core::ffi::c_void; 9] = [
            x.ptr.ptr as *mut core::ffi::c_void,
            w.ptr.ptr as *mut core::ffi::c_void,
            dout.ptr.ptr as *mut core::ffi::c_void,
            d_x.ptr.ptr as *mut core::ffi::c_void,
            d_w.ptr.ptr as *mut core::ffi::c_void,
            d_b.ptr.ptr as *mut core::ffi::c_void,
            &m as *const _ as *mut core::ffi::c_void,
            &k as *const _ as *mut core::ffi::c_void,
            &n as *const _ as *mut core::ffi::c_void,
        ];

        launcher
            .launch(&rt, &func, grid, block, 0, &mut args)
            .map_err(|_| ())?;

        Ok((d_x, d_w, d_b))
    }
}
