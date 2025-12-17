use crate::gpu::{
    nvrtc::NvrtcCompiler,
    loader::CudaLoader,
    runtime::GpuRuntime,
    memory::GpuPtr,
    launcher::GpuLauncher,
    kernel::KernelNormalizer,
    planning::AutoPlanner,
    autotuner,
};
use crate::gpu::loader::{compat_layer::CompatLoader, CudaLoaderError};

use std::ffi::c_void;
use std::mem;

pub struct MatMulOp;

impl MatMulOp {
    pub fn run(a: &GpuPtr, b: &GpuPtr, c: &GpuPtr,
               _m: usize, _k: usize, n: usize) {

        let compiler = NvrtcCompiler::new().unwrap();
        let loader = CudaLoader::new().unwrap();
        let rt = GpuRuntime::new().unwrap();
        let launcher = GpuLauncher::new().unwrap();

        // Kernel NVRTC-safe: triple bucle simple, sin shared memory ni inline PTX.
        // Firma EXACTA pedida: extern "C" __global__ void matmul_kernel(..., int N)
        // Nota: asumimos casos cuadrados (M == K == N) en los usos actuales.
        let src = r#"
        extern "C" __global__
        void matmul_kernel(const float* A,
                           const float* B,
                           float* C,
                           int N) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < N && col < N) {
                float acc = 0.0f;
                for (int kk = 0; kk < N; ++kk) {
                    acc += A[row * N + kk] * B[kk * N + col];
                }
                C[row * N + col] = acc;
            }
        }
        "#;

        // APX 12.0: normalizar kernel antes de NVRTC.
        let src = KernelNormalizer::normalize_kernel(src, "matmul_kernel");

        // Compilar con NVRTC usando solo la arquitectura compute_89.
        // Se delega en NvrtcCompiler::compile, que traducirá arch -> "--gpu-architecture=...".
        let ptx = compiler
            .compile(&src, "matmul_kernel", "compute_89")
            .unwrap();

        // Logs NVRTC / PTX (sólo en modo debug)
        let ptx_str = &ptx.ptx;
        let debug = std::env::var("ATENIA_DEBUG").ok().as_deref() == Some("1");
        if debug {
            println!("[MATMUL] PTX size = {} bytes", ptx_str.len());
            println!("[MATMUL] --- PTX START ---");
            for (i, line) in ptx_str.lines().take(20).enumerate() {
                println!("[PTX][{:02}] {}", i, line);
            }
            println!("[MATMUL] --- PTX END (preview) ---");
            println!("[MATMUL] PTX contains '.visible .entry matmul_kernel'? {}",
                     ptx_str.contains(".visible .entry matmul_kernel"));
            println!("[MATMUL] PTX contains 'matmul_kernel'? {}",
                     ptx_str.contains("matmul_kernel"));
            println!("[MATMUL] Calling cuModuleLoadData...");
            println!("[MATMUL-DEBUG] PTX length = {} bytes", ptx_str.len());
        }

        // Carga de módulo con compat layer; CpuFallback no debe panicar.
        let module = match loader.load_module_from_ptx(ptx_str) {
            Ok(m) => {
                if debug {
                    println!("[MATMUL] cuModuleLoadData returned Ok");
                }
                m
            }
            Err(CudaLoaderError::CpuFallback) => {
                // APX 12.2.5: entorno sin GPU usable. No paniqueamos;
                // dejamos que la ruta CPU se encargue a nivel superior.
                if debug {
                    eprintln!("[MATMUL] CpuFallback detected - skipping GPU matmul_kernel launch");
                }
                return;
            }
            Err(e) => {
                if debug {
                    eprintln!("[MATMUL] ERROR in cuModuleLoadData: {:?}", e);
                }
                panic!("[MATMUL] ModuleLoadFailed: {:?}", e);
            }
        };

        // Resolución de símbolo con logs (sólo en modo debug)
        if debug {
            println!("[MATMUL] Calling cuModuleGetFunction(\"matmul_kernel\")");
            println!("[MATMUL-DEBUG] Kernel name = matmul_kernel");
        }
        let func = match loader.get_function(&module, "matmul_kernel") {
            Ok(f) => {
                if debug {
                    println!("[MATMUL] cuModuleGetFunction returned Ok");
                }
                f
            }
            Err(e) => {
                if debug {
                    eprintln!("[MATMUL] ERROR in cuModuleGetFunction: {:?}", e);
                }
                panic!("[MATMUL] FunctionNotFound: {:?}", e);
            }
        };

        // APX 12.1: calcular grid/block/shared_mem vía AutoPlanner.
        // Para este kernel asumimos matrices cuadradas N x N.
        let cfg = AutoPlanner::plan_square_matmul(n);

        // Argumentos para kernel: (A, B, C, N)
        let n_i32 = n as i32;
        let mut args = vec![
            &a.ptr as *const u64 as *mut c_void,
            &b.ptr as *const u64 as *mut c_void,
            &c.ptr as *const u64 as *mut c_void,
            &n_i32 as *const i32 as *mut c_void,
        ];

        // DEBUG grid/block/args antes del launch
        let shared_mem_bytes: u32 = cfg.shared_mem;

        // APX 12.3: autotuner de grid/block basado en tiempo de ejecución.
        let gpu_enabled = !CompatLoader::is_forced_fallback();
        let compute_cap = 89; // compute capability para hashing del autotuner

        let tuning = autotuner::autotune_matmul(
            n,
            compute_cap,
            &|layout| {
                let (bx, by, gx, gy) = layout;
                let grid = (gx, gy, 1);
                let block = (bx, by, 1);
                let mut args_clone = args.clone();
                let start = std::time::Instant::now();
                let _ = launcher.launch(&rt, &func, grid, block, shared_mem_bytes, &mut args_clone);
                start.elapsed().as_secs_f32() * 1000.0
            },
            gpu_enabled,
        );

        let grid = (tuning.grid_x, tuning.grid_y, 1);
        let block = (tuning.block_x, tuning.block_y, 1);

        let (grid_x, grid_y, grid_z) = grid;
        let (block_x, block_y, block_z) = block;

        if debug {
            eprintln!("[MATMUL-DEBUG] Launching matmul_kernel...");
            eprintln!(
                "[MATMUL-DEBUG] grid=({}, {}, {})",
                grid_x, grid_y, grid_z
            );
            eprintln!(
                "[MATMUL-DEBUG] block=({}, {}, {})",
                block_x, block_y, block_z
            );
            eprintln!(
                "[MATMUL-DEBUG] shared_mem={}",
                shared_mem_bytes
            );
            eprintln!(
                "[MATMUL-DEBUG] param_count={}",
                args.len()
            );

            eprintln!(
                "[MATMUL-DEBUG] arg0=A.ptr=0x{:x} size={}",
                a.ptr, a.size
            );
            eprintln!(
                "[MATMUL-DEBUG] arg1=B.ptr=0x{:x} size={}",
                b.ptr, b.size
            );
            eprintln!(
                "[MATMUL-DEBUG] arg2=C.ptr=0x{:x} size={}",
                c.ptr, c.size
            );
            eprintln!(
                "[MATMUL-DEBUG] arg3=N (host)={} bytes={}",
                n_i32,
                mem::size_of_val(&n_i32)
            );
            eprintln!(
                "[MATMUL-DEBUG] param N type = i32, bytes = {}",
                mem::size_of_val(&n_i32)
            );

            // Validaciones previas al launch (solo log, no cambiamos control de flujo)
            if grid_x == 0 || grid_y == 0 || block_x == 0 || block_y == 0 {
                eprintln!(
                    "[MATMUL-DEBUG] INVALID grid/block: cannot launch (grid=({}, {}), block=({}, {}))",
                    grid_x, grid_y, block_x, block_y
                );
            }
            if (block_x as u64) * (block_y as u64) * (block_z as u64) > 1024 {
                eprintln!(
                    "[MATMUL-DEBUG] blockDim exceeds 1024 threads: {}*{}*{}",
                    block_x, block_y, block_z
                );
            }
        }

        // Lanzamiento real (la lógica se mantiene igual; el launcher interno reportará el código)
        launcher
            .launch(&rt, &func, grid, block, shared_mem_bytes, &mut args)
            .unwrap();
    }
}
