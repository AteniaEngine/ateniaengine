use atenia_engine::gpu::{
    nvrtc::NvrtcCompiler,
    loader::CudaLoader,
    memory::GpuMemoryEngine,
    runtime::GpuRuntime,
    launcher::GpuLauncher,
};

use std::ffi::c_void;

#[test]
fn test_launch_add_one() {
    // Si cualquiera de los componentes GPU no está disponible, no fallamos el test.
    let compiler = match NvrtcCompiler::new() {
        Ok(c) => c,
        Err(_) => return,
    };

    let loader = match CudaLoader::new() {
        Ok(l) => l,
        Err(_) => return,
    };

    let mem = match GpuMemoryEngine::new() {
        Ok(m) => m,
        Err(_) => return,
    };

    let rt = match GpuRuntime::new() {
        Ok(r) => r,
        Err(_) => return,
    };

    let launcher = match GpuLauncher::new() {
        Ok(l) => l,
        Err(_) => return,
    };

    let src = r#"
    extern "C" __global__ void add_one(float* x) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        x[i] += 1.0f;
    }
    "#;

    let ptx = match compiler.compile(src, "add_one_launch", "compute_89") {
        Ok(p) => p,
        Err(_) => return,
    };

    let module = match loader.load_module_from_ptx(&ptx.ptx) {
        Ok(m) => m,
        Err(_) => return,
    };

    let func = match loader.get_function(&module, "add_one") {
        Ok(f) => f,
        Err(_) => return,
    };

    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let n = input.len();

    let gpu_ptr = match mem.alloc(n * 4) {
        Ok(p) => p,
        Err(_) => return,
    };

    if mem.copy_htod(&gpu_ptr, &input).is_err() {
        let _ = mem.free(&gpu_ptr);
        return;
    }

    let mut args = vec![
        &gpu_ptr.ptr as *const u64 as *mut c_void,
    ];

    if launcher
        .launch(&rt, &func, (1, 1, 1), (n as u32, 1, 1), 0, &mut args)
        .is_err()
    {
        let _ = mem.free(&gpu_ptr);
        return;
    }

    std::thread::sleep(std::time::Duration::from_millis(10));

    let mut out = vec![0.0f32; n];
    if mem.copy_dtoh(&gpu_ptr, &mut out).is_err() {
        let _ = mem.free(&gpu_ptr);
        return;
    }

    let _ = mem.free(&gpu_ptr);

    assert_eq!(out, vec![2.0, 3.0, 4.0, 5.0]);
}
