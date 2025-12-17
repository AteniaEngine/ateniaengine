use crate::cuda::cuda_available;

pub fn gpu_enabled() -> bool {
    cuda_available()
}

pub fn log_gpu(msg: &str) {
    if !crate::apx_is_silent() && std::env::var("APX_TRACE").is_ok() {
        eprintln!("[APX 4.3 GPU] {msg}");
    }
}
