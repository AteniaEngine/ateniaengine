use std::fmt;

#[derive(Debug)]
pub enum CudaLoaderError {
    LoadError(String),
    MissingSymbol(String),
    ModuleLoadFailed,
    FunctionNotFound(String),
    CpuFallback,
}

impl fmt::Display for CudaLoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CudaLoaderError::LoadError(e) => write!(f, "Failed to load CUDA driver: {}", e),
            CudaLoaderError::MissingSymbol(s) => write!(f, "CUDA driver missing symbol: {}", s),
            CudaLoaderError::ModuleLoadFailed => write!(f, "cuModuleLoadData failed"),
            CudaLoaderError::FunctionNotFound(k) => write!(f, "Kernel not found: {}", k),
            CudaLoaderError::CpuFallback => write!(f, "CPU fallback activated"),
        }
    }
}

impl std::error::Error for CudaLoaderError {}
