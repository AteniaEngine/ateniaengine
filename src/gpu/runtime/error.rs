use std::fmt;

#[derive(Debug)]
pub enum GpuRuntimeError {
    DriverNotFound,
    MissingSymbol(String),
    InitFailed,
    StreamCreateFailed,
}

impl fmt::Display for GpuRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuRuntimeError::DriverNotFound => write!(f, "CUDA driver not found"),
            GpuRuntimeError::MissingSymbol(s) =>
                write!(f, "CUDA symbol missing: {}", s),
            GpuRuntimeError::InitFailed =>
                write!(f, "Failed to initialize CUDA runtime"),
            GpuRuntimeError::StreamCreateFailed =>
                write!(f, "Failed to create CUDA stream"),
        }
    }
}

impl std::error::Error for GpuRuntimeError {}
