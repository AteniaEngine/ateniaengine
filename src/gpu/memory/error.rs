use std::fmt;

#[derive(Debug)]
pub enum GpuMemoryError {
    DriverLoadFailed,
    MissingSymbol(String),
    AllocationFailed,
    CopyFailed,
    FreeFailed,
}

impl fmt::Display for GpuMemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuMemoryError::DriverLoadFailed =>
                write!(f, "Failed to load CUDA driver"),
            GpuMemoryError::MissingSymbol(s) =>
                write!(f, "Missing CUDA symbol: {}", s),
            GpuMemoryError::AllocationFailed =>
                write!(f, "GPU memory allocation failed"),
            GpuMemoryError::CopyFailed =>
                write!(f, "GPU memory copy failed"),
            GpuMemoryError::FreeFailed =>
                write!(f, "GPU free failed"),
        }
    }
}

impl std::error::Error for GpuMemoryError {}
