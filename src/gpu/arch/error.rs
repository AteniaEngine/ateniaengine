use std::fmt;

#[derive(Debug)]
pub enum ArchError {
    DriverNotFound,
    MissingSymbol(String),
    DetectionFailed,
}

impl fmt::Display for ArchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArchError::DriverNotFound => write!(f, "CUDA driver not found"),
            ArchError::MissingSymbol(s) => write!(f, "Missing CUDA symbol: {}", s),
            ArchError::DetectionFailed => write!(f, "Failed to detect CUDA compute capability"),
        }
    }
}

impl std::error::Error for ArchError {}
