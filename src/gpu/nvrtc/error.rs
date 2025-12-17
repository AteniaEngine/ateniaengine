use std::fmt;

#[derive(Debug)]
pub enum NvrtcError {
    LoadError(&'static str),
    CompilationError(String),
    MissingSymbol(String),
    IoError(std::io::Error),
}

impl From<std::io::Error> for NvrtcError {
    fn from(e: std::io::Error) -> Self {
        NvrtcError::IoError(e)
    }
}

impl fmt::Display for NvrtcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NvrtcError::LoadError(msg) => write!(f, "NVRTC load error: {}", msg),
            NvrtcError::CompilationError(msg) => write!(f, "NVRTC compilation failed:\n{}", msg),
            NvrtcError::MissingSymbol(s) => write!(f, "Missing NVRTC symbol: {}", s),
            NvrtcError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for NvrtcError {}
