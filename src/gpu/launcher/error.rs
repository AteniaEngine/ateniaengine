use std::fmt;

#[derive(Debug)]
pub enum LaunchError {
    MissingSymbol(String),
    LaunchFailed(i32),
}

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LaunchError::MissingSymbol(s) =>
                write!(f, "Missing CUDA symbol: {}", s),
            LaunchError::LaunchFailed(code) =>
                write!(f, "Kernel launch failed with code {}", code),
        }
    }
}

impl std::error::Error for LaunchError {}
