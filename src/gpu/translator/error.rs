use std::fmt;

#[derive(Debug)]
pub enum TranslationError {
    EmptyInput,
    InvalidPtx,
}

impl fmt::Display for TranslationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TranslationError::EmptyInput => write!(f, "PTX input is empty"),
            TranslationError::InvalidPtx => write!(f, "Invalid PTX input"),
        }
    }
}

impl std::error::Error for TranslationError {}
