use crate::v13::memory_types::MoveError;

pub trait VramAdapter {
    fn is_available(&self) -> bool;
    fn upload(&self, id: &str, data: &[u8]) -> Result<(), MoveError>;
    fn download(&self, id: &str) -> Result<Vec<u8>, MoveError>;
    fn free(&self, id: &str) -> Result<(), MoveError>;
}

#[derive(Debug, Clone, Copy)]
pub struct NullVramAdapter;

impl VramAdapter for NullVramAdapter {
    fn is_available(&self) -> bool {
        false
    }

    fn upload(&self, _id: &str, _data: &[u8]) -> Result<(), MoveError> {
        Err(MoveError::BackendUnavailable(
            "VRAM adapter not available".to_string(),
        ))
    }

    fn download(&self, _id: &str) -> Result<Vec<u8>, MoveError> {
        Err(MoveError::BackendUnavailable(
            "VRAM adapter not available".to_string(),
        ))
    }

    fn free(&self, _id: &str) -> Result<(), MoveError> {
        Err(MoveError::BackendUnavailable(
            "VRAM adapter not available".to_string(),
        ))
    }
}
