use std::fs;

use super::compression::{rle_compress, rle_decompress};
use super::memory_types::{CompressionKind, CompressionMeta, MoveError};

#[derive(Debug, Clone)]
pub struct SsdCache {
    dir: String,
}

impl SsdCache {
    pub fn new(dir: &str) -> Self {
        SsdCache {
            dir: dir.to_string(),
        }
    }

    pub fn ensure_dir(&self) -> Result<(), MoveError> {
        if self.dir.is_empty() {
            return Err(MoveError::Unsupported(
                "SSD cache directory path is empty".to_string(),
            ));
        }

        if let Err(e) = fs::create_dir_all(&self.dir) {
            return Err(MoveError::IoError(format!(
                "Failed to create SSD cache directory '{}': {}",
                self.dir, e
            )));
        }

        Ok(())
    }

    pub fn dir(&self) -> &str {
        &self.dir
    }

    pub fn blob_path(&self, tensor_id: &str) -> String {
        let mut sanitized = String::with_capacity(tensor_id.len());
        for ch in tensor_id.chars() {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                sanitized.push(ch);
            } else {
                sanitized.push('_');
            }
        }
        format!("{}/tensor_{}.bin", self.dir, sanitized)
    }

    pub fn write_blob(
        &self,
        path: &str,
        data: &[u8],
        compression: CompressionKind,
    ) -> Result<CompressionMeta, MoveError> {
        let (to_write, meta) = match compression {
            CompressionKind::None => {
                let meta = CompressionMeta {
                    kind: CompressionKind::None,
                    original_bytes: data.len() as u64,
                };
                (data.to_vec(), meta)
            }
            CompressionKind::Rle => rle_compress(data),
        };

        if let Err(e) = fs::write(path, &to_write) {
            return Err(MoveError::IoError(format!(
                "Failed to write SSD blob '{}': {}",
                path, e
            )));
        }

        Ok(meta)
    }

    pub fn read_blob(&self, path: &str) -> Result<Vec<u8>, MoveError> {
        match fs::read(path) {
            Ok(bytes) => Ok(bytes),
            Err(e) => Err(MoveError::IoError(format!(
                "Failed to read SSD blob '{}': {}",
                path, e
            ))),
        }
    }

    pub fn read_blob_with_meta(
        &self,
        path: &str,
        meta: &CompressionMeta,
    ) -> Result<Vec<u8>, MoveError> {
        let bytes = self.read_blob(path)?;
        match meta.kind {
            CompressionKind::None => {
                if bytes.len() as u64 != meta.original_bytes {
                    return Err(MoveError::Unsupported(
                        "Invalid SSD blob: length mismatch for uncompressed data".to_string(),
                    ));
                }
                Ok(bytes)
            }
            CompressionKind::Rle => rle_decompress(&bytes, meta),
        }
    }

    pub fn delete_blob(&self, path: &str) -> Result<(), MoveError> {
        match fs::remove_file(path) {
            Ok(_) => Ok(()),
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Ok(())
                } else {
                    Err(MoveError::IoError(format!(
                        "Failed to delete SSD blob '{}': {}",
                        path, e
                    )))
                }
            }
        }
    }
}
