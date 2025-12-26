use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

// PersistentHybridCache is intentionally decoupled from higher-level memory
// types to keep it vendor-agnostic and usable for tensors, gradients, and
// kernel metadata.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheKind {
    Tensor,
    Gradient,
    KernelMeta,
}

#[derive(Debug, Clone)]
pub struct CacheEntryMeta {
    pub kind: CacheKind,
    pub len_bytes: usize,
    pub checksum32: u32,
    pub created_unix: u64,
}

#[derive(Debug, Clone)]
pub enum CacheError {
    Io(String),
    NotFound,
    Corrupt(String),
    AlreadyExists,
}

#[derive(Debug, Clone)]
pub struct PersistentHybridCache {
    root: PathBuf,
}

impl PersistentHybridCache {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        PersistentHybridCache { root: root.into() }
    }

    pub fn ensure_root(&self) -> Result<(), CacheError> {
        if let Err(e) = fs::create_dir_all(&self.root) {
            return Err(CacheError::Io(format!(
                "Failed to create cache root {:?}: {}",
                &self.root, e
            )));
        }
        Ok(())
    }

    pub fn put_blob(
        &self,
        kind: CacheKind,
        key: &str,
        bytes: &[u8],
        created_unix: u64,
        overwrite: bool,
    ) -> Result<(), CacheError> {
        self.ensure_root()?;

        let (bin_path, meta_path) = self.paths_for(kind, key);

        if !overwrite && bin_path.exists() {
            return Err(CacheError::AlreadyExists);
        }

        if let Some(parent) = bin_path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return Err(CacheError::Io(format!(
                    "Failed to create cache subdir {:?}: {}",
                    parent, e
                )));
            }
        }

        // Write binary blob
        if let Err(e) = fs::write(&bin_path, bytes) {
            return Err(CacheError::Io(format!(
                "Failed to write cache blob {:?}: {}",
                &bin_path, e
            )));
        }

        // Prepare metadata
        let len_bytes = bytes.len();
        let checksum32 = checksum32(bytes);
        let kind_str = kind_to_str(kind);

        let meta_contents = format!(
            "kind={}\nlen={}\nchecksum32={}\ncreated_unix={}\n",
            kind_str, len_bytes, checksum32, created_unix
        );

        if let Some(parent) = meta_path.parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return Err(CacheError::Io(format!(
                    "Failed to create meta subdir {:?}: {}",
                    parent, e
                )));
            }
        }

        if let Err(e) = fs::write(&meta_path, meta_contents.as_bytes()) {
            return Err(CacheError::Io(format!(
                "Failed to write cache meta {:?}: {}",
                &meta_path, e
            )));
        }

        Ok(())
    }

    pub fn get_blob(&self, kind: CacheKind, key: &str) -> Result<Vec<u8>, CacheError> {
        let (bin_path, meta_path) = self.paths_for(kind, key);

        if !bin_path.exists() || !meta_path.exists() {
            return Err(CacheError::NotFound);
        }

        let meta = self.read_meta(&meta_path)?;

        // Read data
        let mut file = match File::open(&bin_path) {
            Ok(f) => f,
            Err(e) => {
                return Err(CacheError::Io(format!(
                    "Failed to open cache blob {:?}: {}",
                    &bin_path, e
                )))
            }
        };
        let mut buf = Vec::new();
        if let Err(e) = file.read_to_end(&mut buf) {
            return Err(CacheError::Io(format!(
                "Failed to read cache blob {:?}: {}",
                &bin_path, e
            )));
        }

        if buf.len() != meta.len_bytes {
            return Err(CacheError::Corrupt(format!(
                "Length mismatch: meta={}, actual={}",
                meta.len_bytes,
                buf.len()
            )));
        }

        let actual_checksum = checksum32(&buf);
        if actual_checksum != meta.checksum32 {
            return Err(CacheError::Corrupt(format!(
                "Checksum mismatch: meta={}, actual={}",
                meta.checksum32, actual_checksum
            )));
        }

        Ok(buf)
    }

    pub fn exists(&self, kind: CacheKind, key: &str) -> bool {
        let (bin_path, meta_path) = self.paths_for(kind, key);
        bin_path.exists() && meta_path.exists()
    }

    fn kind_dir(&self, kind: CacheKind) -> PathBuf {
        let name = kind_to_str(kind);
        self.root.join(name)
    }

    fn paths_for(&self, kind: CacheKind, key: &str) -> (PathBuf, PathBuf) {
        let dir = self.kind_dir(kind);
        let sanitized = sanitize_key_for_path(key);
        let bin = dir.join(format!("{}.bin", sanitized));
        let meta = dir.join(format!("{}.meta", sanitized));
        (bin, meta)
    }

    fn read_meta(&self, path: &Path) -> Result<CacheEntryMeta, CacheError> {
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(e) => {
                return Err(CacheError::Io(format!(
                    "Failed to open cache meta {:?}: {}",
                    path, e
                )))
            }
        };

        let mut buf = String::new();
        if let Err(e) = file.read_to_string(&mut buf) {
            return Err(CacheError::Io(format!(
                "Failed to read cache meta {:?}: {}",
                path, e
            )));
        }

        parse_meta(&buf)
    }
}

fn kind_to_str(kind: CacheKind) -> &'static str {
    match kind {
        CacheKind::Tensor => "tensor",
        CacheKind::Gradient => "gradient",
        CacheKind::KernelMeta => "kernel_meta",
    }
}

fn str_to_kind(s: &str) -> Option<CacheKind> {
    match s {
        "tensor" => Some(CacheKind::Tensor),
        "gradient" => Some(CacheKind::Gradient),
        "kernelmeta" => Some(CacheKind::KernelMeta),
        "kernel_meta" => Some(CacheKind::KernelMeta),
        _ => None,
    }
}

fn sanitize_key_for_path(key: &str) -> String {
    let mut out = String::with_capacity(key.len());
    for ch in key.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    out
}

fn parse_meta(contents: &str) -> Result<CacheEntryMeta, CacheError> {
    let mut kind: Option<CacheKind> = None;
    let mut len_bytes: Option<usize> = None;
    let mut checksum32_val: Option<u32> = None;
    let mut created_unix: Option<u64> = None;

    for line in contents.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let eq_pos = match trimmed.find('=') {
            Some(p) => p,
            None => {
                return Err(CacheError::Corrupt(format!(
                    "Invalid meta line (no '='): {}",
                    trimmed
                )))
            }
        };
        let (k, v) = trimmed.split_at(eq_pos);
        let v = &v[1..]; // skip '='

        match k {
            "kind" => {
                kind = match str_to_kind(v) {
                    Some(knd) => Some(knd),
                    None => {
                        return Err(CacheError::Corrupt(format!(
                            "Unknown kind in meta: {}",
                            v
                        )))
                    }
                };
            }
            "len" => {
                match v.parse::<usize>() {
                    Ok(n) => len_bytes = Some(n),
                    Err(e) => {
                        return Err(CacheError::Corrupt(format!(
                            "Invalid len in meta: {} ({})",
                            v, e
                        )))
                    }
                }
            }
            "checksum32" => {
                match v.parse::<u32>() {
                    Ok(n) => checksum32_val = Some(n),
                    Err(e) => {
                        return Err(CacheError::Corrupt(format!(
                            "Invalid checksum32 in meta: {} ({})",
                            v, e
                        )))
                    }
                }
            }
            "created_unix" => {
                match v.parse::<u64>() {
                    Ok(n) => created_unix = Some(n),
                    Err(e) => {
                        return Err(CacheError::Corrupt(format!(
                            "Invalid created_unix in meta: {} ({})",
                            v, e
                        )))
                    }
                }
            }
            _ => {
                // Unknown key: treat as corrupt to keep format strict.
                return Err(CacheError::Corrupt(format!(
                    "Unknown key in meta: {}",
                    k
                )));
            }
        }
    }

    let k = match kind {
        Some(k) => k,
        None => {
            return Err(CacheError::Corrupt(
                "Missing kind in meta".to_string(),
            ))
        }
    };
    let l = match len_bytes {
        Some(v) => v,
        None => {
            return Err(CacheError::Corrupt(
                "Missing len in meta".to_string(),
            ))
        }
    };
    let c = match checksum32_val {
        Some(v) => v,
        None => {
            return Err(CacheError::Corrupt(
                "Missing checksum32 in meta".to_string(),
            ))
        }
    };
    let cu = match created_unix {
        Some(v) => v,
        None => {
            return Err(CacheError::Corrupt(
                "Missing created_unix in meta".to_string(),
            ))
        }
    };

    Ok(CacheEntryMeta {
        kind: k,
        len_bytes: l,
        checksum32: c,
        created_unix: cu,
    })
}

// Simple deterministic 32-bit checksum (FNV-1a like)
fn checksum32(data: &[u8]) -> u32 {
    let mut hash: u32 = 0x811C9DC5; // FNV offset basis
    let prime: u32 = 0x01000193; // FNV prime

    for b in data {
        hash ^= *b as u32;
        hash = hash.wrapping_mul(prime);
    }

    hash
}
