use std::path::PathBuf;
use std::fs::{self, File};
use std::io::Write;

use super::NvrtcError;

pub struct KernelCache {
    root: PathBuf,
}

impl KernelCache {
    pub fn new() -> Self {
        let root = PathBuf::from("atenia_cache/kernels/");
        // Best-effort: if directory creation fails, we will see it on save.
        let _ = fs::create_dir_all(&root);
        KernelCache { root }
    }

    pub fn get_path(&self, name: &str) -> PathBuf {
        self.root.join(format!("{}.ptx", name))
    }

    pub fn exists(&self, name: &str) -> bool {
        self.get_path(name).exists()
    }

    pub fn save(&self, name: &str, ptx: &str) -> Result<(), NvrtcError> {
        let path = self.get_path(name);
        let mut file = File::create(path)?;
        file.write_all(ptx.as_bytes())?;
        Ok(())
    }

    pub fn load(&self, name: &str) -> Option<String> {
        let path = self.get_path(name);
        if path.exists() {
            fs::read_to_string(path).ok()
        } else {
            None
        }
    }
}
