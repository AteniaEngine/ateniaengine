//! Logic for offloading tensors and activations to secondary storage.

use crate::tensor::tensor::{Device, Tensor};
use std::fs::{self, File};
use std::io::{Read, Write};
use uuid::Uuid;

#[derive(Debug)]
pub enum StorageLocation {
    GPU,
    RAM,
    DISK,
}

#[derive(Debug)]
pub struct OffloadHandle {
    pub id: Uuid,
    pub path: Option<String>,
    pub location: StorageLocation,
}

pub struct Offloader {
    pub disk_directory: String,
}

impl Offloader {
    pub fn new(disk_directory: String) -> Self {
        fs::create_dir_all(&disk_directory).ok();
        Self { disk_directory }
    }

    /// Move tensor from GPU → RAM.
    pub fn to_ram(&self, tensor: &mut Tensor) {
        tensor.device = Device::CPU;
    }

    /// Move tensor from RAM → DISK (serialize as binary f32s).
    pub fn to_disk(&self, tensor: &Tensor) -> OffloadHandle {
        let id = Uuid::new_v4();
        let path = format!("{}/{}.bin", self.disk_directory, id);

        let mut file = File::create(&path).unwrap();
        for value in tensor.as_cpu_slice() {
            file.write_all(&value.to_le_bytes()).unwrap();
        }

        OffloadHandle {
            id,
            path: Some(path),
            location: StorageLocation::DISK,
        }
    }

    /// Read back from disk → RAM.
    pub fn from_disk(&self, handle: &OffloadHandle, shape: Vec<usize>) -> Tensor {
        let mut file = File::open(handle.path.as_ref().unwrap()).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();

        let mut data = Vec::new();
        for chunk in buffer.chunks_exact(4) {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            data.push(f32::from_le_bytes(arr));
        }

        Tensor::new_cpu(shape, data)
    }
}
