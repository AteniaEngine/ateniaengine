use libloading::{Library, Symbol};
use std::ptr;

use super::{GpuRuntimeError, logging::log};

type CUstream = *mut std::os::raw::c_void;

pub struct GpuRuntime {
    driver: Library,
    pub default_stream: CUstream,
}

impl GpuRuntime {
    pub fn new() -> Result<Self, GpuRuntimeError> {
        unsafe {
            log("Loading CUDA driver...");

            let driver = Library::new("nvcuda.dll")
                .or_else(|_| Library::new("libcuda.so"))
                .map_err(|_| GpuRuntimeError::DriverNotFound)?;

            log("CUDA driver loaded");

            let mut rt = Self {
                driver,
                default_stream: ptr::null_mut(),
            };

            rt.init()?;

            Ok(rt)
        }
    }

    unsafe fn get<T>(&self, name: &[u8]) -> Result<Symbol<'_, T>, GpuRuntimeError> {
        unsafe {
            self.driver
                .get(name)
                .map_err(|_| GpuRuntimeError::MissingSymbol(String::from_utf8_lossy(name).into()))
        }
    }

    fn init(&mut self) -> Result<(), GpuRuntimeError> {
        unsafe {
            let cu_init: Symbol<unsafe extern "C" fn(u32) -> i32> =
                self.get(b"cuInit\0")?;

            log("Initializing CUDA driver...");
            let res = cu_init(0);

            if res != 0 {
                return Err(GpuRuntimeError::InitFailed);
            }

            log("CUDA driver initialized");

            let cu_stream_create: Symbol<unsafe extern "C" fn(*mut CUstream, u32) -> i32> =
                self.get(b"cuStreamCreate\0")?;

            let mut stream: CUstream = ptr::null_mut();
            let res = cu_stream_create(&mut stream, 0);

            if res != 0 || stream.is_null() {
                return Err(GpuRuntimeError::StreamCreateFailed);
            }

            log("Default CUDA stream created");

            self.default_stream = stream;

            Ok(())
        }
    }
}
