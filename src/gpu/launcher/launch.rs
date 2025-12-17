use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::ptr;

use crate::gpu::runtime::GpuRuntime;
use crate::gpu::loader::CudaFunction;
use crate::gpu::safety::GpuSafety;
use super::LaunchError;

pub struct GpuLauncher {
    driver: Library,
}

impl GpuLauncher {
    pub fn new() -> Result<Self, LaunchError> {
        unsafe {
            let driver = Library::new("nvcuda.dll")
                .or_else(|_| Library::new("libcuda.so"))
                .map_err(|_| LaunchError::MissingSymbol("driver".into()))?;

            Ok(Self { driver })
        }
    }

    unsafe fn get<T>(&self, name: &[u8]) -> Result<Symbol<'_, T>, LaunchError> {
        unsafe {
            self.driver
                .get(name)
                .map_err(|_| LaunchError::MissingSymbol(String::from_utf8_lossy(name).into()))
        }
    }

    pub fn launch(
        &self,
        rt: &GpuRuntime,
        func: &CudaFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        args: &mut [*mut c_void],
    ) -> Result<(), LaunchError> {
        unsafe {
            let cu_launch: Symbol<
                unsafe extern "C" fn(
                    cufunc: *mut c_void,
                    grid_x: u32, grid_y: u32, grid_z: u32,
                    block_x: u32, block_y: u32, block_z: u32,
                    shared_mem: u32,
                    stream: *mut c_void,
                    args: *mut *mut c_void,
                    extra: *mut *mut c_void,
                ) -> i32
            > = self.get(b"cuLaunchKernel\0")?;

            let res = cu_launch(
                func.handle as *mut _,
                grid.0, grid.1, grid.2,
                block.0, block.1, block.2,
                shared_mem,
                rt.default_stream,
                args.as_mut_ptr(),
                ptr::null_mut(),
            );

            if res != 0 {
                let _ = GpuSafety::check(res, "launch");
                if GpuSafety::should_fallback(res) {
                    return Err(LaunchError::LaunchFailed(res));
                }
                return Err(LaunchError::LaunchFailed(res));
            }

            Ok(())
        }
    }
}
