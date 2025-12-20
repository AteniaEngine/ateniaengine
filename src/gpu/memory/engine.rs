use libloading::{Library, Symbol};

use super::GpuMemoryError;

type CUdeviceptr = u64;

pub struct GpuPtr {
    pub ptr: CUdeviceptr,
    pub size: usize,
}

pub struct GpuMemoryEngine {
    driver: Library,
}

impl GpuMemoryEngine {
    pub fn new() -> Result<Self, GpuMemoryError> {
        unsafe {
            let driver = Library::new("nvcuda.dll")
                .or_else(|_| Library::new("libcuda.so"))
                .map_err(|_| GpuMemoryError::DriverLoadFailed)?;

            // --- Load required CUDA driver functions ---
            let cu_init: Symbol<unsafe extern "C" fn(u32) -> i32> =
                driver
                    .get(b"cuInit\0")
                    .map_err(|_| GpuMemoryError::MissingSymbol("cuInit".into()))?;
            cu_init(0);

            let cu_device_get: Symbol<unsafe extern "C" fn(*mut i32, i32) -> i32> =
                driver
                    .get(b"cuDeviceGet\0")
                    .map_err(|_| GpuMemoryError::MissingSymbol("cuDeviceGet".into()))?;

            let mut device = 0;
            let res = cu_device_get(&mut device, 0);
            if res != 0 {
                return Err(GpuMemoryError::DriverLoadFailed);
            }

            let cu_ctx_create: Symbol<
                unsafe extern "C" fn(*mut *mut std::ffi::c_void, u32, i32) -> i32,
            > = driver
                .get(b"cuCtxCreate_v2\0")
                .map_err(|_| GpuMemoryError::MissingSymbol("cuCtxCreate_v2".into()))?;

            // Create context
            let mut ctx: *mut std::ffi::c_void = std::ptr::null_mut();
            let res = cu_ctx_create(&mut ctx, 0, device);
            if res != 0 || ctx.is_null() {
                return Err(GpuMemoryError::DriverLoadFailed);
            }

            Ok(Self { driver })
        }
    }

    unsafe fn load<T>(&self, name: &[u8]) -> Result<Symbol<'_, T>, GpuMemoryError> {
        unsafe {
            self.driver
                .get(name)
                .map_err(|_| GpuMemoryError::MissingSymbol(String::from_utf8_lossy(name).into()))
        }
    }

    pub fn alloc(&self, size: usize) -> Result<GpuPtr, GpuMemoryError> {
        unsafe {
            let cu_init: Symbol<unsafe extern "C" fn(u32) -> i32> =
                self.load(b"cuInit\0")?;

            let cu_mem_alloc: Symbol<unsafe extern "C" fn(*mut CUdeviceptr, usize) -> i32> =
                self.load(b"cuMemAlloc_v2\0")?;

            let _ = cu_init(0);

            let mut ptr: CUdeviceptr = 0;
            let res = cu_mem_alloc(&mut ptr, size);

            if res != 0 || ptr == 0 {
                return Err(GpuMemoryError::AllocationFailed);
            }

            Ok(GpuPtr { ptr, size })
        }
    }

    pub fn free(&self, gpu: &GpuPtr) -> Result<(), GpuMemoryError> {
        unsafe {
            let cu_free: Symbol<unsafe extern "C" fn(CUdeviceptr) -> i32> =
                self.load(b"cuMemFree_v2\0")?;

            let res = cu_free(gpu.ptr);
            if res != 0 {
                return Err(GpuMemoryError::FreeFailed);
            }

            Ok(())
        }
    }

    pub fn copy_htod(&self, dst: &GpuPtr, src: &[f32]) -> Result<(), GpuMemoryError> {
        unsafe {
            let cu_copy: Symbol<unsafe extern "C" fn(CUdeviceptr, *const std::os::raw::c_void, usize) -> i32> =
                self.load(b"cuMemcpyHtoD_v2\0")?;

            let bytes = src.len() * 4;
            let res = cu_copy(dst.ptr, src.as_ptr() as *const _, bytes);

            if res != 0 {
                return Err(GpuMemoryError::CopyFailed);
            }

            Ok(())
        }
    }

    pub fn copy_dtoh(&self, src: &GpuPtr, dst: &mut [f32]) -> Result<(), GpuMemoryError> {
        unsafe {
            let cu_copy: Symbol<unsafe extern "C" fn(*mut std::os::raw::c_void, CUdeviceptr, usize) -> i32> =
                self.load(b"cuMemcpyDtoH_v2\0")?;

            let bytes = dst.len() * 4;
            let res = cu_copy(dst.as_mut_ptr() as *mut _, src.ptr, bytes);

            if res != 0 {
                return Err(GpuMemoryError::CopyFailed);
            }

            Ok(())
        }
    }
}

/// Convenience alias for the future: a logical GPU memory handle.
pub type GpuMemory = GpuPtr;
