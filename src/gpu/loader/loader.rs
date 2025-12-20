use libloading::{Library, Symbol};
use std::ffi::{CString, c_void};
use std::ptr;

use super::CudaLoaderError;
use super::module_cache::{get_cached_module, insert_cached_module, hash_ptx};
use crate::gpu::loader::compat_layer::CompatLoader;

type CUmodule = *mut std::os::raw::c_void;
type CUfunction = *mut std::os::raw::c_void;

pub struct CudaModule {
    pub handle: CUmodule,
}

pub struct CudaFunction {
    pub handle: CUfunction,
}

pub struct CudaLoader {
    driver: Library,
}

impl CudaLoader {
    pub fn new() -> Result<Self, CudaLoaderError> {
        unsafe {
            let driver = Library::new("nvcuda.dll")
                .or_else(|_| Library::new("libcuda.so"))
                .map_err(|e| CudaLoaderError::LoadError(format!("{:?}", e)))?;

            Ok(Self { driver })
        }
    }

    unsafe fn get_symbol<T>(&self, name: &[u8]) -> Result<Symbol<'_, T>, CudaLoaderError> {
        unsafe {
            self.driver.get(name).map_err(|_| {
                CudaLoaderError::MissingSymbol(String::from_utf8_lossy(name).to_string())
            })
        }
    }

    pub fn load_module_from_ptx(&self, ptx: &str) -> Result<CudaModule, CudaLoaderError> {
        // Keep the PTX-hash cache at the entry.
        let hash = hash_ptx(ptx);

        if let Some(handle) = get_cached_module(hash) {
            println!("[MODULE CACHE] Using cached module for hash {}", hash);
            return Ok(CudaModule { handle: handle as CUmodule });
        }

        let module = CompatLoader::try_all_paths(self, ptx)?;

        // Store the resulting handle in the cache, regardless of the chosen path.
        insert_cached_module(hash, module.handle as u64);
        Ok(module)
    }

    /// Direct PTX load path using cuModuleLoadData (no internal cache).
    pub fn load_module_from_ptx_direct(&self, ptx: &str) -> Result<CudaModule, CudaLoaderError> {
        unsafe {
            let cu_init: Symbol<unsafe extern "C" fn(u32) -> i32> =
                self.get_symbol(b"cuInit\0")?;

            let cu_module_load_data: Symbol<unsafe extern "C" fn(*mut CUmodule, *const std::os::raw::c_void) -> i32> =
                self.get_symbol(b"cuModuleLoadData\0")?;

            let _ = cu_init(0);

            let mut module: CUmodule = ptr::null_mut();
            let res = cu_module_load_data(&mut module, ptx.as_ptr() as *const _);

            if res != 0 || module.is_null() {
                return Err(CudaLoaderError::ModuleLoadFailed);
            }

            Ok(CudaModule { handle: module })
        }
    }

    /// Load a CUDA module from an in-memory CUBIN buffer.
    pub fn load_module_from_cubin(&self, cubin: &[u8]) -> Result<CudaModule, CudaLoaderError> {
        unsafe {
            let cu_init: Symbol<unsafe extern "C" fn(u32) -> i32> =
                self.get_symbol(b"cuInit\0")?;

            let cu_module_load_data: Symbol<unsafe extern "C" fn(*mut CUmodule, *const std::os::raw::c_void) -> i32> =
                self.get_symbol(b"cuModuleLoadData\0")?;

            let _ = cu_init(0);

            let mut module: CUmodule = ptr::null_mut();
            let res = cu_module_load_data(&mut module, cubin.as_ptr() as *const _);

            if res != 0 || module.is_null() {
                return Err(CudaLoaderError::ModuleLoadFailed);
            }

            Ok(CudaModule { handle: module })
        }
    }

    /// Extended version using cuModuleLoadDataEx with JIT options.
    pub fn load_module_from_ptx_ex(&self, ptx: &[u8]) -> Result<CudaModule, CudaLoaderError> {
        unsafe {
            // Initialize driver
            let cu_init: Symbol<unsafe extern "C" fn(u32) -> i32> =
                self.get_symbol(b"cuInit\0")?;

            type CuModuleLoadDataExFn = unsafe extern "C" fn(
                *mut CUmodule,
                *const c_void,
                u32,
                *const u32,
                *const *mut c_void,
            ) -> i32;

            let cu_module_load_data_ex: Symbol<CuModuleLoadDataExFn> =
                self.get_symbol(b"cuModuleLoadDataEx\0")?;

            let _ = cu_init(0);

            // Minimal JIT constants (values per CUjit_option enum)
            const CU_JIT_OPTIMIZATION_LEVEL: u32 = 7;
            const CU_JIT_TARGET_FROM_CUCONTEXT: u32 = 8;

            let jit_options: [u32; 2] = [
                CU_JIT_TARGET_FROM_CUCONTEXT,
                CU_JIT_OPTIMIZATION_LEVEL,
            ];

            let mut opt_level: u32 = 4;
            let jit_values: [*mut c_void; 2] = [
                ptr::null_mut(),
                &mut opt_level as *mut u32 as *mut c_void,
            ];

            let mut module: CUmodule = ptr::null_mut();
            let res = cu_module_load_data_ex(
                &mut module as *mut _,
                ptx.as_ptr() as *const c_void,
                jit_options.len() as u32,
                jit_options.as_ptr(),
                jit_values.as_ptr(),
            );

            if res != 0 || module.is_null() {
                // Basic debug to understand PTX load failures.
                println!(
                    "[GPU LOADER] cuModuleLoadDataEx failed: res={} | module_null={} ",
                    res,
                    module.is_null()
                );
                return Err(CudaLoaderError::ModuleLoadFailed);
            }

            Ok(CudaModule { handle: module })
        }
    }

    pub fn get_function(&self, module: &CudaModule, name: &str)
        -> Result<CudaFunction, CudaLoaderError>
    {
        unsafe {
            let cu_get_function: Symbol<unsafe extern "C" fn(*mut CUfunction, CUmodule, *const i8) -> i32> =
                self.get_symbol(b"cuModuleGetFunction\0")?;

            let mut func: CUfunction = ptr::null_mut();
            let cname = CString::new(name).unwrap();

            let res = cu_get_function(&mut func, module.handle, cname.as_ptr());

            if res != 0 || func.is_null() {
                return Err(CudaLoaderError::FunctionNotFound(name.into()));
            }

            Ok(CudaFunction { handle: func })
        }
    }
}
