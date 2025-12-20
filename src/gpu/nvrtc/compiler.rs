use std::ffi::{CString, CStr};
use std::ptr;
use libloading::{Library, Symbol};

use super::{NvrtcError, cache::KernelCache};
use crate::gpu::arch::CudaArchDetector;

// Opaque type for nvrtcProgram.
#[allow(non_camel_case_types)]
type NvrtcProgramT = *mut std::os::raw::c_void;

// Minimal NVRTC return codes (0 = SUCCESS).
const NVRTC_SUCCESS: i32 = 0;

pub struct NvrtcProgram {
    pub ptx: String,
}

pub struct NvrtcCompiler {
    lib: Library,
    cache: KernelCache,
}

impl NvrtcCompiler {
    pub fn new() -> Result<Self, NvrtcError> {
        // Try to load NVRTC on Windows and Linux.
        let lib = unsafe {
            Library::new("nvrtc64_120_0.dll")
                .or_else(|_| Library::new("nvrtc64_112_0.dll"))
                .or_else(|_| Library::new("libnvrtc.so"))
                .map_err(|_| NvrtcError::LoadError("Cannot load NVRTC"))?
        };

        Ok(Self {
            lib,
            cache: KernelCache::new(),
        })
    }

    pub fn compile(&self, source: &str, kernel_name: &str, arch: &str)
        -> Result<NvrtcProgram, NvrtcError>
    {
        // Important: do not read from the cache here to avoid reusing PTX
        // generated with old kernel signatures. Always compile.

        unsafe {
            // Minimal NVRTC signatures we need.
            type CreateFn = unsafe extern "C" fn(
                *mut NvrtcProgramT,
                *const i8,
                *const i8,
                i32,
                *const *const i8,
                *const *const i8,
            ) -> i32;

            type CompileFn = unsafe extern "C" fn(NvrtcProgramT, i32, *const *const i8) -> i32;
            type GetPtxSizeFn = unsafe extern "C" fn(NvrtcProgramT, *mut usize) -> i32;
            type GetPtxFn = unsafe extern "C" fn(NvrtcProgramT, *mut i8) -> i32;
            type GetLogSizeFn = unsafe extern "C" fn(NvrtcProgramT, *mut usize) -> i32;
            type GetLogFn = unsafe extern "C" fn(NvrtcProgramT, *mut i8) -> i32;
            type DestroyFn = unsafe extern "C" fn(*mut NvrtcProgramT) -> i32;

            let create: Symbol<CreateFn> = self.lib
                .get(b"nvrtcCreateProgram\0")
                .map_err(|_| NvrtcError::MissingSymbol("nvrtcCreateProgram".into()))?;

            let compile: Symbol<CompileFn> = self.lib
                .get(b"nvrtcCompileProgram\0")
                .map_err(|_| NvrtcError::MissingSymbol("nvrtcCompileProgram".into()))?;

            let get_ptx_size: Symbol<GetPtxSizeFn> = self.lib
                .get(b"nvrtcGetPTXSize\0")
                .map_err(|_| NvrtcError::MissingSymbol("nvrtcGetPTXSize".into()))?;

            let get_ptx: Symbol<GetPtxFn> = self.lib
                .get(b"nvrtcGetPTX\0")
                .map_err(|_| NvrtcError::MissingSymbol("nvrtcGetPTX".into()))?;

            let get_log_size: Symbol<GetLogSizeFn> = self.lib
                .get(b"nvrtcGetProgramLogSize\0")
                .map_err(|_| NvrtcError::MissingSymbol("nvrtcGetProgramLogSize".into()))?;

            let get_log: Symbol<GetLogFn> = self.lib
                .get(b"nvrtcGetProgramLog\0")
                .map_err(|_| NvrtcError::MissingSymbol("nvrtcGetProgramLog".into()))?;

            let destroy: Symbol<DestroyFn> = self.lib
                .get(b"nvrtcDestroyProgram\0")
                .map_err(|_| NvrtcError::MissingSymbol("nvrtcDestroyProgram".into()))?;

            let src_c = CString::new(source).unwrap();
            let name_c = CString::new("kernel.cu").unwrap();

            let mut prog: NvrtcProgramT = ptr::null_mut();
            let headers: *const *const i8 = ptr::null();
            let header_names: *const *const i8 = ptr::null();
            let headers_count: i32 = 0;

            let create_res = create(
                &mut prog,
                src_c.as_ptr(),
                name_c.as_ptr(),
                headers_count,
                headers,
                header_names,
            );
            if create_res != NVRTC_SUCCESS || prog.is_null() {
                return Err(NvrtcError::CompilationError("Failed to create NVRTC program".into()));
            }

            // Options: allow arch="auto" for dynamic detection.
            // IMPORTANT: always force compute_XX (generic PTX), never sm_XX.
            let mut arch_to_use = if arch == "auto" {
                match CudaArchDetector::new().and_then(|d| d.arch_flag()) {
                    Ok(a) => a,
                    Err(_) => "compute_61".to_string(),
                }
            } else {
                arch.to_string()
            };

            // Normalize to compute_XX if it came as sm_XX or similar.
            if let Some(sm_suffix) = arch_to_use.strip_prefix("sm_") {
                arch_to_use = format!("compute_{}", sm_suffix);
            }

            // Build a vector of CStrings so references live
            // during the entire nvrtcCompileProgram call.
            let mut opt_cstrings: Vec<CString> = Vec::new();

            // Flags used (per NVRTC -> nvJitLink pipeline):
            //   --gpu-architecture=compute_XX
            //   --std=c++17
            //   --device-c
            //   --relocatable-device-code=true
            //   --fmad=false
            // (No --gpu-code=sm_XX nor variants.)
            opt_cstrings.push(
                CString::new(format!("--gpu-architecture={}", arch_to_use)).unwrap(),
            );
            opt_cstrings.push(CString::new("--std=c++17").unwrap());
            opt_cstrings.push(CString::new("--device-c").unwrap());
            opt_cstrings.push(CString::new("--relocatable-device-code=true").unwrap());
            opt_cstrings.push(CString::new("--fmad=false").unwrap());

            let opt_ptrs: Vec<*const i8> = opt_cstrings
                .iter()
                .map(|s| s.as_ptr())
                .collect();

            let res = compile(prog, opt_ptrs.len() as i32, opt_ptrs.as_ptr());
            if res != NVRTC_SUCCESS {
                // Get detailed compilation log.
                let mut log_size: usize = 0;
                let mut log_msg = String::new();
                if get_log_size(prog, &mut log_size) == NVRTC_SUCCESS && log_size > 1 {
                    let mut buf = vec![0i8; log_size];
                    if get_log(prog, buf.as_mut_ptr()) == NVRTC_SUCCESS {
                        if let Ok(s) = CStr::from_ptr(buf.as_ptr()).to_str() {
                            log_msg = s.to_string();
                        }
                    }
                }
                let _ = destroy(&mut prog);
                return Err(NvrtcError::CompilationError(log_msg));
            }

            // Get PTX
            let mut size: usize = 0;
            let size_res = get_ptx_size(prog, &mut size);
            if size_res != NVRTC_SUCCESS || size == 0 {
                let _ = destroy(&mut prog);
                return Err(NvrtcError::CompilationError("Failed to get PTX size".into()));
            }

            let mut buffer = vec![0i8; size];
            let get_res = get_ptx(prog, buffer.as_mut_ptr());
            if get_res != NVRTC_SUCCESS {
                let _ = destroy(&mut prog);
                return Err(NvrtcError::CompilationError("Failed to get PTX".into()));
            }

            let _ = destroy(&mut prog);

            let ptx_str = CStr::from_ptr(buffer.as_ptr())
                .to_str()
                .unwrap_or("")
                .to_string();

            // Save to cache
            self.cache.save(kernel_name, &ptx_str)?;

            Ok(NvrtcProgram { ptx: ptx_str })
        }
    }

    /// Explicit variant: compile `source` with an exact set of NVRTC flags
    /// provided by the caller. Does not apply additional architecture logic.
    pub fn compile_with_flags(&self, source: &str, kernel_name: &str, flags: &[&str])
        -> Result<NvrtcProgram, NvrtcError>
    {
        // Important: do NOT use the input cache here to avoid reusing PTX
        // generated with a different kernel signature. Always compile.

        unsafe {
            // Minimal NVRTC signatures we need.
            type CreateFn = unsafe extern "C" fn(
                *mut NvrtcProgramT,
                *const i8,
                *const i8,
                i32,
                *const *const i8,
                *const *const i8,
            ) -> i32;

            type CompileFn = unsafe extern "C" fn(NvrtcProgramT, i32, *const *const i8) -> i32;
            type GetPtxSizeFn = unsafe extern "C" fn(NvrtcProgramT, *mut usize) -> i32;
            type GetPtxFn = unsafe extern "C" fn(NvrtcProgramT, *mut i8) -> i32;
            type GetLogSizeFn = unsafe extern "C" fn(NvrtcProgramT, *mut usize) -> i32;
            type GetLogFn = unsafe extern "C" fn(NvrtcProgramT, *mut i8) -> i32;
            type DestroyFn = unsafe extern "C" fn(*mut NvrtcProgramT) -> i32;

            let create: Symbol<CreateFn> = self.lib
                .get(b"nvrtcCreateProgram\0")
                .map_err(|_| NvrtcError::MissingSymbol("nvrtcCreateProgram".into()))?;

            let compile: Symbol<CompileFn> = self.lib
                .get(b"nvrtcCompileProgram\0")
                .map_err(|_| NvrtcError::MissingSymbol("nvrtcCompileProgram".into()))?;

            let get_ptx_size: Symbol<GetPtxSizeFn> = self.lib
                .get(b"nvrtcGetPTXSize\0")
                .map_err(|_| NvrtcError::MissingSymbol("nvrtcGetPTXSize".into()))?;

            let get_ptx: Symbol<GetPtxFn> = self.lib
                .get(b"nvrtcGetPTX\0")
                .map_err(|_| NvrtcError::MissingSymbol("nvrtcGetPTX".into()))?;

            let get_log_size: Symbol<GetLogSizeFn> = self.lib
                .get(b"nvrtcGetProgramLogSize\0")
                .map_err(|_| NvrtcError::MissingSymbol("nvrtcGetProgramLogSize".into()))?;

            let get_log: Symbol<GetLogFn> = self.lib
                .get(b"nvrtcGetProgramLog\0")
                .map_err(|_| NvrtcError::MissingSymbol("nvrtcGetProgramLog".into()))?;

            let destroy: Symbol<DestroyFn> = self.lib
                .get(b"nvrtcDestroyProgram\0")
                .map_err(|_| NvrtcError::MissingSymbol("nvrtcDestroyProgram".into()))?;

            let src_c = CString::new(source).unwrap();
            let name_c = CString::new("kernel.cu").unwrap();

            let mut prog: NvrtcProgramT = ptr::null_mut();
            let headers: *const *const i8 = ptr::null();
            let header_names: *const *const i8 = ptr::null();
            let headers_count: i32 = 0;

            let create_res = create(
                &mut prog,
                src_c.as_ptr(),
                name_c.as_ptr(),
                headers_count,
                headers,
                header_names,
            );
            if create_res != NVRTC_SUCCESS || prog.is_null() {
                return Err(NvrtcError::CompilationError("Failed to create NVRTC program".into()));
            }

            // Build CStrings from the provided flags.
            let mut opt_cstrings: Vec<CString> = Vec::new();
            for f in flags {
                opt_cstrings.push(CString::new(*f).unwrap());
            }
            let opt_ptrs: Vec<*const i8> = opt_cstrings
                .iter()
                .map(|s| s.as_ptr())
                .collect();

            let res = compile(prog, opt_ptrs.len() as i32, opt_ptrs.as_ptr());
            if res != NVRTC_SUCCESS {
                // Get detailed compilation log.
                let mut log_size: usize = 0;
                let mut log_msg = String::new();
                if get_log_size(prog, &mut log_size) == NVRTC_SUCCESS && log_size > 1 {
                    let mut buf = vec![0i8; log_size];
                    if get_log(prog, buf.as_mut_ptr()) == NVRTC_SUCCESS {
                        if let Ok(s) = CStr::from_ptr(buf.as_ptr()).to_str() {
                            log_msg = s.to_string();
                        }
                    }
                }
                let _ = destroy(&mut prog);
                return Err(NvrtcError::CompilationError(log_msg));
            }

            // Get PTX
            let mut size: usize = 0;
            let size_res = get_ptx_size(prog, &mut size);
            if size_res != NVRTC_SUCCESS || size == 0 {
                let _ = destroy(&mut prog);
                return Err(NvrtcError::CompilationError("Failed to get PTX size".into()));
            }

            let mut buffer = vec![0i8; size];
            let get_res = get_ptx(prog, buffer.as_mut_ptr());
            if get_res != NVRTC_SUCCESS {
                let _ = destroy(&mut prog);
                return Err(NvrtcError::CompilationError("Failed to get PTX".into()));
            }

            let _ = destroy(&mut prog);

            let ptx_str = CStr::from_ptr(buffer.as_ptr())
                .to_str()
                .unwrap_or("")
                .to_string();

            // Save to cache
            self.cache.save(kernel_name, &ptx_str)?;

            Ok(NvrtcProgram { ptx: ptx_str })
        }
    }
}
