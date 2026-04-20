use std::ffi::{CString, CStr};
use std::fs;
use std::path::{Path, PathBuf};
use std::ptr;
use libloading::{Library, Symbol};

use super::{NvrtcError, cache::KernelCache};
use crate::gpu::arch::CudaArchDetector;

/// Known NVRTC DLL names, ordered from newest CUDA major version to oldest.
const NVRTC_DLL_NAMES: &[&str] = &[
    "nvrtc64_130_0.dll", // CUDA 13.x
    "nvrtc64_120_0.dll", // CUDA 12.x
    "nvrtc64_112_0.dll", // CUDA 11.x
];

/// Subdirectories under a CUDA Toolkit root where NVRTC DLLs may live.
/// CUDA 13 moved the runtime DLLs to `bin\x64`; earlier versions kept
/// them directly in `bin`.
const CUDA_DLL_SUBDIRS: &[&str] = &["bin\\x64", "bin"];

/// Detects the root directory of the installed CUDA Toolkit.
///
/// Respects `CUDA_PATH` when set; otherwise scans the default Windows
/// install location and picks the highest available version.
///
/// NOTE: This duplicates the logic in `build.rs::detect_cuda_path`. The
/// two implementations must be kept in sync. We cannot share code because
/// `build.rs` is a separate build-time binary and its symbols are not
/// linked into the runtime library.
fn cuda_root() -> Option<String> {
    if let Ok(p) = std::env::var("CUDA_PATH") {
        if Path::new(&p).is_dir() {
            return Some(p);
        }
    }

    let base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA";
    let entries = fs::read_dir(base).ok()?;

    let mut versions: Vec<((u32, u32), PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix('v') else { continue };
        let Some((maj, min)) = rest.split_once('.') else { continue };
        let (Ok(m), Ok(n)) = (maj.parse::<u32>(), min.parse::<u32>()) else { continue };
        versions.push(((m, n), entry.path()));
    }
    versions.sort_by(|a, b| b.0.cmp(&a.0));
    versions
        .into_iter()
        .next()
        .map(|(_, p)| p.to_string_lossy().into_owned())
}

/// Attempts to load NVRTC across multiple strategies and OS conventions.
///
/// Order:
/// 1. Bare DLL names (works when CUDA's bin directory is on `PATH`).
/// 2. Absolute paths constructed from `cuda_root()` — covers shells whose
///    `PATH` does not include CUDA even though the Toolkit is installed.
/// 3. Linux shared object.
///
/// On failure returns a human-readable string listing every path probed
/// and every CUDA directory scanned. Callers are expected to surface this
/// detail via logging before returning the static `NvrtcError::LoadError`.
/// Prepends CUDA's DLL subdirectories to the process `PATH`.
///
/// Required because NVRTC's main DLL transitively loads a companion DLL
/// (`nvrtc-builtins64_*.dll`) via `LoadLibrary`, which searches the
/// process `PATH`. Shells like MSYS/Git Bash often inherit a `PATH` that
/// does not include CUDA even when the Toolkit is installed; this
/// function patches the in-process `PATH` so transitive loads succeed.
fn prepend_cuda_dirs_to_path(cuda_root: &str) {
    let existing = std::env::var("PATH").unwrap_or_default();
    let mut new_entries: Vec<String> = Vec::new();
    for subdir in CUDA_DLL_SUBDIRS {
        let dir = format!(r"{}\{}", cuda_root, subdir);
        if Path::new(&dir).is_dir() && !existing.contains(&dir) {
            new_entries.push(dir);
        }
    }
    if new_entries.is_empty() {
        return;
    }
    let new_path = if existing.is_empty() {
        new_entries.join(";")
    } else {
        format!("{};{}", new_entries.join(";"), existing)
    };
    // SAFETY: Rust 2024 requires `unsafe` for set_var. We only mutate PATH
    // at process-local scope during NVRTC initialization. No other Atenia
    // code mutates PATH concurrently.
    unsafe {
        std::env::set_var("PATH", new_path);
    }
}

fn try_load_nvrtc() -> Result<Library, String> {
    let mut attempted: Vec<String> = Vec::new();

    // Before attempting any load, ensure CUDA's bin dirs are on PATH so
    // transitive dependencies (e.g. nvrtc-builtins) resolve correctly.
    if let Some(root) = cuda_root() {
        prepend_cuda_dirs_to_path(&root);
    }

    // Strategy 1: bare names resolved via process PATH.
    for name in NVRTC_DLL_NAMES {
        attempted.push((*name).to_string());
        if let Ok(lib) = unsafe { Library::new(name) } {
            return Ok(lib);
        }
    }

    // Strategy 2: absolute paths built from the detected CUDA root.
    let mut scanned_dirs: Vec<String> = Vec::new();
    if let Some(root) = cuda_root() {
        for subdir in CUDA_DLL_SUBDIRS {
            let dir = format!(r"{}\{}", root, subdir);
            scanned_dirs.push(dir.clone());
            for name in NVRTC_DLL_NAMES {
                let full = format!(r"{}\{}", dir, name);
                attempted.push(full.clone());
                if let Ok(lib) = unsafe { Library::new(&full) } {
                    return Ok(lib);
                }
            }
        }
    }

    // Strategy 3: Linux shared object.
    attempted.push("libnvrtc.so".to_string());
    if let Ok(lib) = unsafe { Library::new("libnvrtc.so") } {
        return Ok(lib);
    }

    Err(format!(
        "tried: [{}]; scanned CUDA dirs: [{}]",
        attempted.join(", "),
        if scanned_dirs.is_empty() {
            "<none>".to_string()
        } else {
            scanned_dirs.join(", ")
        }
    ))
}

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
        let lib = try_load_nvrtc().map_err(|details| {
            eprintln!("[NVRTC loader] Failed to load NVRTC: {}", details);
            NvrtcError::LoadError("Cannot load NVRTC")
        })?;

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
