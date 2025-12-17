use libloading::{Library, Symbol};
use std::ffi::c_void;
use std::ffi::CString;

#[allow(non_camel_case_types)]
type NvJitLinkHandle = *mut c_void;

#[derive(Debug)]
pub enum NvJitLinkError {
    LoadError(String),
    MissingSymbol(String),
    LinkError(String),
}

pub struct NvJitLinker {
    lib: Library,
}

impl NvJitLinker {
    pub fn new() -> Result<Self, NvJitLinkError> {
        // Intentar cargar nvJitLink en Windows y Linux con nombres reales.
        // En Windows las DLL suelen tener sufijo _0 y variantes 64.
        // En Linux usamos libnvJitLink.so.
        let candidates: [&str; 13] = [
            // Windows bin paths típicos (CUDA 13.0, carpeta x64)
            "C\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin\\x64\\nvJitLink_130_0.dll",
            "C\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.0\\bin\\x64\\nvJitLink64_130_0.dll",
            // Windows bin path típico (CUDA 12.6)
            "C\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin\\nvJitLink_120_0.dll",
            "C\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.6\\bin\\nvJitLink64_120_0.dll",
            // Windows: nombres de DLL en PATH (con y sin sufijo _0, con y sin 64)
            "nvJitLink_130_0.dll",
            "nvJitLink64_130_0.dll",
            "nvJitLink_120_0.dll",
            "nvJitLink64_120_0.dll",
            "nvJitLink_130.dll",
            "nvJitLink_120.dll",
            "nvJitLink.dll",
            // Linux
            "libnvJitLink.so",
            "libnvJitLink.so.1",
        ];

        let mut last_err: Option<String> = None;
        for path in &candidates {
            unsafe {
                match Library::new(path) {
                    Ok(lib) => {
                        println!("[JITLINK] Loaded nvJitLink from: {}", path);
                        return Ok(Self { lib });
                    }
                    Err(e) => {
                        last_err = Some(format!("{:?}", e));
                    }
                }
            }
        }

        Err(NvJitLinkError::LoadError(
            last_err.unwrap_or_else(|| "Failed to load any nvJitLink library".into()),
        ))
    }

    fn get_symbol<T>(&self, name: &[u8]) -> Result<Symbol<'_, T>, NvJitLinkError> {
        unsafe {
            self.lib
                .get(name)
                .map_err(|_| NvJitLinkError::MissingSymbol(String::from_utf8_lossy(name).to_string()))
        }
    }

    /// Enlaza un PTX (como &str) a un CUBIN usando nvJitLink.
    pub fn link_ptx_to_cubin(&self, ptx: &str) -> Result<Vec<u8>, NvJitLinkError> {
        unsafe {
            type NvJitLinkCreateFn = unsafe extern "C" fn(
                *mut NvJitLinkHandle,
                u32,
                *mut *mut i8,
                *mut *mut i8,
                *mut *mut i8,
            ) -> i32;

            type NvJitLinkAddDataFn = unsafe extern "C" fn(
                NvJitLinkHandle,
                i32,
                *const c_void,
                usize,
                *const i8,
                u32,
                *mut *mut i8,
                *mut *mut i8,
            ) -> i32;

            type NvJitLinkCompleteFn = unsafe extern "C" fn(
                NvJitLinkHandle,
                *mut *mut c_void,
                *mut usize,
            ) -> i32;

            type NvJitLinkDestroyFn = unsafe extern "C" fn(
                *mut NvJitLinkHandle,
            ) -> i32;

            let create: Symbol<NvJitLinkCreateFn> = self
                .get_symbol(b"nvJitLinkCreate\0")?;
            let add_data: Symbol<NvJitLinkAddDataFn> = self
                .get_symbol(b"nvJitLinkAddData\0")?;
            let complete: Symbol<NvJitLinkCompleteFn> = self
                .get_symbol(b"nvJitLinkComplete\0")?;
            let destroy: Symbol<NvJitLinkDestroyFn> = self
                .get_symbol(b"nvJitLinkDestroy\0")?;

            let mut handle: NvJitLinkHandle = std::ptr::null_mut();

            // Opciones mínimas para nvJitLink: target sm_89.
            let arch_opt = CString::new("-arch=sm_89").unwrap();
            let mut options: [*mut i8; 1] = [arch_opt.as_ptr() as *mut i8];
            let mut option_vals: [*mut i8; 1] = [std::ptr::null_mut()];

            let create_res = create(
                &mut handle,
                options.len() as u32,
                options.as_mut_ptr(),
                option_vals.as_mut_ptr(),
                std::ptr::null_mut(),
            );
            if create_res != 0 || handle.is_null() {
                return Err(NvJitLinkError::LinkError("nvJitLinkCreate failed".into()));
            }

            // Añadir PTX.
            let ptx_bytes = ptx.as_bytes();
            let name_c = CString::new("matmul_kernel.ptx").unwrap();

            // NVJITLINK_INPUT_PTX = 0 según documentación.
            let input_type_ptx: i32 = 0;
            let add_res = add_data(
                handle,
                input_type_ptx,
                ptx_bytes.as_ptr() as *const c_void,
                ptx_bytes.len(),
                name_c.as_ptr(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            );
            if add_res != 0 {
                let mut h = handle;
                let _ = destroy(&mut h);
                return Err(NvJitLinkError::LinkError("nvJitLinkAddData failed".into()));
            }

            // Completar y recuperar CUBIN.
            let mut cubin_ptr: *mut c_void = std::ptr::null_mut();
            let mut cubin_size: usize = 0;
            let complete_res = complete(handle, &mut cubin_ptr, &mut cubin_size);
            if complete_res != 0 || cubin_ptr.is_null() || cubin_size == 0 {
                let mut h = handle;
                let _ = destroy(&mut h);
                return Err(NvJitLinkError::LinkError("nvJitLinkComplete failed".into()));
            }

            let cubin_slice = std::slice::from_raw_parts(cubin_ptr as *const u8, cubin_size);
            let cubin = cubin_slice.to_vec();

            let mut h = handle;
            let _ = destroy(&mut h);

            Ok(cubin)
        }
    }
}
