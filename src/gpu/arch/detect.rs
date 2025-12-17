use libloading::{Library, Symbol};

use super::ArchError;

pub struct CudaArchDetector {
    driver: Library,
}

impl CudaArchDetector {
    pub fn new() -> Result<Self, ArchError> {
        unsafe {
            let driver = Library::new("nvcuda.dll")
                .or_else(|_| Library::new("libcuda.so"))
                .map_err(|_| ArchError::DriverNotFound)?;

            Ok(Self { driver })
        }
    }

    unsafe fn get<T>(&self, name: &[u8]) -> Result<Symbol<'_, T>, ArchError> {
        unsafe {
            self.driver
                .get(name)
                .map_err(move |_| {
                    let sym = std::str::from_utf8(name).unwrap_or("unknown").to_string();
                    ArchError::MissingSymbol(sym)
                })
        }
    }

    pub fn compute_capability(&self) -> Result<(i32, i32), ArchError> {
        unsafe {
            let cu_init: Symbol<unsafe extern "C" fn(u32) -> i32> =
                self.get(b"cuInit\0")?;

            let cu_device_get: Symbol<unsafe extern "C" fn(*mut i32, i32) -> i32> =
                self.get(b"cuDeviceGet\0")?;

            let cu_device_get_attr: Symbol<unsafe extern "C" fn(*mut i32, i32, i32) -> i32> =
                self.get(b"cuDeviceGetAttribute\0")?;

            let major_attr = 75; // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
            let minor_attr = 76; // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR

            let _ = cu_init(0);

            let mut device: i32 = 0;
            let res = cu_device_get(&mut device, 0);
            if res != 0 {
                return Err(ArchError::DetectionFailed);
            }

            let mut major = 0;
            let mut minor = 0;

            let _ = cu_device_get_attr(&mut major, major_attr, device);
            let _ = cu_device_get_attr(&mut minor, minor_attr, device);

            Ok((major, minor))
        }
    }

    pub fn arch_flag(&self) -> Result<String, ArchError> {
        let (major, minor) = self.compute_capability()?;

        let arch = match (major, minor) {
            (8, 9) => "compute_89",
            (8, _) => "compute_80",
            (7, 5) => "compute_75",
            (6, _) => "compute_61",
            _ => "compute_61",
        };

        Ok(arch.to_string())
    }
}
