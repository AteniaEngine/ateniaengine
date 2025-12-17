use libloading::{Library, Symbol};
use std::ffi::CStr;

#[test]
fn test_cuda_device_info() {
    println!("\n=== [CUDA DEVICE INFO TEST] ===\n");

    // Try to load the CUDA driver explicitly
    let driver = unsafe {
        Library::new("nvcuda.dll")
            .or_else(|_| Library::new("libcuda.so"))
            .expect("❌ CUDA driver not found! GPU backend NOT available.")
    };

    println!("✔ CUDA driver loaded.");

    unsafe {
        // Load symbols
        let cu_init: Symbol<unsafe extern "C" fn(u32) -> i32> =
            driver.get(b"cuInit\0").expect("❌ Missing cuInit");
        let cu_device_get: Symbol<unsafe extern "C" fn(*mut i32, i32) -> i32> =
            driver.get(b"cuDeviceGet\0").expect("❌ Missing cuDeviceGet");
        let cu_device_get_count: Symbol<unsafe extern "C" fn(*mut i32) -> i32> =
            driver.get(b"cuDeviceGetCount\0").expect("❌ Missing cuDeviceGetCount");
        let cu_device_get_name: Symbol<unsafe extern "C" fn(*mut i8, i32, i32) -> i32> =
            driver.get(b"cuDeviceGetName\0").expect("❌ Missing cuDeviceGetName");
        let cu_device_get_attr: Symbol<unsafe extern "C" fn(*mut i32, i32, i32) -> i32> =
            driver.get(b"cuDeviceGetAttribute\0").expect("❌ Missing cuDeviceGetAttribute");

        // Init CUDA
        assert_eq!(cu_init(0), 0, "❌ cuInit failed");
        println!("✔ cuInit OK.");

        // Device count
        let mut count = 0;
        assert_eq!(cu_device_get_count(&mut count), 0, "❌ cuDeviceGetCount failed");
        println!("✔ Devices found: {}", count);
        assert!(count > 0, "❌ No CUDA devices detected!");

        // Select device 0
        let mut device: i32 = 0;
        assert_eq!(cu_device_get(&mut device, 0), 0, "❌ cuDeviceGet failed");
        println!("✔ Using device 0.");

        // Get GPU name
        let mut name_buf = vec![0i8; 100];
        assert_eq!(cu_device_get_name(name_buf.as_mut_ptr(), 100, device), 0);

        let cstr = CStr::from_ptr(name_buf.as_ptr());
        let name = cstr.to_string_lossy();
        println!("✔ Device name: {}", name);

        // Get compute capability
        const MAJOR_ATTR: i32 = 75;
        const MINOR_ATTR: i32 = 76;

        let mut major = 0;
        let mut minor = 0;

        assert_eq!(cu_device_get_attr(&mut major, MAJOR_ATTR, device), 0);
        assert_eq!(cu_device_get_attr(&mut minor, MINOR_ATTR, device), 0);

        println!("✔ Compute capability: {}.{}", major, minor);

        assert!(
            matches!((major, minor),
                (8, 9) | (8, 0) | (8, 6) |
                (7, 5) |
                (6, 1) |
                (7, 0) |
                (7, 2)
            ),
            "❌ Unsupported compute capability: {}.{}",
            major, minor
        );

        // Check VRAM via attribute 200 (TOTAL_MEMORY)
        const GLOBAL_MEM_ATTR: i32 = 200;
        let mut mem_bytes = 0;
        let _ = cu_device_get_attr(&mut mem_bytes, GLOBAL_MEM_ATTR, device);

        if mem_bytes > 0 {
            println!("✔ VRAM detected: {:.2} GB", mem_bytes as f64 / 1024.0 / 1024.0 / 1024.0);
        } else {
            println!("⚠ Could not query VRAM size.");
        }

        println!("\n=== ALL CUDA TESTS PASSED ===\n");
    }
}
