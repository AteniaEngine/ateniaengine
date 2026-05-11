use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Detects the highest installed CUDA Toolkit version under the default
/// NVIDIA install location (Windows) or Linux, or respects the `CUDA_PATH`
/// env var if set.
///
/// Returns the full toolkit directory (e.g.
/// `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2` on Windows,
/// `/usr/local/cuda` on Linux).
fn detect_cuda_path() -> Result<String, String> {
    // Manual override via env var.
    if let Ok(p) = std::env::var("CUDA_PATH") {
        let nvcc_name = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
        if Path::new(&p).join("bin").join(nvcc_name).is_file() {
            return Ok(p);
        }
        return Err(format!(
            "CUDA_PATH is set to '{}' but {} was not found there.",
            p, nvcc_name
        ));
    }

    // Linux: Check common CUDA locations
    if cfg!(unix) {
        // First, try to find nvcc using 'which' command
        if let Ok(output) = Command::new("which").arg("nvcc").output() {
            if output.status.success() {
                let nvcc_path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !nvcc_path.is_empty() {
                    // nvcc found in PATH, derive CUDA_PATH from its location
                    // nvcc is usually in CUDA_PATH/bin or /usr/bin
                    if nvcc_path.contains("/bin/nvcc") {
                        let cuda_dir = nvcc_path.trim_end_matches("/bin/nvcc");
                        if Path::new(cuda_dir).is_dir() {
                            return Ok(cuda_dir.to_string());
                        }
                    } else if nvcc_path == "/usr/bin/nvcc" {
                        // Ubuntu nvidia-cuda-toolkit installs libraries in /usr/lib/x86_64-linux-gnu/
                        // or /usr/lib/cuda/lib64/
                        if Path::new("/usr/lib/cuda/lib64").is_dir() {
                            return Ok("/usr/lib/cuda".to_string());
                        } else if Path::new("/usr/lib/x86_64-linux-gnu").is_dir() {
                            return Ok("/usr/lib/x86_64-linux-gnu".to_string());
                        } else {
                            // Fallback: use /usr as CUDA_PATH
                            return Ok("/usr".to_string());
                        }
                    }
                }
            }
        }

        // Then check standard CUDA locations
        for cuda_dir in &["/usr/local/cuda", "/opt/cuda", "/usr/cuda"] {
            if Path::new(cuda_dir).join("bin").join("nvcc").is_file() {
                return Ok(cuda_dir.to_string());
            }
        }
        return Err(
            "CUDA Toolkit not found. nvcc not found in PATH, /usr/local/cuda, /opt/cuda, or /usr/cuda. \
             Install with: sudo apt install nvidia-cuda-toolkit or set CUDA_PATH."
                .to_string(),
        );
    }

    // Windows: Default Windows install location.
    let base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA";
    let entries = fs::read_dir(base).map_err(|_| {
        format!(
            "CUDA Toolkit not found at {}. Install from \
             https://developer.nvidia.com/cuda-downloads or set CUDA_PATH.",
            base
        )
    })?;

    // Collect every "vMAJOR.MINOR" subdir that actually contains nvcc.exe,
    // then sort by (major, minor) descending so the highest version wins.
    let mut versions: Vec<((u32, u32), PathBuf)> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix('v') else {
            continue;
        };
        let Some((maj, min)) = rest.split_once('.') else {
            continue;
        };
        let (Ok(m), Ok(n)) = (maj.parse::<u32>(), min.parse::<u32>()) else {
            continue;
        };
        let p = entry.path();
        if p.join("bin").join("nvcc.exe").is_file() {
            versions.push(((m, n), p));
        }
    }
    versions.sort_by(|a, b| b.0.cmp(&a.0));

    versions
        .into_iter()
        .next()
        .map(|(_, p)| p.to_string_lossy().into_owned())
        .ok_or_else(|| {
            format!(
                "No valid CUDA version found under {}. Install from \
                 https://developer.nvidia.com/cuda-downloads.",
                base
            )
        })
}

/// Detects the most recent MSVC BuildTools/Community/Professional/Enterprise
/// installation and returns the path to its `bin\Hostx64\x64` directory
/// (which contains `cl.exe` and `lib.exe`).
///
/// On Linux, returns the path to gcc.
///
/// Respects `MSVC_TOOLS_PATH` as a manual override pointing at that same
/// `bin\Hostx64\x64` directory on Windows, or the gcc binary on Linux.
fn detect_compiler_bin() -> Result<String, String> {
    // Linux: Use gcc
    if cfg!(unix) {
        if let Ok(p) = std::env::var("CC") {
            if Path::new(&p).is_file() {
                return Ok(p);
            }
            return Err(format!("CC is set to '{}' but it's not a file.", p));
        }

        // Try to find gcc
        if Path::new("/usr/bin/gcc").is_file() {
            return Ok("/usr/bin/gcc".to_string());
        }

        return Err("gcc not found. Install with: sudo apt install build-essential".to_string());
    }

    // Windows: MSVC detection
    if let Ok(p) = std::env::var("MSVC_TOOLS_PATH") {
        if Path::new(&p).join("lib.exe").is_file() {
            return Ok(p);
        }
        return Err(format!(
            "MSVC_TOOLS_PATH is set to '{}' but lib.exe was not found there.",
            p
        ));
    }

    let roots = [
        r"C:\Program Files (x86)\Microsoft Visual Studio",
        r"C:\Program Files\Microsoft Visual Studio",
    ];
    let editions = ["BuildTools", "Community", "Professional", "Enterprise"];

    let mut candidates: Vec<(Vec<u32>, PathBuf)> = Vec::new();

    for root in roots {
        let Ok(root_entries) = fs::read_dir(root) else {
            continue;
        };
        for vs_dir in root_entries.flatten() {
            let name = vs_dir.file_name().to_string_lossy().into_owned();
            if name.is_empty() || !name.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }

            for ed in editions {
                let msvc_root = vs_dir.path().join(ed).join("VC").join("Tools").join("MSVC");
                let Ok(vers) = fs::read_dir(&msvc_root) else {
                    continue;
                };
                for v in vers.flatten() {
                    let vname = v.file_name().to_string_lossy().into_owned();
                    let parts: Option<Vec<u32>> =
                        vname.split('.').map(|s| s.parse().ok()).collect();
                    let Some(parts) = parts else { continue };

                    let bin = v.path().join("bin").join("Hostx64").join("x64");
                    if bin.join("lib.exe").is_file() && bin.join("cl.exe").is_file() {
                        candidates.push((parts, bin));
                    }
                }
            }
        }
    }
    candidates.sort_by(|a, b| b.0.cmp(&a.0));

    candidates
        .into_iter()
        .next()
        .map(|(_, p)| p.to_string_lossy().into_owned())
        .ok_or_else(|| {
            "MSVC BuildTools/Community/Professional/Enterprise not found. \
             Install Visual Studio with the 'Desktop development with C++' \
             workload, or set MSVC_TOOLS_PATH to the directory containing \
             lib.exe and cl.exe (Hostx64\\x64)."
                .to_string()
        })
}

fn main() {
    let cu_path = "src/cuda/atenia_kernels.cu";

    // Auto-detected toolchain paths. Overridable via CUDA_PATH / MSVC_TOOLS_PATH.
    let cuda_path = match detect_cuda_path() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[build.rs] CUDA detection failed: {}", e);
            eprintln!("[build.rs] Skipping CUDA kernel compilation (CPU-only build)");
            return;
        }
    };

    let compiler_bin = match detect_compiler_bin() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[build.rs] Compiler detection failed: {}", e);
            eprintln!("[build.rs] Skipping CUDA kernel compilation (CPU-only build)");
            return;
        }
    };

    let is_windows = cfg!(windows);
    let nvcc_name = if is_windows { "nvcc.exe" } else { "nvcc" };

    // On Ubuntu with nvidia-cuda-toolkit, nvcc is in /usr/bin, not in CUDA_PATH/bin
    let nvcc_path = if cuda_path == "/usr"
        || cuda_path == "/usr/lib/cuda"
        || cuda_path == "/usr/lib/x86_64-linux-gnu"
    {
        "/usr/bin/nvcc".to_string()
    } else {
        format!("{}/bin/{}", cuda_path, nvcc_name)
    };

    // Rebuild when the user changes toolchain overrides.
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=MSVC_TOOLS_PATH");
    println!("cargo:rerun-if-env-changed=CC");

    eprintln!("[build.rs] Detected CUDA:  {}", cuda_path);
    eprintln!("[build.rs] Detected Compiler:  {}", compiler_bin);

    println!("cargo:rerun-if-changed=src/cuda/atenia_kernels.cu");
    println!("cargo:rerun-if-changed=src/cuda/atenia_kernels.h");
    println!("cargo:rerun-if-changed=src/cuda/matmul_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/linear_cuda.cu");
    println!("cargo:rerun-if-changed=src/cuda/batch_matmul.cu");
    println!("cargo:rerun-if-changed=src/cuda/fused_linear_silu.cu");
    println!("cargo:rerun-if-changed=src/cuda/bf16_to_f32.cu");
    println!("cargo:rerun-if-changed=src/cuda/int8_to_bf16.cu");

    eprintln!("[DEBUG] Using NVCC path: {}", nvcc_path);
    eprintln!("[DEBUG] CU file path: {}", cu_path);

    // Compile CUDA kernels
    compile_cuda_kernel(
        &nvcc_path,
        &compiler_bin,
        cu_path,
        "atenia_kernels.obj",
        is_windows,
    );
    compile_cuda_kernel(
        &nvcc_path,
        &compiler_bin,
        "src/cuda/matmul_kernel.cu",
        "matmul_kernel.obj",
        is_windows,
    );
    compile_cuda_kernel(
        &nvcc_path,
        &compiler_bin,
        "src/cuda/linear_cuda.cu",
        "linear_cuda.obj",
        is_windows,
    );
    compile_cuda_kernel(
        &nvcc_path,
        &compiler_bin,
        "src/cuda/batch_matmul.cu",
        "batch_matmul.obj",
        is_windows,
    );
    compile_cuda_kernel(
        &nvcc_path,
        &compiler_bin,
        "src/cuda/fused_linear_silu.cu",
        "fused_linear_silu.obj",
        is_windows,
    );
    compile_cuda_kernel(
        &nvcc_path,
        &compiler_bin,
        "src/cuda/bf16_to_f32.cu",
        "bf16_to_f32.obj",
        is_windows,
    );
    compile_cuda_kernel(
        &nvcc_path,
        &compiler_bin,
        "src/cuda/int8_to_bf16.cu",
        "int8_to_bf16.obj",
        is_windows,
    );

    // Link kernels into static libraries
    if is_windows {
        let lib_exe = format!("{}/lib.exe", compiler_bin);
        create_static_lib(&lib_exe, "atenia_kernels.obj", "atenia_kernels.lib");
        create_static_lib(&lib_exe, "matmul_kernel.obj", "matmul_kernel.lib");
        create_static_lib(&lib_exe, "linear_cuda.obj", "linear_cuda.lib");
        create_static_lib(&lib_exe, "batch_matmul.obj", "batch_matmul.lib");
        create_static_lib(&lib_exe, "fused_linear_silu.obj", "fused_linear_silu.lib");
        create_static_lib(&lib_exe, "bf16_to_f32.obj", "bf16_to_f32.lib");
        create_static_lib(&lib_exe, "int8_to_bf16.obj", "int8_to_bf16.lib");
    } else {
        // Linux: use ar to create static libraries
        create_static_lib_linux("atenia_kernels.obj", "libatenia_kernels.a");
        create_static_lib_linux("matmul_kernel.obj", "libmatmul_kernel.a");
        create_static_lib_linux("linear_cuda.obj", "liblinear_cuda.a");
        create_static_lib_linux("batch_matmul.obj", "libbatch_matmul.a");
        create_static_lib_linux("fused_linear_silu.obj", "libfused_linear_silu.a");
        create_static_lib_linux("bf16_to_f32.obj", "libbf16_to_f32.a");
        create_static_lib_linux("int8_to_bf16.obj", "libint8_to_bf16.a");
    }

    // Link against cudart from the same toolkit we used to compile kernels.
    let cuda_lib_dir = if is_windows {
        format!(r"{}\lib\x64", cuda_path)
    } else if cuda_path == "/usr" || cuda_path == "/usr/lib/x86_64-linux-gnu" {
        "/usr/lib/x86_64-linux-gnu".to_string()
    } else if cuda_path == "/usr/lib/cuda" {
        "/usr/lib/cuda/lib64".to_string()
    } else {
        format!("{}/lib64", cuda_path)
    };
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=cublas");

    // Link our static libraries
    println!("cargo:rustc-link-search=.");
    if is_windows {
        println!("cargo:rustc-link-lib=static=atenia_kernels");
        println!("cargo:rustc-link-lib=static=matmul_kernel");
        println!("cargo:rustc-link-lib=static=linear_cuda");
        println!("cargo:rustc-link-lib=static=batch_matmul");
        println!("cargo:rustc-link-lib=static=fused_linear_silu");
        println!("cargo:rustc-link-lib=static=bf16_to_f32");
    } else {
        println!("cargo:rustc-link-lib=static=atenia_kernels");
        println!("cargo:rustc-link-lib=static=matmul_kernel");
        println!("cargo:rustc-link-lib=static=linear_cuda");
        println!("cargo:rustc-link-lib=static=batch_matmul");
        println!("cargo:rustc-link-lib=static=fused_linear_silu");
        println!("cargo:rustc-link-lib=static=bf16_to_f32");
    }
}

fn compile_cuda_kernel(
    nvcc_path: &str,
    compiler_bin: &str,
    cu_file: &str,
    output_obj: &str,
    is_windows: bool,
) {
    let mut args = vec![];

    if is_windows {
        args.extend(&["-ccbin", compiler_bin]);
        args.extend(&["-Xcompiler", "/MD"]);
        args.extend(&["-Xcompiler", "/EHsc"]);
    } else {
        args.extend(&["-ccbin", compiler_bin]);
    }

    args.extend(&["-c", cu_file, "-o", output_obj]);

    if is_windows {
        args.push("-v");
    }

    let output = Command::new(nvcc_path)
        .args(&args)
        .output()
        .expect("Failed to run NVCC");

    eprintln!("---- NVCC STATUS: {:?} ----", output.status.code());
    eprintln!(
        "---- NVCC STDOUT ----\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    eprintln!(
        "---- NVCC STDERR ----\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    if !output.status.success() {
        panic!("CUDA kernel compilation failed for {}", cu_file);
    }
}

fn create_static_lib(lib_exe: &str, obj_file: &str, lib_name: &str) {
    let output = Command::new(lib_exe)
        .args(&["/OUT:".to_string() + lib_name, obj_file.to_string()])
        .output()
        .expect("Failed to create static library.");

    eprintln!("---- LIB STATUS: {:?} ----", output.status.code());
    eprintln!(
        "---- LIB STDOUT ----\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    eprintln!(
        "---- LIB STDERR ----\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn create_static_lib_linux(obj_file: &str, lib_name: &str) {
    let output = Command::new("ar")
        .args(&["rcs", lib_name, obj_file])
        .output()
        .expect("Failed to create static library with ar.");

    eprintln!("---- AR STATUS: {:?} ----", output.status.code());
    eprintln!(
        "---- AR STDOUT ----\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    eprintln!(
        "---- AR STDERR ----\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
