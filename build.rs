use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Detects the highest installed CUDA Toolkit version under the default
/// NVIDIA install root, or respects the `CUDA_PATH` env var if set.
///
/// Returns the full toolkit directory (e.g.
/// `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`).
fn detect_cuda_path() -> Result<String, String> {
    // Manual override via env var.
    if let Ok(p) = std::env::var("CUDA_PATH") {
        if Path::new(&p).join("bin").join("nvcc.exe").is_file() {
            return Ok(p);
        }
        return Err(format!(
            "CUDA_PATH is set to '{}' but nvcc.exe was not found there.",
            p
        ));
    }

    // Default Windows install location.
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
        let Some(rest) = name.strip_prefix('v') else { continue };
        let Some((maj, min)) = rest.split_once('.') else { continue };
        let (Ok(m), Ok(n)) = (maj.parse::<u32>(), min.parse::<u32>()) else { continue };
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
/// Respects `MSVC_TOOLS_PATH` as a manual override pointing at that same
/// `bin\Hostx64\x64` directory.
fn detect_msvc_bin() -> Result<String, String> {
    // Manual override via env var.
    if let Ok(p) = std::env::var("MSVC_TOOLS_PATH") {
        if Path::new(&p).join("lib.exe").is_file() {
            return Ok(p);
        }
        return Err(format!(
            "MSVC_TOOLS_PATH is set to '{}' but lib.exe was not found there.",
            p
        ));
    }

    // Visual Studio roots. Modern VS uses a numeric version (18, 19, ...);
    // older layouts use the year (2019, 2022). Accept any purely numeric
    // subfolder so the detection covers both schemes.
    let roots = [
        r"C:\Program Files (x86)\Microsoft Visual Studio",
        r"C:\Program Files\Microsoft Visual Studio",
    ];
    let editions = ["BuildTools", "Community", "Professional", "Enterprise"];

    // Score entries by MSVC version parts descending; the highest wins.
    let mut candidates: Vec<(Vec<u32>, PathBuf)> = Vec::new();

    for root in roots {
        let Ok(root_entries) = fs::read_dir(root) else { continue };
        for vs_dir in root_entries.flatten() {
            let name = vs_dir.file_name().to_string_lossy().into_owned();
            if name.is_empty() || !name.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }

            for ed in editions {
                let msvc_root = vs_dir
                    .path()
                    .join(ed)
                    .join("VC")
                    .join("Tools")
                    .join("MSVC");
                let Ok(vers) = fs::read_dir(&msvc_root) else { continue };
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
    let cuda_path = detect_cuda_path().unwrap_or_else(|e| panic!("{}", e));
    let nvcc_path = format!(r"{}\bin\nvcc.exe", cuda_path);
    let msvc_bin = detect_msvc_bin().unwrap_or_else(|e| panic!("{}", e));
    let lib_exe = format!(r"{}\lib.exe", msvc_bin);

    // Rebuild when the user changes toolchain overrides.
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=MSVC_TOOLS_PATH");

    eprintln!("[build.rs] Detected CUDA:  {}", cuda_path);
    eprintln!("[build.rs] Detected MSVC:  {}", msvc_bin);

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

    let output = Command::new(&nvcc_path)
        .args(&[
            "-ccbin",
            msvc_bin.as_str(),
            "-c",
            cu_path,
            "-o", "atenia_kernels.obj",
            "-Xcompiler", "/MD",
            "-Xcompiler", "/EHsc",
            "-v",
        ])
        .output()
        .expect("Failed to run NVCC");

    eprintln!("---- NVCC STATUS: {:?} ----", output.status.code());
    eprintln!("---- NVCC STDOUT ----\n{}", String::from_utf8_lossy(&output.stdout));
    eprintln!("---- NVCC STDERR ----\n{}", String::from_utf8_lossy(&output.stderr));

    if !output.status.success() {
        panic!("CUDA kernel compilation failed");
    }

    let output_lib = Command::new(&lib_exe)
        .args(&["/OUT:atenia_kernels.lib", "atenia_kernels.obj"])
        .output()
        .expect("Failed to create static library.");

    eprintln!("---- LIB STATUS: {:?} ----", output_lib.status.code());
    eprintln!("---- LIB STDOUT ----\n{}", String::from_utf8_lossy(&output_lib.stdout));
    eprintln!("---- LIB STDERR ----\n{}", String::from_utf8_lossy(&output_lib.stderr));

    // Compile matmul kernel
    let matmul_out = Command::new(&nvcc_path)
        .args(&[
            "-ccbin",
            msvc_bin.as_str(),
            "-c",
            "src/cuda/matmul_kernel.cu",
            "-o",
            "matmul_kernel.obj",
            "-Xcompiler",
            "/MD",
            "-Xcompiler",
            "/EHsc",
        ])
        .output()
        .expect("Failed to run NVCC (matmul)");

    eprintln!("---- MATMUL NVCC STATUS: {:?} ----", matmul_out.status.code());
    eprintln!(
        "---- MATMUL NVCC STDOUT ----\n{}",
        String::from_utf8_lossy(&matmul_out.stdout)
    );
    eprintln!(
        "---- MATMUL NVCC STDERR ----\n{}",
        String::from_utf8_lossy(&matmul_out.stderr)
    );

    if !matmul_out.status.success() {
        panic!("CUDA matmul kernel compilation failed");
    }

    let matmul_lib = Command::new(&lib_exe)
        .args(&["/OUT:matmul_kernel.lib", "matmul_kernel.obj"])
        .output()
        .expect("Failed to create matmul library.");

    eprintln!("---- MATMUL LIB STATUS: {:?} ----", matmul_lib.status.code());
    eprintln!(
        "---- MATMUL LIB STDOUT ----\n{}",
        String::from_utf8_lossy(&matmul_lib.stdout)
    );
    eprintln!(
        "---- MATMUL LIB STDERR ----\n{}",
        String::from_utf8_lossy(&matmul_lib.stderr)
    );

    // Compile linear CUDA kernel into separate static library
    let linear_out = Command::new(&nvcc_path)
        .args(&[
            "-ccbin",
            msvc_bin.as_str(),
            "-c",
            "src/cuda/linear_cuda.cu",
            "-o",
            "linear_cuda.obj",
            "-Xcompiler",
            "/MD",
            "-Xcompiler",
            "/EHsc",
        ])
        .output()
        .expect("Failed to run NVCC (linear)");

    eprintln!("---- LINEAR NVCC STATUS: {:?} ----", linear_out.status.code());
    eprintln!(
        "---- LINEAR NVCC STDOUT ----\n{}",
        String::from_utf8_lossy(&linear_out.stdout)
    );
    eprintln!(
        "---- LINEAR NVCC STDERR ----\n{}",
        String::from_utf8_lossy(&linear_out.stderr)
    );

    if !linear_out.status.success() {
        panic!("CUDA linear kernel compilation failed");
    }

    let linear_lib = Command::new(&lib_exe)
        .args(&["/OUT:linear_cuda.lib", "linear_cuda.obj"])
        .output()
        .expect("Failed to create linear_cuda library.");

    eprintln!("---- LINEAR LIB STATUS: {:?} ----", linear_lib.status.code());
    eprintln!(
        "---- LINEAR LIB STDOUT ----\n{}",
        String::from_utf8_lossy(&linear_lib.stdout)
    );
    eprintln!(
        "---- LINEAR LIB STDERR ----\n{}",
        String::from_utf8_lossy(&linear_lib.stderr)
    );

    // Compile batch_matmul CUDA kernel into separate static library
    let bmm_out = Command::new(&nvcc_path)
        .args(&[
            "-ccbin",
            msvc_bin.as_str(),
            "-c",
            "src/cuda/batch_matmul.cu",
            "-o",
            "batch_matmul.obj",
            "-Xcompiler",
            "/MD",
            "-Xcompiler",
            "/EHsc",
        ])
        .output()
        .expect("Failed to run NVCC (batch_matmul)");

    eprintln!("---- BATCH_MATMUL NVCC STATUS: {:?} ----", bmm_out.status.code());
    eprintln!(
        "---- BATCH_MATMUL NVCC STDOUT ----\n{}",
        String::from_utf8_lossy(&bmm_out.stdout)
    );
    eprintln!(
        "---- BATCH_MATMUL NVCC STDERR ----\n{}",
        String::from_utf8_lossy(&bmm_out.stderr)
    );

    if !bmm_out.status.success() {
        panic!("CUDA batch_matmul kernel compilation failed");
    }

    let bmm_lib = Command::new(&lib_exe)
        .args(&["/OUT:batch_matmul.lib", "batch_matmul.obj"])
        .output()
        .expect("Failed to create batch_matmul library.");

    eprintln!("---- BATCH_MATMUL LIB STATUS: {:?} ----", bmm_lib.status.code());
    eprintln!(
        "---- BATCH_MATMUL LIB STDOUT ----\n{}",
        String::from_utf8_lossy(&bmm_lib.stdout)
    );
    eprintln!(
        "---- BATCH_MATMUL LIB STDERR ----\n{}",
        String::from_utf8_lossy(&bmm_lib.stderr)
    );

    // Compile fused_linear_silu CUDA kernel into separate static library
    let fls_out = Command::new(&nvcc_path)
        .args(&[
            "-ccbin",
            msvc_bin.as_str(),
            "-c",
            "src/cuda/fused_linear_silu.cu",
            "-o",
            "fused_linear_silu.obj",
            "-Xcompiler",
            "/MD",
            "-Xcompiler",
            "/EHsc",
        ])
        .output()
        .expect("Failed to run NVCC (fused_linear_silu)");

    eprintln!("---- FUSED_LINEAR_SILU NVCC STATUS: {:?} ----", fls_out.status.code());
    eprintln!(
        "---- FUSED_LINEAR_SILU NVCC STDOUT ----\n{}",
        String::from_utf8_lossy(&fls_out.stdout)
    );
    eprintln!(
        "---- FUSED_LINEAR_SILU NVCC STDERR ----\n{}",
        String::from_utf8_lossy(&fls_out.stderr)
    );

    if !fls_out.status.success() {
        panic!("CUDA fused_linear_silu kernel compilation failed");
    }

    let fls_lib = Command::new(&lib_exe)
        .args(&["/OUT:fused_linear_silu.lib", "fused_linear_silu.obj"])
        .output()
        .expect("Failed to create fused_linear_silu library.");

    eprintln!("---- FUSED_LINEAR_SILU LIB STATUS: {:?} ----", fls_lib.status.code());
    eprintln!(
        "---- FUSED_LINEAR_SILU LIB STDOUT ----\n{}",
        String::from_utf8_lossy(&fls_lib.stdout)
    );
    eprintln!(
        "---- FUSED_LINEAR_SILU LIB STDERR ----\n{}",
        String::from_utf8_lossy(&fls_lib.stderr)
    );

    // Compile bf16_to_f32 CUDA kernel into separate static library
    let bf16_out = Command::new(&nvcc_path)
        .args(&[
            "-ccbin",
            msvc_bin.as_str(),
            "-c",
            "src/cuda/bf16_to_f32.cu",
            "-o",
            "bf16_to_f32.obj",
            "-Xcompiler",
            "/MD",
            "-Xcompiler",
            "/EHsc",
        ])
        .output()
        .expect("Failed to run NVCC (bf16_to_f32)");

    eprintln!("---- BF16_TO_F32 NVCC STATUS: {:?} ----", bf16_out.status.code());
    eprintln!(
        "---- BF16_TO_F32 NVCC STDOUT ----\n{}",
        String::from_utf8_lossy(&bf16_out.stdout)
    );
    eprintln!(
        "---- BF16_TO_F32 NVCC STDERR ----\n{}",
        String::from_utf8_lossy(&bf16_out.stderr)
    );

    if !bf16_out.status.success() {
        panic!("CUDA bf16_to_f32 kernel compilation failed");
    }

    let bf16_lib = Command::new(&lib_exe)
        .args(&["/OUT:bf16_to_f32.lib", "bf16_to_f32.obj"])
        .output()
        .expect("Failed to create bf16_to_f32 library.");

    eprintln!("---- BF16_TO_F32 LIB STATUS: {:?} ----", bf16_lib.status.code());
    eprintln!(
        "---- BF16_TO_F32 LIB STDOUT ----\n{}",
        String::from_utf8_lossy(&bf16_lib.stdout)
    );
    eprintln!(
        "---- BF16_TO_F32 LIB STDERR ----\n{}",
        String::from_utf8_lossy(&bf16_lib.stderr)
    );

    // M9.0 — INT8 → BF16 per-channel dequant kernel for the
    // `examples/bench_int8_w8a16.rs` microbench (gating data
    // for M9 INT8 weight quantisation).
    let int8_out = Command::new(&nvcc_path)
        .args(&[
            "-ccbin",
            msvc_bin.as_str(),
            "-c",
            "src/cuda/int8_to_bf16.cu",
            "-o",
            "int8_to_bf16.obj",
            "-Xcompiler",
            "/MD",
            "-Xcompiler",
            "/EHsc",
        ])
        .output()
        .expect("Failed to run NVCC (int8_to_bf16)");

    eprintln!("---- INT8_TO_BF16 NVCC STATUS: {:?} ----", int8_out.status.code());
    eprintln!(
        "---- INT8_TO_BF16 NVCC STDOUT ----\n{}",
        String::from_utf8_lossy(&int8_out.stdout)
    );
    eprintln!(
        "---- INT8_TO_BF16 NVCC STDERR ----\n{}",
        String::from_utf8_lossy(&int8_out.stderr)
    );

    if !int8_out.status.success() {
        panic!("CUDA int8_to_bf16 kernel compilation failed");
    }

    let int8_lib = Command::new(&lib_exe)
        .args(&["/OUT:int8_to_bf16.lib", "int8_to_bf16.obj"])
        .output()
        .expect("Failed to create int8_to_bf16 library.");

    eprintln!("---- INT8_TO_BF16 LIB STATUS: {:?} ----", int8_lib.status.code());
    eprintln!(
        "---- INT8_TO_BF16 LIB STDOUT ----\n{}",
        String::from_utf8_lossy(&int8_lib.stdout)
    );
    eprintln!(
        "---- INT8_TO_BF16 LIB STDERR ----\n{}",
        String::from_utf8_lossy(&int8_lib.stderr)
    );

    // Link against cudart from the same toolkit we used to compile kernels.
    let cuda_lib_dir = format!(r"{}\lib\x64", cuda_path);
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    println!("cargo:rustc-link-lib=dylib=cudart");
    // M8.0 — link cuBLAS for the BF16 TC bench (`examples/bench_cublas_bf16.rs`)
    // and any future production path that needs cublasGemmEx. Same toolkit as
    // cudart so the versions stay coherent.
    println!("cargo:rustc-link-lib=dylib=cublas");

    // Nuestra lib está en el cwd
    println!("cargo:rustc-link-search=.");
    println!("cargo:rustc-link-lib=static=atenia_kernels");
    println!("cargo:rustc-link-lib=static=matmul_kernel");
    println!("cargo:rustc-link-lib=static=linear_cuda");
    println!("cargo:rustc-link-lib=static=batch_matmul");
    println!("cargo:rustc-link-lib=static=fused_linear_silu");
    println!("cargo:rustc-link-lib=static=bf16_to_f32");
}