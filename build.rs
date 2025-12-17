use std::process::Command;

fn main() {
    // Usar ruta NORMAL, NO canonicalize()
    let cu_path = "src/cuda/atenia_kernels.cu";

    // Ruta explícita al compilador MSVC (cl.exe) según tu instalación
    let msvc_bin = r#"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"#;
    let lib_exe = format!(r"{}\lib.exe", msvc_bin);

    // Ruta explícita a nvcc (ajusta si quieres usar v13.0)
    let nvcc_path = r#"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe"#;
    // Alternativa:
    // let nvcc_path = r#"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\nvcc.exe"#;

    println!("cargo:rerun-if-changed=src/cuda/atenia_kernels.cu");
    println!("cargo:rerun-if-changed=src/cuda/atenia_kernels.h");
    println!("cargo:rerun-if-changed=src/cuda/matmul_kernel.cu");
    println!("cargo:rerun-if-changed=src/cuda/matmul_kernel.h");
    println!("cargo:rerun-if-changed=src/cuda/linear_cuda.cu");
    println!("cargo:rerun-if-changed=src/cuda/batch_matmul.cu");
    println!("cargo:rerun-if-changed=src/cuda/fused_linear_silu.cu");

    eprintln!("[DEBUG] Using NVCC path: {}", nvcc_path);
    eprintln!("[DEBUG] CU file path: {}", cu_path);

    let output = Command::new(nvcc_path)
        .args(&[
            "-ccbin",
            msvc_bin,
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
    let matmul_out = Command::new(nvcc_path)
        .args(&[
            "-ccbin",
            msvc_bin,
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
    let linear_out = Command::new(nvcc_path)
        .args(&[
            "-ccbin",
            msvc_bin,
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
    let bmm_out = Command::new(nvcc_path)
        .args(&[
            "-ccbin",
            msvc_bin,
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
    let fls_out = Command::new(nvcc_path)
        .args(&[
            "-ccbin",
            msvc_bin,
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

    // Añadir ruta a cudart.lib usando CUDA_PATH
    let cuda_path =
        std::env::var("CUDA_PATH").unwrap_or_else(|_| r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6".into());
    let cuda_lib_dir = format!(r"{}\lib\x64", cuda_path);
    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    println!("cargo:rustc-link-lib=dylib=cudart");

    // Nuestra lib está en el cwd
    println!("cargo:rustc-link-search=.");
    println!("cargo:rustc-link-lib=static=atenia_kernels");
    println!("cargo:rustc-link-lib=static=matmul_kernel");
    println!("cargo:rustc-link-lib=static=linear_cuda");
    println!("cargo:rustc-link-lib=static=batch_matmul");
    println!("cargo:rustc-link-lib=static=fused_linear_silu");
}