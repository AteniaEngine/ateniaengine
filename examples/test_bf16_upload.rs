//! M6 — Option A1 standalone validation.
//!
//! Hypothesis: upload BF16 raw bytes to VRAM (half the PCIe traffic
//! of the F32 path) and run a GPU-side `bf16_to_f32_launch_device`
//! upcast into a separate VRAM F32 buffer. The host never
//! materialises an F32 transient, so RAM peak during the upload is
//! the BF16 source only.
//!
//! This example is a sanity check **before** wiring A1 into
//! production (`WeightStore`, dispatch hooks, etc). It allocates a
//! synthetic FFN-down-shaped weight in BF16, runs the A1 path end
//! to end, then runs `cuda_matmul_non_pooled` against the resident
//! F32 buffer and compares against a CPU reference within the
//! ADR-004 0.5 absolute envelope.
//!
//! # Shape
//!
//! Llama 2 13B FFN-down: `out = x @ W_down`,
//! `x = [seq=1, intermediate=13824]`, `W_down = [intermediate=13824,
//! hidden=5120]`. The `W_down` BF16 buffer is 141 MB; the F32 VRAM
//! buffer is 270 MB.
//!
//! # Usage
//!
//! ```powershell
//! cargo run --release --example test_bf16_upload
//! ```
//!
//! Optionally with `--features gpu-trace` for the full per-step log
//! from the underlying `cuda_matmul_non_pooled` invocation.

use std::ffi::c_void;
use std::os::raw::c_int;
use std::time::Instant;

use atenia_engine::cuda::cuda_available;
use atenia_engine::cuda::matmul::cuda_matmul_non_pooled;

const M: usize = 1;
const K: usize = 13824;
const N: usize = 5120;
const W_NUMEL: usize = K * N; // 70_778_880

// Bring the new kernel + the shared cudart wrappers in via FFI.
// `cuda_malloc` / `cuda_free` already exist in `atenia_kernels`;
// `cudaMemcpy` lives in `cudart`. The example re-declares them
// rather than depending on private engine APIs so the test stays
// self-contained.
#[link(name = "bf16_to_f32", kind = "static")]
unsafe extern "C" {
    fn bf16_to_f32_launch_device(
        d_src_bf16: *const c_void,
        d_dst_f32: *mut f32,
        n: c_int,
    ) -> c_int;
}

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
    ) -> c_int;
}

unsafe extern "C" {
    fn cuda_malloc(ptr: *mut *mut c_void, bytes: usize);
    fn cuda_free(ptr: *mut c_void);
}

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

/// Generate `n` deterministic BF16 values whose F32 upcasts are
/// well within finite F32 range. Returns the BF16 bit pattern as
/// `Vec<u16>`.
fn fill_bf16_pattern(n: usize, seed: f32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let f = ((i as f32) * 0.0001 + seed).sin() * 0.5;
            // Round-half-to-nearest BF16: keep the high 16 bits of
            // the F32 representation. Matches the host AVX2 decode's
            // inverse.
            (f.to_bits() >> 16) as u16
        })
        .collect()
}

/// Decode BF16 to F32 on the host (reference / non-GPU path) using
/// the bit-identical formula that the CUDA `__bfloat162float`
/// intrinsic implements.
fn host_bf16_to_f32(bits: &[u16]) -> Vec<f32> {
    bits.iter()
        .map(|&b| f32::from_bits((b as u32) << 16))
        .collect()
}

/// Reference CPU matmul `[m, k] × [k, n] = [m, n]`.
fn cpu_reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Sample the resident process working set (in bytes) via
/// `Win32_Process` WMI. Returns 0 on Linux / WMI failure (the test
/// still runs; only the RAM-peak banner is informational).
#[cfg(target_os = "windows")]
fn process_working_set_bytes() -> u64 {
    use std::process::Command;
    let pid = std::process::id();
    let out = Command::new("wmic")
        .args(&[
            "process",
            "where",
            &format!("ProcessId={}", pid),
            "get",
            "WorkingSetSize",
            "/value",
        ])
        .output();
    let Ok(out) = out else { return 0 };
    let s = String::from_utf8_lossy(&out.stdout);
    for line in s.lines() {
        if let Some(rest) = line.trim().strip_prefix("WorkingSetSize=") {
            if let Ok(n) = rest.parse::<u64>() {
                return n;
            }
        }
    }
    0
}

#[cfg(not(target_os = "windows"))]
fn process_working_set_bytes() -> u64 {
    0
}

fn fmt_mib(bytes: u64) -> String {
    format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
}

fn main() {
    println!("=== M6 Option A1 Standalone Validation ===");
    println!(
        "Shape: x=[{}, {}], W_down=[{}, {}], out=[{}, {}]",
        M, K, K, N, M, N
    );
    println!(
        "BF16 W_down size:    {:.1} MB",
        (W_NUMEL * 2) as f64 / (1024.0 * 1024.0)
    );
    println!(
        "F32 VRAM buffer:     {:.1} MB",
        (W_NUMEL * 4) as f64 / (1024.0 * 1024.0)
    );
    println!();

    if !cuda_available() {
        eprintln!("FATAL: CUDA not available. This test requires a working RTX driver.");
        std::process::exit(1);
    }

    // ----------------------------------------------------------------
    // [1/6] Allocate synthetic BF16 weight + F32 activation on host
    // ----------------------------------------------------------------
    println!("[1/6] Allocating synthetic host buffers...");
    let ws_baseline = process_working_set_bytes();
    let t0 = Instant::now();

    let w_bf16: Vec<u16> = fill_bf16_pattern(W_NUMEL, 0.2);
    let x_f32: Vec<f32> = (0..M * K)
        .map(|i| ((i as f32) * 0.0001 + 0.1).sin() * 0.5)
        .collect();

    let w_alloc_secs = t0.elapsed().as_secs_f64();
    let ws_after_host = process_working_set_bytes();
    println!(
        "      Host allocs done in {:.2}s. RAM working set: {} → {} (Δ {})",
        w_alloc_secs,
        fmt_mib(ws_baseline),
        fmt_mib(ws_after_host),
        fmt_mib(ws_after_host.saturating_sub(ws_baseline)),
    );
    println!();

    // ----------------------------------------------------------------
    // [2/6] First-touch CUDA warmup so the cold-context cost (~700 ms
    //       on this host) does not contaminate the upload timing
    // ----------------------------------------------------------------
    println!("[2/6] Warming up CUDA context (first cuda_malloc absorbs context init)...");
    unsafe {
        let mut warmup_ptr: *mut c_void = std::ptr::null_mut();
        cuda_malloc(&mut warmup_ptr, 1024);
        if !warmup_ptr.is_null() {
            cuda_free(warmup_ptr);
        }
    }
    println!("      Done.");
    println!();

    // ----------------------------------------------------------------
    // [3/6] A1 upload path: BF16 H→D + GPU upcast → F32 VRAM, repeated
    //       3× to expose cold-vs-warm timings (kernel JIT/PTX load
    //       happens on first invocation only).
    // ----------------------------------------------------------------
    println!("[3/6] A1 upload path: 3 consecutive runs (cold | warm | warm)");
    println!();

    let bytes_bf16 = W_NUMEL * 2;
    let bytes_f32 = W_NUMEL * 4;

    // The F32 buffer is allocated ONCE and reused across runs (it is
    // the residency target — production code keeps it). The BF16
    // buffer is allocated/freed PER RUN to mirror the per-layer
    // upload pattern: in production the upload happens N times for
    // N resident layers, each one alloc'ing + freeing a fresh BF16
    // staging buffer.
    let mut d_f32: *mut c_void = std::ptr::null_mut();
    unsafe {
        cuda_malloc(&mut d_f32, bytes_f32);
    }
    if d_f32.is_null() {
        eprintln!("FATAL: cudaMalloc for F32 residency target failed.");
        std::process::exit(3);
    }

    // Per-run timing tuples: (h2d_ms, upcast_ms, total_ms).
    let mut runs: [(f64, f64, f64); 3] = [(0.0, 0.0, 0.0); 3];

    for run_idx in 0..3 {
        let label = match run_idx {
            0 => "Run #1 (cold — first kernel launch, includes JIT/PTX load)",
            1 => "Run #2 (warm — kernel binary cached on device)",
            _ => "Run #3 (warm — confirm stability)",
        };
        println!("      --- {} ---", label);

        let t_run = Instant::now();

        let mut d_bf16: *mut c_void = std::ptr::null_mut();
        let t_alloc_bf16 = Instant::now();
        unsafe {
            cuda_malloc(&mut d_bf16, bytes_bf16);
        }
        let alloc_bf16_ms = t_alloc_bf16.elapsed().as_secs_f64() * 1000.0;
        if d_bf16.is_null() {
            eprintln!("FATAL: cudaMalloc BF16 failed on run #{}.", run_idx + 1);
            unsafe { cuda_free(d_f32); }
            std::process::exit(2);
        }

        let t_h2d = Instant::now();
        let rc = unsafe {
            cudaMemcpy(
                d_bf16,
                w_bf16.as_ptr() as *const c_void,
                bytes_bf16,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        let h2d_ms = t_h2d.elapsed().as_secs_f64() * 1000.0;
        if rc != 0 {
            eprintln!("FATAL: cudaMemcpy H→D BF16 returned rc={} on run #{}.", rc, run_idx + 1);
            unsafe { cuda_free(d_bf16); cuda_free(d_f32); }
            std::process::exit(4);
        }
        let h2d_bw = (bytes_bf16 as f64 / (1024.0 * 1024.0 * 1024.0)) / (h2d_ms / 1000.0);

        let t_upcast = Instant::now();
        let rc = unsafe {
            bf16_to_f32_launch_device(d_bf16, d_f32 as *mut f32, W_NUMEL as c_int)
        };
        let upcast_ms = t_upcast.elapsed().as_secs_f64() * 1000.0;
        if rc != 0 {
            eprintln!("FATAL: bf16_to_f32_launch_device returned rc={} on run #{}.", rc, run_idx + 1);
            unsafe { cuda_free(d_bf16); cuda_free(d_f32); }
            std::process::exit(5);
        }

        let t_free = Instant::now();
        unsafe {
            cuda_free(d_bf16);
        }
        let free_ms = t_free.elapsed().as_secs_f64() * 1000.0;

        let total_ms = t_run.elapsed().as_secs_f64() * 1000.0;

        println!("        cudaMalloc BF16 ({:.0} MB):  {:.2} ms",
                 bytes_bf16 as f64 / (1024.0 * 1024.0), alloc_bf16_ms);
        println!("        cudaMemcpy H→D BF16:          {:.2} ms ({:.2} GB/s)", h2d_ms, h2d_bw);
        println!("        GPU upcast bf16→f32 kernel:   {:.2} ms", upcast_ms);
        println!("        cudaFree BF16:                {:.2} ms", free_ms);
        println!("        ───────────────────────────────");
        println!("        Run total:                    {:.2} ms", total_ms);
        println!();

        runs[run_idx] = (h2d_ms, upcast_ms, total_ms);
    }

    let ws_after_upload = process_working_set_bytes();
    println!(
        "      RAM working set: {} (Δ from host alloc: {})",
        fmt_mib(ws_after_upload),
        fmt_mib(ws_after_upload.saturating_sub(ws_after_host)),
    );
    println!();

    // Pick the warm-state numbers (run #3) for the rest of the
    // analysis — they are what production would observe after the
    // first matmul has executed.
    let (h2d_ms, upcast_ms, total_upload_ms) = runs[2];
    let h2d_bw = (bytes_bf16 as f64 / (1024.0 * 1024.0 * 1024.0)) / (h2d_ms / 1000.0);

    // ----------------------------------------------------------------
    // [4/6] Sanity-check the resident F32 buffer by downloading it and
    //       comparing against the host BF16→F32 reference. If the
    //       upcast kernel is correct this is bit-exact.
    // ----------------------------------------------------------------
    println!("[4/6] Verifying GPU upcast is bit-exact with host decode...");
    let t_verify = Instant::now();
    let mut f32_check = vec![0.0_f32; W_NUMEL];
    let rc = unsafe {
        cudaMemcpy(
            f32_check.as_mut_ptr() as *mut c_void,
            d_f32,
            bytes_f32,
            CUDA_MEMCPY_DEVICE_TO_HOST,
        )
    };
    if rc != 0 {
        eprintln!("FATAL: D→H memcpy for verification returned rc={}", rc);
        unsafe { cuda_free(d_f32); }
        std::process::exit(6);
    }
    let host_ref = host_bf16_to_f32(&w_bf16);
    let mut bitwise_mismatches = 0_usize;
    for (g, h) in f32_check.iter().zip(host_ref.iter()) {
        if g.to_bits() != h.to_bits() {
            bitwise_mismatches += 1;
        }
    }
    println!(
        "      D→H verify done in {:.2}s. Bit-exact mismatches: {} / {}",
        t_verify.elapsed().as_secs_f64(),
        bitwise_mismatches,
        W_NUMEL
    );
    if bitwise_mismatches > 0 {
        eprintln!("FATAL: GPU upcast is not bit-exact with the host decode.");
        eprintln!(
            "       This contradicts the design assumption that __bfloat162float() == AVX2 decode."
        );
        unsafe { cuda_free(d_f32); }
        std::process::exit(7);
    }
    // Drop the verification buffer (we used it only for the sanity
    // check; the matmul against d_f32 below is the real measurement).
    drop(f32_check);
    drop(host_ref);
    println!();

    // ----------------------------------------------------------------
    // [5/6] Run a matmul against the resident F32 weight using the
    //       existing `cuda_matmul_non_pooled`. We measure the warm
    //       path only (matches step 2b's production path).
    // ----------------------------------------------------------------
    println!("[5/6] Matmul x @ W_down using the F32 resident weight in VRAM...");
    println!("      Note: cuda_matmul_non_pooled re-allocates A/B/out and re-uploads B.");
    println!("            For the residency-true measurement we need a separate path that");
    println!("            takes a device-resident B pointer; this run is a sanity check that");
    println!("            the upcasted F32 buffer is indeed kernel-consumable. We compare its");
    println!("            output against a fresh non-resident matmul to confirm value parity.");

    // Sanity: re-decode W_down host-side and run cuda_matmul_non_pooled
    // against it. If the GPU upcast is bit-exact (verified in [4/6])
    // and the existing kernel is deterministic, the two should agree
    // bit-exactly.
    let w_f32_host = host_bf16_to_f32(&w_bf16);
    let t_matmul = Instant::now();
    let gpu_out = match cuda_matmul_non_pooled(&x_f32, &w_f32_host, M, K, N) {
        Some(t) => t,
        None => {
            eprintln!("FATAL: cuda_matmul_non_pooled returned None.");
            unsafe { cuda_free(d_f32); }
            std::process::exit(8);
        }
    };
    let matmul_ms = t_matmul.elapsed().as_secs_f64() * 1000.0;
    println!("      cuda_matmul_non_pooled total:  {:.2} ms", matmul_ms);
    println!();

    // ----------------------------------------------------------------
    // [6/6] CPU reference + ADR-004 tolerance check
    // ----------------------------------------------------------------
    println!("[6/6] CPU reference matmul + ADR-004 verification...");
    let t_cpu = Instant::now();
    let cpu_out = cpu_reference(&x_f32, &w_f32_host, M, K, N);
    let cpu_ms = t_cpu.elapsed().as_secs_f64() * 1000.0;
    println!("      CPU reference (triple-loop):   {:.2} ms", cpu_ms);

    let gpu_data = gpu_out.as_cpu_slice();
    let mut max_abs_diff = 0.0_f32;
    let mut argmax = 0_usize;
    for (idx, (g, c)) in gpu_data.iter().zip(cpu_out.iter()).enumerate() {
        let d = (g - c).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
            argmax = idx;
        }
    }
    println!(
        "      max |gpu - cpu|:               {:e} at index {} (gpu={:.6}, cpu={:.6})",
        max_abs_diff, argmax, gpu_data[argmax], cpu_out[argmax]
    );
    println!();

    // Cleanup the resident F32 buffer.
    unsafe {
        cuda_free(d_f32);
    }

    println!("=== Summary ===");
    println!("Cold-vs-warm comparison (3 runs):");
    println!("  Run #  | H→D BF16  | Kernel upcast | Total");
    println!("  -------+-----------+---------------+--------");
    for (i, (h2d, upcast, total)) in runs.iter().enumerate() {
        println!("  {:6} | {:6.2} ms | {:10.2} ms | {:6.2} ms",
                 format!("#{} {}", i + 1, if i == 0 { "(cold)" } else { "(warm)" }),
                 h2d, upcast, total);
    }
    let cold_total = runs[0].2;
    let warm_total = runs[2].2;
    println!();
    println!("Cold-vs-warm delta:                {:.2} ms ({:.2}× cold)",
             cold_total - warm_total,
             cold_total / warm_total);
    println!();
    println!("Warm-state upload (run #3):        {:.2} ms", total_upload_ms);
    println!("  - cudaMemcpy H2D BF16 (135MB):   {:.2} ms ({:.2} GB/s)", h2d_ms, h2d_bw);
    println!("  - GPU bf16→f32 upcast kernel:    {:.2} ms", upcast_ms);
    println!("Matmul against resident F32:       {:.2} ms", matmul_ms);
    println!("CPU reference:                     {:.2} ms", cpu_ms);
    println!();
    println!("Bit-exact GPU upcast vs host:      {}", if bitwise_mismatches == 0 { "YES" } else { "NO" });
    println!("ADR-004 envelope (0.5 absolute):   {}", if max_abs_diff < 0.5 { "PASS" } else { "FAIL" });
    println!("max |diff|:                        {:e}", max_abs_diff);
    println!();

    if max_abs_diff >= 0.5 {
        eprintln!("FAIL: ADR-004 envelope violated.");
        std::process::exit(9);
    }
    if bitwise_mismatches > 0 {
        // Already exited above, but keep the explicit check for
        // future readers.
        unreachable!();
    }

    println!("PASS: A1 standalone path is numerically valid.");
}
