//! M8.0b — NVMe → PCIe → GPU pipeline async bench.
//!
//! **Second gating data point for the M8 milestone.** M8.0
//! demonstrated that the GPU matmul itself is already at ~95 % of
//! the RTX 4070 Laptop's peak memory bandwidth on decode-step
//! shapes — so the 5×-on-kernel speedup we hoped for from BF16
//! Tensor Cores does not exist (max 2.3× from bandwidth ratio).
//!
//! That left the M8 ROI question hinging on whether the **disk-
//! resident weights** (29 layers in the M7.3 13B smoke) can be
//! pumped through `NVMe → PCIe → GPU compute` fast enough to
//! reach the headline 5–6 s/tok target. This bench measures the
//! pipeline overlap empirically on the operator's hardware.
//!
//! # The shape
//!
//! Single shape — Llama 2 13B FFN-down:
//!
//! ```text
//! [1, 13824] × [13824, 5120] = [1, 5120]
//! ```
//!
//! The `B` weight is 135 MiB BF16 — the largest weight per layer
//! in 13B and the dominant per-layer disk-tier read.
//!
//! # The three configurations
//!
//! ## Config 1 — Sequential (baseline)
//!
//! For every iteration:
//! ```text
//!   read NVMe  →  cudaMemcpy H→D  →  cublasGemmEx  →  syncronize
//! ```
//! No overlap. Worst case. The number M7.3 production effectively
//! pays today minus the CPU matmul cost (which is even slower
//! than this baseline). Establishes the ceiling above which the
//! pipeline configurations need to land.
//!
//! ## Config 2 — Two-buffer pipeline
//!
//! Two staging buffers in VRAM, two pinned host buffers. CUDA
//! `copy_stream` and `compute_stream` run independently. While
//! the GPU is computing on slot `i`, the CPU is reading + DMA-
//! uploading slot `i+1`. Steady-state throughput ≈ max(read,
//! upload+compute_wait).
//!
//! ## Config 3 — Triple-buffer pipeline
//!
//! Three staging buffers. Slot `i` is computing, slot `i+1` is
//! uploading, slot `i+2` is being read. Fully decoupled stages.
//! Steady-state throughput ≈ max(read, upload, compute).
//!
//! # Decision criteria (for Config 2)
//!
//! - **≤ 200 ms / iter**: PLAN A. Pipeline async delivers the
//!   target. Proceed M8.1-M8.4 (BF16 VRAM infrastructure) +
//!   M8.7 (Disk → GPU JIT path).
//! - **200–400 ms / iter**: PARTIAL. Pipeline is partially
//!   effective. Compare ROI vs M9 (INT8 quantization).
//! - **> 400 ms / iter**: FAIL. Pipeline does not help on this
//!   hardware. Either the NVMe is the cap, or PCIe overlap is
//!   absent. Pivot to M9 INT8 or accept M7 as the v20 closure.
//!
//! # Cache behaviour
//!
//! Files are read with `FILE_FLAG_NO_BUFFERING` on Windows so the
//! OS page cache cannot mask NVMe latency. This is the worst-case
//! scenario for the pipeline. Production reads (no NO_BUFFERING)
//! will be **at least as fast** because cache hits eliminate the
//! NVMe stage entirely on warm reads. We intentionally bias
//! pessimistic — if the pipeline survives cold-cache, it will
//! survive warm-cache trivially.
//!
//! # Usage
//!
//! ```powershell
//! $env:ATENIA_M8_BENCH_DIR = "D:\atenia-m8-pipeline"   # NVMe dir
//! cargo run --release --example bench_disk_gpu_pipeline
//! ```
//!
//! The bench creates 10 distinct 135 MiB files in `ATENIA_M8_BENCH_DIR`
//! (1.35 GiB total) on first run. Subsequent runs reuse them.
//! Default: `%LOCALAPPDATA%\Atenia\m8_pipeline_bench`.

use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::raw::c_int;
use std::path::{Path, PathBuf};
use std::time::Instant;

use atenia_engine::tensor::tensor::f32_to_bf16_bits;

#[cfg(windows)]
use std::os::windows::fs::OpenOptionsExt;

// ---------------------------------------------------------------------------
// Bench shape — Llama 2 13B FFN-down [1, 13824] × [13824, 5120]
// ---------------------------------------------------------------------------

const M_DIM: usize = 1;
const K_DIM: usize = 13824;
const N_DIM: usize = 5120;
const NUMEL_B: usize = K_DIM * N_DIM; // weight numel
const BYTES_B: usize = NUMEL_B * 2; // BF16 = 141_557_760 bytes (135 MiB)

const N_FILES: usize = 10;
const WARMUP_ITERS: usize = 3;
const MEASURED_ITERS: usize = 10;

#[cfg(windows)]
const FILE_FLAG_NO_BUFFERING: u32 = 0x20000000;
#[cfg(windows)]
const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x08000000;

// ---------------------------------------------------------------------------
// CUDA / cuBLAS bindings (only what the bench needs)
// ---------------------------------------------------------------------------

#[link(name = "cudart")]
unsafe extern "C" {
    fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> c_int;
    fn cudaFree(ptr: *mut c_void) -> c_int;
    fn cudaMallocHost(ptr: *mut *mut c_void, bytes: usize) -> c_int;
    fn cudaFreeHost(ptr: *mut c_void) -> c_int;
    fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int;
    fn cudaMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
        stream: cudaStream_t,
    ) -> c_int;
    fn cudaDeviceSynchronize() -> c_int;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int;

    fn cudaStreamCreate(stream: *mut cudaStream_t) -> c_int;
    fn cudaStreamDestroy(stream: cudaStream_t) -> c_int;
    fn cudaStreamWaitEvent(stream: cudaStream_t, event: cudaEvent_t, flags: c_int) -> c_int;

    fn cudaEventCreate(event: *mut cudaEvent_t) -> c_int;
    fn cudaEventDestroy(event: cudaEvent_t) -> c_int;
    fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> c_int;
    fn cudaEventSynchronize(event: cudaEvent_t) -> c_int;
}

#[allow(non_camel_case_types)]
type cudaStream_t = *mut c_void;
#[allow(non_camel_case_types)]
type cudaEvent_t = *mut c_void;
#[allow(non_camel_case_types)]
type cublasHandle_t = *mut c_void;

const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
const CUBLAS_OP_N: c_int = 0;
const CUDA_R_32F: c_int = 0;
const CUDA_R_16BF: c_int = 14;
const CUBLAS_COMPUTE_32F: c_int = 68;
const CUBLAS_GEMM_DEFAULT: c_int = -1;

#[link(name = "cublas")]
unsafe extern "C" {
    fn cublasCreate_v2(handle: *mut cublasHandle_t) -> c_int;
    fn cublasDestroy_v2(handle: cublasHandle_t) -> c_int;
    fn cublasSetStream_v2(handle: cublasHandle_t, stream: cudaStream_t) -> c_int;
    fn cublasGemmEx(
        handle: cublasHandle_t,
        transa: c_int,
        transb: c_int,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: c_int,
        lda: c_int,
        b: *const c_void,
        b_type: c_int,
        ldb: c_int,
        beta: *const c_void,
        c: *mut c_void,
        c_type: c_int,
        ldc: c_int,
        compute_type: c_int,
        algo: c_int,
    ) -> c_int;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn check_cuda(rc: c_int, what: &str) {
    if rc != 0 {
        eprintln!("FATAL: {} returned cuda rc={}", what, rc);
        std::process::exit(1);
    }
}

fn check_cublas(rc: c_int, what: &str) {
    if rc != 0 {
        eprintln!("FATAL: {} returned cuBLAS status={}", what, rc);
        std::process::exit(1);
    }
}

fn fmt_size(bytes: usize) -> String {
    let mib = bytes as f64 / (1024.0 * 1024.0);
    if mib >= 1024.0 {
        format!("{:.2} GiB", mib / 1024.0)
    } else {
        format!("{:.1} MiB", mib)
    }
}

fn check_vram_budget(label: &str) {
    let (mut free_now, mut total_now): (usize, usize) = (0, 0);
    unsafe {
        check_cuda(
            cudaMemGetInfo(&mut free_now, &mut total_now),
            "cudaMemGetInfo",
        );
    }
    let used_now = total_now.saturating_sub(free_now);
    if used_now > 7 * 1024 * 1024 * 1024 {
        eprintln!(
            "FATAL [{}]: VRAM in use exceeded 7 GiB rollback floor: {} used / {} total",
            label,
            fmt_size(used_now),
            fmt_size(total_now)
        );
        std::process::exit(2);
    }
}

fn default_bench_dir() -> PathBuf {
    if let Ok(v) = std::env::var("ATENIA_M8_BENCH_DIR") {
        return PathBuf::from(v);
    }
    if let Ok(v) = std::env::var("LOCALAPPDATA") {
        return PathBuf::from(v).join("Atenia").join("m8_pipeline_bench");
    }
    PathBuf::from("./atenia_m8_pipeline_bench")
}

/// Generate `N_FILES` distinct 135 MiB BF16 files. Each file's
/// content is deterministic but unique. Skips files that already
/// exist with the right size.
///
/// The generation phase deliberately avoids `FILE_FLAG_NO_BUFFERING`
/// — we let the OS page cache populate during writes. Reads in the
/// timing loop, however, do use NO_BUFFERING to bypass the cache.
fn ensure_bench_files(dir: &Path) -> Vec<PathBuf> {
    std::fs::create_dir_all(dir).expect("create bench dir");
    let mut paths = Vec::with_capacity(N_FILES);
    for i in 0..N_FILES {
        let p = dir.join(format!("layer_{}.bin", i));
        let regen = match std::fs::metadata(&p) {
            Ok(m) => (m.len() as usize) != BYTES_B,
            Err(_) => true,
        };
        if regen {
            // Synthesize unique content per file.
            let mut bits: Vec<u16> = Vec::with_capacity(NUMEL_B);
            for j in 0..NUMEL_B {
                let f = ((j as f32) * 0.0001 + (i as f32) * 0.137).sin() * 0.5
                    + ((j as f32) * 0.00007 + (i as f32) * 0.42).cos() * 0.4;
                bits.push(f32_to_bf16_bits(f));
            }
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(bits.as_ptr() as *const u8, BYTES_B) };
            let mut f = File::create(&p).expect("create file");
            f.write_all(bytes).expect("write file");
            f.sync_all().expect("sync file");
        }
        paths.push(p);
    }
    paths
}

/// Open a file with NO_BUFFERING + SEQUENTIAL_SCAN flags on Windows
/// to bypass the OS page cache for a worst-case NVMe-only read
/// measurement. On non-Windows this falls back to a regular open.
fn open_nvmebench(path: &Path) -> File {
    #[cfg(windows)]
    {
        OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN)
            .open(path)
            .expect("open file with NO_BUFFERING")
    }
    #[cfg(not(windows))]
    {
        File::open(path).expect("open file")
    }
}

// ---------------------------------------------------------------------------
// Configurations
// ---------------------------------------------------------------------------

/// Result tuple per config: (avg_ms_per_iter, total_ms_for_all_iters).
#[allow(dead_code)]
struct ConfigResult {
    avg_ms: f64,
    total_ms: f64,
    nvme_gbs: f64,
    pcie_gbs_implied: f64,
}

/// Issue one cuBLAS gemm matching the bench shape on the given
/// VRAM `B` buffer. `d_a_bf16`, `d_c_f32` are pre-allocated
/// activation/output buffers reused across iters.
unsafe fn issue_gemm(
    handle: cublasHandle_t,
    stream: cudaStream_t,
    d_a_bf16: *const c_void,
    d_b_bf16: *const c_void,
    d_c_f32: *mut c_void,
) {
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    unsafe {
        check_cublas(cublasSetStream_v2(handle, stream), "cublasSetStream");
        check_cublas(
            cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N_DIM as c_int,
                M_DIM as c_int,
                K_DIM as c_int,
                &alpha as *const f32 as *const c_void,
                d_b_bf16,
                CUDA_R_16BF,
                N_DIM as c_int,
                d_a_bf16,
                CUDA_R_16BF,
                K_DIM as c_int,
                &beta as *const f32 as *const c_void,
                d_c_f32,
                CUDA_R_32F,
                N_DIM as c_int,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT,
            ),
            "cublasGemmEx",
        );
    }
}

// -------------------------------- Config 1 ---------------------------------

fn run_config_1_sequential(
    paths: &[PathBuf],
    handle: cublasHandle_t,
    pinned_host: *mut c_void,
    d_a_bf16: *const c_void,
    d_b_vram: *mut c_void,
    d_c_f32: *mut c_void,
    default_stream: cudaStream_t,
) -> ConfigResult {
    let pinned_slice: &mut [u8] =
        unsafe { std::slice::from_raw_parts_mut(pinned_host as *mut u8, BYTES_B) };

    // Warmup.
    for w in 0..WARMUP_ITERS {
        let path = &paths[w % N_FILES];
        let mut f = open_nvmebench(path);
        f.read_exact(pinned_slice).expect("read warmup");
        unsafe {
            check_cuda(
                cudaMemcpy(d_b_vram, pinned_host, BYTES_B, CUDA_MEMCPY_HOST_TO_DEVICE),
                "memcpy H2D warmup",
            );
            issue_gemm(handle, default_stream, d_a_bf16, d_b_vram, d_c_f32);
            check_cuda(cudaDeviceSynchronize(), "sync warmup");
        }
    }
    check_vram_budget("Config 1 post-warmup");

    let t0 = Instant::now();
    let mut total_read_ms = 0.0_f64;
    for i in 0..MEASURED_ITERS {
        let path = &paths[i % N_FILES];
        let mut f = open_nvmebench(path);

        let t_read = Instant::now();
        f.read_exact(pinned_slice).expect("read measured");
        total_read_ms += t_read.elapsed().as_secs_f64() * 1000.0;

        unsafe {
            check_cuda(
                cudaMemcpy(d_b_vram, pinned_host, BYTES_B, CUDA_MEMCPY_HOST_TO_DEVICE),
                "memcpy H2D measured",
            );
            issue_gemm(handle, default_stream, d_a_bf16, d_b_vram, d_c_f32);
            check_cuda(cudaDeviceSynchronize(), "sync measured");
        }
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let avg_ms = total_ms / (MEASURED_ITERS as f64);
    let avg_read_ms = total_read_ms / (MEASURED_ITERS as f64);
    let nvme_gbs = (BYTES_B as f64 / 1.0e9) / (avg_read_ms / 1000.0);
    // PCIe implied = (avg_ms - read - compute_minimum) — too noisy
    // to compute meaningfully for Config 1. Report 0 here.
    ConfigResult {
        avg_ms,
        total_ms,
        nvme_gbs,
        pcie_gbs_implied: 0.0,
    }
}

// -------------------------------- Config 2 ---------------------------------

/// Two staging slots in VRAM + two pinned host buffers + two CUDA
/// streams. The per-slot dependency:
///
/// 1. NVMe read into `pinned[slot]` (CPU thread, blocking)
/// 2. `cudaMemcpyAsync(pinned[slot] -> vram[slot])` on `copy_stream`
/// 3. Record `upload_done[slot]` event on `copy_stream`
/// 4. `cudaStreamWaitEvent(compute_stream, upload_done[slot])`
/// 5. `cublasGemmEx` on `vram[slot]` via `compute_stream`
/// 6. Record `compute_done[slot]` event on `compute_stream`
///
/// Before reusing a slot for the next file, we `cudaEventSynchronize`
/// on its previous `compute_done` event. That bounds the in-flight
/// work to N_SLOTS at any time.
fn run_config_n_pipeline<const N_SLOTS: usize>(
    paths: &[PathBuf],
    handle: cublasHandle_t,
    pinned_hosts: &[*mut c_void],
    d_a_bf16: *const c_void,
    d_b_slots: &[*mut c_void],
    d_c_f32: *mut c_void,
    copy_stream: cudaStream_t,
    compute_stream: cudaStream_t,
) -> ConfigResult {
    assert_eq!(pinned_hosts.len(), N_SLOTS);
    assert_eq!(d_b_slots.len(), N_SLOTS);

    // Per-slot events.
    let mut upload_done: Vec<cudaEvent_t> = Vec::with_capacity(N_SLOTS);
    let mut compute_done: Vec<cudaEvent_t> = Vec::with_capacity(N_SLOTS);
    for _ in 0..N_SLOTS {
        let mut e1: cudaEvent_t = std::ptr::null_mut();
        let mut e2: cudaEvent_t = std::ptr::null_mut();
        unsafe {
            check_cuda(cudaEventCreate(&mut e1), "eventCreate upload");
            check_cuda(cudaEventCreate(&mut e2), "eventCreate compute");
        }
        upload_done.push(e1);
        compute_done.push(e2);
    }
    let mut compute_recorded: Vec<bool> = vec![false; N_SLOTS];

    // Closure that drives one iter — reused for warmup + measured.
    let mut drive_iter = |iter: usize| {
        let slot = iter % N_SLOTS;

        // Wait until this slot's previous compute completes (if any).
        if compute_recorded[slot] {
            unsafe {
                check_cuda(
                    cudaEventSynchronize(compute_done[slot]),
                    "eventSync prev compute",
                );
            }
        }

        // 1. NVMe read into pinned host (this thread, blocking).
        let path = &paths[iter % N_FILES];
        let mut f = open_nvmebench(path);
        let pinned_slice: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(pinned_hosts[slot] as *mut u8, BYTES_B) };
        f.read_exact(pinned_slice).expect("read iter");

        // 2. PCIe upload async on copy_stream.
        unsafe {
            check_cuda(
                cudaMemcpyAsync(
                    d_b_slots[slot],
                    pinned_hosts[slot],
                    BYTES_B,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                    copy_stream,
                ),
                "memcpyAsync H2D",
            );
            check_cuda(
                cudaEventRecord(upload_done[slot], copy_stream),
                "eventRecord upload",
            );

            // 3. compute_stream waits for upload of this slot.
            check_cuda(
                cudaStreamWaitEvent(compute_stream, upload_done[slot], 0),
                "streamWaitEvent",
            );

            // 4. cublasGemmEx on compute_stream.
            issue_gemm(
                handle,
                compute_stream,
                d_a_bf16,
                d_b_slots[slot] as *const c_void,
                d_c_f32,
            );

            // 5. Record compute_done.
            check_cuda(
                cudaEventRecord(compute_done[slot], compute_stream),
                "eventRecord compute",
            );
        }
        compute_recorded[slot] = true;
    };

    // Warmup.
    for w in 0..WARMUP_ITERS {
        drive_iter(w);
    }
    unsafe {
        check_cuda(cudaDeviceSynchronize(), "sync after warmup");
    }
    check_vram_budget("Pipeline post-warmup");

    // Measured.
    let t0 = Instant::now();
    for i in 0..MEASURED_ITERS {
        drive_iter(WARMUP_ITERS + i);
    }
    unsafe {
        check_cuda(cudaDeviceSynchronize(), "sync after measured");
    }
    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let avg_ms = total_ms / (MEASURED_ITERS as f64);
    let nvme_gbs = (BYTES_B as f64 * MEASURED_ITERS as f64 / 1.0e9) / (total_ms / 1000.0);
    // PCIe runs concurrently with NVMe in this config; implied
    // bandwidth is the same wallclock — report it so the table can
    // be inspected for bottleneck attribution.
    let pcie_gbs_implied = nvme_gbs;

    // Cleanup events.
    for e in upload_done.drain(..) {
        unsafe {
            check_cuda(cudaEventDestroy(e), "eventDestroy upload");
        }
    }
    for e in compute_done.drain(..) {
        unsafe {
            check_cuda(cudaEventDestroy(e), "eventDestroy compute");
        }
    }

    ConfigResult {
        avg_ms,
        total_ms,
        nvme_gbs,
        pcie_gbs_implied,
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== M8.0b — NVMe → PCIe → GPU pipeline async bench ===");
    println!();
    println!(
        "Shape:  [{}, {}] × [{}, {}] (FFN-down Llama 13B; B = {} BF16)",
        M_DIM,
        K_DIM,
        K_DIM,
        N_DIM,
        fmt_size(BYTES_B)
    );
    println!(
        "Iters:  {} warmup + {} measured per config",
        WARMUP_ITERS, MEASURED_ITERS
    );
    println!(
        "Files:  {} distinct files of {} each ({} total)",
        N_FILES,
        fmt_size(BYTES_B),
        fmt_size(N_FILES * BYTES_B)
    );

    let bench_dir = default_bench_dir();
    println!("Dir:    {}", bench_dir.display());
    println!();

    // Initial VRAM probe.
    let (mut free0, mut total0): (usize, usize) = (0, 0);
    unsafe {
        check_cuda(
            cudaMemGetInfo(&mut free0, &mut total0),
            "cudaMemGetInfo (start)",
        );
    }
    println!(
        "VRAM at start: free {} / total {}",
        fmt_size(free0),
        fmt_size(total0)
    );

    // Generate / verify files.
    println!("[1/4] Ensuring bench files...");
    let t_gen = Instant::now();
    let paths = ensure_bench_files(&bench_dir);
    println!(
        "      Files ready in {:.1}s ({} files)",
        t_gen.elapsed().as_secs_f64(),
        paths.len()
    );

    // Driver warmup.
    let mut warmup_ptr: *mut c_void = std::ptr::null_mut();
    unsafe {
        check_cuda(
            cudaMalloc(&mut warmup_ptr, 1024),
            "cudaMalloc (driver warmup)",
        );
        check_cuda(cudaFree(warmup_ptr), "cudaFree (driver warmup)");
    }

    // cuBLAS handle.
    let mut handle: cublasHandle_t = std::ptr::null_mut();
    unsafe {
        check_cublas(cublasCreate_v2(&mut handle), "cublasCreate");
    }

    // Streams.
    let mut copy_stream: cudaStream_t = std::ptr::null_mut();
    let mut compute_stream: cudaStream_t = std::ptr::null_mut();
    unsafe {
        check_cuda(cudaStreamCreate(&mut copy_stream), "streamCreate copy");
        check_cuda(
            cudaStreamCreate(&mut compute_stream),
            "streamCreate compute",
        );
    }
    let default_stream: cudaStream_t = std::ptr::null_mut();

    // Activation A (small, [1, K] BF16, fixed across iters).
    let mut a_bits: Vec<u16> = Vec::with_capacity(K_DIM);
    for i in 0..K_DIM {
        a_bits.push(f32_to_bf16_bits(((i as f32) * 0.001).sin() * 0.3));
    }
    let mut d_a_bf16: *mut c_void = std::ptr::null_mut();
    unsafe {
        check_cuda(cudaMalloc(&mut d_a_bf16, K_DIM * 2), "malloc A bf16");
        check_cuda(
            cudaMemcpy(
                d_a_bf16,
                a_bits.as_ptr() as *const c_void,
                K_DIM * 2,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            ),
            "memcpy A bf16",
        );
    }

    // Output C (small, [1, N] F32).
    let mut d_c_f32: *mut c_void = std::ptr::null_mut();
    unsafe {
        check_cuda(cudaMalloc(&mut d_c_f32, N_DIM * 4), "malloc C f32");
    }

    // Pinned host buffers (3 — enough for Config 3, Config 1/2 use 1/2).
    let mut pinned: [*mut c_void; 3] = [
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    ];
    for i in 0..3 {
        unsafe {
            check_cuda(cudaMallocHost(&mut pinned[i], BYTES_B), "mallocHost pinned");
        }
    }

    // VRAM staging buffers (3 — same).
    let mut d_b_slots: [*mut c_void; 3] = [
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    ];
    for i in 0..3 {
        unsafe {
            check_cuda(cudaMalloc(&mut d_b_slots[i], BYTES_B), "malloc B slot");
        }
    }

    check_vram_budget("After allocations");

    // ------------------------------- Run --------------------------------
    println!();
    println!("[2/4] Running Config 1 (sequential baseline)...");
    let r1 = run_config_1_sequential(
        &paths,
        handle,
        pinned[0],
        d_a_bf16 as *const c_void,
        d_b_slots[0],
        d_c_f32,
        default_stream,
    );
    println!(
        "      Avg {:.2} ms / iter   (NVMe {:.2} GB/s)",
        r1.avg_ms, r1.nvme_gbs
    );

    println!();
    println!("[3/4] Running Config 2 (two-buffer pipeline)...");
    let pinned_2 = [pinned[0], pinned[1]];
    let slots_2 = [d_b_slots[0], d_b_slots[1]];
    let r2 = run_config_n_pipeline::<2>(
        &paths,
        handle,
        &pinned_2,
        d_a_bf16 as *const c_void,
        &slots_2,
        d_c_f32,
        copy_stream,
        compute_stream,
    );
    println!(
        "      Avg {:.2} ms / iter   (eff NVMe + PCIe {:.2} GB/s wallclock)",
        r2.avg_ms, r2.nvme_gbs
    );

    println!();
    println!("[4/4] Running Config 3 (triple-buffer pipeline)...");
    let pinned_3 = [pinned[0], pinned[1], pinned[2]];
    let slots_3 = [d_b_slots[0], d_b_slots[1], d_b_slots[2]];
    let r3 = run_config_n_pipeline::<3>(
        &paths,
        handle,
        &pinned_3,
        d_a_bf16 as *const c_void,
        &slots_3,
        d_c_f32,
        copy_stream,
        compute_stream,
    );
    println!(
        "      Avg {:.2} ms / iter   (eff NVMe + PCIe {:.2} GB/s wallclock)",
        r3.avg_ms, r3.nvme_gbs
    );

    // ------------------------------- Report --------------------------------
    println!();
    println!("==============================================================");
    println!(
        "{:<26}  {:>10}  {:>12}  {:>12}",
        "Config", "ms / iter", "NVMe GB/s", "wallclock GB/s"
    );
    println!("--------------------------------------------------------------");
    println!(
        "{:<26}  {:>10.2}  {:>12.2}  {:>12.2}",
        "1 — Sequential", r1.avg_ms, r1.nvme_gbs, r1.nvme_gbs
    );
    println!(
        "{:<26}  {:>10.2}  {:>12}  {:>12.2}",
        "2 — Two-buffer pipeline", r2.avg_ms, "(overlap)", r2.nvme_gbs
    );
    println!(
        "{:<26}  {:>10.2}  {:>12}  {:>12.2}",
        "3 — Triple-buffer pipeline", r3.avg_ms, "(overlap)", r3.nvme_gbs
    );
    println!("==============================================================");
    println!();

    // Pipeline overlap inspection.
    let speedup_2 = r1.avg_ms / r2.avg_ms;
    let speedup_3 = r1.avg_ms / r3.avg_ms;
    println!("Pipeline overlap factor (Config N vs Config 1):");
    println!("  Config 2 / Config 1 = {:.2}× speedup", speedup_2);
    println!("  Config 3 / Config 1 = {:.2}× speedup", speedup_3);
    println!();

    // Decision banner.
    println!("=== Decision (Config 2 ms / iter) ===");
    println!("Config 2: {:.2} ms / iter", r2.avg_ms);
    println!("Plan A threshold:    ≤ 200 ms  (proceed M8.1-M8.4 + M8.7)");
    println!("Partial threshold:   200–400 ms");
    println!("Plan B threshold:    > 400 ms  (PARAR; pivot to M9 INT8 or close M7 as v20)");
    if r2.avg_ms <= 200.0 {
        println!("Decision:            PLAN A — pipeline async delivers, proceed M8.1-M8.4 + M8.7");
    } else if r2.avg_ms <= 400.0 {
        println!(
            "Decision:            PARTIAL — pipeline partially effective; revisit M8 vs M9 ROI"
        );
    } else {
        println!(
            "Decision:            FAIL — pipeline does NOT help; pivot to M9 INT8 or accept M7 closure"
        );
    }
    println!();

    // Project per-token cost — 29 disk layers in M7.3, this shape is FFN-down
    // (the largest of the 7 per-layer matmuls). Real per-token cost will be
    // lower than 29 × ms_per_iter because Q/K/V/O proj are smaller.
    let per_token_ms_pipelined = (r2.avg_ms * 29.0) + (3.0 * 11.0);
    let per_token_ms_serial = (r1.avg_ms * 29.0) + (3.0 * 11.0);
    println!("Projection at M7.3 tier mix (29 Disk + 11 VRAM layers, this shape only):");
    println!(
        "  Sequential (Config 1):  {:.1} s / token",
        per_token_ms_serial / 1000.0
    );
    println!(
        "  Pipelined  (Config 2):  {:.1} s / token  (vs M7.3 baseline 36.6 s/tok)",
        per_token_ms_pipelined / 1000.0
    );
    println!("  (Real per-token will be lower — Q/K/V/O projs are 4x smaller than FFN-down.)");
    println!();

    // ------------------------------- Cleanup --------------------------------
    for p in pinned.iter() {
        unsafe {
            check_cuda(cudaFreeHost(*p), "freeHost");
        }
    }
    for d in d_b_slots.iter() {
        unsafe {
            check_cuda(cudaFree(*d), "free B slot");
        }
    }
    unsafe {
        check_cuda(cudaFree(d_a_bf16), "free A bf16");
        check_cuda(cudaFree(d_c_f32), "free C f32");
        check_cuda(cudaStreamDestroy(copy_stream), "streamDestroy copy");
        check_cuda(cudaStreamDestroy(compute_stream), "streamDestroy compute");
        check_cublas(cublasDestroy_v2(handle), "cublasDestroy");
    }

    let (mut free_end, mut total_end): (usize, usize) = (0, 0);
    unsafe {
        check_cuda(
            cudaMemGetInfo(&mut free_end, &mut total_end),
            "cudaMemGetInfo (end)",
        );
    }
    println!(
        "VRAM at end:   free {} / total {}",
        fmt_size(free_end),
        fmt_size(total_end)
    );
}
