//! M7.0 — NVMe bench for the Disk tier on the FFN-down shape.
//!
//! This bench is the **gating data point** for M7's overall design.
//! It measures how fast `disk_tier::read_bf16_tensor` can restore
//! a 13B-class FFN-down weight (270 MB on disk in BF16) from NVMe
//! to RAM, both cold (first read) and warm (second read with OS
//! page cache primed). The result decides whether
//!
//! - **Plan A** (proceed with M7.1 → M7.3 directly): viable if cold
//!   read ≤ 50 ms (per-token cost ≤ 14 s for 280 matmuls), or
//! - **Plan B** (insert M7.4 cache + prefetch before M7.1): required
//!   if cold read ≥ 200 ms (per-token cost ≥ 56 s, infeasible
//!   without a residency cache).
//!
//! # Usage
//!
//! ```powershell
//! # Pin the disk-tier cache directory to the operator's NVMe drive
//! # (D: on the dev box). Without this, the cache lands in
//! # `%LOCALAPPDATA%\Atenia\cache` which may resolve to the C: drive.
//! $env:ATENIA_DISK_TIER_DIR = "D:\Atenia\disk_tier_bench"
//! cargo run --release --example bench_disk_weight
//! ```
//!
//! # What is measured
//!
//! Three full read iterations of the same handle, sequentially:
//! 1. **Cold**: handle has just been written; OS page cache is
//!    primed by the write but the data is still being committed
//!    to disk. First read is what matters most for the
//!    end-to-end M7.3 smoke (loader writes → load done → forward
//!    starts reading).
//! 2. **Warm 1**: second read, after the cache has had time to
//!    settle. Represents the steady-state hot path if a Disk-tier
//!    weight is touched repeatedly.
//! 3. **Warm 2**: third read, confirming the warm timing is
//!    stable, not an outlier.
//!
//! After each read we run a single tiny matmul against the
//! reconstructed F32 buffer to confirm the bytes are
//! consumable end-to-end. The matmul itself is wall-clock-tiny
//! (single row × column) — it exists only to detect
//! corruption, not to profile.
//!
//! # Output
//!
//! - Per-iteration ms of `read_bf16_tensor`.
//! - Per-iteration MB/s effective NVMe throughput.
//! - Bit-exact verification vs a host-decoded reference.
//! - Projection of per-token cost at 280 matmuls (a Llama 2 13B
//!   decode-step) for two scenarios:
//!   - "all weights cold" (worst case — every matmul is a fresh
//!     NVMe read).
//!   - "all weights warm" (best case — OS page cache absorbs
//!     repeated reads).

use std::path::PathBuf;
use std::time::Instant;

use atenia_engine::tensor::disk_tier;
use atenia_engine::tensor::tensor::{bf16_bits_to_f32, f32_to_bf16_bits};

/// Llama 2 13B FFN-down shape: `[hidden=5120, intermediate=13824]`
/// laid out so the disk file is 5120 × 13824 × 2 bytes = 141.6 MB.
/// (The "270 MB" figure cited elsewhere refers to the F32 in-VRAM
/// representation; on disk the BF16 layout is exactly half.)
const HIDDEN: usize = 5120;
const INTERMEDIATE: usize = 13824;

fn ms(elapsed: std::time::Duration) -> f64 {
    elapsed.as_secs_f64() * 1000.0
}

fn fmt_size(bytes: usize) -> String {
    let mib = bytes as f64 / (1024.0 * 1024.0);
    format!("{:.1} MiB", mib)
}

fn fmt_throughput(bytes: usize, ms_elapsed: f64) -> String {
    if ms_elapsed <= 0.0 {
        return "n/a".to_string();
    }
    let mib = bytes as f64 / (1024.0 * 1024.0);
    let mib_per_s = mib / (ms_elapsed / 1000.0);
    format!("{:.1} MiB/s", mib_per_s)
}

/// Build a deterministic synthetic FFN-down weight as `Vec<u16>` of
/// BF16 bits. Values are sampled from a stable analytic function so
/// the bench is reproducible across runs and can be diff-checked.
fn synth_bf16_weight(numel: usize) -> Vec<u16> {
    (0..numel)
        .map(|i| {
            // Mix two phases of a low-amplitude sine to get a
            // value in roughly [-1.0, 1.0] without too many
            // BF16 round-half-to-even artefacts.
            let f = ((i as f32) * 0.0001 + 0.137).sin() * 0.5
                + ((i as f32) * 0.00007 + 0.42).cos() * 0.4;
            f32_to_bf16_bits(f)
        })
        .collect()
}

/// Decode a BF16 buffer to F32 on the host. Used to build the
/// reference values for the bit-exact check.
fn decode_bf16(bits: &[u16]) -> Vec<f32> {
    bits.iter().map(|&b| bf16_bits_to_f32(b)).collect()
}

fn main() {
    println!("=== M7.0 — NVMe bench for Disk tier (FFN-down shape) ===");
    let numel = HIDDEN * INTERMEDIATE;
    let bytes_bf16 = numel * 2;
    let bytes_f32 = numel * 4;
    println!(
        "Shape:        [{}, {}] = {} numel",
        HIDDEN, INTERMEDIATE, numel
    );
    println!(
        "BF16 size:    {} ({} bytes)",
        fmt_size(bytes_bf16),
        bytes_bf16
    );
    println!(
        "F32 size:     {} ({} bytes)",
        fmt_size(bytes_f32),
        bytes_f32
    );

    let cache_dir: PathBuf = disk_tier::default_cache_dir();
    println!("Cache dir:    {}", cache_dir.display());
    println!();

    // ------------------------------------------------------------
    // [1/4] Synthesize the BF16 buffer in RAM.
    // ------------------------------------------------------------
    println!("[1/4] Synthesizing BF16 weight in RAM...");
    let t0 = Instant::now();
    let bits: Vec<u16> = synth_bf16_weight(numel);
    println!(
        "      Done in {:.1} ms ({:.1} MiB allocated host-side).",
        ms(t0.elapsed()),
        bytes_bf16 as f64 / (1024.0 * 1024.0)
    );
    println!();

    // ------------------------------------------------------------
    // [2/4] Write to disk via disk_tier::write_bf16_tensor.
    // ------------------------------------------------------------
    println!("[2/4] Writing to NVMe (disk_tier::write_bf16_tensor)...");
    let t1 = Instant::now();
    let handle = match disk_tier::write_bf16_tensor(&cache_dir, &bits) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("FATAL: write_bf16_tensor failed: {}", e);
            std::process::exit(1);
        }
    };
    let write_ms = ms(t1.elapsed());
    println!(
        "      Wrote {} in {:.1} ms ({})",
        fmt_size(bytes_bf16),
        write_ms,
        fmt_throughput(bytes_bf16, write_ms)
    );
    println!("      Path: {}", handle.path().display());
    println!();

    // ------------------------------------------------------------
    // [3/4] Read three times: cold (first), warm 1, warm 2.
    // ------------------------------------------------------------
    let host_ref: Vec<f32> = decode_bf16(&bits);

    let labels = ["Cold (first read after write)", "Warm 1", "Warm 2"];
    let mut read_ms: [f64; 3] = [0.0, 0.0, 0.0];

    println!("[3/4] Reading + bit-exact verification (3 iterations)...");
    for (i, label) in labels.iter().enumerate() {
        let t = Instant::now();
        let restored = match disk_tier::read_bf16_tensor(&handle) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("FATAL: read_bf16_tensor failed on iter {}: {}", i + 1, e);
                std::process::exit(2);
            }
        };
        read_ms[i] = ms(t.elapsed());

        // Bit-exact check: every BF16 bit pattern must round-trip
        // exactly, i.e. the read-back vector equals the source
        // `bits` byte-for-byte.
        if restored.len() != bits.len() {
            eprintln!(
                "FATAL: read_bf16_tensor returned {} elements, expected {}",
                restored.len(),
                bits.len()
            );
            std::process::exit(3);
        }
        let mut bit_mismatches = 0_usize;
        for (a, b) in restored.iter().zip(bits.iter()) {
            if a != b {
                bit_mismatches += 1;
            }
        }
        if bit_mismatches > 0 {
            eprintln!(
                "FATAL: {} BF16 bit mismatches on iter {} (read produced \
                 corrupted bytes)",
                bit_mismatches,
                i + 1
            );
            std::process::exit(4);
        }

        // Decode-and-spot-check: confirm the F32 upcast still
        // matches the host reference. Done per-iteration to catch
        // any path that decodes correctly the first time but
        // corrupts on a second read (e.g. a hypothetical mmap
        // misalignment that surfaces under cache pressure).
        let decoded = decode_bf16(&restored);
        let mut max_abs_diff = 0.0_f32;
        for (a, b) in decoded.iter().zip(host_ref.iter()) {
            let d = (a - b).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
        // BF16 round-trip is bit-exact by construction (high 16
        // bits of F32). A non-zero max_abs_diff signals a bug.
        if max_abs_diff != 0.0 {
            eprintln!(
                "FATAL: F32 decode drift {:e} on iter {} (BF16 \
                 round-trip should be bit-exact)",
                max_abs_diff,
                i + 1
            );
            std::process::exit(5);
        }

        println!(
            "      {} {:>32}: {:7.1} ms ({})",
            i + 1,
            label,
            read_ms[i],
            fmt_throughput(bytes_bf16, read_ms[i])
        );
    }
    println!();

    // ------------------------------------------------------------
    // [4/4] Project per-token cost at 280 matmuls.
    // ------------------------------------------------------------
    println!("[4/4] Per-token cost projection (Llama 2 13B decode step):");
    let cold_ms = read_ms[0];
    let warm_ms = (read_ms[1] + read_ms[2]) / 2.0;

    let n_matmuls = 280_usize;

    println!("      Per-matmul read (cold):       {:7.1} ms", cold_ms);
    println!("      Per-matmul read (warm avg):   {:7.1} ms", warm_ms);
    println!();
    println!("      Worst case: every matmul cold-reads from NVMe");
    println!(
        "        {} matmuls × {:.1} ms = {:.1} s/token",
        n_matmuls,
        cold_ms,
        (n_matmuls as f64 * cold_ms) / 1000.0
    );
    println!();
    println!("      Best case: every matmul hits OS page cache");
    println!(
        "        {} matmuls × {:.1} ms = {:.1} s/token",
        n_matmuls,
        warm_ms,
        (n_matmuls as f64 * warm_ms) / 1000.0
    );
    println!();

    // ------------------------------------------------------------
    // Plan A vs Plan B decision banner.
    // ------------------------------------------------------------
    println!("=== Plan A vs Plan B decision ===");
    println!("Cold read ms:          {:.1}", cold_ms);
    println!("Plan A threshold:      ≤ 50 ms");
    println!("Plan B threshold:      ≥ 200 ms");
    if cold_ms <= 50.0 {
        println!("Decision:              PLAN A (proceed M7.1 → M7.3 directly)");
    } else if cold_ms >= 200.0 {
        println!("Decision:              PLAN B (insert M7.4 cache + prefetch first)");
    } else {
        println!(
            "Decision:              GREY ZONE ({:.1} ms is between thresholds; \
             operator decides)",
            cold_ms
        );
    }
}
