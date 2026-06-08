//! **MOE-PERF-1** — performance audit bench (measurement only).
//!
//! A reproducible, test-only timing harness for the three certified MoE families
//! on their **topology-representative scale fixtures** (the same fixtures the
//! scale-cert test uses). It measures **load / prefill / decode / tokens-per-sec**
//! at reduced dim — the routing / attention / expert **compute** cost, isolated
//! from the multi-GB real-weight disk I/O (which is measured separately from the
//! certification runs; see `docs/MOE_PERF_AUDIT.md`).
//!
//! **No runtime / numerics change** — it only *calls* the existing `MoeRuntime`
//! and times it (`Instant`). `#[ignore]` so CI never runs it (timing is not a
//! gate); run on demand:
//!   cargo test --release --test moe_perf_scale_bench -- --ignored --nocapture
//! For a per-op breakdown, add `ATENIA_NODE_TIMING=1`.

use std::path::PathBuf;
use std::time::Instant;

use atenia_engine::moe::MoeRuntime;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn opt_in() {
    // SAFETY: single-threaded bench; owns the env toggle.
    unsafe { std::env::set_var("ATENIA_ENABLE_MOE", "1") };
}

/// Time load / prefill / decode for one scale fixture; print a one-line report.
fn bench(name: &str) {
    opt_in();
    let cfg = fixture_dir().join(format!("{name}_config.json"));
    let st = fixture_dir().join(format!("{name}.safetensors"));

    let t = Instant::now();
    let rt = MoeRuntime::load_from_files(&cfg, &st)
        .unwrap_or_else(|e| panic!("{name} must load: {e}"));
    let load_ms = t.elapsed().as_secs_f64() * 1e3;

    // A small canonical prompt (the scale fixtures use vocab 32).
    let ids: Vec<u32> = vec![1, 5, 9, 13, 17];

    // Prefill = one full forward over the prompt (≈ first-token compute).
    let t = Instant::now();
    let logits = rt.forward_logits(&ids);
    let prefill_ms = t.elapsed().as_secs_f64() * 1e3;
    assert!(!logits.is_empty() && logits.iter().all(|v| v.is_finite()));

    // Decode = generate K new tokens (KV-cache decode after prefill).
    let k = 8usize;
    let t = Instant::now();
    let out = rt.generate(&ids, k);
    let gen_s = t.elapsed().as_secs_f64();
    let new_tokens = out.len().saturating_sub(ids.len()).max(1);
    let tok_s = new_tokens as f64 / gen_s;

    eprintln!(
        "MOE-PERF {name:<14} family={:<11} | load={load_ms:8.2} ms | prefill(seq={})={prefill_ms:8.2} ms \
         | generate({k})={:7.3} s ({new_tokens} new tok, {tok_s:6.2} tok/s)",
        rt.family().name(),
        ids.len(),
        gen_s,
    );
}

#[test]
#[ignore = "performance bench (timing, not a correctness gate) — run with --ignored --nocapture"]
fn moe_scale_perf_bench() {
    eprintln!("=== MOE-PERF-1 scale-fixture compute bench (reduced dim; disk-tier I/O measured separately) ===");
    for name in ["mixtral_scale", "qwen_scale", "deepseek_scale"] {
        bench(name);
    }
}
