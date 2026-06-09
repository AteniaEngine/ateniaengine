//! **MOE-PERF-5-REAL-MEASURE** — real telemetry baseline via the PERF-5
//! instrumentation, **measurement only** (no runtime / numerics / routing /
//! cache / cert change). `#[ignore]` (timing, mutates process env), run with:
//!   cargo test --release --test moe_perf5_real_measure -- --ignored --nocapture
//!
//! Workload classification (see HANDOFF_MOE_PERF_5_REAL_MEASURE.md):
//!   * Mixtral-8x7B real (87 GB)  → TOO HEAVY / SKIP (host has ~12 GB free;
//!     cache=1 forward peaks ~29 GB → documented OOM/hang hazard).
//!   * DeepSeek-V2-Lite real (29 GB) → SKIP (load-time spikes on a constrained
//!     host; MLA streams experts uncached → cache telemetry N/A regardless).
//!   * Qwen-MoE real → N/A (no whole-model transformer path; block-level cert).
//!   * The three topology-representative **scale fixtures** → SAFE. They drive
//!     the EXACT certified runtime path (`forward_cached`, disk tier, prefetch,
//!     LRU) at the real routing fan-out (Mixtral top-2 / Qwen top-4 / DeepSeek
//!     top-6), so the cache / prefetch / tier telemetry is real; only the hidden
//!     dim (hence absolute time) is reduced.
//!
//! The fixtures hit EOS on the first greedy token, so `decode ≈ 0`; the I/O
//! signal lives in the prefill, which resolves every layer's selected experts.

use std::path::PathBuf;

use atenia_engine::moe::{MoeFamily, MoeGenTelemetry, MoeRuntime};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn input_ids(name: &str) -> Vec<u32> {
    let j: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join(format!("{name}.json"))).unwrap(),
    )
    .unwrap();
    j["input_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

/// Set the env for one config, load fresh, run instrumented generate.
fn run(name: &str, tier: &str, cache: Option<&str>, prefetch: bool, tier_dir: &std::path::Path) -> MoeGenTelemetry {
    unsafe {
        std::env::set_var("ATENIA_ENABLE_MOE", "1");
        std::env::set_var("ATENIA_MOE_EXPERT_TIER", tier);
        std::env::set_var("ATENIA_DISK_TIER_DIR", tier_dir);
        std::env::set_var("ATENIA_MOE_PREFETCH", if prefetch { "1" } else { "0" });
        match cache {
            Some(c) => std::env::set_var("ATENIA_MOE_EXPERT_CACHE", c),
            None => std::env::remove_var("ATENIA_MOE_EXPERT_CACHE"),
        }
    }
    let rt = MoeRuntime::load_from_files(
        &fixture_dir().join(format!("{name}_config.json")),
        &fixture_dir().join(format!("{name}.safetensors")),
    )
    .unwrap_or_else(|e| panic!("{name} must load: {e}"));
    let ids = input_ids(name);
    rt.generate_instrumented(&ids, 8).1
}

fn kib(b: usize) -> f64 {
    b as f64 / 1024.0
}

#[test]
#[ignore = "PERF-5 real-measure sweep (timing, mutates env) — run with --ignored --nocapture"]
fn perf5_real_measure_sweep() {
    let tier_dir = std::env::temp_dir().join(format!("atenia_perf5_rm_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tier_dir);

    println!(
        "{:<14} {:<5} {:<6} {:<6} | {:>9} {:>8} | {:>5} {:>6} {:>5} {:>9} | {:>5} {:>9} {:>7} {:>9} {:>9} | {:>5}",
        "fixture", "tier", "cache", "pref", "prefill_ms", "tok/s", "hits", "miss", "evic", "resid_KiB",
        "reads", "mat_KiB", "par", "ovl_ms", "resv_ms", "cache?"
    );
    let line = |name: &str, family: MoeFamily, tier: &str, cache: &str, pref: bool, t: &MoeGenTelemetry| {
        println!(
            "{:<14} {:<5} {:<6} {:<6} | {:>9.2} {:>8.2} | {:>5} {:>6} {:>5} {:>9.1} | {:>5} {:>9.1} {:>7} {:>9.2} {:>9.2} | {:>5}",
            name,
            tier,
            cache,
            if pref { "on" } else { "off" },
            t.prefill_ms,
            t.tokens_per_second,
            t.cache_hits,
            t.cache_misses,
            t.evictions,
            kib(t.resident_bytes),
            t.tier_reads,
            kib(t.materialized_bytes),
            t.parallel_prefetches,
            t.overlap_saved_ms,
            t.resolve_time_ms,
            if t.cache_telemetry_available { "yes" } else { "NO" },
        );
        let _ = family;
    };

    let families = [
        ("mixtral_scale", MoeFamily::Mixtral),
        ("qwen_scale", MoeFamily::QwenMoe),
        ("deepseek_scale", MoeFamily::DeepSeekMoe),
    ];

    for (name, family) in families {
        // RAM-tier timing baseline (experts resident → zero tier I/O).
        let t = run(name, "ram", None, false, &tier_dir);
        line(name, family, "ram", "auto", false, &t);
        // Disk tier: prefetch off/on × cache {auto, 1}.
        for (cache_label, cache_env) in [("auto", None), ("1", Some("1"))] {
            for pref in [false, true] {
                let t = run(name, "disk", cache_env, pref, &tier_dir);
                line(name, family, "disk", cache_label, pref, &t);
            }
        }
        println!();
    }

    unsafe {
        std::env::remove_var("ATENIA_MOE_EXPERT_TIER");
        std::env::remove_var("ATENIA_DISK_TIER_DIR");
        std::env::remove_var("ATENIA_MOE_PREFETCH");
        std::env::remove_var("ATENIA_MOE_EXPERT_CACHE");
    }
    let _ = std::fs::remove_dir_all(&tier_dir);
    println!(
        "NOTE: fixtures EOS after 1 token (decode≈0); the I/O signal is in prefill (all layers'\n\
         selected experts resolved). RAM tier = experts resident (true zero tier I/O). DeepSeek\n\
         (MLA) streams experts UNCACHED → cache? = NO (timing only). Absolute times are reduced-dim;\n\
         the real-scale anchor is PERF-1/C5 (Mixtral 87 GB: load 4.5 s, forward 402.7 s, cache=4→OOM)."
    );
}
