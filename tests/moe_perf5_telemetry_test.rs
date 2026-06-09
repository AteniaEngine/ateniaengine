//! **MOE-PERF-5** — MoE-generation telemetry (observability parity with dense).
//!
//! Verifies the instrumented generate path on the topology-representative scale
//! fixtures (Mixtral / Qwen-MoE / DeepSeek-MLA):
//!   * the instrumented tokens are **bit-identical** to `generate` (no behaviour
//!     change),
//!   * timing fields populate and are self-consistent,
//!   * it is deterministic,
//!   * the `cache_telemetry_available` flag is correct per family (graph = true,
//!     DeepSeek/MLA = false — it streams experts uncached).
//!
//! A separate `#[ignore]` demo (`perf5_disk_tier_cache_metrics_demo`) runs the
//! graph fixtures on the **disk tier** with prefetch on to show the expert-cache
//! / prefetch / tier I/O metrics actually populate (PHASE 6). It mutates process
//! env, so it is `#[ignore]` (run explicitly, not in the parallel CI set).

use std::path::PathBuf;

use atenia_engine::moe::{MoeFamily, MoeRuntime};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn opt_in() {
    unsafe {
        std::env::set_var("ATENIA_ENABLE_MOE", "1");
    }
}

fn load(name: &str) -> MoeRuntime {
    opt_in();
    MoeRuntime::load_from_files(
        &fixture_dir().join(format!("{name}_config.json")),
        &fixture_dir().join(format!("{name}.safetensors")),
    )
    .unwrap_or_else(|e| panic!("{name} must load: {e}"))
}

fn input_ids(name: &str) -> Vec<u32> {
    let j: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join(format!("{name}.json"))).unwrap(),
    )
    .unwrap();
    j["input_ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

/// For each family: instrumented == plain generate, timings populate, det.
fn check_family(name: &str, family: MoeFamily, cache_available: bool) {
    let rt = load(name);
    assert_eq!(rt.family(), family, "{name} family");
    let ids = input_ids(name);

    let plain = rt.generate(&ids, 8);
    let (instr, tele) = rt.generate_instrumented(&ids, 8);

    // 1. Bit-identical output — instrumentation must not change generation.
    assert_eq!(instr, plain, "{name}: instrumented tokens must equal generate()");

    // 2. Timing populates and is self-consistent.
    assert_eq!(tele.generated_tokens, instr.len(), "{name}: token count");
    assert!(tele.total_generation_ms >= 0.0, "{name}: total >= 0");
    assert!(tele.prefill_ms >= 0.0 && tele.decode_ms >= 0.0, "{name}: stage times >= 0");
    // first token == prefill cost (the first token comes out of prefill).
    assert!(
        (tele.first_token_ms - tele.prefill_ms).abs() < 1e-9,
        "{name}: first_token == prefill"
    );
    if tele.total_generation_ms > 0.0 {
        let expected_tps = tele.generated_tokens as f64 / (tele.total_generation_ms / 1e3);
        assert!(
            (tele.tokens_per_second - expected_tps).abs() < 1e-3,
            "{name}: tok/s consistent ({} vs {})",
            tele.tokens_per_second,
            expected_tps
        );
    }

    // 3. Coverage flag is honest about the family.
    assert_eq!(
        tele.cache_telemetry_available, cache_available,
        "{name}: cache telemetry availability"
    );

    // 4. Deterministic token output across instrumented runs.
    let (instr2, _) = rt.generate_instrumented(&ids, 8);
    assert_eq!(instr, instr2, "{name}: instrumented determinism");

    // 5. RAM-tier render is well-formed.
    let r = tele.render();
    assert!(r.contains("MoE generation telemetry"), "{name}: render header");
    assert!(r.contains("tok/s"), "{name}: render tok/s");

    eprintln!("PERF-5 {name} telemetry:\n{r}");
}

#[test]
fn perf5_mixtral_telemetry_parity() {
    check_family("mixtral_scale", MoeFamily::Mixtral, true);
}

#[test]
fn perf5_qwen_telemetry_parity() {
    check_family("qwen_scale", MoeFamily::QwenMoe, true);
}

#[test]
fn perf5_deepseek_telemetry_timing_only() {
    // DeepSeek streams experts through the uncached MLA path → no cache metrics.
    check_family("deepseek_scale", MoeFamily::DeepSeekMoe, false);
}

/// **PHASE 6 demo** — disk-tier graph fixture with prefetch on, to show the
/// expert-cache / prefetch / tier I/O metrics populate (not just timing).
/// `#[ignore]` because it mutates process-global env (tier + prefetch).
///   cargo test --release --test moe_perf5_telemetry_test -- --ignored --nocapture
#[test]
#[ignore = "PERF-5 disk-tier telemetry demo (mutates env) — run with --ignored --nocapture"]
fn perf5_disk_tier_cache_metrics_demo() {
    let tmp = std::env::temp_dir().join(format!("atenia_perf5_tier_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);
    unsafe {
        std::env::set_var("ATENIA_ENABLE_MOE", "1");
        std::env::set_var("ATENIA_MOE_EXPERT_TIER", "disk");
        std::env::set_var("ATENIA_DISK_TIER_DIR", &tmp);
        std::env::set_var("ATENIA_MOE_PREFETCH", "1");
    }

    for name in ["mixtral_scale", "qwen_scale"] {
        let rt = load(name);
        let ids = input_ids(name);
        let (_out, tele) = rt.generate_instrumented(&ids, 8);
        eprintln!("=== PERF-5 disk-tier demo: {name} ===\n{}", tele.render());
        assert!(tele.cache_telemetry_available, "{name}: graph family cache telemetry");
        // The disk tier must have produced real tier reads (the experts are on
        // NVMe; the first touch of each selected expert is a miss).
        assert!(tele.cache_misses > 0, "{name}: disk tier must register misses");
        assert!(tele.materialized_bytes > 0, "{name}: disk tier must read bytes");
    }

    unsafe {
        std::env::remove_var("ATENIA_MOE_EXPERT_TIER");
        std::env::remove_var("ATENIA_DISK_TIER_DIR");
        std::env::remove_var("ATENIA_MOE_PREFETCH");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}
