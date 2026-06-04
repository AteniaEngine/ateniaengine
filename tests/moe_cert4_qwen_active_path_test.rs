//! **MOE-CERT-4** — Qwen-MoE C5 active-path parity (ADR-007), reaching L3.
//!
//! ADR-007 C5 certifies the **real model, as actually run**, end-to-end on a
//! canonical input vs an external reference over the active subgraph. A global
//! F64 forward is infeasible on this host (full weights 114 GB F64 / 57 GB F32 /
//! 27 GB bf16 vs 34 GB RAM), so the reference is computed **one decoder layer at
//! a time in float64** using HuggingFace's own Qwen2Moe layer module
//! (`fixtures/moe/generate_qwen_moe_c5_reference.py`) — the F64 form of C5,
//! never holding the whole model in F64 (so it is NOT L4). The reference driver
//! is validated against the committed tiny Qwen-MoE HF fixture before use.
//!
//! This harness runs Atenia's controlled `MoeRuntime` full forward on the real
//! Qwen1.5-MoE-A2.7B (disk-tier, ~few GB RAM) and gates it against that f64
//! reference: end-to-end `max_abs_diff < 0.5` (ADR-004 bar, unchanged) + exact
//! per-position argmax. It is **test-only** — no runtime / loader / MoeRuntime /
//! Adapter Toolkit / numerics change; it only *calls* the existing runtime.
//!
//! `#[ignore]` + env `QWEN_MOE_DIR`. Reproduce the reference + run:
//!   python fixtures/moe/generate_qwen_moe_c5_reference.py tiny      # validate driver
//!   python fixtures/moe/generate_qwen_moe_c5_reference.py real <dir> fixtures/moe
//!   QWEN_MOE_DIR=models/Qwen1.5-MoE-A2.7B-Chat cargo test \
//!     --test moe_cert4_qwen_active_path_test --release -- \
//!     --ignored cert4_real_qwen_moe_active_path --nocapture

use std::path::PathBuf;

use atenia_engine::moe::MoeRuntime;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

const ADR_004_GATE: f32 = 0.5;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}

#[test]
#[ignore = "needs the real ~27 GB Qwen1.5-MoE checkpoint via QWEN_MOE_DIR"]
fn cert4_real_qwen_moe_active_path() {
    let model_dir = std::env::var("QWEN_MOE_DIR")
        .expect("set QWEN_MOE_DIR to the Qwen1.5-MoE checkout");

    // Controlled MoE opt-in + disk-tier residency (so the 14.3B model fits RAM).
    // SAFETY: single-threaded test owning these process-global env toggles.
    unsafe {
        std::env::set_var("ATENIA_ENABLE_MOE", "1");
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
        std::env::set_var("ATENIA_MOE_EXPERT_TIER", "disk");
    }

    // --- committed C5 f64 reference (one layer at a time) ---
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("qwen_moe_c5_ref.json")).unwrap(),
    )
    .unwrap();
    let input_ids: Vec<u32> = meta["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let vocab = meta["vocab_size"].as_u64().unwrap() as usize;
    let seq = meta["seq"].as_u64().unwrap() as usize;
    let ref_argmax: Vec<usize> = meta["argmax_per_position"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let refr = SafetensorsReader::open(&fixture_dir().join("qwen_moe_c5_ref.safetensors")).unwrap();
    let ref_logits = refr.get("logits").unwrap().to_vec_f32().unwrap();
    assert_eq!(ref_logits.len(), seq * vocab, "reference logits dims");

    // --- Atenia: real full forward through the controlled MoE runtime ---
    let rt = MoeRuntime::load_from_dir(&PathBuf::from(&model_dir))
        .unwrap_or_else(|e| panic!("MoeRuntime must load the real Qwen-MoE: {e}"));
    let got = rt.forward_logits(&input_ids);
    assert_eq!(got.len(), seq * vocab, "atenia logits dims");

    // === C5 — active-path parity (gate < 0.5 + exact per-position argmax) ===
    let mut worst = 0.0_f32;
    let mut worst_pos = 0usize;
    let mut argmax_mismatch: Vec<(usize, usize, usize)> = Vec::new();
    for pos in 0..seq {
        let a = &got[pos * vocab..(pos + 1) * vocab];
        let r = &ref_logits[pos * vocab..(pos + 1) * vocab];
        let d = a.iter().zip(r).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
        if d > worst {
            worst = d;
            worst_pos = pos;
        }
        let (ga, ra) = (argmax(a), ref_argmax[pos]);
        if ga != ra {
            argmax_mismatch.push((pos, ga, ra));
        }
    }
    println!(
        "CERT4-C5 real Qwen-MoE: seq={seq} input_ids={input_ids:?} | worst max_abs_diff={worst:.3e} \
         (pos {worst_pos}) | ref argmax={ref_argmax:?} | argmax_mismatches={argmax_mismatch:?}"
    );

    // Determinism: a second forward is bit-identical.
    let got2 = rt.forward_logits(&input_ids);
    assert_eq!(got, got2, "MoeRuntime forward must be deterministic");

    // Hard gates — if either fails, do NOT certify L3 (asserts fail loudly).
    assert!(
        argmax_mismatch.is_empty(),
        "C5 FAILED — per-position argmax differs from the f64 reference (not certifying L3): {argmax_mismatch:?}"
    );
    assert!(
        worst < ADR_004_GATE,
        "C5 FAILED — end-to-end max_abs_diff {worst:.3e} >= {ADR_004_GATE} (not certifying L3)"
    );

    println!(
        "CERT4 RESULT: Qwen-MoE C5 PASS (active-path, real weights, worst {worst:.3e} < {ADR_004_GATE}; \
         argmax exact {seq}/{seq}). F64 reference, one layer at a time (NOT L4). → ADR-007 L3 for Qwen-MoE."
    );
}
