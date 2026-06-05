//! **MLA-1 / C5-DIAG** — localise where the C5 divergence (worst 2.032, argmax
//! 3/4) originates in the real DeepSeek-V2-Lite active path. **Diagnostic only —
//! NOT a certification.** No threshold is enforced; it prints per-component /
//! per-layer `max_abs_diff` against the f64 reference so the failure can be
//! classified (Atenia bug vs reference float32-islands vs expected accumulation
//! vs convention).
//!
//! Two views, both vs `fixtures/moe/deepseek_v2lite_c5_diag.safetensors`
//! (embeddings + per-layer post-attention `x1` + per-layer post-FFN output, F64):
//!   * **ISOLATION** — feed the *reference's* layer input into Atenia's single
//!     layer (`debug_deepseek_layer`) and compare its `x1` / `out` to the
//!     reference. This is the layer's INTRINSIC drift, free of accumulation, so
//!     a spike pinpoints the faulty component (MLA attention vs FFN/MoE).
//!   * **ACCUMULATION** — chain Atenia's own layer outputs from the embeddings
//!     and compare the running hidden to the reference per layer (the real C5
//!     error growth curve).
//! Plus embeddings parity and an lm_head-in-isolation check.
//!
//! `#[ignore]` + env `DEEPSEEK_V2_LITE_DIR` (+ `ATENIA_DISK_TIER_DIR` on NVMe).
//!   python fixtures/moe/generate_deepseek_v2lite_c5_reference.py diag models/DeepSeek-V2-Lite fixtures/moe
//!   $env:ATENIA_DISK_TIER_DIR="F:\atenia_c5_tier"; $env:DEEPSEEK_V2_LITE_DIR="models/DeepSeek-V2-Lite"
//!   cargo test --test moe_mla1_deepseek_c5_diag_test --release -- --ignored c5_diag --nocapture

use std::path::PathBuf;

use atenia_engine::moe::MoeRuntime;
use atenia_engine::v17::loader::safetensors_reader::SafetensorsReader;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures").join("moe")
}

/// `[seq][hidden]` slice of a flat `[..,seq,hidden]` reference tensor at layer
/// offset `base` (in rows of `hidden`).
fn rows(flat: &[f32], base_row: usize, seq: usize, hidden: usize) -> Vec<Vec<f32>> {
    (0..seq)
        .map(|p| flat[(base_row + p) * hidden..(base_row + p + 1) * hidden].to_vec())
        .collect()
}

/// Worst per-token `max_abs_diff` between `got[seq][hidden]` and reference rows,
/// returning `(worst_diff, worst_token)`.
fn worst(got: &[Vec<f32>], reff: &[Vec<f32>]) -> (f32, usize) {
    let mut w = 0.0_f32;
    let mut wp = 0usize;
    for (p, (a, b)) in got.iter().zip(reff.iter()).enumerate() {
        let d = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max);
        if d > w {
            w = d;
            wp = p;
        }
    }
    (w, wp)
}

#[test]
#[ignore = "needs the real DeepSeek-V2-Lite checkout + the c5_diag reference"]
fn c5_diag() {
    let model_dir = std::env::var("DEEPSEEK_V2_LITE_DIR")
        .expect("set DEEPSEEK_V2_LITE_DIR to the DeepSeek-V2-Lite checkout");
    // SAFETY: single-threaded test owning these process-global env toggles.
    unsafe {
        std::env::set_var("ATENIA_ENABLE_MOE", "1");
        std::env::set_var("ATENIA_EXPERIMENTAL_MOE", "1");
        std::env::set_var("ATENIA_MOE_EXPERT_TIER", "disk");
        if std::env::var("ATENIA_DISK_TIER_DIR").is_err() {
            std::env::set_var("ATENIA_DISK_TIER_DIR", std::env::temp_dir().join("atenia_c5_tier"));
        }
    }

    // --- reference ---
    let meta: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture_dir().join("deepseek_v2lite_c5_diag.json")).unwrap(),
    )
    .expect("c5 diag json (run the `diag` generator mode first)");
    let ids: Vec<u32> =
        meta["input_ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let seq = meta["seq"].as_u64().unwrap() as usize;
    let hidden = meta["hidden"].as_u64().unwrap() as usize;
    let n_layers = meta["num_layers"].as_u64().unwrap() as usize;

    let r = SafetensorsReader::open(&fixture_dir().join("deepseek_v2lite_c5_diag.safetensors")).unwrap();
    let ref_emb = r.get("embeddings").unwrap().to_vec_f32().unwrap(); // [seq, hidden]
    let ref_pa = r.get("post_attn").unwrap().to_vec_f32().unwrap(); // [L, seq, hidden]
    let ref_pf = r.get("post_ffn").unwrap().to_vec_f32().unwrap(); // [L, seq, hidden]

    let emb_rows = rows(&ref_emb, 0, seq, hidden);
    let pa_layer = |l: usize| rows(&ref_pa, l * seq, seq, hidden);
    let pf_layer = |l: usize| rows(&ref_pf, l * seq, seq, hidden);

    // --- Atenia (disk tier) ---
    let rt = MoeRuntime::load_from_dir(&PathBuf::from(&model_dir))
        .unwrap_or_else(|e| panic!("MoeRuntime must load DeepSeek-V2-Lite: {e}"));

    // 0) embeddings parity.
    let emb_atenia = rt.debug_deepseek_embeddings(&ids).expect("deepseek backend");
    let (de, dep) = worst(&emb_atenia, &emb_rows);
    println!("C5-DIAG embeddings: max_abs_diff={de:.3e} (token {dep})");

    // 1) ISOLATION — feed the reference's layer input, compare Atenia's layer out.
    println!("C5-DIAG ISOLATION (reference input per layer; intrinsic drift):");
    println!("  layer |  post_attn(x1)  |  post_ffn(out)  | worst_tok");
    let mut worst_iso_pa = (0.0_f32, 0usize);
    let mut worst_iso_pf = (0.0_f32, 0usize);
    for l in 0..n_layers {
        let input = if l == 0 { emb_rows.clone() } else { pf_layer(l - 1) };
        let (x1, out) = rt.debug_deepseek_layer(l, &input).expect("deepseek backend");
        let (dpa, _) = worst(&x1, &pa_layer(l));
        let (dpf, tpf) = worst(&out, &pf_layer(l));
        if dpa > worst_iso_pa.0 {
            worst_iso_pa = (dpa, l);
        }
        if dpf > worst_iso_pf.0 {
            worst_iso_pf = (dpf, l);
        }
        println!("  {l:5} |  {dpa:.3e}    |  {dpf:.3e}    | {tpf}");
    }
    println!(
        "C5-DIAG ISOLATION worst: post_attn={:.3e} (layer {}) | post_ffn={:.3e} (layer {})",
        worst_iso_pa.0, worst_iso_pa.1, worst_iso_pf.0, worst_iso_pf.1
    );

    // 2) ACCUMULATION — Atenia's own running hidden vs reference per layer.
    println!("C5-DIAG ACCUMULATION (Atenia self-chained; real C5 error growth):");
    let mut xs = emb_atenia.clone();
    for l in 0..n_layers {
        let (_x1, out) = rt.debug_deepseek_layer(l, &xs).expect("deepseek backend");
        xs = out;
        let (d, tp) = worst(&xs, &pf_layer(l));
        println!("  layer {l:2}: accumulated max_abs_diff={d:.3e} (token {tp})");
    }

    // 3) lm_head in isolation — feed the reference's final hidden state.
    let head_logits = rt.debug_deepseek_head(&pf_layer(n_layers - 1)).expect("deepseek backend");
    let ref_logits = r.get("logits").unwrap().to_vec_f32().unwrap(); // [seq, vocab]
    let vocab = ref_logits.len() / seq;
    let head_rows = rows(&ref_logits, 0, seq, vocab);
    let (dh, thp) = worst(&head_logits, &head_rows);
    println!("C5-DIAG lm_head (reference final hidden): max_abs_diff={dh:.3e} (token {thp})");

    println!(
        "C5-DIAG DONE — classify: isolation spike => component bug/precision in that layer; \
         flat-then-growing accumulation => numeric drift; embeddings/head tiny => I/O+head OK."
    );
}
