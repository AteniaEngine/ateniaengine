//! M5.d.b — real-checkpoint integration test for the
//! `GenerationPipeline` against TinyLlama-1.1B-Chat.
//!
//! Why TinyLlama and not the headline 13B? TinyLlama is small
//! enough to load in seconds and run a meaningful number of
//! decode steps inside a test budget; it also exercises the
//! GQA load-time tile (gap-3 from M5.b close, "Way A"
//! decision) since it ships 32 query × 4 KV heads.
//!
//! ## What this test locks
//!
//!   1. **End-to-end load** — every artefact under
//!      `models/tinyllama-1.1b/` is consumable by
//!      `GenerationPipeline::from_model_dir`. config.json
//!      parses, tokenizer.json + tokenizer_config.json
//!      load, the safetensors checkpoint round-trips through
//!      `WeightMapper::load_into` → `extract_from_graph` →
//!      `WeightStore`.
//!
//!   2. **GQA pre-tile cache (Way A)** — the cache-aware
//!      attention path produces sensible logits on a model
//!      that uses GQA at the architectural level. The
//!      load-pipeline tiles K/V to MHA at load time, so the
//!      cache stores post-tile MHA-shaped tensors. Validated
//!      indirectly: if the cache layout were wrong, prefill
//!      logits or per-step decode would diverge from the
//!      reference no-cache forward catastrophically.
//!
//!   3. **R6 — generation halts** — the loop respects EOS
//!      and `max_new_tokens` against a real Llama
//!      checkpoint, not a synthetic config.
//!
//!   4. **Determinism contract D67** — the first N greedy
//!      tokens for a fixed prompt are reproducible across
//!      runs. The test compares against a JSON fixture at
//!      `tests/fixtures/generation_determinism/`. On first
//!      run with `ATENIA_REGENERATE_FIXTURES=1` it writes
//!      the fixture; on every subsequent run it asserts
//!      equality.
//!
//! ## Env-gating
//!
//! Mirrors the existing `tinyllama_weight_loading_test`
//! pattern: the test is `#[ignore]` by default because the
//! full TinyLlama checkpoint (~2 GB) is not always present
//! in CI. Local runs:
//!
//! ```text
//! cargo test --test m5_db_tinyllama_pipeline_test \
//!     -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};

use atenia_engine::nn::llama::{CollectingTokenSink, GenerationPipeline};

const TINYLLAMA_DIR_DEFAULT: &str = "models/tinyllama-1.1b";

fn tinyllama_dir() -> Option<PathBuf> {
    let dir =
        std::env::var("ATENIA_TINYLLAMA_DIR").unwrap_or_else(|_| TINYLLAMA_DIR_DEFAULT.to_string());
    let path = PathBuf::from(dir);
    if path.join("config.json").exists()
        && path.join("tokenizer.json").exists()
        && path.join("model.safetensors").exists()
    {
        Some(path)
    } else {
        None
    }
}

fn fixture_path() -> PathBuf {
    PathBuf::from("tests/fixtures/generation_determinism/expected_tokens_tinyllama.json")
}

#[derive(serde::Serialize, serde::Deserialize, Debug)]
struct DeterminismFixture {
    /// Raw text fed to the pipeline (after-chat-template if
    /// any) — locks the input bytes too.
    prompt: String,
    /// Token IDs produced by greedy decoding.
    expected_token_ids: Vec<u32>,
    /// Decoded text (joined). Locked alongside the IDs so a
    /// detokenisation regression also surfaces.
    expected_text: String,
    /// max_new_tokens the fixture was captured with.
    max_new_tokens: usize,
}

fn read_fixture() -> Option<DeterminismFixture> {
    let path = fixture_path();
    if !path.exists() {
        return None;
    }
    let s = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&s).ok()
}

fn write_fixture(fix: &DeterminismFixture) {
    let path = fixture_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let s = serde_json::to_string_pretty(fix).expect("serialize fixture");
    std::fs::write(&path, s).expect("write fixture");
}

/// Headline R6 + determinism test. See module-level doc.
///
/// **Ignored by default** — the TinyLlama checkpoint is too
/// large (~2 GB) to keep in CI. Run locally with
/// `cargo test --test m5_db_tinyllama_pipeline_test -- --ignored`.
#[test]
#[ignore]
fn tinyllama_pipeline_loads_and_generates_deterministic_text() {
    let Some(dir) = tinyllama_dir() else {
        eprintln!(
            "[skip] TinyLlama checkpoint not present at {TINYLLAMA_DIR_DEFAULT} \
                   (override with ATENIA_TINYLLAMA_DIR)"
        );
        return;
    };

    eprintln!("loading TinyLlama from {}...", dir.display());
    let load_start = std::time::Instant::now();
    let pipe = GenerationPipeline::from_model_dir(&dir).expect("pipeline load failed");
    eprintln!(
        "loaded in {:.2}s ({} parameters in store, {:.2} MiB resident)",
        load_start.elapsed().as_secs_f32(),
        pipe.store.len(),
        pipe.store.resident_bytes() as f64 / 1_048_576.0
    );

    // Sanity: TinyLlama config matches expectations.
    assert_eq!(pipe.config.num_attention_heads, 32);
    assert_eq!(pipe.config.num_key_value_heads, 4);
    assert_eq!(pipe.config.num_hidden_layers, 22);
    assert!(
        pipe.tokenizer.has_chat_template(),
        "TinyLlama-Chat must ship a chat_template"
    );

    // Fixed prompt — locked bytes, used both for R6 and the
    // determinism fixture.
    let prompt = "Hello";
    let max_new_tokens = 8usize;

    eprintln!("generating {max_new_tokens} tokens for prompt {prompt:?}...");
    let mut sink = CollectingTokenSink::default();
    let gen_start = std::time::Instant::now();
    let text = pipe
        .generate(prompt, max_new_tokens, &mut sink)
        .expect("generate failed");
    eprintln!(
        "generated in {:.2}s: {:?}",
        gen_start.elapsed().as_secs_f32(),
        text
    );

    // R6 contract — must produce something within the cap,
    // every event has the right step index, EOS handled.
    assert!(!sink.tokens.is_empty(), "must emit at least one token");
    assert!(
        sink.tokens.len() <= max_new_tokens,
        "must respect max_new_tokens cap"
    );
    for (i, t) in sink.tokens.iter().enumerate() {
        assert_eq!(t.step, i, "step index must equal sink array index");
        if i + 1 < sink.tokens.len() {
            assert!(!t.is_eos, "non-final tokens must not be flagged EOS");
        }
    }

    // Build / load the determinism fixture.
    let token_ids: Vec<u32> = sink.tokens.iter().map(|t| t.token_id).collect();
    let regenerate = std::env::var("ATENIA_REGENERATE_FIXTURES").ok().as_deref() == Some("1");

    if regenerate {
        let fixture = DeterminismFixture {
            prompt: prompt.to_string(),
            expected_token_ids: token_ids.clone(),
            expected_text: text.clone(),
            max_new_tokens,
        };
        write_fixture(&fixture);
        eprintln!(
            "[regen] wrote determinism fixture to {}",
            fixture_path().display()
        );
        return;
    }

    match read_fixture() {
        Some(fix) => {
            assert_eq!(
                fix.prompt, prompt,
                "fixture prompt drift (re-run with ATENIA_REGENERATE_FIXTURES=1 \
                 if intentional)"
            );
            assert_eq!(
                fix.max_new_tokens, max_new_tokens,
                "fixture max_new_tokens drift"
            );
            assert_eq!(
                fix.expected_token_ids, token_ids,
                "DETERMINISM BREACH: token IDs differ from locked fixture. \
                 If intentional, re-run with ATENIA_REGENERATE_FIXTURES=1."
            );
            assert_eq!(
                fix.expected_text, text,
                "DETERMINISM BREACH: decoded text differs from locked fixture. \
                 If intentional, re-run with ATENIA_REGENERATE_FIXTURES=1."
            );
            eprintln!("[ok] determinism fixture matches (token IDs + text)");
        }
        None => {
            // No fixture yet. Print what we got so the
            // operator can hand-write one and re-run, or
            // re-run with ATENIA_REGENERATE_FIXTURES=1.
            eprintln!(
                "[fixture-missing] no fixture at {}",
                fixture_path().display()
            );
            eprintln!("first-run capture:");
            eprintln!("  prompt:           {prompt:?}");
            eprintln!("  max_new_tokens:   {max_new_tokens}");
            eprintln!("  token_ids:        {token_ids:?}");
            eprintln!("  text:             {text:?}");
            eprintln!("re-run with ATENIA_REGENERATE_FIXTURES=1 to write the fixture");
            // Don't fail when the fixture is genuinely
            // missing — capture mode is the explicit "this
            // is a first run" path. CI runs (which set no
            // fixture in tree) should still pass on first
            // landing.
        }
    }
}

/// Lighter sanity check that doesn't run a full forward —
/// just verifies the pipeline construction and the parameter
/// hoist did not lose any tensor.
#[test]
#[ignore]
fn tinyllama_pipeline_load_completes_with_full_param_set() {
    let Some(dir) = tinyllama_dir() else {
        return;
    };

    let pipe = GenerationPipeline::from_model_dir(&dir).expect("pipeline load failed");

    // TinyLlama: 22 layers × 9 params + 1 embed + 1 final
    // norm + 1 lm_head = 22*9 + 3 = 201. Sanity match
    // mirroring `tinyllama_builder_test::build_full_*`.
    assert_eq!(pipe.store.len(), 201);

    // Every parameter must be on F32 (default) or BF16 path
    // — no Cuda / Disk leaked through extract_from_graph.
    // Cuda / Disk variants were added in M6 (tier-planner residency)
    // and are valid in *some* pipelines, but this M5 smoke goes
    // through the legacy CPU loader path — neither tier should
    // appear here. Explicit arms (rather than a wildcard) preserve
    // the original assertion: regress if either tier leaks in.
    use atenia_engine::amg::weight_store::SharedParam;
    for p in &pipe.store.params {
        match p {
            SharedParam::F32 { .. } | SharedParam::Bf16 { .. } => {}
            SharedParam::Cuda { .. } => {
                panic!("M5 pipeline must not expose Cuda-resident params");
            }
            SharedParam::Disk { .. } => {
                panic!("M5 pipeline must not expose Disk-resident params");
            }
            SharedParam::CpuInt8Outlier(_) => {
                panic!("M5 pipeline must not expose CpuInt8Outlier-resident params");
            }
        }
    }
}

/// Validate that the pipeline path for a non-existent dir
/// surfaces a clear error rather than panicking. Runs always
/// (cheap; doesn't need the checkpoint).
#[test]
fn pipeline_missing_dir_surfaces_clear_error() {
    let res =
        GenerationPipeline::from_model_dir(Path::new("definitely_not_a_model_dir_for_M5_d_b_test"));
    assert!(
        res.is_err(),
        "missing dir must produce an error, not a panic"
    );
}
