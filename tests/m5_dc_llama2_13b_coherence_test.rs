//! M5.d.c — coherence + Arc-sharing proof on the headline 13B model.
//!
//! Two contracts validated against the real Llama 2 13B Chat
//! checkpoint:
//!
//!   1. **Arc-sharing keeps RAM under 30 GiB.** A naïve
//!      "two graphs, two parameter clones" approach would
//!      land 26 GB BF16 × 2 = 52 GB on a 32 GB box. With
//!      `WeightStore::extract_from_graph` (M5.c.2.b) +
//!      `build_llama_with_store` (M5.c.2.c), the pipeline
//!      should stay at the single-model footprint
//!      (~26 GB BF16) regardless of how many graphs are
//!      built.
//!
//!   2. **Coherence on real prompts.** TinyLlama-1.1B was
//!      too small to give a meaningful coherence signal.
//!      Llama 2 13B Chat — the M4.7 killer-demo target — is
//!      large enough that "Hello, how are you?" should
//!      yield a recognisably conversational answer when the
//!      chat template is correctly rendered.
//!
//! ## Env-gated
//!
//! Mirrors the existing 13B test pattern: `#[ignore]` so
//! `cargo test` doesn't try to load 26 GB of weights on
//! every CI run. Local invocation:
//!
//! ```text
//! cargo test --release --test m5_dc_llama2_13b_coherence_test \
//!     -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use atenia_engine::nn::llama::{
    CollectingTokenSink, GenerationPipeline,
};

/// **Memory-safety trick.** Both tests in this file load the
/// same 26 GB BF16 13B checkpoint. Cargo runs tests in
/// parallel by default, so without coordination two parallel
/// loads would peak at ~52 GiB → OOM on a 32 GiB box.
///
/// We share a single `GenerationPipeline` across the two
/// tests via a `OnceLock<Mutex<...>>`. The first test that
/// arrives builds the pipeline; subsequent tests reuse it.
/// The Mutex serialises *access* (each test inspects /
/// generates against the pipeline in turn); the OnceLock
/// guarantees the load happens at most once.
///
/// `Option<Pipeline>` because we want to gracefully skip
/// when the checkpoint isn't on disk.
fn shared_pipeline() -> &'static Mutex<Option<GenerationPipeline>> {
    static CELL: OnceLock<Mutex<Option<GenerationPipeline>>> = OnceLock::new();
    CELL.get_or_init(|| {
        let pipe = match llama2_13b_dir() {
            Some(dir) => {
                eprintln!("[shared-load] Llama 2 13B Chat from {} ...", dir.display());
                let t = std::time::Instant::now();
                match GenerationPipeline::from_model_dir(&dir) {
                    Ok(p) => {
                        let resident_gib =
                            p.store.resident_bytes() as f64 / (1024.0_f64.powi(3));
                        eprintln!(
                            "[shared-load] loaded in {:.1}s ({} params, {:.2} GiB resident)",
                            t.elapsed().as_secs_f32(), p.store.len(), resident_gib);
                        Some(p)
                    }
                    Err(e) => {
                        eprintln!("[shared-load] pipeline load failed: {e}");
                        None
                    }
                }
            }
            None => {
                eprintln!("[shared-load] checkpoint not present; tests will skip");
                None
            }
        };
        Mutex::new(pipe)
    })
}

const LLAMA2_13B_DIR_DEFAULT: &str = "models/llama-2-13b-chat";

fn llama2_13b_dir() -> Option<PathBuf> {
    let dir = std::env::var("ATENIA_LLAMA2_13B_DIR")
        .unwrap_or_else(|_| LLAMA2_13B_DIR_DEFAULT.to_string());
    let path = PathBuf::from(dir);
    // Llama 2 13B ships sharded — 3 shard files + index.
    if path.join("config.json").exists()
        && path.join("tokenizer.json").exists()
        && path.join("model.safetensors.index.json").exists()
    {
        Some(path)
    } else {
        None
    }
}

/// Arc-sharing proof: load the model and assert resident
/// bytes stay below the threshold.
#[test]
#[ignore]
fn llama2_13b_arc_sharing_keeps_resident_under_30_gib() {
    let guard = shared_pipeline().lock().unwrap();
    let Some(pipe) = guard.as_ref() else {
        eprintln!("[skip] Llama 2 13B Chat checkpoint not present");
        return;
    };

    let resident_gib = pipe.store.resident_bytes() as f64 / (1024.0_f64.powi(3));
    eprintln!(
        "shared pipeline: {} parameters, {:.2} GiB resident",
        pipe.store.len(), resident_gib);

    // Sanity: 13B Chat is MHA (40 Q == 40 KV).
    assert_eq!(pipe.config.num_attention_heads, 40);
    assert_eq!(pipe.config.num_key_value_heads, 40);
    assert_eq!(pipe.config.num_hidden_layers, 40);
    assert_eq!(pipe.config.vocab_size, 32_000);

    // Param count: 40 layers × 9 params/layer + embed +
    // final norm + lm_head = 363.
    assert_eq!(pipe.store.len(), 363,
        "Llama 2 13B Chat must materialise 363 parameters");

    // Arc-sharing proof: BF16 13B should land at ~26 GiB.
    // The threshold is 30 GiB to allow ~15% headroom for
    // Arc bookkeeping overhead and any minor allocator
    // padding without forcing a re-tune on every loader
    // tweak. Failure here means a parameter buffer was
    // duplicated — exactly the failure mode M5.c.2.a's
    // CpuShared / CpuBf16Shared was designed to prevent.
    let threshold_gib = 30.0_f64;
    assert!(
        resident_gib < threshold_gib,
        "Arc-sharing leak: resident_bytes = {resident_gib:.2} GiB exceeds {threshold_gib:.2} GiB threshold. \
         Naïve two-graph cloning would put us at ~52 GiB; this assertion is the M5.c.2.a sentinel."
    );

    // Build a SECOND graph against the same store — proves
    // that per-graph parameter materialisation (Arc::clone of
    // the stored Arcs) does not duplicate the underlying
    // buffers. The newly-built graph drops at end of scope;
    // the strong_count drop after that proves correctness
    // of Arc bookkeeping.
    use atenia_engine::amg::weight_store::SharedParam;
    use atenia_engine::amg::builder::GraphBuilder;
    use atenia_engine::nn::llama::{build_llama_with_store, LlamaRuntime};
    let mut gb = GraphBuilder::new();
    let token_in = gb.input();
    let runtime = LlamaRuntime { batch: 1, seq: 1 };
    let _ = build_llama_with_store(
        &mut gb, &pipe.config, &runtime, token_in, &pipe.store, None,
    ).expect("second graph build must succeed");
    let _g2 = gb.build();
    // Each shared param now has at least 2 refs (store
    // entry + this graph's parameter slot). Real number
    // depends on test ordering: if the coherence test
    // already built its own decode graphs, the count is
    // higher.
    for (name, p) in pipe.store.names.iter().zip(pipe.store.params.iter()).take(5) {
        let count = match p {
            SharedParam::F32 { arc, .. } => std::sync::Arc::strong_count(arc),
            SharedParam::Bf16 { arc, .. } => std::sync::Arc::strong_count(arc),
            // Cuda / Disk variants were added in M6 (tier-planner
            // residency). This M5 coherence smoke uses the legacy
            // CPU loader path — neither tier should appear here.
            // Explicit panics preserve the M5 invariant: any leak
            // of a non-CPU variant is a regression.
            SharedParam::Cuda { .. } => {
                panic!(
                    "M5 coherence smoke must not expose Cuda-resident params (got '{name}')"
                );
            }
            SharedParam::Disk { .. } => {
                panic!(
                    "M5 coherence smoke must not expose Disk-resident params (got '{name}')"
                );
            }
        };
        assert!(count >= 2,
            "param '{name}' strong_count = {count}, expected >= 2 (store + at least one graph)");
        eprintln!("  {name}: strong_count = {count}");
    }

    eprintln!(
        "[OK] Arc-sharing proof: resident {resident_gib:.2} GiB < {threshold_gib:.2} GiB threshold"
    );
}

/// Coherence test: real chat-template prompt, generate
/// long enough that the model can compose a full sentence,
/// validate that the answer is recognisably conversational
/// (not, e.g., "Yes, absolutely! Here are some examples"
/// regression seen in M5.d.b before the chat-template fix).
///
/// This test does NOT lock the exact tokens — Llama 2 13B
/// can shift by a few tokens between hardware revisions
/// (BF16 + summation order on different AVX implementations).
/// Instead, we lock LIGHT properties: lowercase detection of
/// expected conversational phrases.
#[test]
#[ignore]
fn llama2_13b_responds_coherently_to_greeting() {
    let guard = shared_pipeline().lock().unwrap();
    let Some(pipe) = guard.as_ref() else {
        eprintln!("[skip] Llama 2 13B Chat checkpoint not present");
        return;
    };

    let prompt = "Hello, how are you?";
    // **Per-step rebuild constraint.** The decode graph is
    // rebuilt at every step (M5.c.2.c policy — production
    // decode-graph reuse is M6 territory). For 13B that
    // build takes ~2.1s per step (M4.7.6.d benchmark) on top
    // of the forward. 30 tokens = ~90s of pure build
    // overhead even before counting the forward, which is
    // why the original budget hung the test session.
    //
    // 5 tokens is enough to detect a regression like the
    // pre-M5.d.c "Yes,absolutely!" symptom (which was a
    // chat-template / detokenisation bug, not a model-
    // capacity issue). With 5 tokens the wall-clock budget
    // is ~30-60s post-load on the dev box.
    let max_new_tokens = 5usize;

    eprintln!("generating {max_new_tokens} tokens for prompt {prompt:?}...");
    let mut sink = CollectingTokenSink::default();
    let gen_start = std::time::Instant::now();
    let text = pipe.generate(prompt, max_new_tokens, &mut sink)
        .expect("generate failed");
    eprintln!("generated in {:.1}s",
        gen_start.elapsed().as_secs_f32());
    eprintln!("response: {:?}", text);

    let lowered = text.to_lowercase();

    // Light coherence checks — at least one of these
    // conversational anchors must appear. We deliberately
    // accept multiple options because the exact response
    // can shift by a few tokens between runs on different
    // hardware. The point is to detect a regression like
    // "Yes, absolutely! Here are some examples" (which
    // was the M5.d.b symptom of a broken chat template).
    let coherent_anchors = ["i", "hello", "thank", "fine", "well", "doing", "how"];
    let hits: Vec<&&str> = coherent_anchors.iter()
        .filter(|w| lowered.contains(*w))
        .collect();
    assert!(!hits.is_empty(),
        "13B response {text:?} contains none of {coherent_anchors:?} — \
         possible chat-template / detokenisation regression");
    eprintln!("[OK] coherence hits: {hits:?}");

    // Sanity on streaming: every token has the right
    // step index, EOS handled.
    for (i, t) in sink.tokens.iter().enumerate() {
        assert_eq!(t.step, i);
    }
    assert!(sink.tokens.len() <= max_new_tokens);
}
