//! M5.d.a — Greedy generation loop for Llama-family models.
//!
//! Ties together every M5.+ primitive shipped so far:
//!
//!   - [`crate::tokenizer::AteniaTokenizer`] (M5.a) for
//!     prompt encoding and per-token decoding.
//!   - [`crate::amg::weight_store::WeightStore`] (M5.c.2.a)
//!     for Arc-shared weights between prefill and decode
//!     graphs (no RAM duplication on 13B).
//!   - [`super::builder_shared::build_llama_with_store`]
//!     (M5.c.2.c) to build the prefill (cached_len=0) and
//!     per-step decode (cached_len growing) graphs.
//!   - [`crate::amg::kv_cache::KvLayerHandle`] +
//!     [`crate::amg::graph::Graph::overwrite_parameter`]
//!     (M5.b D60) for the runtime cache state machine.
//!
//! ## Decisions encoded here (M5 plan ledger)
//!
//!   - **D61 — Greedy only.** No temperature, top-k, top-p.
//!     Sampling lands in M5.5 / M6 once correctness is
//!     locked. Greedy preserves the M4.6→M4.9 bit-exactness
//!     brand for generated text.
//!   - **D64 — Streaming as MVP.** A `TokenSink` callback
//!     is invoked per generated token; the default sink
//!     prints to stdout and flushes immediately. The
//!     callback shape lets tests harvest the stream
//!     deterministically without an stdout capture
//!     dependency.
//!   - **R6 stop conditions.** Generation halts on EOS or
//!     when `max_new_tokens` is reached (whichever first).
//!
//! ## Per-step rebuild policy
//!
//! M5.d.a ships the per-step rebuild policy from the M5
//! research report: the decode graph is rebuilt at every
//! step with the new `cached_len`. Build cost on TinyLlama
//! (22 layers, hidden 2048) is ~50–100 ms. Acceptable for
//! correctness and for the M5.d.b real-checkpoint smoke
//! test. Production decode-graph reuse (fixed-size cache +
//! valid_len mask) is a M6 optimisation; the runtime API
//! shipped here is forward-compatible.

use std::io::Write;

use crate::amg::builder::GraphBuilder;
use crate::amg::graph::Graph;
use crate::amg::kv_cache::{KvCacheBuildSpec, KvCacheHandles};
use crate::amg::weight_store::WeightStore;
use crate::nn::llama::config::LlamaConfig;
use crate::tensor::Tensor;
use super::builder::LlamaRuntime;
use super::builder_shared::{build_llama_with_store, BuildError, LlamaHandlesShared};

/// Configuration for one generation call.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of NEW tokens to generate (cap).
    pub max_new_tokens: usize,
    /// Stop generation when this token id is produced.
    /// Typical: `tokenizer.eos_id()`.
    pub eos_token_id: u32,
}

/// One generated token. The runtime hands this to the sink
/// after each decode step so callers can stream output as
/// it's produced.
#[derive(Debug, Clone)]
pub struct GeneratedToken {
    /// Step index relative to the start of generation
    /// (0-based; 0 == first token after the prompt).
    pub step: usize,
    /// Vocabulary id picked by greedy argmax.
    pub token_id: u32,
    /// The decoded text fragment (may be empty for special
    /// tokens; never includes leading SentencePiece word
    /// markers — those are buffered until the next regular
    /// token arrives).
    pub text: String,
    /// True iff this token is the EOS sentinel and
    /// generation will halt on it.
    pub is_eos: bool,
}

/// Sink for streaming tokens. Implementations:
///
///   - [`StdoutTokenSink`] — production default. Prints
///     each token's text and flushes stdout.
///   - [`CollectingTokenSink`] — buffers everything in a
///     `Vec<GeneratedToken>` for tests.
///   - any user-supplied `FnMut(&GeneratedToken)`.
pub trait TokenSink {
    fn on_token(&mut self, t: &GeneratedToken);
}

impl<F: FnMut(&GeneratedToken)> TokenSink for F {
    fn on_token(&mut self, t: &GeneratedToken) { self(t) }
}

/// Default production sink: print + flush per token.
///
/// Skips empty `text` (special tokens that decoded to ""),
/// so users see the conversation text and not the BOS
/// sentinel.
pub struct StdoutTokenSink;

impl TokenSink for StdoutTokenSink {
    fn on_token(&mut self, t: &GeneratedToken) {
        if !t.text.is_empty() {
            print!("{}", t.text);
            let _ = std::io::stdout().flush();
        }
    }
}

/// Test-friendly sink that collects every token into a
/// `Vec` for post-hoc inspection.
#[derive(Debug, Default)]
pub struct CollectingTokenSink {
    pub tokens: Vec<GeneratedToken>,
}

impl TokenSink for CollectingTokenSink {
    fn on_token(&mut self, t: &GeneratedToken) { self.tokens.push(t.clone()); }
}

/// Errors surfaced during generation.
#[derive(Debug)]
pub enum GenerateError {
    Build(BuildError),
}

impl std::fmt::Display for GenerateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerateError::Build(e) => write!(f, "generate: build error: {e}"),
        }
    }
}

impl std::error::Error for GenerateError {}

impl From<BuildError> for GenerateError {
    fn from(e: BuildError) -> Self { GenerateError::Build(e) }
}

/// **M5.d.a** — single generation entry point.
///
/// Runs prefill on `prompt_ids`, then loops per-step decode
/// with KV cache until EOS or `max_new_tokens`. Returns the
/// list of generated token ids in order. Each token is also
/// streamed to `sink` as soon as it's produced, so callers
/// that want UX visibility see output land token-by-token.
///
/// The `decode_token` callback is invoked once per generated
/// id to produce the streaming text fragment. Splitting
/// detokenization out of this function keeps `nn::llama`
/// independent of `tokenizer` (no circular deps; the
/// crate-level orchestration layer is the M5.e CLI shim).
///
/// ## Cache management
///
/// Prefill runs at `seq = prompt_len` with
/// `cached_len = 0` and an empty per-layer cache (zero-
/// shaped). After prefill we harvest each layer's
/// `k_full_node_id` / `v_full_node_id` outputs (shape
/// `[1, n_heads, prompt_len, head_dim]`) — those become
/// the resident cache for step 0.
///
/// Each subsequent decode step:
///   1. Build a fresh decode graph at `seq = 1`,
///      `cached_len = current cache length`.
///   2. Patch the per-layer `cache_k_param_id` /
///      `cache_v_param_id` slots with the resident cache
///      tensors via [`Graph::overwrite_parameter`] (D60).
///   3. Forward at the new token id.
///   4. Greedy argmax over `logits[0]` → next token id.
///   5. Harvest the new `k_full` / `v_full` (shape grew by
///      one along axis 2) and store them as the next step's
///      cache.
pub fn generate_greedy<DT, S>(
    cfg: &LlamaConfig,
    store: &WeightStore,
    prompt_ids: &[u32],
    gen_cfg: &GenerationConfig,
    mut decode_token: DT,
    sink: &mut S,
) -> Result<Vec<u32>, GenerateError>
where
    DT: FnMut(u32) -> String,
    S: TokenSink,
{
    assert!(!prompt_ids.is_empty(), "generate_greedy: prompt_ids must not be empty");

    let prompt_len = prompt_ids.len();
    let vocab = cfg.vocab_size;

    // ---- Prefill ----
    let prefill_tokens: Vec<f32> = prompt_ids.iter().map(|&t| t as f32).collect();
    let prefill_input = Tensor::new_cpu(vec![1, prompt_len], prefill_tokens);

    let mut gb = GraphBuilder::new();
    let token_in = gb.input();
    let runtime = LlamaRuntime { batch: 1, seq: prompt_len };
    let spec = KvCacheBuildSpec { cached_len: 0 };
    let h: LlamaHandlesShared = build_llama_with_store(
        &mut gb, cfg, &runtime, token_in, store, Some(&spec),
    )?;
    let _ = gb.output(h.logits_id);
    let mut g_prefill = gb.build();

    let logits_prefill = g_prefill.execute(vec![prefill_input])
        .into_iter().next().expect("prefill: missing output tensor");

    // Greedy argmax for the first generated token (logits at
    // last prompt position predicts the next token).
    let prefill_logits_data = logits_prefill.copy_to_cpu_vec();
    let mut next_token = greedy_argmax_at(
        &prefill_logits_data, prompt_len - 1, vocab,
    );

    // Harvest cache state from prefill.
    let kv_handles = h.kv_handles.expect("prefill must produce kv_handles");
    let mut cache_k = harvest_cache_k(&g_prefill, &kv_handles);
    let mut cache_v = harvest_cache_v(&g_prefill, &kv_handles);

    let mut generated: Vec<u32> = Vec::with_capacity(gen_cfg.max_new_tokens);
    // Drop the prefill graph — its weights are Arc-shared
    // with the store and will live as long as the store
    // does. Local cache_K / cache_V tensors keep the
    // residency reference for the next step.
    drop(g_prefill);

    // ---- Decode loop ----
    for step in 0..gen_cfg.max_new_tokens {
        let cached_len = prompt_len + step;

        // Stream the token we're about to feed in (the
        // newly-predicted one from the previous step / the
        // prefill output).
        let token_id = next_token;
        let is_eos = token_id == gen_cfg.eos_token_id;
        let text = if is_eos {
            String::new()
        } else {
            decode_token(token_id)
        };
        let event = GeneratedToken {
            step,
            token_id,
            text,
            is_eos,
        };
        sink.on_token(&event);
        generated.push(token_id);

        if is_eos {
            return Ok(generated);
        }

        // Build the next decode graph at seq=1 with the
        // current cached_len.
        let token_input =
            Tensor::new_cpu(vec![1, 1], vec![token_id as f32]);
        let mut gb_d = GraphBuilder::new();
        let token_in_d = gb_d.input();
        let runtime_d = LlamaRuntime { batch: 1, seq: 1 };
        let spec_d = KvCacheBuildSpec { cached_len };
        let h_d = build_llama_with_store(
            &mut gb_d, cfg, &runtime_d, token_in_d, store, Some(&spec_d),
        )?;
        let _ = gb_d.output(h_d.logits_id);
        let mut g_d = gb_d.build();

        // Patch cache slots.
        let kv_d = h_d.kv_handles.as_ref()
            .expect("decode build_llama_with_store must produce kv_handles");
        for (li, layer) in kv_d.per_layer.iter().enumerate() {
            g_d.overwrite_parameter(layer.cache_k_param_id, cache_k[li].clone())
                .expect("decode: overwrite cache_K");
            g_d.overwrite_parameter(layer.cache_v_param_id, cache_v[li].clone())
                .expect("decode: overwrite cache_V");
        }

        let logits_d = g_d.execute(vec![token_input])
            .into_iter().next().expect("decode: missing output tensor");
        let logits_d_data = logits_d.copy_to_cpu_vec();

        // Greedy argmax over the single seq position.
        next_token = greedy_argmax_at(&logits_d_data, 0, vocab);

        // Harvest new cache state.
        cache_k = harvest_cache_k(&g_d, kv_d);
        cache_v = harvest_cache_v(&g_d, kv_d);
    }

    Ok(generated)
}

fn greedy_argmax_at(logits_data: &[f32], position: usize, vocab: usize) -> u32 {
    let row = &logits_data[position * vocab .. (position + 1) * vocab];
    row.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap()
}

fn harvest_cache_k(g: &Graph, kv: &KvCacheHandles) -> Vec<Tensor> {
    kv.per_layer.iter()
        .map(|h| g.nodes[h.k_full_node_id].output.as_ref()
            .expect("k_full not materialised").clone())
        .collect()
}

fn harvest_cache_v(g: &Graph, kv: &KvCacheHandles) -> Vec<Tensor> {
    kv.per_layer.iter()
        .map(|h| g.nodes[h.v_full_node_id].output.as_ref()
            .expect("v_full not materialised").clone())
        .collect()
}
