//! M5.d.b — `GenerationPipeline`: end-to-end Llama-family
//! inference orchestration.
//!
//! Wraps every M5.+ primitive into one operator-facing API:
//!
//! ```ignore
//! let pipe = GenerationPipeline::from_model_dir("models/tinyllama-1.1b")?;
//! let text = pipe.generate("Hello, who are you?", 20, &mut StdoutTokenSink)?;
//! ```
//!
//! ## What `from_model_dir` does
//!
//!   1. Parse `config.json` into [`LlamaConfig`].
//!   2. Load `tokenizer.json` + `tokenizer_config.json` into
//!      [`AteniaTokenizer`] (M5.a, including chat-template
//!      Jinja2 rendering).
//!   3. Build a **scratch graph** via [`super::build_llama`]
//!      with zero-init parameter slots. This graph exists
//!      only to serve as the destination for
//!      [`WeightMapper::load_into`] — its parameter slots
//!      have the right shapes so the loader's per-name
//!      transform pipeline lands the correct bytes.
//!   4. Resolve the checkpoint shape (single
//!      `model.safetensors` or sharded
//!      `model.safetensors.index.json`) and run
//!      [`WeightMapper::load_into`] /
//!      [`ShardedSafetensorsReader::load_into`].
//!   5. Hoist the loaded parameters out of the scratch graph
//!      into a fresh [`WeightStore`] via
//!      [`WeightStore::extract_from_graph`] (M5.c.2.b).
//!      After this call the scratch graph's parameter slots
//!      are `CpuShared` / `CpuBf16Shared` over the same
//!      `Arc`s the store holds. We **drop the scratch
//!      graph** to release its node table — the parameter
//!      Arcs survive in the store.
//!   6. Cache `LlamaConfig` + `AteniaTokenizer` + `WeightStore`
//!      on the `GenerationPipeline`.
//!
//! ## What `generate` does
//!
//!   1. Apply the chat template to `prompt` (or skip if the
//!      checkpoint has no template — raw prompt mode).
//!   2. Tokenise with `add_bos = true`.
//!   3. Call [`super::generate_greedy`] with prefill +
//!      per-step decode, streaming each generated token to
//!      `sink` and accumulating the joined text into the
//!      returned `String`.
//!
//! ## GQA pre-tile cache (gap-3 from M5.b close — Way A confirmed)
//!
//! TinyLlama-1.1B is 32 query heads × 4 KV heads (GQA factor
//! 8). The existing weight-loading pipeline applies
//! `TileGroupedDim` to `k_proj` / `v_proj` weights at load
//! time, expanding them to `[hidden, hidden]` so the graph
//! sees post-tile MHA-shaped K, V projections. The cache-aware
//! attention path in [`super::build_llama_with_store`] thus
//! stores **post-tile** K, V — Way A from the gap-3 decision
//! lock. Way B (pre-tile cache, more memory-efficient under
//! GQA) requires a different load-pipeline that defers tile
//! to read-time; that's a M6 optimisation paired with the
//! GPU offload work.
//!
//! Practical consequence for M5.d.b: the cache memory is
//! ~8× larger than necessary on TinyLlama (post-tile vs
//! pre-tile), but `1.6 KiB/token × 22 layers = 36 KiB/token`
//! is negligible against any reasonable context length.
//! Llama 2 13B is MHA (40Q == 40KV), so the post-tile path
//! is already the optimal storage for the headline 13B target.

use std::path::{Path, PathBuf};

use crate::amg::builder::GraphBuilder;
use crate::amg::weight_store::{
    extract_layer_index_from_param_name, WeightStore, WeightStoreError,
};
use crate::gpu::backend::{Backend, CudaBackend};
use crate::gpu::residency_planner::{parse_gpu_layers_env, plan_residency, PlannerInput};
use crate::nn::llama::config::{ConfigError, LlamaConfig};
use crate::nn::llama::weight_loading::llama_weight_mapper;
use crate::tokenizer::{AteniaTokenizer, ChatMessage, TokenizerError};
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::safetensors_reader::SafetensorsReader;
use crate::v17::loader::sharded_reader::ShardedSafetensorsReader;
use super::builder::{build_llama, LlamaRuntime};
use super::generator::{
    generate_greedy, GenerateError, GeneratedToken, GenerationConfig, TokenSink,
};

/// Bundles everything needed to run inference against one
/// checkpoint. Cheap to clone the references out of, but the
/// store itself is large — keep one pipeline per loaded model.
pub struct GenerationPipeline {
    pub config: LlamaConfig,
    pub tokenizer: AteniaTokenizer,
    pub store: WeightStore,
    pub model_dir: PathBuf,
}

#[derive(Debug)]
pub enum PipelineError {
    Io(std::io::Error),
    Config(ConfigError),
    Tokenizer(TokenizerError),
    Loader(LoaderError),
    WeightStore(WeightStoreError),
    Generate(GenerateError),
    MissingFile(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Io(e) => write!(f, "pipeline io: {e}"),
            PipelineError::Config(e) => write!(f, "pipeline config: {e}"),
            PipelineError::Tokenizer(e) => write!(f, "pipeline tokenizer: {e}"),
            PipelineError::Loader(e) => write!(f, "pipeline loader: {e:?}"),
            PipelineError::WeightStore(e) => write!(f, "pipeline weight_store: {e}"),
            PipelineError::Generate(e) => write!(f, "pipeline generate: {e}"),
            PipelineError::MissingFile(s) => write!(f, "pipeline: required file not found: {s}"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<std::io::Error> for PipelineError {
    fn from(e: std::io::Error) -> Self { PipelineError::Io(e) }
}
impl From<ConfigError> for PipelineError {
    fn from(e: ConfigError) -> Self { PipelineError::Config(e) }
}
impl From<TokenizerError> for PipelineError {
    fn from(e: TokenizerError) -> Self { PipelineError::Tokenizer(e) }
}
impl From<LoaderError> for PipelineError {
    fn from(e: LoaderError) -> Self { PipelineError::Loader(e) }
}
impl From<WeightStoreError> for PipelineError {
    fn from(e: WeightStoreError) -> Self { PipelineError::WeightStore(e) }
}
impl From<GenerateError> for PipelineError {
    fn from(e: GenerateError) -> Self { PipelineError::Generate(e) }
}

impl GenerationPipeline {
    /// Construct a pipeline by reading every artefact under
    /// `model_dir`. See module-level doc for the load
    /// procedure. Heavy: BF16 storage shrinks the on-host
    /// footprint to ~half the F32 path; expect 1–2 seconds
    /// for TinyLlama and ~3 minutes for Llama 2 13B on the
    /// dev box (matches the M4.7.6 / M4.8 numbers — load
    /// dominated by safetensors decode + transform).
    pub fn from_model_dir<P: AsRef<Path>>(
        model_dir: P,
    ) -> Result<Self, PipelineError> {
        Self::from_model_dir_with_options(model_dir, true)
    }

    /// Same as [`Self::from_model_dir`] but with explicit
    /// control over BF16 storage. `bf16 = false` keeps F32
    /// resident — useful for tests that want bit-exact
    /// comparison against an F32-trained reference.
    pub fn from_model_dir_with_options<P: AsRef<Path>>(
        model_dir: P,
        bf16_storage: bool,
    ) -> Result<Self, PipelineError> {
        let model_dir = model_dir.as_ref().to_path_buf();

        // 1. Config.
        let config_path = model_dir.join("config.json");
        if !config_path.exists() {
            return Err(PipelineError::MissingFile(config_path.display().to_string()));
        }
        let config = LlamaConfig::from_json_file(&config_path)?;

        // 2. Tokenizer.
        let tokenizer = AteniaTokenizer::from_model_dir(&model_dir)?;

        // 3. Scratch graph (zero-init parameters; will get
        //    populated by the loader and then hoisted into
        //    the store).
        let runtime = LlamaRuntime { batch: 1, seq: 1 };
        let mut gb = GraphBuilder::new();
        let token_input_id = gb.input();
        let handles = build_llama(&mut gb, &config, &runtime, token_input_id);
        let _ = gb.output(handles.logits_id);
        let mut scratch_graph = gb.build();

        // 4. Load weights.
        let mut mapper = llama_weight_mapper(&config, &handles.param_names, &handles.param_ids)?;
        mapper.set_store_params_as_bf16(bf16_storage);

        let index_path = model_dir.join("model.safetensors.index.json");
        let single_path = model_dir.join("model.safetensors");

        if index_path.exists() {
            let sharded = ShardedSafetensorsReader::open(&index_path)?;
            let _report = sharded.load_into(&mut scratch_graph, &mapper)?;
        } else if single_path.exists() {
            let reader = SafetensorsReader::open(&single_path)?;
            let _report = mapper.load_into(&mut scratch_graph, &reader)?;
        } else {
            return Err(PipelineError::MissingFile(format!(
                "{} or {}",
                single_path.display(), index_path.display()
            )));
        }

        // 5. Hoist into Arc-shared store. After this call the
        //    scratch graph's parameter slots are `Shared`
        //    views, so dropping the graph is safe — the
        //    parameter Arcs survive in the store.
        let mut store = WeightStore::extract_from_graph(
            &mut scratch_graph,
            &handles.param_ids,
            &handles.param_names,
        )?;

        // **M6.c.3** — opportunistic GPU residency upload.
        //
        // When a CUDA backend is available and the operator
        // hasn't disabled it via `ATENIA_GPU=0`, run the
        // M6.c.2 residency planner to pick the resident
        // layer set, then upload those layers' weights to
        // VRAM via `WeightStore::upload_resident_layers`.
        // The decode-step graph's `to_tensor()` calls then
        // materialise the resident params as
        // `TensorStorage::Cuda` slots, and the M6.c.4
        // mixed-storage matmul path takes care of the
        // (a=Cpu, b=Cuda) executor case.
        //
        // Failures here are non-fatal: any error logs to
        // stderr and the pipeline continues with a CPU-only
        // store (bit-exact identical to the M5.f.a build).
        if std::env::var("ATENIA_GPU").as_deref() != Ok("0") {
            try_upload_resident_layers(&mut store, &config, &handles.param_names);
        }

        // After M6.c.3 the scratch graph's parameter slots
        // for the resident layers are still `CpuShared` /
        // `CpuBf16Shared` (the upload only updated the
        // store's entries). The scratch graph is about to
        // drop, so this divergence is harmless. Future
        // graph builds via `build_llama_with_store` pull
        // fresh `to_tensor()` results, which return
        // `Cuda`-storage tensors for resident layers.

        // 6. Drop scratch graph. The Llama graph also holds
        //    the causal-mask Parameter (zero-shape-on-build)
        //    which is not in the store — it's recomputed by
        //    each new build via `build_llama_with_store`. No
        //    leak.
        drop(scratch_graph);

        Ok(Self { config, tokenizer, store, model_dir })
    }

    /// Run greedy generation against the loaded model.
    ///
    /// Applies the model's chat template if one is present.
    /// Otherwise treats `prompt` as raw text and prepends BOS
    /// per `tokenizer.add_bos_token`.
    pub fn generate<S: TokenSink>(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        sink: &mut S,
    ) -> Result<String, PipelineError> {
        let prompt_text = if self.tokenizer.has_chat_template() {
            self.tokenizer.apply_chat_template(&[
                ChatMessage::user(prompt.to_string()),
            ])?
        } else {
            prompt.to_string()
        };
        self.generate_raw(&prompt_text, max_new_tokens, sink)
    }

    /// Run greedy generation against `prompt_text` directly,
    /// **without** applying any chat template. Useful for
    /// completion-style prompts and tests that want
    /// deterministic input bytes.
    pub fn generate_raw<S: TokenSink>(
        &self,
        prompt_text: &str,
        max_new_tokens: usize,
        sink: &mut S,
    ) -> Result<String, PipelineError> {
        let prompt_ids = self.tokenizer.encode(prompt_text, true)?;
        let gen_cfg = GenerationConfig {
            max_new_tokens,
            eos_token_id: self.tokenizer.eos_id(),
        };

        // **M5.d.c — incremental-context detokenisation.**
        //
        // SentencePiece tokens carry a leading `▁` (U+2581)
        // marker for word starts. Calling `decode(&[id])` on
        // a single id strips the marker without inserting a
        // space — which is what the M5.d.b run produced
        // ("Yes,absolutely!Herearesomeexamples"). Production
        // streaming detokenisers run the decode in context:
        // decode `tokens[..i+1]`, subtract `decode(tokens[..i])`,
        // emit the diff as the new chunk. That yields the
        // SentencePiece-correct spacing without per-token
        // heuristics.
        //
        // We use FnMut state to track the running text so the
        // closure surface stays trivial. Special tokens
        // (BOS/EOS) decode to empty so the user stream stays
        // clean.
        let tokenizer = &self.tokenizer;
        let mut emitted_text = String::new();
        let mut generated_ids: Vec<u32> = Vec::new();
        let decode = |id: u32| -> String {
            if tokenizer.is_special(id) {
                return String::new();
            }
            generated_ids.push(id);
            let full = tokenizer.decode(&generated_ids, true)
                .unwrap_or_default();
            // Diff: bytes added by the latest token. Robust
            // to multi-byte UTF-8 because we slice on
            // `emitted_text.len()` (byte length) and the
            // suffix is guaranteed to be a complete
            // continuation of the prior decode.
            let new_chunk = if full.len() > emitted_text.len() {
                full[emitted_text.len()..].to_string()
            } else {
                // Defensive: if the new decode is shorter
                // (token coalesced into a smaller string),
                // emit nothing and reset state on next
                // iteration. Should not happen for
                // SentencePiece BPE.
                String::new()
            };
            emitted_text = full;
            new_chunk
        };

        // Wrap the user's sink with a recorder that builds
        // up the joined text we return.
        let mut joined = String::new();
        let mut recording = TextRecordingSink {
            inner: sink,
            joined: &mut joined,
        };

        let _ids = generate_greedy(
            &self.config, &self.store, &prompt_ids, &gen_cfg,
            decode, &mut recording,
        )?;

        Ok(joined)
    }
}

/// **M6.c.3** — best-effort GPU residency upload.
///
/// Runs the residency planner over `config` + the live
/// `CudaBackend`'s reported VRAM budget, then asks the
/// store to upload every weight whose layer index is in
/// the resident set.
///
/// Non-fatal: any error prints to stderr and returns —
/// the caller's pipeline continues with whatever subset
/// of layers successfully landed on GPU (zero if the very
/// first upload failed). The kill-switch `ATENIA_GPU=0`
/// is checked by the caller.
///
/// Per-layer byte size estimate at the model's storage
/// dtype (BF16 by default after `from_model_dir`):
///   - Q/K/V/O proj × 4 = 4 × hidden² bytes/dtype-byte
///   - FFN gate/up × 2 + down × 1 = 3 × hidden×intermediate bytes/dtype-byte
///   - input_layernorm + post_attention_layernorm = 2 × hidden bytes/dtype-byte
///
/// The estimate is printed to stderr alongside the planner
/// decision so the operator can verify the resident set
/// fits the VRAM budget before generation kicks off.
fn try_upload_resident_layers(
    store: &mut WeightStore,
    config: &LlamaConfig,
    param_names: &[String],
) {
    let backend = CudaBackend::global();
    let Some(vram_bytes) = backend.available_vram_bytes() else {
        eprintln!("[M6.c.3] CUDA not available — skipping residency upload \
                   (CPU-only mode, bit-exact with M5.f.a).");
        return;
    };

    // Reserve ~20% of available VRAM for working buffers
    // (lm_head, transient activations, output downloads).
    // M6.c is conservative; M6.f can re-tune after live
    // measurement.
    let working_buffer_bytes: u64 = vram_bytes / 5;
    let budget_bytes = vram_bytes.saturating_sub(working_buffer_bytes);

    // Bytes per resident layer at the store's storage dtype.
    // After `from_model_dir(bf16=true)` the host params are
    // BF16 (2 B/elt), but `upload_resident_layers` converts
    // to F32 (4 B/elt) on upload via `Tensor::ensure_gpu`
    // (the BF16 GPU kernel is M6.f / v21). So the planner
    // budgets in F32 bytes — what actually lands in VRAM.
    let h = config.hidden_size as u64;
    let i = config.intermediate_size as u64;
    let bytes_per_layer_f32: u64 =
        (4 * h * h + 3 * h * i + 2 * h) * 4 /* F32 */;

    let user_override = parse_gpu_layers_env().unwrap_or(None);
    let plan = plan_residency(&PlannerInput {
        num_layers: config.num_hidden_layers,
        bytes_per_layer: bytes_per_layer_f32,
        vram_budget_bytes: budget_bytes,
        user_override_count: user_override,
    });

    eprintln!(
        "[M6.c.3] CUDA detected — {:.2} GiB VRAM free.",
        vram_bytes as f64 / (1024.0_f64.powi(3))
    );
    eprintln!(
        "[M6.c.3] Residency plan: {} resident, {} streamed (~{:.2} GiB resident at F32).",
        plan.resident.len(), plan.streamed.len(),
        plan.resident_bytes as f64 / (1024.0_f64.powi(3)),
    );

    if plan.resident.is_empty() {
        eprintln!("[M6.c.3] No resident layers picked — every matmul streams (M6.b path).");
        return;
    }

    // Build a HashSet for O(1) layer-index lookup. The
    // upload predicate maps a param name → resident? via
    // `extract_layer_index_from_param_name`.
    let resident_set: std::collections::HashSet<usize> =
        plan.resident.iter().copied().collect();
    let _ = param_names; // names live inside the store; passed for symmetry / future hooks

    match store.upload_resident_layers(|name| {
        match extract_layer_index_from_param_name(name) {
            Some(layer) => resident_set.contains(&layer),
            None => false, // embed/norm/lm_head stay on CPU in M6.c
        }
    }) {
        Ok((count, bytes)) => {
            eprintln!(
                "[M6.c.3] Uploaded {} parameters ({:.2} GiB) to VRAM.",
                count, bytes as f64 / (1024.0_f64.powi(3))
            );
        }
        Err(e) => {
            eprintln!(
                "[M6.c.3] GPU upload failed: {e}. Falling back to CPU dispatch \
                 (M6.b non-pooled streaming for matmuls; same correctness, \
                 lower throughput)."
            );
        }
    }
}

/// Internal sink wrapper that mirrors events to the user's
/// sink AND accumulates the rendered text for the `generate`
/// return value.
struct TextRecordingSink<'a, S: TokenSink + ?Sized> {
    inner: &'a mut S,
    joined: &'a mut String,
}

impl<'a, S: TokenSink + ?Sized> TokenSink for TextRecordingSink<'a, S> {
    fn on_token(&mut self, t: &GeneratedToken) {
        self.joined.push_str(&t.text);
        self.inner.on_token(t);
    }
}
