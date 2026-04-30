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
use crate::amg::weight_store::{WeightStore, WeightStoreError};
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
        let store = WeightStore::extract_from_graph(
            &mut scratch_graph,
            &handles.param_ids,
            &handles.param_names,
        )?;

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

        // Detokenisation hook fed to `generate_greedy`. We do
        // single-token decoding here (no SentencePiece word-
        // boundary buffering yet — that's a small follow-up
        // for M5.d.c when streaming UX is polished). Special
        // tokens render as empty so the user-visible stream
        // stays clean.
        let tokenizer = &self.tokenizer;
        let decode = |id: u32| -> String {
            if tokenizer.is_special(id) {
                String::new()
            } else {
                tokenizer.decode(&[id], true).unwrap_or_default()
            }
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
