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
use crate::amg::weight_store::{UploadReport, WeightStore, WeightStoreError};
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

        // **M6 replan sub-fase 3** — tier-aware loader (now default).
        //
        // Originally introduced as opt-in via `ATENIA_TIER_AWARE_LOADER=1`
        // (D74 of `HANDOFF_APX_V20_M6.md`) until the placement policy was
        // re-validated per environment. That validation completed across
        // M6 (1.46× on 7B, bit-exact), M7 (13B without BSOD, automatic
        // tiers), and M8 (1.31× on 7B, 1.36× on 13B, ADR-004 preserved),
        // and the operator confirmed a 21.9 s/token 13B baseline through
        // the tier-aware path.
        //
        // Therefore the policy is now inverted: the tier-aware loader
        // runs by default, and operators set `ATENIA_LEGACY_LOADER=1` to
        // force the pre-M6 `WeightMapper::load_into` +
        // `WeightStore::extract_from_graph` path. D74's "default off"
        // requirement is **superseded** by this commit.
        //
        // The deprecated `ATENIA_TIER_AWARE_LOADER` is still recognised
        // (it's a no-op now since the path is default) with a one-line
        // deprecation warning so existing scripts keep working through
        // a grace period.
        if std::env::var("ATENIA_TIER_AWARE_LOADER").as_deref() == Ok("1") {
            eprintln!(
                "[ATENIA] ATENIA_TIER_AWARE_LOADER is now the default and \
                 will be removed in a future version. Use \
                 ATENIA_LEGACY_LOADER=1 to opt out instead."
            );
        }
        let tier_aware =
            std::env::var("ATENIA_LEGACY_LOADER").as_deref() != Ok("1");
        if !tier_aware {
            eprintln!(
                "[ATENIA] Legacy loader active (ATENIA_LEGACY_LOADER=1): \
                 tier-aware placement disabled."
            );
        }
        let gpu_residency =
            std::env::var("ATENIA_GPU_RESIDENCY").as_deref() == Ok("1");

        if tier_aware && gpu_residency {
            eprintln!(
                "[ATENIA] WARNING: ATENIA_GPU_RESIDENCY is set but the \
                 tier-aware loader is now the default. The tier-aware \
                 loader supersedes the post-load upload — the legacy \
                 ATENIA_GPU_RESIDENCY block will be skipped. Set \
                 ATENIA_LEGACY_LOADER=1 if you specifically need the \
                 legacy upload path."
            );
        }

        let mut store: WeightStore;
        if tier_aware {
            // Probe live machine state once, build a single plan,
            // then dispatch to the multi-shard or single-shard
            // tier-aware loader.
            let free_ram_bytes =
                crate::gpu::safety::resource_check::probe_free_ram_bytes();
            let free_vram_bytes =
                crate::gpu::safety::resource_check::probe_free_vram_bytes();
            let total_ram_bytes =
                crate::gpu::safety::resource_check::probe_total_ram_bytes();

            // **M8.3 / M8.4c** — kernel dtype for the VRAM-resident
            // matmul path. The env var `ATENIA_M8_BF16_KERNEL=1` is
            // the operator's *request*; whether the request takes
            // effect depends on the **adaptive heuristic** computed
            // below from `model_total_bytes` and `free_ram_bytes`.
            //
            // Rationale: the M8.4c "Path B" matmul (BF16 weight in
            // VRAM, upcast to F32 transient per-matmul, F32 GEMM)
            // is ~3× slower per matmul than the M6 path. For models
            // that fit comfortably in RAM (e.g. Llama 2 7B Chat in
            // 32 GiB), this slowdown is a pure regression. M8 only
            // pays off when the doubled VRAM capacity translates to
            // moving weights off the Disk tier (e.g. Llama 2 13B
            // on 32 GiB).
            //
            // Threshold: `model_total > 0.7 × free_ram` — the same
            // threshold the M7.2 adaptive headroom already uses to
            // decide whether to inflate the RAM headroom. Below
            // it, the model dominates RAM and overflowing to Disk
            // is structural; above it (model fits in RAM with
            // headroom), the M8 BF16 path is gratuitous.
            let env_requested = std::env::var("ATENIA_M8_BF16_KERNEL")
                .as_deref()
                == Ok("1");
            // `model_total_bytes` is computed inside each load
            // branch; we wire the conditional inline there so both
            // branches share the same logic.
            let m8_bf16_resolver = |model_total_bytes: u64| -> bool {
                if !env_requested {
                    return false;
                }
                let threshold = (free_ram_bytes / 10).saturating_mul(7);
                let model_dominates = model_total_bytes > threshold;
                if !model_dominates {
                    eprintln!(
                        "[ATENIA] M8 BF16 kernel requested but model fits \
                         comfortably in RAM (model {:.2} GiB ≤ 0.7 × free RAM \
                         {:.2} GiB); falling back to F32 path for stability \
                         (M8 BF16 adds per-matmul upcast overhead that's only \
                         worth paying when capacity-constrained).",
                        (model_total_bytes as f64) / 1024.0_f64.powi(3),
                        (free_ram_bytes as f64) / 1024.0_f64.powi(3),
                    );
                }
                model_dominates
            };

            if index_path.exists() {
                let sharded = ShardedSafetensorsReader::open(&index_path)?;
                let metas = sharded.collect_tensor_metas()?;
                let model_total_bytes = sum_model_bytes(&metas);
                // **M8.4c** — resolve the effective M8 BF16 kernel flag
                // for THIS load. Pipeline owns the conditional;
                // the loader stays a pure consumer of the resolved
                // flag via `mapper.set_bf16_kernel_active`.
                let m8_bf16_effective = m8_bf16_resolver(model_total_bytes);
                let kernel_dtype = if m8_bf16_effective {
                    crate::tensor::DType::BF16
                } else {
                    crate::tensor::DType::F32
                };
                log_m8_kernel_dtype(kernel_dtype);
                mapper.set_bf16_kernel_active(Some(m8_bf16_effective));
                let plan_input = crate::gpu::tier_plan::TierPlanInput {
                    tensors: metas,
                    free_vram_bytes,
                    free_ram_bytes,
                    model_total_bytes,
                    total_ram_bytes,
                    kernel_dtype,
                };
                let plan = crate::gpu::tier_plan::plan(&plan_input);
                log_adaptive_headroom(
                    plan_input.model_total_bytes,
                    plan_input.free_ram_bytes,
                    plan_input.total_ram_bytes,
                    &plan,
                );
                log_tier_plan(&plan);
                let (s, _report) = sharded.load_into_with_residency_plan(
                    &mut scratch_graph,
                    &mapper,
                    &plan,
                    &handles.param_ids,
                    &handles.param_names,
                )?;
                store = s;
            } else if single_path.exists() {
                let reader = SafetensorsReader::open(&single_path)?;
                let metas: Vec<crate::gpu::tier_plan::TensorMeta> = reader
                    .iter()
                    .map(|e| crate::gpu::tier_plan::TensorMeta {
                        name: e.name.to_string(),
                        shape: e.shape.to_vec(),
                        dtype: e.dtype,
                    })
                    .collect();
                let model_total_bytes = sum_model_bytes(&metas);
                // **M8.4c** — same conditional as the sharded
                // branch above. Kept inline (not factored into a
                // helper closure) so the data dependency on
                // `model_total_bytes` is locally visible.
                let m8_bf16_effective = m8_bf16_resolver(model_total_bytes);
                let kernel_dtype = if m8_bf16_effective {
                    crate::tensor::DType::BF16
                } else {
                    crate::tensor::DType::F32
                };
                log_m8_kernel_dtype(kernel_dtype);
                mapper.set_bf16_kernel_active(Some(m8_bf16_effective));
                let plan_input = crate::gpu::tier_plan::TierPlanInput {
                    tensors: metas,
                    free_vram_bytes,
                    free_ram_bytes,
                    model_total_bytes,
                    total_ram_bytes,
                    kernel_dtype,
                };
                let plan = crate::gpu::tier_plan::plan(&plan_input);
                log_adaptive_headroom(
                    plan_input.model_total_bytes,
                    plan_input.free_ram_bytes,
                    plan_input.total_ram_bytes,
                    &plan,
                );
                log_tier_plan(&plan);
                let (s, _report) = mapper.load_into_with_residency_plan(
                    &mut scratch_graph,
                    &reader,
                    &plan,
                    &handles.param_ids,
                    &handles.param_names,
                )?;
                store = s;
            } else {
                return Err(PipelineError::MissingFile(format!(
                    "{} or {}",
                    single_path.display(),
                    index_path.display()
                )));
            }
        } else {
            // Legacy load path — unchanged from M5.f.a.
            if index_path.exists() {
                let sharded = ShardedSafetensorsReader::open(&index_path)?;
                let _report = sharded.load_into(&mut scratch_graph, &mapper)?;
            } else if single_path.exists() {
                let reader = SafetensorsReader::open(&single_path)?;
                let _report = mapper.load_into(&mut scratch_graph, &reader)?;
            } else {
                return Err(PipelineError::MissingFile(format!(
                    "{} or {}",
                    single_path.display(),
                    index_path.display()
                )));
            }

            // 5. Hoist into Arc-shared store. After this call the
            //    scratch graph's parameter slots are `Shared`
            //    views, so dropping the graph is safe — the
            //    parameter Arcs survive in the store.
            store = WeightStore::extract_from_graph(
                &mut scratch_graph,
                &handles.param_ids,
                &handles.param_names,
            )?;
        }

        // 6. Drop scratch graph. The Llama graph also holds
        //    the causal-mask Parameter (zero-shape-on-build)
        //    which is not in the store — it's recomputed by
        //    each new build via `build_llama_with_store`. No
        //    leak.
        drop(scratch_graph);

        // 7. **M6 step 4c** — opt-in GPU residency.
        //
        // Default-off so production smokes stay bit-exact with
        // the M5.f.a baseline. Operators set
        // `ATENIA_GPU_RESIDENCY=1` to upload the first N layers
        // of the model to VRAM. Each layer's BF16 parameters
        // become `SharedParam::Cuda` and the original
        // `Arc<Vec<u16>>` is dropped from the store, freeing
        // host RAM (subject to whether any sibling reference
        // survives — see `UploadReport.ram_bytes_freed` doc).
        //
        // The dispatch hook in `gpu/dispatch/hooks.rs::try_gpu_matmul`
        // (M6 step 4d) detects the `Cuda` weight at matmul
        // time and routes to a residency path that uploads the
        // small activation (~20 KB) instead of the 270 MB
        // weight, which is the throughput unlock this whole
        // milestone is targeting.
        //
        // `n_layers` is hardcoded conservatively at 5 for the
        // first runs — 5 × ~1.21 GiB F32 ≈ 6 GiB persistent
        // VRAM, comfortably under the RTX 4070 Laptop's 8 GiB
        // dedicated. A future sub-step can wire this to the
        // safety gate's `DegradeToLayers` decision and to a
        // user-facing env (e.g. `ATENIA_GPU_LAYERS=N`).
        if !tier_aware && std::env::var("ATENIA_GPU_RESIDENCY").as_deref() == Ok("1") {
            let n_layers: usize = 5;
            let mut total_report = UploadReport::default();
            for layer_idx in 0..n_layers {
                match store.upload_layer_bf16_to_vram(layer_idx) {
                    Ok(report) => {
                        eprintln!(
                            "[M6] Layer {}: {} params to VRAM, \
                             {:.2} GiB RAM freed",
                            layer_idx,
                            report.params_uploaded,
                            report.ram_bytes_freed as f64 / 1024.0_f64.powi(3),
                        );
                        total_report.params_uploaded += report.params_uploaded;
                        total_report.vram_bytes_used += report.vram_bytes_used;
                        total_report.ram_bytes_freed += report.ram_bytes_freed;
                    }
                    Err(e) => {
                        eprintln!(
                            "[M6] Layer {} upload failed: {} \
                             — staying on CPU",
                            layer_idx, e,
                        );
                    }
                }
            }
            eprintln!(
                "[M6] Residency total: {} params, {:.2} GiB in VRAM, \
                 {:.2} GiB RAM freed",
                total_report.params_uploaded,
                total_report.vram_bytes_used as f64 / 1024.0_f64.powi(3),
                total_report.ram_bytes_freed as f64 / 1024.0_f64.powi(3),
            );
        }

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

/// **M7.2** — sum the source-dtype byte size across a tensor
/// meta list. Feeds `TierPlanInput::model_total_bytes`. Mirrors
/// `TensorMeta::ram_cost_bytes` (private to the planner module)
/// — kept inline here so the loader path doesn't depend on
/// internal helpers.
fn sum_model_bytes(metas: &[crate::gpu::tier_plan::TensorMeta]) -> u64 {
    metas
        .iter()
        .map(|m| {
            let numel: u64 = m.shape.iter().product::<usize>() as u64;
            numel * (m.dtype.size_in_bytes() as u64)
        })
        .sum()
}

/// **M8.3** — operator-facing log of the kernel dtype that the
/// tier-aware planner is using for VRAM cost computation. Emits
/// nothing on the F32 default (avoids cluttering the legacy
/// stderr); emits a one-liner banner when the M8 BF16 path is
/// enabled, so the operator can confirm the flag was picked up.
fn log_m8_kernel_dtype(dtype: crate::tensor::DType) {
    if crate::apx_is_silent() {
        return;
    }
    if dtype == crate::tensor::DType::BF16 {
        eprintln!(
            "[ATENIA] M8 BF16 kernel active: VRAM budget doubles \
             (numel×2 vs numel×4). Loader uploads BF16 weights \
             via bf16_to_vram_no_upcast (M8.1); dispatcher routes \
             through cuda_matmul_bf16_inplace which upcasts the \
             BF16 weight to an F32 transient on-device per-matmul \
             and runs cublasGemmEx(F32, F32, F32) — Path B (M8.4c) \
             preserves M4.7.2.e numerics (drift ≤ 2.4e-2 vs F64 \
             4-model fixture, ADR-004 ≤ 0.5)."
        );
    }
}

/// **M7.2** — operator-facing log of the adaptive RAM headroom
/// decision. Emits one line summarising:
/// model size / free RAM / total RAM / headroom (base + overflow).
/// Suppressed in `--silent` builds.
fn log_adaptive_headroom(
    model_total_bytes: u64,
    free_ram_bytes: u64,
    total_ram_bytes: u64,
    plan: &crate::gpu::tier_plan::TierPlan,
) {
    if crate::apx_is_silent() {
        return;
    }
    let gib = |b: u64| (b as f64) / (1024.0_f64.powi(3));
    let base_gib = gib(crate::gpu::tier_plan::RAM_HEADROOM_BASE_BYTES);
    let overflow_gib = gib(plan.ram_headroom_overflow_bytes);
    let headroom_gib = gib(plan.ram_headroom_bytes);
    eprintln!(
        "[ATENIA] Adaptive headroom: model {:.2} GiB, free RAM {:.2} GiB, \
         total RAM {:.2} GiB → RAM headroom {:.2} GiB ({:.2} base + {:.2} \
         overflow protection)",
        gib(model_total_bytes),
        gib(free_ram_bytes),
        gib(total_ram_bytes),
        headroom_gib,
        base_gib,
        overflow_gib,
    );
}

/// **M6 replan sub-fase 3** — operator-facing log of the
/// tier-aware load decision. Suppressed in `--silent` builds via
/// `crate::apx_is_silent`.
fn log_tier_plan(plan: &crate::gpu::tier_plan::TierPlan) {
    if crate::apx_is_silent() {
        return;
    }
    let gib = |b: u64| (b as f64) / (1024.0_f64.powi(3));
    eprintln!("[ATENIA] Tier-aware loader plan:");
    eprintln!(
        "  VRAM: {} tensors ({:.2} GiB)",
        plan.vram_count(),
        gib(plan.vram_bytes_assigned)
    );
    eprintln!(
        "  RAM:  {} tensors ({:.2} GiB)",
        plan.ram_count(),
        gib(plan.ram_bytes_assigned)
    );
    eprintln!(
        "  Disk: {} tensors ({:.2} GiB)",
        plan.disk_count(),
        gib(plan.disk_bytes_assigned)
    );
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
