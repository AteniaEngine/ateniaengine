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

use super::builder::LlamaRuntime;
use super::numcert;
use crate::amg::builder::GraphBuilder;
use crate::amg::weight_store::{UploadReport, WeightStore, WeightStoreError};
use crate::model_adapters::{
    AteniaModelAdapter, ModelFormat, ResidencyPolicyHints, model_metadata_from_parts,
    resolve_adapter,
};
use crate::nn::llama::config::{ConfigError, LlamaConfig};
use crate::tokenizer::{AteniaTokenizer, ChatMessage, TokenizerError};
use crate::v17::loader::gguf_config::{architecture_from_gguf, llama_config_from_gguf};
use crate::v17::loader::gguf_reader::GgufReader;
use crate::v17::loader::gguf_to_hf_naming::is_gguf_non_weight_tensor;
use crate::v17::loader::loader_errors::LoaderError;
use crate::v17::loader::safetensors_reader::SafetensorsReader;
use crate::v17::loader::sharded_reader::ShardedSafetensorsReader;

/// **M10.2.1 / M10.3.1.0** — fast-mode gate, exposed for the GPU
/// dispatcher (`gpu::dispatch::hooks`) to choose between
/// `cuda_matmul_bf16_inplace` (certified, Path B M8.4c +
/// COMPUTE_32F_FAST_TF32) and `cuda_matmul_bf16_native_inplace`
/// (fast, BF16-TC native, drift industrial, ADR-005 envelope).
///
/// Resolution priority (M10.3.1.0):
///   1. `ATENIA_FAST_MODE=1` env var — operator override, wins
///      unconditionally.
///   2. `<model_dir>/model.numcert.json` `recommended_mode` —
///      checkpoint-default chosen by the certification author.
///   3. `false` (certified) — safe fallback when the env var is
///      not set and no manifest is present.
///
/// The pipeline calls [`init_fast_mode_active`] during model load
/// to resolve the value. The first read after init returns the
/// resolved boolean; reads before init fall through to the env
/// var so direct callers (tests, micro-benches) that bypass the
/// pipeline keep the M10.2.1 semantics unchanged.
///
/// Stored in a `OnceLock` so the value is process-stable and the
/// hot-path read in the dispatcher is a single atomic load.
pub static FAST_MODE_ACTIVE: FastModeGate = FastModeGate::new();

/// Wrapper around the `OnceLock<bool>` that exposes a `Deref`-
/// shaped `*FAST_MODE_ACTIVE` read identical to the M10.2.1
/// `LazyLock<bool>` API the dispatcher uses today, while
/// supporting one-shot initialisation from the pipeline.
pub struct FastModeGate {
    cell: std::sync::OnceLock<bool>,
}

impl FastModeGate {
    const fn new() -> Self {
        Self {
            cell: std::sync::OnceLock::new(),
        }
    }

    /// Read the resolved value. If the pipeline has called
    /// [`init_fast_mode_active`] the resolved boolean is
    /// returned; otherwise the env var is consulted on the fly
    /// (preserves the M10.2.1 contract for direct-dispatcher
    /// callers that never go through the pipeline, e.g. unit
    /// tests in `cuda::matmul::tests`).
    pub fn get(&self) -> bool {
        if let Some(v) = self.cell.get() {
            return *v;
        }
        std::env::var("ATENIA_FAST_MODE").as_deref() == Ok("1")
    }
}

impl std::ops::Deref for FastModeGate {
    type Target = bool;
    fn deref(&self) -> &bool {
        // Match the LazyLock<bool>::deref shape so existing call
        // sites (`*FAST_MODE_ACTIVE`) keep compiling. Returns a
        // reference to a static — safe because both branches
        // produce a `'static` reference.
        if let Some(v) = self.cell.get() {
            return v;
        }
        // Lazy fallback: cache the env-derived value the first
        // time anyone reads through `Deref` without explicit
        // init. Subsequent reads observe the cached value.
        let env = std::env::var("ATENIA_FAST_MODE").as_deref() == Ok("1");
        let _ = self.cell.set(env);
        // Safe: `set` either succeeded (we just stored it) or
        // raced and someone else stored the value first; either
        // way `get` now returns `Some`.
        self.cell.get().expect("OnceLock populated above")
    }
}

/// **M10.3.1.0** — pipeline calls this once per process when it
/// loads the first model; subsequent loads see the same gate
/// (process-stable). Resolution priority is documented on
/// [`FAST_MODE_ACTIVE`]. Returns the resolved boolean for the
/// caller to log.
///
/// Idempotent: calling more than once with the same resolved
/// value is a no-op; calling again with a different value is
/// silently ignored (the first model loaded wins, which matches
/// the operator's intent — they pointed `--model` at one
/// checkpoint per process).
pub fn init_fast_mode_active(manifest: Option<&numcert::NumcertManifest>) -> bool {
    let env_override = std::env::var("ATENIA_FAST_MODE").as_deref() == Ok("1");
    let resolved = if env_override {
        true
    } else if let Some(m) = manifest {
        matches!(m.recommended_mode, numcert::MatmulMode::Fast)
    } else {
        false
    };
    let _ = FAST_MODE_ACTIVE.cell.set(resolved);
    resolved
}
use super::generator::{
    GenerateError, GeneratedToken, GenerationConfig, TokenSink, generate_greedy,
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
            PipelineError::Loader(e) => write!(f, "pipeline loader: {e}"),
            PipelineError::WeightStore(e) => write!(f, "pipeline weight_store: {e}"),
            PipelineError::Generate(e) => write!(f, "pipeline generate: {e}"),
            PipelineError::MissingFile(s) => write!(f, "pipeline: required file not found: {s}"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<std::io::Error> for PipelineError {
    fn from(e: std::io::Error) -> Self {
        PipelineError::Io(e)
    }
}
impl From<ConfigError> for PipelineError {
    fn from(e: ConfigError) -> Self {
        PipelineError::Config(e)
    }
}
impl From<TokenizerError> for PipelineError {
    fn from(e: TokenizerError) -> Self {
        PipelineError::Tokenizer(e)
    }
}
impl From<LoaderError> for PipelineError {
    fn from(e: LoaderError) -> Self {
        PipelineError::Loader(e)
    }
}
impl From<WeightStoreError> for PipelineError {
    fn from(e: WeightStoreError) -> Self {
        PipelineError::WeightStore(e)
    }
}
impl From<GenerateError> for PipelineError {
    fn from(e: GenerateError) -> Self {
        PipelineError::Generate(e)
    }
}

/// **M11.B step 4b** — peek at `architectures[0]` from a HF
/// `config.json` without going through the full `LlamaConfig`
/// parser. Returns `Ok(Some("Phi3ForCausalLM"))`,
/// `Ok(Some("LlamaForCausalLM"))`, etc. when the field is
/// present and a non-empty array of strings; `Ok(None)` when the
/// field is absent / null / empty / non-array (legacy configs);
/// `Err` only on I/O or top-level JSON syntax errors.
///
/// Used by the pipeline routing branch to choose between
/// `build_llama` and `phi3::build_phi3`. The `LlamaConfig`
/// parser proper continues to ignore the `architectures` field
/// (it consumes only the math-relevant fields), so the peek
/// happens against the raw JSON rather than an extension to
/// `LlamaConfig`.
fn read_architectures_first(config_path: &Path) -> Result<Option<String>, PipelineError> {
    let bytes = std::fs::read(config_path).map_err(PipelineError::Io)?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| {
        PipelineError::Config(ConfigError::Parse(format!(
            "config.json JSON syntax error: {e}"
        )))
    })?;
    Ok(v.get("architectures")
        .and_then(|x| x.as_array())
        .and_then(|arr| arr.first())
        .and_then(|x| x.as_str())
        .map(|s| s.to_string()))
}

fn find_single_gguf(model_dir: &Path) -> Result<Option<PathBuf>, PipelineError> {
    let mut ggufs = Vec::new();
    if !model_dir.exists() {
        return Ok(None);
    }
    for entry in std::fs::read_dir(model_dir).map_err(PipelineError::Io)? {
        let path = entry.map_err(PipelineError::Io)?.path();
        if path.extension().and_then(|x| x.to_str()) == Some("gguf") {
            ggufs.push(path);
        }
    }
    match ggufs.len() {
        0 => Ok(None),
        1 => Ok(ggufs.pop()),
        _ => Err(PipelineError::Loader(LoaderError::InvalidFormat(format!(
            "GGUF load expects exactly one .gguf file in {}; found {}",
            model_dir.display(),
            ggufs.len()
        )))),
    }
}

fn build_gguf_name_map(
    reader: &GgufReader,
    adapter: &dyn AteniaModelAdapter,
) -> Result<std::collections::HashMap<String, String>, PipelineError> {
    // **Phase 16.3** — `arch` is kept only for the diagnostic
    // message (and to validate `general.architecture` is present);
    // the GGUF→HF name *structure* is now decided by the resolved
    // adapter (`GgufNameMapper`), not by an `if arch == "..."`
    // branch in the core.
    let arch = reader.architecture().ok_or_else(|| {
        PipelineError::Loader(LoaderError::InvalidFormat(
            "GGUF file is missing general.architecture metadata".to_string(),
        ))
    })?;
    let mut map = std::collections::HashMap::with_capacity(reader.tensors.len());
    for descriptor in &reader.tensors {
        // GGUF config-input tensors (e.g. Phi-3 LongRope
        // `rope_factors_{short,long}.weight`) have no HF graph
        // parameter — they are consumed at config-parse time by
        // `gguf_config::gguf_rope_scaling_json`. Skip them here
        // instead of hard-erroring as "no known HF name mapping";
        // absent from the map, the downstream loader skips them.
        if is_gguf_non_weight_tensor(&descriptor.name) {
            continue;
        }
        let hf_name = adapter.gguf_to_hf_name(&descriptor.name).ok_or_else(|| {
            PipelineError::Loader(LoaderError::InvalidFormat(format!(
                "GGUF tensor '{}' has no known HuggingFace name mapping for architecture '{}'",
                descriptor.name, arch
            )))
        })?;
        if map
            .insert(descriptor.name.clone(), hf_name.clone())
            .is_some()
        {
            return Err(PipelineError::Loader(LoaderError::InvalidFormat(format!(
                "GGUF tensor '{}' appears more than once",
                descriptor.name
            ))));
        }
    }
    Ok(map)
}

fn gguf_tensor_metas(
    reader: &GgufReader,
    name_map: &std::collections::HashMap<String, String>,
    bf16_storage: bool,
) -> Result<Vec<crate::gpu::tier_plan::TensorMeta>, PipelineError> {
    let mut metas = Vec::with_capacity(reader.tensors.len());
    for descriptor in &reader.tensors {
        // Mirror the `build_gguf_name_map` skip: config-input
        // tensors are not graph weights and are intentionally
        // absent from `name_map` — exclude them from the
        // tier-planner metas rather than erroring "missing from
        // name map".
        if is_gguf_non_weight_tensor(&descriptor.name) {
            continue;
        }
        let hf_name = name_map.get(&descriptor.name).ok_or_else(|| {
            PipelineError::Loader(LoaderError::InvalidFormat(format!(
                "GGUF tensor '{}' missing from name map",
                descriptor.name
            )))
        })?;
        match descriptor.tensor_type {
            crate::v17::loader::gguf_reader::GgufTensorType::F32
            | crate::v17::loader::gguf_reader::GgufTensorType::F16
            | crate::v17::loader::gguf_reader::GgufTensorType::Q8_0
            | crate::v17::loader::gguf_reader::GgufTensorType::Q4_K
            | crate::v17::loader::gguf_reader::GgufTensorType::Q6_K => {}
            other => {
                return Err(PipelineError::Loader(LoaderError::UnsupportedDType(
                    format!(
                        "GGUF tensor '{}' has unsupported type {:?} for M11.D.3",
                        descriptor.name, other
                    ),
                )));
            }
        };
        // GGUF quantized tensors are decoded into an F32 working buffer by
        // `decode_tensor` before the residency write. The steady-state RAM/Disk
        // footprint is therefore Atenia's storage dtype, not the source GGUF
        // quantization width. Keeping Q4/Q8 as `Int8` here makes the tier
        // planner over-pack memory even though no quantized residency exists.
        let dtype = if bf16_storage {
            crate::tensor::DType::BF16
        } else {
            crate::tensor::DType::F32
        };
        metas.push(crate::gpu::tier_plan::TensorMeta {
            name: hf_name.clone(),
            shape: descriptor.dimensions.iter().map(|d| *d as usize).collect(),
            dtype,
        });
    }
    Ok(metas)
}

/// **M12.4 H2** — render the M6 legacy-residency summary line.
/// Pulled out as a pure function so the degrade-visibility contract
/// (failed-layer count appended when `failed > 0`) is unit-testable
/// without a live `WeightStore`. The success-only shape is
/// byte-identical to the pre-M12.4 line.
fn format_m6_residency_summary(total: &UploadReport, failed: usize) -> String {
    let mut s = format!(
        "[M6] Residency total: {} params, {:.2} GiB in VRAM, {:.2} GiB RAM freed",
        total.params_uploaded,
        total.vram_bytes_used as f64 / 1024.0_f64.powi(3),
        total.ram_bytes_freed as f64 / 1024.0_f64.powi(3),
    );
    if failed > 0 {
        s.push_str(&format!(
            "; {failed} layer(s) failed and stayed on CPU (degraded — see \
             the [M6] Layer … upload failed lines above)"
        ));
    }
    s
}

impl GenerationPipeline {
    /// Construct a pipeline by reading every artefact under
    /// `model_dir`. See module-level doc for the load
    /// procedure. Heavy: BF16 storage shrinks the on-host
    /// footprint to ~half the F32 path; expect 1–2 seconds
    /// for TinyLlama and ~3 minutes for Llama 2 13B on the
    /// dev box (matches the M4.7.6 / M4.8 numbers — load
    /// dominated by safetensors decode + transform).
    pub fn from_model_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self, PipelineError> {
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

        let gguf_path = find_single_gguf(&model_dir)?;
        let gguf_reader = match gguf_path.as_ref() {
            Some(path) => Some(GgufReader::read_from_path(path)?),
            None => None,
        };
        let is_gguf = gguf_reader.is_some();

        // 1. Config.
        let config_path = model_dir.join("config.json");
        if !is_gguf && !config_path.exists() {
            return Err(PipelineError::MissingFile(
                config_path.display().to_string(),
            ));
        }
        let config = if let Some(reader) = gguf_reader.as_ref() {
            llama_config_from_gguf(reader)?
        } else {
            LlamaConfig::from_json_file(&config_path)?
        };

        // 2. Tokenizer.
        let tokenizer = AteniaTokenizer::from_model_dir(&model_dir)?;

        // 2.5. **M10.3.1.0** — numeric contract manifest. Read the
        //      sibling `model.numcert.json` (if present) before the
        //      M8 / fast-mode resolver runs below; the manifest's
        //      `recommended_mode` is the per-checkpoint default that
        //      replaces the bare `ATENIA_FAST_MODE` global. Resolution
        //      priority lives on `FAST_MODE_ACTIVE`; here we just
        //      surface the file we found (or didn't) and pin the
        //      value into the OnceLock so every subsequent dispatcher
        //      read is process-stable.
        let manifest = numcert::load_manifest(&model_dir);
        let env_fast_override = std::env::var("ATENIA_FAST_MODE").as_deref() == Ok("1");
        let fast_resolved = init_fast_mode_active(manifest.as_ref());
        match (&manifest, env_fast_override) {
            (Some(m), false) => eprintln!(
                "[ATENIA] Numeric contract: {} found — recommended mode: {} \
                 (schema {}). FAST_MODE_ACTIVE = {}.",
                m.source.display(),
                if matches!(m.recommended_mode, numcert::MatmulMode::Fast) {
                    "fast"
                } else if matches!(m.recommended_mode, numcert::MatmulMode::Quantized) {
                    "quantized"
                } else {
                    "certified"
                },
                m.schema_version,
                fast_resolved,
            ),
            (Some(m), true) => eprintln!(
                "[ATENIA] Numeric contract: {} found — recommended mode: {} \
                 (schema {}). ATENIA_FAST_MODE=1 env override active; \
                 FAST_MODE_ACTIVE = true.",
                m.source.display(),
                if matches!(m.recommended_mode, numcert::MatmulMode::Fast) {
                    "fast"
                } else if matches!(m.recommended_mode, numcert::MatmulMode::Quantized) {
                    "quantized"
                } else {
                    "certified"
                },
                m.schema_version,
            ),
            (None, false) => eprintln!(
                "[ATENIA] Numeric contract: no manifest at {}/{} — \
                 defaulting to certified mode (FAST_MODE_ACTIVE = false). \
                 Set ATENIA_FAST_MODE=1 to opt into fast without a manifest, \
                 or ship a model.numcert.json sibling for per-checkpoint \
                 selection.",
                model_dir.display(),
                numcert::MANIFEST_FILENAME,
            ),
            (None, true) => eprintln!(
                "[ATENIA] Numeric contract: no manifest at {}/{}; \
                 ATENIA_FAST_MODE=1 env override active; \
                 FAST_MODE_ACTIVE = true.",
                model_dir.display(),
                numcert::MANIFEST_FILENAME,
            ),
        }

        // **M11.B step 4** — architecture detection. Before
        // building the scratch graph, peek at the raw config JSON
        // to read `architectures[0]`. Phi-3 / Phi-3.5 ships
        // `Phi3ForCausalLM`; the Llama-family checkpoints ship
        // `LlamaForCausalLM`. The two paths build different
        // graphs and use different weight mappers; everything
        // downstream of the build (graph wiring, weight loading,
        // tier-aware planning) is the same shape because both
        // builders return `(token_input_id, logits_id, param_ids,
        // param_names)` with HF-convention names that the
        // mapper consumes.
        let architecture = if let Some(reader) = gguf_reader.as_ref() {
            architecture_from_gguf(reader)?
        } else {
            read_architectures_first(&config_path)?.unwrap_or_default()
        };
        let metadata = model_metadata_from_parts(
            (!architecture.is_empty()).then_some(architecture.as_str()),
            config.model_type.as_deref(),
            if is_gguf {
                ModelFormat::Gguf
            } else {
                ModelFormat::HfSafetensors
            },
        );
        let adapter = resolve_adapter(&metadata).ok_or_else(|| {
            PipelineError::Loader(LoaderError::InvalidFormat(format!(
                "unsupported architecture/model_type: architecture=\"{}\", model_type={:?}; {}",
                metadata.architecture,
                config.model_type,
                crate::model_adapters::supported_architectures_message()
            )))
        })?;
        adapter.log_selection();
        let residency_hints = adapter.residency_hints(&config);
        log_adapter_residency_policy(adapter, residency_hints);

        // 3. Scratch graph (zero-init parameters; will get
        //    populated by the loader and then hoisted into
        //    the store).
        let runtime = LlamaRuntime { batch: 1, seq: 1 };
        let mut gb = GraphBuilder::new();
        let token_input_id = gb.input();
        // Build the adapter-specific graph and extract the shared
        // (logits, param_ids, param_names) shape.
        let scratch = adapter.build_scratch_graph(&mut gb, &config, &runtime, token_input_id);
        let logits_id = scratch.logits_id;
        let param_ids = scratch.param_ids;
        let param_names = scratch.param_names;
        let _ = gb.output(logits_id);
        let mut scratch_graph = gb.build();

        // 4. Load weights. Mapper selection is adapter-specific because
        //    families differ in names, layout transforms, and fused tensors.
        let mut mapper = if is_gguf {
            adapter.map_gguf_weights(&config, &param_names, &param_ids)?
        } else {
            adapter.map_hf_weights(&config, &param_names, &param_ids)?
        };
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
        let tier_aware = std::env::var("ATENIA_LEGACY_LOADER").as_deref() != Ok("1");
        if !tier_aware {
            eprintln!(
                "[ATENIA] Legacy loader active (ATENIA_LEGACY_LOADER=1): \
                 tier-aware placement disabled."
            );
        }
        let gpu_residency = std::env::var("ATENIA_GPU_RESIDENCY").as_deref() == Ok("1");

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
            let free_ram_bytes = crate::gpu::safety::resource_check::probe_free_ram_bytes();
            // **M12.1** — surface a VRAM-probe failure rather than
            // silently treating it as "0 free VRAM" (which looks
            // identical to a real GPU with no spare VRAM). Value
            // stays `0` on failure so the tier plan is unchanged.
            let free_vram_bytes =
                match crate::gpu::safety::resource_check::probe_free_vram_bytes_detailed() {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!(
                            "[ATENIA][warn] VRAM probe failed: {e}; treating free VRAM \
                             as 0 — weights will be placed in RAM/Disk. If this box has \
                             an NVIDIA GPU, check that `nvidia-smi` is on PATH."
                        );
                        0
                    }
                };
            let total_ram_bytes = crate::gpu::safety::resource_check::probe_total_ram_bytes();

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
            let env_requested = std::env::var("ATENIA_M8_BF16_KERNEL").as_deref() == Ok("1");
            // **M10.2.1** — fast mode (`ATENIA_FAST_MODE=1`) implies
            // M8 BF16 storage (the native fast kernel consumes a
            // BF16-resident weight) and overrides the M7.2 RAM-based
            // M8 heuristic. Read once at first load via the
            // `FAST_MODE_ACTIVE` LazyLock below.
            let fast_mode = *FAST_MODE_ACTIVE;
            if fast_mode {
                eprintln!(
                    "[ATENIA] Fast mode active (BF16-TC native, drift \
                     industrial, no ADR-004 strict guarantee). \
                     Routes through cuda_matmul_bf16_native_inplace \
                     instead of cuda_matmul_bf16_inplace; cublasGemmEx \
                     with CUDA_R_16BF inputs + COMPUTE_32F. See ADR-005 \
                     for the drift envelope and per-checkpoint \
                     certification policy."
                );
            }
            // `model_total_bytes` is computed inside each load
            // branch; we wire the conditional inline there so both
            // branches share the same logic.
            let m8_bf16_resolver = |model_total_bytes: u64| -> bool {
                if fast_mode {
                    // Fast mode requires BF16 storage in VRAM —
                    // override the M7.2 RAM-based heuristic.
                    return true;
                }
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

            if let Some(reader) = gguf_reader.as_ref() {
                let name_map = build_gguf_name_map(reader, adapter)?;
                let metas = gguf_tensor_metas(reader, &name_map, bf16_storage)?;
                let model_total_bytes = sum_model_bytes(&metas);
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
                let plan = crate::gpu::tier_plan::plan_with_hints(&plan_input, residency_hints);
                log_adaptive_headroom(
                    plan_input.model_total_bytes,
                    plan_input.free_ram_bytes,
                    plan_input.total_ram_bytes,
                    &plan,
                );
                log_tier_plan(&plan_input, &plan);
                let (s, report) = mapper.load_gguf_with_residency_plan(
                    &mut scratch_graph,
                    reader,
                    &name_map,
                    &plan,
                    &param_ids,
                    &param_names,
                )?;
                // Allow rope_freqs as skipped tensor (RoPE scaling metadata, not a model weight)
                let acceptable_skipped: Vec<&str> = vec!["rope_freqs"];
                let unexpected_skipped: Vec<_> = report
                    .skipped
                    .iter()
                    .filter(|s| !acceptable_skipped.iter().any(|a| s.contains(a)))
                    .cloned()
                    .collect();
                if !unexpected_skipped.is_empty() || !report.missing.is_empty() {
                    return Err(PipelineError::Loader(LoaderError::InvalidFormat(format!(
                        "GGUF load incomplete: unexpected_skipped={:?}, missing={:?}",
                        unexpected_skipped, report.missing
                    ))));
                }
                store = s;
            } else if index_path.exists() {
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
                let plan = crate::gpu::tier_plan::plan_with_hints(&plan_input, residency_hints);
                log_adaptive_headroom(
                    plan_input.model_total_bytes,
                    plan_input.free_ram_bytes,
                    plan_input.total_ram_bytes,
                    &plan,
                );
                log_tier_plan(&plan_input, &plan);
                let (s, _report) = sharded.load_into_with_residency_plan(
                    &mut scratch_graph,
                    &mapper,
                    &plan,
                    &param_ids,
                    &param_names,
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
                let plan = crate::gpu::tier_plan::plan_with_hints(&plan_input, residency_hints);
                log_adaptive_headroom(
                    plan_input.model_total_bytes,
                    plan_input.free_ram_bytes,
                    plan_input.total_ram_bytes,
                    &plan,
                );
                log_tier_plan(&plan_input, &plan);
                let (s, _report) = mapper.load_into_with_residency_plan(
                    &mut scratch_graph,
                    &reader,
                    &plan,
                    &param_ids,
                    &param_names,
                )?;
                store = s;
            } else {
                return Err(PipelineError::MissingFile(format!(
                    "{} or {} or a single .gguf file",
                    single_path.display(),
                    index_path.display()
                )));
            }
        } else {
            // Legacy load path — unchanged from M5.f.a.
            if let Some(reader) = gguf_reader.as_ref() {
                let name_map = build_gguf_name_map(reader, adapter)?;
                let report = mapper.load_gguf_into(&mut scratch_graph, reader, &name_map)?;
                // Allow rope_freqs as skipped tensor (RoPE scaling metadata, not a model weight)
                let acceptable_skipped: Vec<&str> = vec!["rope_freqs"];
                let unexpected_skipped: Vec<_> = report
                    .skipped
                    .iter()
                    .filter(|s| !acceptable_skipped.iter().any(|a| s.contains(a)))
                    .cloned()
                    .collect();
                if !unexpected_skipped.is_empty() || !report.missing.is_empty() {
                    return Err(PipelineError::Loader(LoaderError::InvalidFormat(format!(
                        "GGUF load incomplete: unexpected_skipped={:?}, missing={:?}",
                        unexpected_skipped, report.missing
                    ))));
                }
            } else if index_path.exists() {
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
            store = WeightStore::extract_from_graph(&mut scratch_graph, &param_ids, &param_names)?;
        }

        // 5.5. **M10.3.1.1** — stamp per-tensor matmul policy onto
        //      each VRAM-resident TensorGPU from the manifest's
        //      `per_tensor_policy` (or the whole-model
        //      `recommended_mode` fallback if the manifest is
        //      v1.0.0). Walks names + params once; every Cuda
        //      variant gets a precision byte the dispatcher
        //      reads at matmul time. Tensors not yet uploaded to
        //      VRAM (Bf16 / F32 / Disk) are stamped lazily — once
        //      they transition to Cuda via
        //      `upload_layer_bf16_to_vram` (M6) the dispatcher's
        //      None-fallback to FAST_MODE_ACTIVE preserves the
        //      M10.3.1.0 contract.
        //
        //      Placed outside the tier-aware vs legacy if/else
        //      so all three paths (tier-aware sharded,
        //      tier-aware single, legacy extract_from_graph)
        //      produce a stamped store identically.
        if let Some(ref m) = manifest {
            store.apply_per_tensor_policy(m);
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
            // **M12.4 H2** — count layers that failed to upload so the
            // summary line can surface the degrade. Without this an
            // operator scanning logs sees "Residency total: N params"
            // and may not notice that some layers silently stayed on
            // CPU. Bounded loop (≤5), one-time at load — not a hot
            // path, so the per-layer line stays as-is.
            let mut failed: usize = 0;
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
                        failed += 1;
                        eprintln!(
                            "[M6] Layer {} upload failed: {} \
                             — staying on CPU",
                            layer_idx, e,
                        );
                    }
                }
            }
            eprintln!("{}", format_m6_residency_summary(&total_report, failed));
        }

        Ok(Self {
            config,
            tokenizer,
            store,
            model_dir,
        })
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
            self.tokenizer
                .apply_chat_template(&[ChatMessage::user(prompt.to_string())])?
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
            let full = tokenizer.decode(&generated_ids, true).unwrap_or_default();
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
            &self.config,
            &self.store,
            &prompt_ids,
            &gen_cfg,
            decode,
            &mut recording,
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

fn log_adapter_residency_policy(adapter: &dyn AteniaModelAdapter, hints: ResidencyPolicyHints) {
    if crate::apx_is_silent() {
        return;
    }
    let caps = adapter.capabilities();
    eprintln!(
        "[ATENIA] Adapter residency policy: adapter={} family={:?} \
         hf={} gguf={} store={} fused_qkv={} fused_gate_up={} gemma2_softcaps={} \
         hints(proj_vram={}, lm_head={:?}, embeddings_cpu={}, norms_cpu={}, layer_local={})",
        adapter.id(),
        adapter.family(),
        caps.hf_safetensors,
        caps.gguf,
        caps.store_backed_generation,
        caps.fused_qkv_weight_mapping,
        caps.fused_gate_up_weight_mapping,
        caps.gemma2_softcaps,
        hints.projection_weights_vram_eligible,
        hints.lm_head,
        hints.embeddings_cpu_only,
        hints.norms_cpu_only,
        hints.prefer_layer_local_projection_packing,
    );
}

/// **M6 replan sub-fase 3** — operator-facing log of the
/// tier-aware load decision. Suppressed in `--silent` builds via
/// `crate::apx_is_silent`.
fn log_tier_plan(
    input: &crate::gpu::tier_plan::TierPlanInput,
    plan: &crate::gpu::tier_plan::TierPlan,
) {
    if crate::apx_is_silent() {
        return;
    }
    let gib = |b: u64| (b as f64) / (1024.0_f64.powi(3));
    let total_resident_if_vram: u64 = input
        .tensors
        .iter()
        .map(|m| m.vram_cost_bytes(input.kernel_dtype))
        .sum();
    let gpu_eligible_count = input
        .tensors
        .iter()
        .filter(|m| crate::gpu::tier_plan::is_gpu_eligible(m))
        .count();
    let gpu_eligible_source_bytes: u64 = input
        .tensors
        .iter()
        .filter(|m| crate::gpu::tier_plan::is_gpu_eligible(m))
        .map(|m| m.ram_cost_bytes())
        .sum();
    let gpu_eligible_resident_bytes: u64 = input
        .tensors
        .iter()
        .filter(|m| crate::gpu::tier_plan::is_gpu_eligible(m))
        .map(|m| m.vram_cost_bytes(input.kernel_dtype))
        .sum();
    let certified_upload_reserve = if input.kernel_dtype == crate::tensor::DType::F32 {
        crate::gpu::tier_plan::CERTIFIED_BF16_UPLOAD_STAGING_BYTES
    } else {
        0
    };
    let vram_budget = input
        .free_vram_bytes
        .saturating_sub(crate::gpu::tier_plan::VRAM_HEADROOM_BYTES)
        .saturating_sub(certified_upload_reserve);
    let final_vram_budget = if plan.disk_bytes_assigned > 0 {
        vram_budget.saturating_sub(crate::gpu::tier_plan::DISK_PIPELINE_STAGING_BYTES)
    } else {
        vram_budget
    };

    let mut ram_not_gpu_eligible = 0usize;
    let mut ram_gpu_eligible_budget = 0usize;
    let mut disk_not_gpu_eligible = 0usize;
    let mut disk_gpu_eligible_budget = 0usize;

    for meta in &input.tensors {
        let Some(tier) = plan.get(&meta.name) else {
            continue;
        };
        let eligible = crate::gpu::tier_plan::is_gpu_eligible(meta);
        match (tier, eligible) {
            (crate::gpu::tier_plan::Tier::Ram, false) => ram_not_gpu_eligible += 1,
            (crate::gpu::tier_plan::Tier::Ram, true) => ram_gpu_eligible_budget += 1,
            (crate::gpu::tier_plan::Tier::Disk, false) => disk_not_gpu_eligible += 1,
            (crate::gpu::tier_plan::Tier::Disk, true) => disk_gpu_eligible_budget += 1,
            _ => {}
        }
    }

    eprintln!(
        "[ATENIA] Tier planner inputs: source {:.2} GiB, all-resident estimate {:.2} GiB \
         at {:?}, free VRAM {:.2} GiB, VRAM budget {:.2} GiB ({:.2} GiB after staging; \
         certified upload reserve {:.2} GiB), \
         GPU-eligible {} / {} tensors ({:.2} GiB source -> {:.2} GiB resident).",
        gib(input.model_total_bytes),
        gib(total_resident_if_vram),
        input.kernel_dtype,
        gib(input.free_vram_bytes),
        gib(vram_budget),
        gib(final_vram_budget),
        gib(certified_upload_reserve),
        gpu_eligible_count,
        input.tensors.len(),
        gib(gpu_eligible_source_bytes),
        gib(gpu_eligible_resident_bytes),
    );
    // **M12.3** — shared summary (also used by the `atenia run`
    // loader). The per-reason line + PLAN_TRACE below stay here.
    crate::gpu::tier_plan::log_tier_plan_summary(&plan);
    eprintln!(
        "  Reasons: RAM not_gpu_eligible={}, RAM vram_budget_exceeded={}, \
         Disk not_gpu_eligible={}, Disk vram_or_ram_budget_exceeded={}",
        ram_not_gpu_eligible,
        ram_gpu_eligible_budget,
        disk_not_gpu_eligible,
        disk_gpu_eligible_budget,
    );

    if std::env::var("ATENIA_PLAN_TRACE").as_deref() == Ok("1") {
        let mut ram_tensors: Vec<_> = input
            .tensors
            .iter()
            .filter_map(|meta| {
                let tier = plan.get(&meta.name)?;
                if tier != crate::gpu::tier_plan::Tier::Ram {
                    return None;
                }
                let reason = if crate::gpu::tier_plan::is_gpu_eligible(meta) {
                    "vram_budget_exceeded"
                } else {
                    "not_gpu_eligible"
                };
                Some((
                    meta.vram_cost_bytes(input.kernel_dtype),
                    meta.ram_cost_bytes(),
                    reason,
                    meta,
                ))
            })
            .collect();
        ram_tensors.sort_by(|a, b| b.0.cmp(&a.0));

        eprintln!("[ATENIA] PLAN_TRACE top RAM tensors:");
        for (resident_bytes, source_bytes, reason, meta) in ram_tensors.into_iter().take(12) {
            eprintln!(
                "  RAM reason={} name={} dtype={:?} shape={:?} source={:.3} GiB resident_if_vram={:.3} GiB",
                reason,
                meta.name,
                meta.dtype,
                meta.shape,
                gib(source_bytes),
                gib(resident_bytes),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_tinyllama_q8_0_gguf_pipeline_without_generation() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/tinyllama-q8_0");
        if !model_dir
            .join("tinyllama-1.1b-chat-v1.0.Q8_0.gguf")
            .exists()
            || !model_dir.join("tokenizer.json").exists()
            || !model_dir.join("tokenizer_config.json").exists()
        {
            eprintln!(
                "[skip] TinyLlama Q8_0 GGUF pipeline test requires .gguf, tokenizer.json, \
                 and tokenizer_config.json in {}",
                model_dir.display()
            );
            return;
        }
        let pipe = GenerationPipeline::from_model_dir_with_options(&model_dir, true)
            .expect("TinyLlama Q8_0 GGUF pipeline load");
        assert_eq!(pipe.config.num_hidden_layers, 22);
        assert_eq!(pipe.config.hidden_size, 2048);
        assert_eq!(pipe.config.vocab_size, 32_000);
        assert!(!pipe.config.tie_word_embeddings);
        assert!(
            pipe.store
                .names
                .iter()
                .any(|n| n == "model.embed_tokens.weight"),
            "HF-mapped embed tensor must be present in WeightStore"
        );
        assert!(
            pipe.store.names.iter().any(|n| n == "lm_head.weight"),
            "HF-mapped untied lm_head tensor must be present in WeightStore"
        );
        assert!(
            pipe.store
                .names
                .iter()
                .any(|n| n == "model.layers.0.self_attn.q_proj.weight"),
            "HF-mapped block tensor must be present in WeightStore"
        );
    }

    #[test]
    fn generates_one_token_from_tinyllama_q8_0_gguf() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/tinyllama-q8_0");
        if !model_dir
            .join("tinyllama-1.1b-chat-v1.0.Q8_0.gguf")
            .exists()
            || !model_dir.join("tokenizer.json").exists()
            || !model_dir.join("tokenizer_config.json").exists()
        {
            eprintln!(
                "[skip] TinyLlama Q8_0 GGUF generation test requires .gguf, tokenizer.json, \
                 and tokenizer_config.json in {}",
                model_dir.display()
            );
            return;
        }
        let pipe = GenerationPipeline::from_model_dir_with_options(&model_dir, true)
            .expect("TinyLlama Q8_0 GGUF pipeline load");
        let mut sink = crate::nn::llama::generator::CollectingTokenSink::default();
        let text = pipe
            .generate_raw("Hello", 1, &mut sink)
            .expect("TinyLlama Q8_0 GGUF one-token generation");
        assert_eq!(sink.tokens.len(), 1);
        assert!(sink.tokens[0].token_id < pipe.config.vocab_size as u32);
        assert_eq!(text, sink.tokens[0].text);
    }

    #[test]
    fn loads_tinyllama_q4_k_m_gguf_pipeline_without_generation() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF");
        if !model_dir
            .join("tinyllama-1.1b-chat-v1.0-q4_k_m.gguf")
            .exists()
            || !model_dir.join("tokenizer.json").exists()
            || !model_dir.join("tokenizer_config.json").exists()
        {
            eprintln!(
                "[skip] TinyLlama Q4_K_M GGUF pipeline test requires .gguf, tokenizer.json, \
                 and tokenizer_config.json in {}",
                model_dir.display()
            );
            return;
        }
        let pipe = GenerationPipeline::from_model_dir_with_options(&model_dir, true)
            .expect("TinyLlama Q4_K_M GGUF pipeline load");
        assert_eq!(pipe.config.num_hidden_layers, 22);
        assert_eq!(pipe.config.hidden_size, 2048);
        assert_eq!(pipe.config.vocab_size, 32_000);
        assert!(!pipe.config.tie_word_embeddings);
        assert!(
            pipe.store
                .names
                .iter()
                .any(|n| n == "model.embed_tokens.weight"),
            "HF-mapped embed tensor must be present in WeightStore"
        );
        assert!(
            pipe.store.names.iter().any(|n| n == "lm_head.weight"),
            "HF-mapped untied lm_head tensor must be present in WeightStore"
        );
        assert!(
            pipe.store
                .names
                .iter()
                .any(|n| n == "model.layers.0.self_attn.q_proj.weight"),
            "HF-mapped block tensor must be present in WeightStore"
        );
    }

    #[test]
    fn generates_one_token_from_tinyllama_q4_k_m_gguf() {
        let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("models/TinyLlama-1.1B-Chat-v1.0-Q4_K_M-GGUF");
        if !model_dir
            .join("tinyllama-1.1b-chat-v1.0-q4_k_m.gguf")
            .exists()
            || !model_dir.join("tokenizer.json").exists()
            || !model_dir.join("tokenizer_config.json").exists()
        {
            eprintln!(
                "[skip] TinyLlama Q4_K_M GGUF generation test requires .gguf, tokenizer.json, \
                 and tokenizer_config.json in {}",
                model_dir.display()
            );
            return;
        }
        let pipe = GenerationPipeline::from_model_dir_with_options(&model_dir, true)
            .expect("TinyLlama Q4_K_M GGUF pipeline load");
        let mut sink = crate::nn::llama::generator::CollectingTokenSink::default();
        let text = pipe
            .generate_raw("Hello", 1, &mut sink)
            .expect("TinyLlama Q4_K_M GGUF one-token generation");
        assert_eq!(sink.tokens.len(), 1);
        assert!(sink.tokens[0].token_id < pipe.config.vocab_size as u32);
        assert_eq!(text, sink.tokens[0].text);
        eprintln!(
            "[M11.D.4] TinyLlama Q4_K_M GGUF one-token generation: token_id={}, text=\"{}\"",
            sink.tokens[0].token_id, text
        );
    }

    // ----- M12.4 -----

    /// **M12.4 H2** — success-only summary is byte-identical to the
    /// pre-M12.4 line (no behavioural drift for a healthy upload).
    #[test]
    fn m6_residency_summary_no_failures_is_unchanged() {
        let total = UploadReport {
            params_uploaded: 1234,
            vram_bytes_used: 2 * 1024 * 1024 * 1024,
            ram_bytes_freed: 1024 * 1024 * 1024,
        };
        let s = format_m6_residency_summary(&total, 0);
        assert_eq!(
            s,
            "[M6] Residency total: 1234 params, 2.00 GiB in VRAM, 1.00 GiB RAM freed"
        );
        assert!(!s.contains("failed"));
    }

    /// **M12.4 H2** — a partial degrade appends an explicit
    /// failed-layer count so the operator sees it in the summary.
    #[test]
    fn m6_residency_summary_appends_failed_count() {
        let total = UploadReport {
            params_uploaded: 10,
            vram_bytes_used: 0,
            ram_bytes_freed: 0,
        };
        let s = format_m6_residency_summary(&total, 3);
        assert!(s.starts_with("[M6] Residency total: 10 params,"));
        assert!(s.contains("3 layer(s) failed and stayed on CPU"));
    }

    /// **M12.4 H5** — `PipelineError::Loader` now renders the inner
    /// `LoaderError` via `Display` (human text), not the raw `{:?}`
    /// debug form, so a CLI load failure is operator-readable.
    #[test]
    fn pipeline_error_loader_uses_display_not_debug() {
        let e = PipelineError::Loader(LoaderError::InvalidFormat(
            "config.json missing required field 'hidden_size'".to_string(),
        ));
        let s = e.to_string();
        assert_eq!(
            s,
            "pipeline loader: invalid format: config.json missing required \
             field 'hidden_size'"
        );
        // Must not leak the Rust enum-debug shape.
        assert!(!s.contains("InvalidFormat("));
    }
}
