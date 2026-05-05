//! Public reproduction-surface helpers for the v20 killer demo.
//!
//! M4.9.b extracts the helpers that previously lived inline in
//! `tests/m4_7_6_e_llama2_13b_modes_b_c_test.rs` so the
//! `atenia run` CLI subcommand and the test harness share one
//! source of truth. Three groups:
//!
//!   - **Pressure probes** for the M3-e SignalBus. Two pairs:
//!     low-pressure stubs (no autonomous trigger) and one-shot
//!     high-pressure probes that fire `Degrade → DeepDegrade`
//!     exactly once before falling back to low-pressure
//!     readings (the canonical Mode B mechanism — keeps the
//!     guard verdict from re-firing on every node checkpoint
//!     and tripping the M4.7.5.e activation-arm gap).
//!
//!   - **`make_context`** + **`permissive_contract`**: factory
//!     functions for `ReactiveExecutionContext` instances
//!     wired with the right probes + guard set + execution
//!     contract for the demo's three modes.
//!
//!   - **`build_and_load_llama`**: end-to-end graph build +
//!     sharded BF16 load for any Llama-family checkpoint
//!     (Llama 2 13B Chat is the demo target; the function is
//!     parameterised on `LlamaRuntime { batch, seq }` so the
//!     test harness's seq=1 and the demo runner's seq=4
//!     callers share one path).
//!
//!   - **`argmax_row`**: the canonical reduction the
//!     transparency contract is gated on.
//!
//! The module is gated behind the `demo` Cargo feature
//! (default-enabled). Power users wanting a minimal library
//! build can disable it via `--no-default-features`; the
//! engine crate's core surface compiles without `demo`. The
//! `atenia run` binary requires the feature; the M4.7.6.e
//! integration test imports it directly so its existing
//! `cargo test --test m4_7_6_e_llama2_13b_modes_b_c_test`
//! invocation keeps working without explicit feature flags.
//!
//! No new engine logic lives here — every primitive is
//! existing engine code surfaced behind a stable public API.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;
use uuid::Uuid;

use crate::amg::builder::GraphBuilder;
use crate::amg::graph::Graph;
use crate::amg::reactive::ReactiveExecutionContext;
use crate::amm::ram_probe::{RamProbeApi, RamProbeError, RamSnapshot};
use crate::amm::signal_bus::SignalBus;
use crate::amm::vram_probe::{VramProbeApi, VramProbeError, VramSnapshot};
use crate::amg::weight_store::WeightStore;
use crate::nn::llama::{
    build_llama, build_llama_with_store, llama_weight_mapper, LlamaConfig,
    LlamaRuntime,
};
use crate::v15::policy::types::DecisionBias;
use crate::v16::contract::constraints::{Constraints, RuntimeState};
use crate::v16::contract::execution_contract::{
    ExecutionBackend, ExecutionContract,
};
use crate::v16::guards::execution_guard::ExecutionGuard;
use crate::v16::guards::guard_manager::GuardManager;
use crate::v16::guards::simple_memory_pressure_guard::SimpleMemoryPressureGuard;
use crate::v17::loader::sharded_reader::ShardedSafetensorsReader;

// ============================================================
// Pressure probes
// ============================================================

/// Low-pressure VRAM probe — returns total=1000, used=100
/// → memory_pressure ≈ 0.10. Below the M4.6
/// `SimpleMemoryPressureGuard` threshold (0.65) and the
/// M4.7.5 `dual_memory_pressure` threshold (0.85), so no
/// autonomous migration fires while this probe is active.
pub struct LowPressureVramProbe;
impl VramProbeApi for LowPressureVramProbe {
    fn snapshot(&self) -> Result<VramSnapshot, VramProbeError> {
        Ok(VramSnapshot {
            total_bytes: 1000,
            free_bytes: 900,
            used_bytes: 100,
        })
    }
}

/// Low-pressure RAM probe. Mirror of [`LowPressureVramProbe`]
/// for the RAM channel of the SignalBus.
pub struct LowPressureRamProbe;
impl RamProbeApi for LowPressureRamProbe {
    fn snapshot(&self) -> Result<RamSnapshot, RamProbeError> {
        Ok(RamSnapshot {
            total_bytes: 1000,
            available_bytes: 900,
            used_bytes: 100,
        })
    }
}

/// One-shot high-pressure VRAM probe. Returns 0.95 used (above
/// the 0.85 dual_memory_pressure threshold) on the FIRST
/// `high_remaining` snapshot calls; returns 0.10 used (below
/// threshold) thereafter.
///
/// Why: a permanently high probe causes the M4.6 guard to fire
/// `Degrade` (and the `dual_memory_pressure` site to promote
/// it to `DeepDegrade`) at *every* node checkpoint, not just
/// once. The continuous spill churns activation tensors
/// mid-forward and exposes the known M4.7.5.e gap where one
/// consumer arm reads `as_cpu_slice` without first calling
/// `ensure_cpu` on a tensor the spill freshly migrated to
/// disk. That gap is M5+ scope; the one-shot pattern keeps
/// the killer-demo Mode B contract intact: DeepDegrade fires
/// *autonomously* at the first guard checkpoint, migrates the
/// graph's parameters to disk, then pressure drops back to
/// low and the forward completes via the well-tested
/// lazy-restore path (the same path M4.7.5.f re-validated on
/// the four 1B-class models).
pub struct OneShotHighPressureVramProbe {
    pub high_remaining: AtomicUsize,
}
impl VramProbeApi for OneShotHighPressureVramProbe {
    fn snapshot(&self) -> Result<VramSnapshot, VramProbeError> {
        let prev = self.high_remaining.load(Ordering::Relaxed);
        if prev > 0 {
            self.high_remaining.fetch_sub(1, Ordering::Relaxed);
            Ok(VramSnapshot {
                total_bytes: 1000,
                free_bytes: 50,
                used_bytes: 950,
            })
        } else {
            Ok(VramSnapshot {
                total_bytes: 1000,
                free_bytes: 900,
                used_bytes: 100,
            })
        }
    }
}

/// One-shot high-pressure RAM probe. Mirror of
/// [`OneShotHighPressureVramProbe`] for the RAM channel.
pub struct OneShotHighPressureRamProbe {
    pub high_remaining: AtomicUsize,
}
impl RamProbeApi for OneShotHighPressureRamProbe {
    fn snapshot(&self) -> Result<RamSnapshot, RamProbeError> {
        let prev = self.high_remaining.load(Ordering::Relaxed);
        if prev > 0 {
            self.high_remaining.fetch_sub(1, Ordering::Relaxed);
            Ok(RamSnapshot {
                total_bytes: 1000,
                available_bytes: 50,
                used_bytes: 950,
            })
        } else {
            Ok(RamSnapshot {
                total_bytes: 1000,
                available_bytes: 900,
                used_bytes: 100,
            })
        }
    }
}

// ============================================================
// Execution contract + reactive context factory
// ============================================================

/// Permissive `ExecutionContract`: bias toward stability,
/// allows `ExecutionBackend::Local`, no constraints. The
/// killer-demo runner attaches this contract to every reactive
/// context — the demo's interest is in the M3-e reaction
/// loop's spill / restore behaviour, not in policy gating.
pub fn permissive_contract() -> ExecutionContract {
    ExecutionContract {
        bias: DecisionBias {
            risk_weight: 0.3,
            latency_weight: 0.4,
            stability_weight: 0.5,
            memory_pressure_weight: 0.5,
            offload_cost_weight: 0.4,
        },
        runtime_snapshot: RuntimeState {
            memory_headroom: 0.8,
            is_stable: true,
            recent_recovery: false,
            offload_supported: true,
        },
        allowed_backends: vec![ExecutionBackend::Local],
        forbidden_backends: vec![],
        max_aggressiveness: 0.5,
        require_fallback: false,
        require_stability: false,
        constraints: Constraints { items: vec![] },
    }
}

/// Build a reactive context for the demo. `cache_dir` is the
/// disk-tier directory that the LRU spill writes into;
/// `high_pressure` selects between the low-pressure (no
/// autonomous trigger) probe pair and the one-shot
/// high-pressure pair (Mode B / autonomous trigger semantics).
///
/// The high-pressure variant pre-allocates 4 high-pressure
/// reads — covers the bus's "collect all signals"
/// pre-checkpoint sweep (vram + ram + a couple of derived
/// reads) and lets the FIRST guard checkpoint produce a
/// `Degrade → DeepDegrade` promotion. Subsequent checkpoints
/// see low-pressure values and return `Continue`.
pub fn make_context(
    cache_dir: PathBuf,
    high_pressure: bool,
) -> ReactiveExecutionContext {
    let bus = if high_pressure {
        Arc::new(SignalBus::with_probes(
            None,
            None,
            None,
            None,
            Some(Arc::new(OneShotHighPressureVramProbe {
                high_remaining: AtomicUsize::new(4),
            })),
            Some(Arc::new(OneShotHighPressureRamProbe {
                high_remaining: AtomicUsize::new(4),
            })),
        ))
    } else {
        Arc::new(SignalBus::with_probes(
            None,
            None,
            None,
            None,
            Some(Arc::new(LowPressureVramProbe)),
            Some(Arc::new(LowPressureRamProbe)),
        ))
    };
    let guards: Vec<Box<dyn ExecutionGuard>> =
        vec![Box::new(SimpleMemoryPressureGuard::new())];
    let gm = GuardManager::new(guards);
    ReactiveExecutionContext::new_without_gc(bus, permissive_contract(), gm)
        .with_cache_dir(cache_dir)
}

/// Helper: cache directory for a labelled run. Uses the
/// `ATENIA_DISK_TIER_DIR` environment variable as the base,
/// then falls back to the engine's
/// [`crate::tensor::disk_tier::default_cache_dir`] under a
/// `m4_7_6_e` sub-directory. Each call appends a `label_<UUID>`
/// suffix so concurrent runs do not collide.
pub fn cache_dir_for(label: &str) -> PathBuf {
    let base = match std::env::var("ATENIA_DISK_TIER_DIR") {
        Ok(v) => PathBuf::from(v),
        Err(_) => crate::tensor::disk_tier::default_cache_dir().join("m4_7_6_e"),
    };
    base.join(format!("{}_{}", label, Uuid::new_v4()))
}

// ============================================================
// Loader (build + sharded BF16 load)
// ============================================================

/// Metrics returned by [`build_and_load_llama`]. Lets the
/// caller print whichever fields it wants (the function
/// itself is silent unless `verbose = true`).
pub struct LlamaLoadMetrics {
    /// Wall-clock duration of `GraphBuilder::build`, seconds.
    pub build_secs: f32,
    /// Wall-clock duration of
    /// `ShardedSafetensorsReader::load_into`, seconds.
    pub load_secs: f32,
    /// Number of named tensors successfully loaded.
    pub tensors_loaded: usize,
    /// Number of tensors in the safetensors index that the
    /// mapper did not consume (e.g. per-layer
    /// `rotary_emb.inv_freq` buffers Atenia computes at
    /// runtime).
    pub tensors_skipped: usize,
    /// Number of named parameter nodes in the built graph
    /// (the `LlamaHandles::param_ids.len()`).
    pub param_count: usize,
    /// Parsed `config.json`. Returned so the caller does not
    /// have to re-parse it for vocab / layer counts.
    pub config: LlamaConfig,
}

/// Build the standard Llama-family graph at the given
/// `runtime` (`{ batch, seq }`) and load its weights from a
/// sharded safetensors checkpoint with BF16 storage active.
///
/// The model directory is expected to contain `config.json`
/// and the sharded safetensors files (`model-NNNNN-of-NNNNN.safetensors`
/// + `model.safetensors.index.json`). Single-file checkpoints
/// (where the index file is absent) are not supported by this
/// helper — callers needing single-file loading should use
/// `SafetensorsReader` directly.
///
/// `verbose = true` enables the test harness's existing
/// progress prints (used by `tests/m4_7_6_e_*`); `false` keeps
/// the helper silent so the CLI runner can drive its own
/// progress UI.
///
/// # Panics
///
/// Panics if `config.json` cannot be parsed, the index file
/// is missing, or the load reports any missing tensors. These
/// are demo-fatal conditions that should fail loudly during
/// development — production-grade error handling lands as
/// part of M4.9.c's `--mode a` runner.
pub fn build_and_load_llama(
    model_dir: &Path,
    runtime: LlamaRuntime,
    verbose: bool,
) -> (Graph, WeightStore, LlamaLoadMetrics) {
    if verbose {
        println!("Reading config.json ...");
    }
    let cfg = LlamaConfig::from_json_file(&model_dir.join("config.json"))
        .expect("config.json must parse");

    if verbose {
        println!(
            "Building Llama graph at seq={} ({} layers, hidden {}, vocab {}) ...",
            runtime.seq, cfg.num_hidden_layers, cfg.hidden_size, cfg.vocab_size,
        );
    }
    // ---- Build scratch graph ----
    //
    // The tier-aware loader needs a `Graph` with the full
    // parameter schema to validate shapes / dtypes against
    // the safetensors header. After the load it returns a
    // populated `WeightStore` and the scratch graph is
    // discarded — for VRAM/Disk-tier weights, the loader
    // never writes the parameter slots in the graph it
    // received (only Ram-tier weights are extracted via
    // `finalize_ram_extract`). The forward graph is rebuilt
    // below with `build_llama_with_store` so every parameter
    // node is connected to the store via
    // `register_param_from_store`. Same canonical pattern
    // as `LlamaPipeline::load` / `generate_greedy`.
    let build_start = Instant::now();
    let mut gb = GraphBuilder::new();
    let token_input_id = gb.input();
    let handles = build_llama(&mut gb, &cfg, &runtime, token_input_id);
    let _ = gb.output(handles.logits_id);
    let mut scratch_graph = gb.build();
    let build_secs = build_start.elapsed().as_secs_f32();
    if verbose {
        println!(
            "Graph built in {:.2}s ({} parameter nodes)",
            build_secs,
            handles.param_ids.len(),
        );
    }
    let param_count = handles.param_ids.len();

    if verbose {
        println!(
            "Loading weights from {} (BF16 storage) ...",
            model_dir.display(),
        );
    }
    let load_start = Instant::now();
    let sharded = ShardedSafetensorsReader::open(
        &model_dir.join("model.safetensors.index.json"),
    )
    .expect("open sharded reader");
    let mut mapper =
        llama_weight_mapper(&cfg, &handles.param_names, &handles.param_ids)
            .expect("llama weight mapper");
    mapper.set_store_params_as_bf16(true);

    // Tier-aware load (M6 + M7 + M8 + M8.7) — same path
    // `LlamaPipeline::load` takes by default. Probes free
    // RAM/VRAM, builds a `TierPlan` from the safetensors
    // header metadata, and routes each parameter to its
    // assigned tier (`Vram` / `Ram` / `Disk`) via
    // `load_into_with_residency_plan`.
    let metas: Vec<crate::gpu::tier_plan::TensorMeta> = sharded
        .collect_tensor_metas()
        .expect("collect_tensor_metas");
    let model_total_bytes: u64 = metas
        .iter()
        .map(|m| {
            let numel = m.shape.iter().product::<usize>() as u64;
            numel * (m.dtype.size_in_bytes() as u64)
        })
        .sum();
    let kernel_dtype = if std::env::var("ATENIA_M8_BF16_KERNEL")
        .as_deref()
        == Ok("1")
    {
        crate::tensor::DType::BF16
    } else {
        crate::tensor::DType::F32
    };
    mapper.set_bf16_kernel_active(Some(
        kernel_dtype == crate::tensor::DType::BF16,
    ));
    let plan_input = crate::gpu::tier_plan::TierPlanInput {
        tensors: metas,
        free_vram_bytes: crate::gpu::safety::resource_check::probe_free_vram_bytes(),
        free_ram_bytes: crate::gpu::safety::resource_check::probe_free_ram_bytes(),
        model_total_bytes,
        total_ram_bytes: crate::gpu::safety::resource_check::probe_total_ram_bytes(),
        kernel_dtype,
    };
    let plan = crate::gpu::tier_plan::plan(&plan_input);
    let (store, report) = sharded
        .load_into_with_residency_plan(
            &mut scratch_graph,
            &mapper,
            &plan,
            &handles.param_ids,
            &handles.param_names,
        )
        .expect("load_into_with_residency_plan");
    drop(scratch_graph);
    let load_secs = load_start.elapsed().as_secs_f32();
    if verbose {
        println!(
            "Loaded {} tensors in {:.2}s",
            report.loaded, load_secs,
        );
    }

    // ---- Rebuild graph wired to the store ----
    //
    // Every parameter node in the rebuilt graph is attached
    // to its `SharedParam::{Cuda, Disk, Bf16, F32}` via
    // `register_param_from_store`. The dispatch hooks
    // (`gpu/dispatch/hooks.rs::try_gpu_matmul`,
    // `cuda::matmul::cuda_matmul_disk_streamed_bf16`) read
    // the `SharedParam` variant directly off the parameter's
    // storage at execute time. The `WeightStore` itself is
    // returned alongside the graph so the caller keeps it
    // alive for the lifetime of the graph (the parameter
    // tensors hold `Arc` clones of the store's contents,
    // but VRAM / disk handles are owned by the store).
    let mut gb2 = GraphBuilder::new();
    let token_input_id_2 = gb2.input();
    let handles2 = build_llama_with_store(
        &mut gb2,
        &cfg,
        &runtime,
        token_input_id_2,
        &store,
        None,
    )
    .expect("build_llama_with_store");
    let _ = gb2.output(handles2.logits_id);
    let graph = gb2.build();

    let metrics = LlamaLoadMetrics {
        build_secs,
        load_secs,
        tensors_loaded: report.loaded,
        tensors_skipped: report.skipped.len(),
        param_count,
        config: cfg,
    };

    (graph, store, metrics)
}

// ============================================================
// Output reductions
// ============================================================

/// Argmax over a single logits row of length `vocab`. Returns
/// `(token_id, logit_value)`. Panics if `slice.len() != vocab`
/// or if the row contains only NaN values.
pub fn argmax_row(slice: &[f32], vocab: usize) -> (usize, f32) {
    assert_eq!(slice.len(), vocab);
    slice
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| {
            x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, v)| (i, *v))
        .unwrap()
}
