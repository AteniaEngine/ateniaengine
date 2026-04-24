#[allow(unused_imports)]
use crate::tensor::ops::*;
use crate::apx3_9::op_router::{route, ExecTarget};
use crate::apx4::gpu_dispatch::{dispatch_matmul as dispatch_matmul_gpu, ApxExecTarget as Apx4ExecTarget};
use crate::apx4_5::dispatcher::dispatch_batch_matmul_cuda;
use crate::apx4_3::GpuPlan;
use crate::apx4_11::gpu_hooks;
use crate::apx4_7::{PersistentPlan, FusionPlan};
use crate::apx4_13::fusion_engine::FusedOp;
use crate::apx5::kernel_planner::{KernelPlanner, KernelTarget};
use crate::apx5::apx_5_3_planner::{Planner5_3, NodeExecInfo};
use crate::apx5_4::{Sample, DeviceTarget};
use crate::tensor::{StorageTransferError, Layout, Tensor, TensorStorage};
use crate::cpu_features::cpu_features;
use super::chunking::{chunk_tensor, merge_chunks};
use super::nodes::{Node, NodeType};
use super::scheduler::{build_execution_plan, ExecStep, ExecutionPlan};
use crate::amg::grad_store::GradStore;
use crate::amg::reactive::{ExecutionAbortReason, ReactiveExecutionContext};
use crate::autograd::{BackOp, BackwardTape};
use crate::v16::guards::guard_action::GuardAction;
use crate::v16::guards::guard_errors::GuardError;
use rayon::prelude::*;
use crate::nn::activations as nn_act;
use crate::nn::linear as nn_linear;
use std::time::Instant;
use crate::nn::normalization as nn_norm;
use crate::nn::softmax as nn_softmax;
use crate::optim::adamw::AdamW;
#[cfg(debug_assertions)]
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub enum FusedOutput {
    QKV { q: Tensor, k: Tensor, v: Tensor },
    /// APX 4.17: output of fused Self-Attention (forward-only).
    SelfAttention {
        q: Tensor,
        k: Tensor,
        v: Tensor,
        att: Tensor,
        out: Tensor,
    },
}

pub struct Graph {
    pub nodes: Vec<Node>,
    pub plan: ExecutionPlan,
    pub tape: BackwardTape,
    pub grad_store: GradStore,
    pub gpu_plan: Option<GpuPlan>,
    pub persistent_plan: Option<PersistentPlan>,
    pub fusion_plan: Option<FusionPlan>,
    pub fusions_applied: usize,
    pub fused_ops: std::collections::HashMap<usize, FusedOp>,
    pub fused_outputs: std::collections::HashMap<usize, FusedOutput>,
    /// APX 8.1: optional CPU+GPU dual graph. It is not used for execution yet;
    /// it is only built as structural metadata.
    pub dual_graph: Option<crate::apx8::dualgraph::DualGraph>,
    /// APX v20 M2: optional reactive execution layer. When set,
    /// schedulers consult `check_guard_before_node` before each node
    /// and abort cleanly if guards fire. Opt-in; default is `None`
    /// and execution behavior is identical to pre-M2.
    pub(crate) reactive_context: Option<ReactiveExecutionContext>,
    /// APX v20 M2: diagnostic — reason of the last abort during a
    /// checked execution, cleared at the start of each `execute_checked`.
    pub(crate) last_abort: Option<ExecutionAbortReason>,
}

impl Graph {
    /// Returns true if the given node is the secondary Linear (K or V) of
    /// some FusedQKV pattern detected in the graph.
    pub fn is_qkv_secondary(&self, id: usize) -> bool {
        for fused in self.fused_ops.values() {
            if let FusedOp::FusedQKV { q_id, k_id, v_id, .. } = fused {
                if id == *k_id || id == *v_id {
                    return true;
                }
                // The representative node (q_id) is handled via exec_fused.
                let _ = q_id; // avoid warnings in builds without direct usage
            }
        }
        false
    }

    /// Returns true if the given node is one of the Q/K/V linear nodes
    /// involved in a FusedSelfAttention pattern.
    pub fn is_sa_linear(&self, id: usize) -> bool {
        for fused in self.fused_ops.values() {
            if let FusedOp::FusedSelfAttention { q, k, v, .. } = fused {
                if id == *q || id == *k || id == *v {
                    return true;
                }
            }
        }
        false
    }

    /// Attaches a reactive execution context so that subsequent
    /// checked executions consult guards before each node. Replaces
    /// any previously set context.
    pub fn set_reactive_context(&mut self, ctx: ReactiveExecutionContext) {
        self.reactive_context = Some(ctx);
    }

    /// Removes any previously set reactive context. Subsequent
    /// executions run without guard checks.
    pub fn clear_reactive_context(&mut self) {
        self.reactive_context = None;
    }

    /// Returns the most recent abort reason produced by a checked
    /// execution path, if any. Cleared at the start of every
    /// `execute_checked`.
    pub fn last_abort(&self) -> Option<&ExecutionAbortReason> {
        self.last_abort.as_ref()
    }

    /// Borrow the attached `ReactiveExecutionContext`, if any. Lets
    /// callers read runtime counters (e.g. `degrade_events_count`) and
    /// inspect the signal bus without owning the context.
    pub fn reactive_context(&self) -> Option<&ReactiveExecutionContext> {
        self.reactive_context.as_ref()
    }

    /// Consults the reactive context (if set) and returns `Err` when
    /// guards produce an `Abort` verdict or the manager rejects the
    /// verdict as illegal. Called by schedulers before each node.
    ///
    /// `Ok(())` is returned when:
    /// - No `reactive_context` is set.
    /// - The signal bus returns `None` for `collect_guard_conditions`
    ///   (fail-open: absence of telemetry does not block execution).
    /// - The combined guard verdict is `Continue` or `Degrade`. In
    ///   M2, `Degrade` is treated as `Continue` with a debug log;
    ///   strategy selection arrives in M3.
    ///
    /// When `Err` is returned, `self.last_abort` is also populated
    /// with the same reason for post-execution inspection.
    pub(crate) fn check_guard_before_node(
        &mut self,
        node_id: usize,
    ) -> Result<(), ExecutionAbortReason> {
        // TODO(PERF): `collect_guard_conditions` runs two subprocess
        // reads (nvidia-smi + sysinfo) per call, ~40 ms. Without a
        // cache this makes a 1000-node graph take minutes just on
        // probing. Must be resolved before APX v20 M5 (first real
        // model end-to-end). See ROADMAP.
        //
        // The verdict is collected in an inner block so the shared
        // borrow of `self.reactive_context` ends before the arms below
        // (notably the Degrade arm) take `&mut self` to migrate.
        let (verdict, conditions) = {
            let Some(ctx) = self.reactive_context.as_ref() else {
                return Ok(());
            };
            let Some(conditions) = ctx.signal_bus.collect_guard_conditions() else {
                // Fail-open: absence of telemetry doesn't block execution.
                return Ok(());
            };
            (
                ctx.guard_manager.evaluate(&ctx.contract, &conditions),
                conditions,
            )
        };

        // M3-e.11.5: dual-pressure promotion. If the guard manager
        // returned `Degrade` but both VRAM and RAM are saturated
        // beyond the deep-spill thresholds, promote to
        // `DeepDegrade` so the reaction site will spill tensors
        // all the way to disk instead of only moving them from
        // VRAM to RAM (which would not help when RAM is already
        // saturated). The policy lives here rather than inside a
        // guard to keep guards single-responsibility — the guard
        // answers "is the pressure high?", the reaction site
        // decides "what to do about it?", same pattern as the
        // M3-e.6 CPU veto.
        let verdict = match verdict {
            Ok(GuardAction::Degrade)
                if crate::amg::reactive::dual_memory_pressure(&conditions) =>
            {
                Ok(GuardAction::DeepDegrade)
            }
            other => other,
        };

        match verdict {
            Ok(GuardAction::Abort) => {
                let reason = ExecutionAbortReason::GuardAborted {
                    at_node: node_id,
                    conditions,
                };
                self.last_abort = Some(reason.clone());
                Err(reason)
            }
            Ok(GuardAction::DeepDegrade) => {
                // M3-e.11.5: spill every eligible tensor to disk
                // via `migrate_all_to_disk` (composite that first
                // brings Cuda → Cpu via the pre-existing shallow
                // migration, then spills Cpu → Disk). Reached
                // either via direct guard emission or via promotion
                // from `Degrade` on the preceding line. The CPU-
                // veto of M3-e.6 intentionally does NOT apply here:
                // disk migration does not add CPU load the way Cpu-
                // tier migration does, so the argument for vetoing
                // (other processes saturating CPU) does not
                // translate.
                let memory_pressure = conditions.memory_pressure;
                let probes_so_far = self
                    .reactive_context
                    .as_ref()
                    .map(|ctx| ctx.signal_bus.probe_calls_count())
                    .unwrap_or(0);
                let timestamp_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);

                // Capture the cache dir before the reactive-context
                // borrow ends, so the mutable borrow for the
                // migration below does not conflict.
                let cache_dir = self
                    .reactive_context
                    .as_ref()
                    .map(|ctx| ctx.cache_dir.clone())
                    .unwrap_or_else(
                        || crate::tensor::disk_tier::default_cache_dir(),
                    );

                if let Some(ctx) = self.reactive_context.as_ref() {
                    ctx.record_deep_degrade_event();
                }

                let migrate_result = self.migrate_all_to_disk(&cache_dir);

                if !crate::apx_is_silent() {
                    let mem_frag =
                        crate::amg::reactive::format_memory_fragment(&conditions);
                    let gpu_frag =
                        crate::amg::reactive::format_gpu_util_fragment(&conditions);
                    let fg_frag =
                        crate::amg::reactive::format_foreground_fragment(&conditions);
                    let bat_frag =
                        crate::amg::reactive::format_battery_fragment(&conditions);
                    let lat_frag =
                        crate::amg::reactive::format_latency_fragment(&conditions);
                    match &migrate_result {
                        Ok(report) => {
                            eprintln!(
                                "[AMG Guard][t_ms={}] DeepDegrade triggered at node {}: \
                                 memory_pressure={:.2}, probes_so_far={},\
                                 {}{}{}{}{} {}",
                                timestamp_ms,
                                node_id,
                                memory_pressure,
                                probes_so_far,
                                mem_frag,
                                gpu_frag,
                                fg_frag,
                                bat_frag,
                                lat_frag,
                                report,
                            );
                        }
                        Err(e) => {
                            eprintln!(
                                "[AMG Guard][t_ms={}] DeepDegrade migration FAILED at node {}: \
                                 memory_pressure={:.2}, probes_so_far={},\
                                 {}{}{}{}{} error: {:?}. \
                                 Continuing; subsequent nodes may still fail.",
                                timestamp_ms,
                                node_id,
                                memory_pressure,
                                probes_so_far,
                                mem_frag,
                                gpu_frag,
                                fg_frag,
                                bat_frag,
                                lat_frag,
                                e,
                            );
                        }
                    }
                }
                // A migration failure here does NOT abort the
                // graph — the CPU/Disk state may be inconsistent
                // across nodes after a partial migrate, but the
                // subsequent execution will lazily pull data via
                // `ensure_cpu` on access. Same policy as the
                // Degrade arm above.
                Ok(())
            }
            Ok(GuardAction::Degrade) => {
                // M3-e.1: act on Degrade by migrating every Cuda-resident
                // `node.output` back to CPU. Execution continues with
                // CPU storage after this; subsequent nodes that would
                // have taken the GPU path now run on CPU, and the VRAM
                // held by the migrated tensors is released as soon as
                // their `Arc<InnerGpuPtr>` refcount reaches zero.
                //
                // M3-e.5: count the attempt (success or failure) and
                // capture observability fields before migration runs,
                // so logs and counters reflect the state that triggered
                // the reaction, not post-migration state.
                let memory_pressure = conditions.memory_pressure;
                let probes_so_far = self
                    .reactive_context
                    .as_ref()
                    .map(|ctx| ctx.signal_bus.probe_calls_count())
                    .unwrap_or(0);
                let timestamp_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);

                // M3-e.6: CPU-availability precondition. If the
                // system is under CPU pressure caused by external
                // processes (not Atenia), migrating more work to the
                // CPU would worsen the external pressure and likely
                // hurt both the user's foreground workload and
                // Atenia itself. Skip the migration in that case and
                // record the veto. Decision lives here (not in the
                // guard) so the guard remains single-responsibility
                // ("is memory pressure high?") while policy
                // ("what do we do about it?") stays at the reaction
                // site — see the M3-e.6 handoff Option (c).
                if crate::amg::reactive::cpu_saturated_externally(&conditions) {
                    if let Some(ctx) = self.reactive_context.as_ref() {
                        ctx.record_degrade_veto_by_cpu();
                    }
                    if !crate::apx_is_silent() {
                        let cpu_total = conditions.cpu_pressure_total.unwrap_or(f32::NAN);
                        let cpu_self = conditions.cpu_pressure_self.unwrap_or(f32::NAN);
                        let share = if cpu_total > 0.0 {
                            cpu_self / cpu_total
                        } else {
                            f32::NAN
                        };
                        let gpu_frag =
                            crate::amg::reactive::format_gpu_util_fragment(&conditions);
                        let fg_frag =
                            crate::amg::reactive::format_foreground_fragment(&conditions);
                        let bat_frag =
                            crate::amg::reactive::format_battery_fragment(&conditions);
                        let lat_frag =
                            crate::amg::reactive::format_latency_fragment(&conditions);
                        eprintln!(
                            "[AMG Guard][t_ms={}] Degrade VETOED at node {} \
                             (external CPU pressure): memory_pressure={:.2}, \
                             cpu_total={:.2}, cpu_self={:.2}, self_share={:.2},\
                             {}{}{}{} thresholds total>{}, share<{}. \
                             Skipping migration; execution continues on VRAM.",
                            timestamp_ms,
                            node_id,
                            memory_pressure,
                            cpu_total,
                            cpu_self,
                            share,
                            gpu_frag,
                            fg_frag,
                            bat_frag,
                            lat_frag,
                            crate::amg::reactive::CPU_PRESSURE_TOTAL_THRESHOLD,
                            crate::amg::reactive::CPU_SELF_CONTRIBUTION_MIN,
                        );
                    }
                    return Ok(());
                }

                if let Some(ctx) = self.reactive_context.as_ref() {
                    ctx.record_degrade_event();
                }
                match self.migrate_all_cuda_to_cpu() {
                    Ok(report) => {
                        if !crate::apx_is_silent() {
                            let mib = report.bytes_freed_estimate as f64
                                / (1024.0 * 1024.0);
                            let gpu_frag =
                                crate::amg::reactive::format_gpu_util_fragment(&conditions);
                            let fg_frag =
                                crate::amg::reactive::format_foreground_fragment(&conditions);
                            let bat_frag =
                                crate::amg::reactive::format_battery_fragment(&conditions);
                            let lat_frag =
                                crate::amg::reactive::format_latency_fragment(&conditions);
                            eprintln!(
                                "[AMG Guard][t_ms={}] Degrade triggered at node {}: \
                                 memory_pressure={:.2}, probes_so_far={},\
                                 {}{}{}{} migrated {} tensors, freed ~{:.2} MiB.",
                                timestamp_ms,
                                node_id,
                                memory_pressure,
                                probes_so_far,
                                gpu_frag,
                                fg_frag,
                                bat_frag,
                                lat_frag,
                                report.tensors_migrated,
                                mib,
                            );
                        }
                    }
                    Err(e) => {
                        // Degrade itself failed (e.g. a D->H transfer
                        // returned an error). Log and continue; the
                        // next node may still fail, but we do not
                        // cascade the transfer error out of the guard.
                        if !crate::apx_is_silent() {
                            let gpu_frag =
                                crate::amg::reactive::format_gpu_util_fragment(&conditions);
                            let fg_frag =
                                crate::amg::reactive::format_foreground_fragment(&conditions);
                            let bat_frag =
                                crate::amg::reactive::format_battery_fragment(&conditions);
                            let lat_frag =
                                crate::amg::reactive::format_latency_fragment(&conditions);
                            eprintln!(
                                "[AMG Guard][t_ms={}] Degrade migration FAILED at node {}: \
                                 memory_pressure={:.2}, probes_so_far={},\
                                 {}{}{}{} error: {:?}. \
                                 Continuing; subsequent nodes may still fail.",
                                timestamp_ms,
                                node_id,
                                memory_pressure,
                                probes_so_far,
                                gpu_frag,
                                fg_frag,
                                bat_frag,
                                lat_frag,
                                e,
                            );
                        }
                    }
                }
                Ok(())
            }
            Ok(GuardAction::Continue) => Ok(()),
            Err(GuardError::IllegalAction(msg))
            | Err(GuardError::InconsistentGuards(msg)) => {
                let reason = ExecutionAbortReason::GuardEvaluationFailed {
                    at_node: node_id,
                    message: msg,
                };
                self.last_abort = Some(reason.clone());
                Err(reason)
            }
        }
    }

    /// Migrate every `node.output` currently in `TensorStorage::Cuda`
    /// back to `TensorStorage::Cpu` via `Tensor::ensure_cpu`. Used by
    /// the `GuardAction::Degrade` reaction strategy introduced in M3-e.1;
    /// also callable standalone for tests and future policies.
    ///
    /// Grads (`output.grad`) are left untouched — they are CPU-resident
    /// by construction (M3-d.3 backward pre-pass contract) and do not
    /// live in VRAM. Input and Parameter nodes are treated identically
    /// to intermediate nodes: any cached output with Cuda storage is
    /// migrated.
    ///
    /// `bytes_freed_estimate` is `numel * size_of::<f32>()` summed over
    /// migrated tensors. It is a best-effort estimate; the real VRAM
    /// release depends on the `Arc<InnerGpuPtr>` refcount at drop time.
    ///
    /// Returns `Err` on the first `ensure_cpu` failure and stops; earlier
    /// migrations in the same call are not rolled back (the tensors that
    /// already moved stay on CPU, which is the safe direction).
    pub fn migrate_all_cuda_to_cpu(
        &mut self,
    ) -> Result<crate::amg::reactive::DegradeReport, StorageTransferError> {
        let mut tensors_migrated = 0usize;
        let mut bytes_freed_estimate = 0usize;

        for node in &mut self.nodes {
            if let Some(ref mut output) = node.output {
                if matches!(output.storage, TensorStorage::Cuda(_)) {
                    let numel = output.numel();
                    output.ensure_cpu()?;
                    tensors_migrated += 1;
                    bytes_freed_estimate += numel * std::mem::size_of::<f32>();
                }
            }
        }

        Ok(crate::amg::reactive::DegradeReport {
            tensors_migrated,
            bytes_freed_estimate,
        })
    }

    /// M3-e.11.4: migrate every `node.output` currently in
    /// `TensorStorage::Cpu` to `TensorStorage::Disk` via the disk
    /// tier introduced in M3-e.11.1. Used by the `DeepDegrade`
    /// reaction strategy that lands in M3-e.11.5; also callable
    /// standalone for tests and future policies.
    ///
    /// # Atomicity
    ///
    /// Per-tensor **write-before-swap**: the data is written to
    /// disk first, and only after `write_f32_tensor` returns
    /// `Ok(handle)` is the tensor's `storage` swapped to
    /// `TensorStorage::Disk(handle)`. If the write fails the
    /// storage is left untouched — the tensor stays fully in
    /// `Cpu` and remains usable. Other tensors already migrated
    /// in the same call are **not rolled back** (consistent with
    /// `migrate_all_cuda_to_cpu`: the safe direction is "away
    /// from the pressured tier", so preserving earlier progress
    /// is always the right call).
    ///
    /// # Orphan files
    ///
    /// A crash in the narrow window between a successful
    /// `write_f32_tensor` and the `storage = ...` assignment
    /// below leaves an orphan file in `cache_dir`. That file
    /// cannot be cleaned up via Arc-Drop because the handle
    /// never reached its owner. The disk tier's periodic GC
    /// (`disk_tier::gc_orphan_disk_tensors`, wired in a later
    /// sub-milestone) sweeps these files on the next process
    /// start. The window is on the order of nanoseconds, so
    /// orphans are rare in practice.
    ///
    /// # Return policy
    ///
    /// - All tensors migrated (or nothing to do): `Ok(report)`
    ///   with `report.is_complete() == true`.
    /// - First tensor failed, none migrated: `Err(error)`. No
    ///   side effects on graph state.
    /// - Partial: at least one tensor migrated, then a later one
    ///   failed: `Ok(report)` with `report.is_partial() == true`.
    ///   The migrated tensors stay in `Disk`; the caller can
    ///   inspect `report.failure` to log the snag or retry.
    ///
    /// # What is not touched
    ///
    /// - `TensorStorage::Cuda(_)` nodes are **skipped** (counted in
    ///   `report.tensors_skipped`). Callers that want to migrate
    ///   Cuda tensors to disk should use
    ///   [`Graph::migrate_all_to_disk`], which chains Cuda → Cpu
    ///   → Disk via the pre-existing
    ///   [`Graph::migrate_all_cuda_to_cpu`].
    /// - `TensorStorage::Disk(_)` nodes are skipped (already in
    ///   the target tier).
    /// - `node.output == None` nodes are skipped (never a tensor
    ///   to migrate).
    /// - `output.grad` is untouched — grads are CPU-resident by
    ///   construction under the M3-d.3 backward pre-pass
    ///   contract.
    pub fn migrate_all_cpu_to_disk(
        &mut self,
        cache_dir: &std::path::Path,
    ) -> Result<crate::amg::reactive::MigrationReport, StorageTransferError> {
        use crate::amg::reactive::MigrationReport;

        let mut report = MigrationReport::new();

        // Ensure the cache dir exists up front. A failure here is
        // a total failure — no tensor has been touched.
        if let Err(e) = std::fs::create_dir_all(cache_dir) {
            return Err(StorageTransferError::DiskWriteFailed(format!(
                "cannot create cache dir {:?}: {}",
                cache_dir, e
            )));
        }

        for (idx, node) in self.nodes.iter_mut().enumerate() {
            let Some(tensor) = node.output.as_mut() else {
                continue;
            };

            // Only Cpu tensors are migration candidates. Cuda gets
            // skipped (caller should compose with
            // migrate_all_cuda_to_cpu first via migrate_all_to_disk).
            // Disk is already in the target tier.
            if !matches!(tensor.storage, TensorStorage::Cpu(_)) {
                report.tensors_skipped += 1;
                continue;
            }

            // Extract a copy of the Cpu data WITHOUT mutating the
            // storage yet. If the write below fails, the tensor
            // must remain fully readable.
            let data: Vec<f32> = match &tensor.storage {
                TensorStorage::Cpu(v) => v.clone(),
                _ => unreachable!("matches! above rules out non-Cpu"),
            };

            // Write-before-swap: the disk file is created first;
            // only after we have a valid handle do we replace
            // `tensor.storage`. If the write errors, the tensor
            // stays in Cpu and the report either becomes partial
            // (if earlier tensors migrated) or we return Err (if
            // nothing migrated yet).
            let handle = match crate::tensor::disk_tier::write_f32_tensor(
                cache_dir, &data,
            ) {
                Ok(h) => h,
                Err(e) => {
                    let err = StorageTransferError::DiskWriteFailed(e.to_string());
                    if report.tensors_migrated > 0 {
                        // Partial progress — preserve what we have,
                        // surface the failure inside the report.
                        report.failure = Some((idx, err));
                        return Ok(report);
                    } else {
                        // Nothing done — zero side effects.
                        return Err(err);
                    }
                }
            };

            // Atomic swap: the Arc<InnerDiskFile> from `handle`
            // becomes the owner. The previous Cpu Vec<f32> drops
            // (its memory is released as usual).
            tensor.storage = TensorStorage::Disk(handle);
            report.tensors_migrated += 1;
        }

        Ok(report)
    }

    /// M3-e.11.4: composite migration for the dual-pressure case.
    /// Chains the pre-existing Cuda → Cpu migration (from M3-e.1)
    /// with the new Cpu → Disk migration (above), producing a
    /// unified [`MigrationReport`] that accounts for both steps.
    ///
    /// Intended as the migration primitive behind
    /// `GuardAction::DeepDegrade` (landing in M3-e.11.5). Two-step
    /// structure:
    ///
    /// 1. [`Graph::migrate_all_cuda_to_cpu`] — brings every VRAM-
    ///    resident output back to host memory. Returns a
    ///    [`DegradeReport`](crate::amg::reactive::DegradeReport)
    ///    whose `tensors_migrated` field contributes to the
    ///    composite report's count.
    /// 2. [`Graph::migrate_all_cpu_to_disk`] — spills the now-
    ///    coalesced Cpu pool (which includes the ones we just
    ///    brought back from Cuda AND any tensors that were
    ///    already Cpu before this call) to disk.
    ///
    /// # Accounting
    ///
    /// The returned report's `tensors_migrated` counts **disk
    /// writes**, not tier transitions. A tensor that starts as
    /// Cuda and ends as Disk contributes 1 (the disk write); a
    /// tensor that was already Cpu and ends as Disk also
    /// contributes 1. A tensor already on Disk contributes to
    /// `tensors_skipped`.
    ///
    /// # Error propagation
    ///
    /// Failures in step 1 propagate unchanged — the disk step
    /// never runs and no files are written. Failures in step 2
    /// follow the policy documented on `migrate_all_cpu_to_disk`
    /// (partial-progress via `Ok(_)`, total-failure via `Err`).
    pub fn migrate_all_to_disk(
        &mut self,
        cache_dir: &std::path::Path,
    ) -> Result<crate::amg::reactive::MigrationReport, StorageTransferError> {
        // Step 1: Cuda → Cpu. The DegradeReport contains the
        // per-call count but no failure surface (this method
        // propagates errors by `?` — if Cuda → Cpu fails, the
        // disk step never runs and we bail with zero side
        // effects on disk).
        let _cuda_report = self.migrate_all_cuda_to_cpu()?;

        // Step 2: Cpu → Disk. The returned report already
        // includes the ones we just brought back from Cuda in
        // its `tensors_migrated` (they're Cpu now, so
        // `migrate_all_cpu_to_disk` will migrate them). Return
        // as-is — the composite semantics match the caller's
        // intent ("everything that could reach disk did").
        self.migrate_all_cpu_to_disk(cache_dir)
    }
}

/// M3-e.10: RAII-style timer that records each node's execution
/// time to both the reaction-loop's `LatencyMonitor` (feeding
/// `GuardConditions::latency_spike` / `latency_ewma` / `latency_ratio`)
/// and the APX 7.6+ per-node EWMA history consumed by the HPGE
/// priority scheduler.
///
/// A single `Instant::now()` at construction and a single `.elapsed()`
/// at Drop cover all exit paths from `Graph::execute_single_inner` —
/// the 4 early-return branches (fused_ops, FusedLinearActivationChain,
/// fusion_plan pair, gpu_plan segment) and the default
/// `match node_type` tail. Before M3-e.10 the hpge timing was
/// scattered across three of the branches with varying placement
/// (one fired *before* the actual `exec_fused` call, measuring
/// dispatch time rather than execution time); this unification
/// fixes that incidentally.
///
/// Recording to `LatencyMonitor` requires a live `SignalBus`; the
/// recorder captures an `Option<Arc<SignalBus>>` at construction
/// time so the Drop path only visits the monitor when a
/// `reactive_context` is attached. `reactive_context: None` runs
/// the graph exactly as before M3-e.10, modulo the hpge timing
/// coverage improvement.
struct NodeTimingRecorder {
    start: std::time::Instant,
    node_id: usize,
    node_count: usize,
    bus: Option<std::sync::Arc<crate::amm::signal_bus::SignalBus>>,
    record_hpge: bool,
}

impl Drop for NodeTimingRecorder {
    fn drop(&mut self) {
        let dt = self.start.elapsed();
        if let Some(bus) = self.bus.as_ref() {
            bus.latency_monitor().record_latency(dt);
        }
        if self.record_hpge {
            crate::apx7::hpge_priority::record_node_time(
                self.node_id,
                dt.as_micros() as f64,
                self.node_count,
            );
        }
    }
}

fn log_softmax_last_dim(x: &Tensor) -> Tensor {
    assert!(x.shape.len() >= 1, "LogSoftmax requires tensors with rank >= 1");
    let ndim = x.shape.len();
    let cols = *x.shape.last().expect("log_softmax needs last dim");
    let rows = if ndim == 1 {
        1
    } else {
        x.shape[..ndim - 1].iter().product()
    };

    let mut out = Tensor::with_layout(
        x.shape.clone(),
        0.0,
        x.device,
        Layout::Contiguous,
        x.dtype,
    );

    let x_data = x.as_cpu_slice();
    let out_data = out.as_cpu_slice_mut();
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let slice = &x_data[start..end];
        let mut max_val = f32::NEG_INFINITY;
        for &v in slice {
            if v > max_val {
                max_val = v;
            }
        }
        let mut sum_exp = 0.0f32;
        let mut temp = Vec::with_capacity(cols);
        for &v in slice {
            let e = (v - max_val).exp();
            temp.push(e);
            sum_exp += e;
        }
        let log_denom = max_val + sum_exp.ln();
        for i in 0..cols {
            out_data[start + i] = slice[i] - log_denom;
        }
    }

    out
}

fn gather_last_dim(data: &Tensor, indices: &Tensor) -> Tensor {
    assert!(
        data.shape.len() >= 1,
        "Gather data must have at least one dimension"
    );
    let last_dim = *data.shape.last().expect("gather last dim");
    let data_rows = data.numel() / last_dim;
    assert_eq!(
        data_rows,
        indices.numel(),
        "Gather indices must match data rows"
    );

    let mut out = Tensor::with_layout(
        indices.shape.clone(),
        0.0,
        data.device,
        Layout::Contiguous,
        data.dtype,
    );

    let data_src = data.as_cpu_slice();
    let indices_src = indices.as_cpu_slice();
    let out_dst = out.as_cpu_slice_mut();
    for row in 0..data_rows {
        let idx = indices_src[row].round() as isize;
        assert!(idx >= 0 && (idx as usize) < last_dim, "Gather index out of bounds");
        let src = row * last_dim + idx as usize;
        out_dst[row] = data_src[src];
    }

    out
}

impl Clone for Graph {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            plan: self.plan.clone(),
            tape: BackwardTape::new(),
            grad_store: GradStore::new(),
            gpu_plan: self.gpu_plan.clone(),
            persistent_plan: self.persistent_plan.clone(),
            fusion_plan: self.fusion_plan.clone(),
            fusions_applied: self.fusions_applied,
            fused_ops: self.fused_ops.clone(),
            fused_outputs: self.fused_outputs.clone(),
            dual_graph: None,
            // Clones start without a reactive context; the caller
            // attaches one explicitly if needed.
            reactive_context: None,
            last_abort: None,
        }
    }
}

impl Graph {
    pub fn new(nodes: Vec<Node>) -> Self {
        Self::build(nodes)
    }

    /// Build a graph from pre-constructed nodes.
    /// Initializes execution plan, backward tape and an empty GradStore.
    pub fn build(nodes: Vec<Node>) -> Self {
        let (plan, fused_ops) = build_execution_plan(&nodes);
        let node_types: Vec<_> = nodes.iter().map(|n| n.node_type.clone()).collect();
        let gpu_plan = Some(GpuPlan::build(&node_types));

        let mut graph = Self {
            nodes,
            plan,
            tape: BackwardTape::new(),
            grad_store: GradStore::new(),
            gpu_plan,
            persistent_plan: None,
            fusion_plan: None,
            fusions_applied: 0,
            fused_ops,
            fused_outputs: std::collections::HashMap::new(),
            dual_graph: None,
            reactive_context: None,
            last_abort: None,
        };

        graph.persistent_plan = Some(PersistentPlan::analyze(&graph));
        graph.fusion_plan = Some(FusionPlan::analyze(&graph));

        // APX 7.8: generate temporal-locality hints per node. This does not
        // modify the graph nor its math; it only provides metadata so that
        // TLO can reorder independent nodes in HPGE.
        if !graph.nodes.is_empty() {
            use crate::apx7::tlo::{LocalityHint, set_locality_hints};

            let mut hints = vec![LocalityHint { branch_id: 0, depth: 0 }; graph.nodes.len()];
            for i in 0..graph.nodes.len() {
                let node = &graph.nodes[i];
                if node.inputs.is_empty() {
                    hints[i] = LocalityHint { branch_id: i, depth: 0 };
                } else {
                    let parent = node.inputs[0];
                    if parent < hints.len() {
                        hints[i].branch_id = hints[parent].branch_id;
                        hints[i].depth = hints[parent].depth.saturating_add(1);
                    } else {
                        hints[i] = LocalityHint { branch_id: i, depth: 0 };
                    }
                }
            }

            set_locality_hints(hints);
        }

        // APX 8.1: if the active mode requires it, also build the structural
        // DualGraph. This does not touch backward nor execute GPU.
        if crate::apx_mode_at_least("8.1") {
            graph.build_plan();
        }

        graph
    }

    /// Rebuild the execution plan from the current nodes. APX 8.1: when the
    /// mode allows it, also builds the structural CPU+GPU DualGraph.
    pub fn build_plan(&mut self) {
        let (plan, fused_ops) = build_execution_plan(&self.nodes);
        self.plan = plan;
        self.fused_ops = fused_ops;

        if crate::apx_mode_at_least("8.1") {
            self.dual_graph = Some(crate::apx8::dualgraph::DualGraphBuilder::build(self));
        }
    }

    /// Validate structural consistency of the graph.
    ///
    /// Checks:
    /// - Per-node inputs match what is expected for the NodeType.
    /// - All input indices reference existing nodes.
    /// - There are no cycles in the directed graph (from inputs to consumers).
    /// - All nodes are reachable from at least one Input node.
    pub fn validate(&self) -> Result<(), String> {
        // 1) Index range and expected number of inputs.
        for (id, node) in self.nodes.iter().enumerate() {
            for &inp in &node.inputs {
                if inp >= self.nodes.len() {
                    return Err(format!(
                        "Node {id} ({:?}) references a non-existent input: {inp}",
                        node.node_type
                    ));
                }
            }

            let in_len = node.inputs.len();
            let ok = match &node.node_type {
                NodeType::Input | NodeType::Parameter => in_len == 0,
                NodeType::Add
                | NodeType::Sub
                | NodeType::Mul
                | NodeType::MatMul
                | NodeType::BatchMatMul
                | NodeType::BroadcastAdd
                | NodeType::Gather
                | NodeType::CrossEntropyLoss => in_len == 2,
                NodeType::Reshape { .. }
                | NodeType::Transpose2D
                | NodeType::TransposeLastTwo
                | NodeType::RmsNorm
                | NodeType::SiLU
                | NodeType::Softmax
                | NodeType::LogSoftmax
                | NodeType::Output => in_len == 1,
                NodeType::IndexSelect => in_len == 2,
                NodeType::Linear => in_len == 2 || in_len == 3,
                NodeType::Activation(_) => in_len == 1,
                NodeType::FusedLinearActivation(_) => in_len == 2 || in_len == 3,
                // M3 debt cleanup: accept 3, 4 or 5 inputs. The APX 4.9
                // fusion detector produces 3-input chains for
                // Linear→acts→Linear patterns without biases; the
                // previous `in_len == 4 || 5` validator was
                // inconsistent with the executor's actual behavior.
                NodeType::FusedLinearActivationChain(_) => (3..=5).contains(&in_len),
                NodeType::Conv2D(_) => in_len == 2 || in_len == 3,
                NodeType::MaxPool2D(_) => in_len == 1,
                NodeType::NoOp => in_len == 1,
            };

            if !ok {
                return Err(format!(
                    "Node {id} ({:?}) has {} inputs, but a different number was expected",
                    node.node_type, in_len
                ));
            }
        }

        // 2) Build children list for cycle and reachability analysis.
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        for (id, node) in self.nodes.iter().enumerate() {
            for &inp in &node.inputs {
                children[inp].push(id);
            }
        }

        // 3) Cycle detection with DFS (colors: 0=white, 1=gray, 2=black).
        let mut color = vec![0u8; self.nodes.len()];
        fn dfs_cycle(
            u: usize,
            children: &Vec<Vec<usize>>,
            color: &mut [u8],
        ) -> Option<(usize, usize)> {
            color[u] = 1;
            for &v in &children[u] {
                match color[v] {
                    0 => {
                        if let Some(c) = dfs_cycle(v, children, color) {
                            return Some(c);
                        }
                    }
                    1 => {
                        return Some((u, v));
                    }
                    _ => {}
                }
            }
            color[u] = 2;
            None
        }

        for start in 0..self.nodes.len() {
            if color[start] == 0 {
                if let Some((u, v)) = dfs_cycle(start, &children, &mut color) {
                    return Err(format!(
                        "A cycle was detected in the graph between nodes {u} and {v}",
                    ));
                }
            }
        }

        // 4) Reachability from Input nodes.
        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = Vec::new();
        for (id, node) in self.nodes.iter().enumerate() {
            if matches!(node.node_type, NodeType::Input) {
                reachable[id] = true;
                stack.push(id);
            }
        }

        while let Some(u) = stack.pop() {
            for &v in &children[u] {
                if !reachable[v] {
                    reachable[v] = true;
                    stack.push(v);
                }
            }
        }

        for (id, node) in self.nodes.iter().enumerate() {
            if !reachable[id]
                && !matches!(node.node_type, NodeType::Parameter)
            {
                return Err(format!(
                    "Node {id} ({:?}) is not reachable from any Input node",
                    node.node_type
                ));
            }
        }

        Ok(())
    }

    /// Dynamically append a node to the graph and rebuild the execution plan.
    pub fn add_node_of_type(&mut self, node_type: NodeType, inputs: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node::new(id, node_type, inputs));
        let (plan, fused_ops) = build_execution_plan(&self.nodes);
        self.plan = plan;
        self.fused_ops = fused_ops;
        id
    }

    /// Convenience helper for parameter (constant/trainable) tensors.
    pub fn add_parameter(&mut self, tensor: Tensor) -> usize {
        let id = self.add_node_of_type(NodeType::Parameter, vec![]);
        self.nodes[id].output = Some(tensor);
        id
    }

    /// Standard execution with automatic execution plan (including
    /// fusion). Backward-compatible wrapper over
    /// [`execute_checked`](Self::execute_checked).
    ///
    /// Panics if a reactive context is set and a guard triggers an
    /// abort. By construction no caller that existed before APX v20
    /// M2 sets a reactive context, so existing code paths never
    /// panic. Callers that set a reactive context should use
    /// `execute_checked` to handle aborts gracefully.
    pub fn execute(&mut self, inputs: Vec<Tensor>) -> Vec<Tensor> {
        match self.execute_checked(inputs) {
            Ok(outputs) => outputs,
            Err(reason) => panic!(
                "Graph::execute triggered an abort: {:?}. \
                 Callers that set reactive_context should use \
                 `execute_checked` to handle aborts gracefully.",
                reason
            ),
        }
    }

    /// Checked execution: consults the reactive context (if set)
    /// before each node and returns `Err(ExecutionAbortReason)` when
    /// a guard triggers or a guard evaluation fails. When no
    /// reactive context is set, behavior is identical to the
    /// pre-M2 `execute` (never returns `Err`).
    pub fn execute_checked(
        &mut self,
        inputs: Vec<Tensor>,
    ) -> Result<Vec<Tensor>, ExecutionAbortReason> {
        self.last_abort = None;
        self.tape.clear();
        self.set_input_outputs(inputs);
        if crate::apx_mode_at_least("7.12") {
            crate::apx7::ule::ule_execute_graph(self)?;
        } else if crate::apx_mode_at_least("7.11") {
            crate::apx7::hls_deep::execute_graph_hls_deep(self)?;
        } else if crate::apx_mode_at_least("7.10") {
            crate::apx7::hls_deep::execute_graph_hls_deep(self)?;
        } else if crate::apx_mode_at_least("7.7") {
            crate::apx7::hpge_priority::execute_graph_parallel_priority(self)?;
        } else if crate::apx_mode_at_least("7.5") {
            crate::apx7::hpge::execute_graph_parallel(self)?;
        } else {
            self.run_plan(true)?;
        }

        // APX 6.11 / 6.12 / 6.13: update global execution policies based on
        // measurements accumulated by the FusionSelector 6.10.
        if crate::apx_mode_at_least("6.11") {
            if let Ok(sel) = crate::apx6_10::global_fusion_selector().lock() {
                if let Some(global_decision) = sel.best_decision() {
                    use crate::apx6_10::GlobalDecision;
                    use crate::apx6_11::runtime_policy::{set_runtime_policy, FusionRuntimePolicy};

                    // Deterministic 6.11 policy, also used as the base for 6.12
                    // scheduling hints.
                    match global_decision {
                        GlobalDecision::PreferFull => {
                            set_runtime_policy(FusionRuntimePolicy::PreferFull);
                        }
                        GlobalDecision::PreferQKV => {
                            set_runtime_policy(FusionRuntimePolicy::PreferQKV);
                        }
                        GlobalDecision::NoPreference => {
                            set_runtime_policy(FusionRuntimePolicy::Baseline);
                        }
                    }

                    // APX 6.12: also derive the pure scheduling bias.
                    if crate::apx_mode_at_least("6.12") {
                        use crate::apx6_12::adaptive_scheduler::{
                            AdaptiveScheduleBias,
                            set_schedule_bias,
                        };

                        match global_decision {
                            GlobalDecision::PreferQKV =>
                                set_schedule_bias(AdaptiveScheduleBias::QKVHeavy),
                            GlobalDecision::PreferFull =>
                                set_schedule_bias(AdaptiveScheduleBias::AttentionHeavy),
                            GlobalDecision::NoPreference =>
                                set_schedule_bias(AdaptiveScheduleBias::None),
                        }
                    }

                    // APX 6.13: probabilistic "tempered" policy that replaces
                    // the 6.11 global policy, but still only affects planning hints.
                    if crate::apx_mode_at_least("6.13") {
                        use crate::apx6_11::runtime_policy::{set_runtime_policy, FusionRuntimePolicy};

                        // APX 6.14: update the adaptive temperature before
                        // computing the tempered softmax. For now we use a
                        // synthetic integration-level step (0); the 6.14 tests
                        // exercise the real decay by explicitly calling
                        // update_temperature.
                        if crate::apx_mode_at_least("6.14") {
                            crate::apx6_14::temperature_manager::update_temperature(0);
                        }

                        let (full_s, qkv_s, base_s) = sel.normalized_scores();
                        let t = if crate::apx_mode_at_least("6.14") {
                            crate::apx6_14::temperature_schedule::get_current_temperature()
                        } else {
                            0.8 // fixed value in APX 6.13
                        };

                        let td = crate::softmax3(full_s, qkv_s, base_s, t);
                        let choice = crate::sample_decision(&td);

                        match choice {
                            "full" => set_runtime_policy(FusionRuntimePolicy::PreferFull),
                            "qkv"  => set_runtime_policy(FusionRuntimePolicy::PreferQKV),
                            _       => set_runtime_policy(FusionRuntimePolicy::Baseline),
                        }
                    }

                    // APX 6.15: stabilize the global decision using the
                    // current temperature as an exploration control, without
                    // touching forward/backward nor real tensors.
                    if crate::apx_mode_at_least("6.15") {
                        use crate::apx6_15::stabilizer::ApxTemperature;

                        let t_val = if crate::apx_mode_at_least("6.14") {
                            crate::apx6_14::temperature_schedule::get_current_temperature()
                        } else {
                            1.0
                        };

                        let temp = ApxTemperature::from_value(t_val);

                        if let Ok(mut stab) = crate::apx6_15::stabilizer::global_stabilizer().write() {
                            let decision = sel.best_decision();
                            let _final = stab.stabilize(decision, &temp);
                        }
                    }
                }
            }
        }

        Ok(self.collect_outputs())
    }

    /// APX 7.5: opt-in execution using the Hierarchical Parallel Graph
    /// Executor (HPGE). This API does not replace the standard path; it only
    /// forces mode 7.5 before delegating to `execute`.
    pub fn execute_hpge(&mut self, inputs: Vec<Tensor>) -> Vec<Tensor> {
        unsafe {
            std::env::set_var("ATENIA_APX_MODE", "7.5");
        }
        self.execute(inputs)
    }

    pub fn apply_optimizer(&mut self, optim: &mut AdamW, param_ids: &[usize]) {
        let mut params = self.get_params_mut(param_ids);
        optim.update(params.as_mut_slice());
    }

    pub fn last_output_id(&self) -> usize {
        self
            .nodes
            .iter()
            .rev()
            .find(|node| matches!(node.node_type, NodeType::Output))
            .map(|node| node.id)
            .expect("Graph must contain at least one Output node")
    }

    pub fn clear_all_grads(&mut self) {
        for node in &mut self.nodes {
            if let Some(out) = &mut node.output {
                out.clear_grad();
            }
        }
    }

    /// Execute graph in chunks for 1D element-wise style graphs.
    /// - `inputs`: same number of tensors as Input nodes
    /// - `max_chunk_elements`: maximum number of elements per chunk
    pub fn execute_chunked(
        &mut self,
        inputs: Vec<Tensor>,
        max_chunk_elements: usize,
    ) -> Vec<Tensor> {
        assert!(
            !inputs.is_empty(),
            "execute_chunked requires at least one input tensor"
        );

        let original_shape = inputs[0].shape.clone();

        let mut chunked_inputs: Vec<Vec<Tensor>> = Vec::new();
        for t in &inputs {
            chunked_inputs.push(chunk_tensor(t, max_chunk_elements));
        }

        let num_chunks = chunked_inputs[0].len();
        for ci in &chunked_inputs {
            assert_eq!(
                ci.len(),
                num_chunks,
                "all inputs must produce the same number of chunks"
            );
        }

        self.tape.clear();
        let mut output_chunks: Vec<Tensor> = Vec::new();

        for chunk_idx in 0..num_chunks {
            let mut per_chunk_inputs = Vec::new();
            for ci in &chunked_inputs {
                per_chunk_inputs.push(ci[chunk_idx].clone());
            }

            self.clear_intermediate_outputs();
            self.set_input_outputs(per_chunk_inputs);
            // `execute_chunked` predates APX v20 M2 and returns
            // `Vec<Tensor>` (not `Result`). If a reactive context is
            // attached and a guard fires inside the chunk, surface it
            // as a panic rather than silently swallowing the abort.
            self.run_plan(false)
                .expect("execute_chunked: guard triggered inside chunk; \
                         chunked execution does not yet support reactive_context");

            let mut chunk_outputs = self.collect_outputs();
            assert_eq!(
                chunk_outputs.len(),
                1,
                "execute_chunked currently supports a single Output node"
            );

            output_chunks.push(chunk_outputs.remove(0));
        }

        vec![merge_chunks(output_chunks, original_shape)]
    }

    fn clear_intermediate_outputs(&mut self) {
        for node in &mut self.nodes {
            if matches!(node.node_type, NodeType::Input | NodeType::Parameter) {
                continue;
            }
            node.output = None;
        }
    }

    fn set_input_outputs(&mut self, inputs: Vec<Tensor>) {
        let mut provided = inputs.into_iter();
        for node in &mut self.nodes {
            if matches!(node.node_type, NodeType::Input) {
                let tensor = provided
                    .next()
                    .expect("not enough input tensors provided for graph execution");
                node.set_output(tensor);
            }
        }

        if provided.next().is_some() {
            panic!("too many input tensors provided for graph execution");
        }
    }

    pub(crate) fn run_plan(
        &mut self,
        record_tape: bool,
    ) -> Result<(), ExecutionAbortReason> {
        let steps = self.plan.steps.clone();
        for step in steps {
            match step {
                ExecStep::Single(node_id) => {
                    self.check_guard_before_node(node_id)?;
                    self.execute_single(node_id, record_tape);
                }
                ExecStep::FusedAddMul { add_node, mul_node } => {
                    if record_tape {
                        self.check_guard_before_node(add_node)?;
                        self.execute_single(add_node, true);
                        self.check_guard_before_node(mul_node)?;
                        self.execute_single(mul_node, true);
                    } else {
                        self.check_guard_before_node(add_node)?;
                        self.execute_fused_add_mul(add_node, mul_node);
                    }
                }
            }
        }
        Ok(())
    }

    fn exec_fused(&mut self, id: usize, fused: FusedOp, record_tape: bool) {
        let apx_mode = crate::apx_mode();
        let is_69_or_higher = apx_mode.starts_with("6.9") || apx_mode > "6.9".to_string();
        let is_610_or_higher = apx_mode.starts_with("6.10") || apx_mode > "6.10".to_string();

        match fused {
            FusedOp::LinearSilu { x, w, b } => {
                unsafe {
                    crate::apx4_11::gpu_hooks::fused_linear_silu_gpu(
                        x,
                        w,
                        b,
                        id,
                        self,
                        record_tape,
                    );
                }
            }
            FusedOp::FusedQKV { x, wq, wk, wv, bq, bk, bv, q_id, k_id, v_id } => {
                use std::time::Instant;

                // Real forward (prototype) for Q/K/V sharing the same X.
                if crate::apx_debug_enabled() {
                    eprintln!("[APX 4.14] executing fused QKV at node {}", id);
                }

                let x_t = self.nodes[x]
                    .output
                    .as_ref()
                    .expect("FusedQKV: missing x output")
                    .clone();
                let wq_t = self.nodes[wq]
                    .output
                    .as_ref()
                    .expect("FusedQKV: missing wq output")
                    .clone();
                let wk_t = self.nodes[wk]
                    .output
                    .as_ref()
                    .expect("FusedQKV: missing wk output")
                    .clone();
                let wv_t = self.nodes[wv]
                    .output
                    .as_ref()
                    .expect("FusedQKV: missing wv output")
                    .clone();

                let bq_t = bq.and_then(|i| self.nodes[i].output.as_ref()).cloned();
                let bk_t = bk.and_then(|i| self.nodes[i].output.as_ref()).cloned();
                let bv_t = bv.and_then(|i| self.nodes[i].output.as_ref()).cloned();

                // In 6.9+ mode, we optionally measure naive vs fused timings.
                let mut use_fused = true;

                if is_69_or_higher {
                    // Naive (non-fused) measurement: execute Q, K, V separately.
                    let t0 = Instant::now();
                    let q_naive = nn_linear::linear(&x_t, &wq_t, bq_t.as_ref());
                    let k_naive = nn_linear::linear(&x_t, &wk_t, bk_t.as_ref());
                    let v_naive = nn_linear::linear(&x_t, &wv_t, bv_t.as_ref());
                    let unfused_time_us = t0.elapsed().as_micros() as u64;

                    // Fused measurement: reuse the current implementation.
                    let t1 = Instant::now();
                    let q_fused = q_naive.clone();
                    let k_fused = k_naive.clone();
                    let v_fused = v_naive.clone();
                    let fused_time_us = t1.elapsed().as_micros() as u64;

                    if let Ok(mut fp) = crate::apx6_9::fusion_profiler::fusion_profiler().lock() {
                        fp.record("FusedQKV", unfused_time_us, fused_time_us);
                        if let Some(decision) = fp.should_use_fused("FusedQKV") {
                            use_fused = decision;
                        }
                    }

                    // Depending on the decision, we use the already-computed
                    // naive tensors or let the fused path generate them. To avoid
                    // duplicate computation, if we decide not to use fused, we
                    // directly reuse q_naive/k_naive/v_naive.
                    if !use_fused {
                        self.nodes[q_id].set_output(q_naive.clone());
                        self.nodes[k_id].set_output(k_naive.clone());
                        self.nodes[v_id].set_output(v_naive.clone());
                        self.fused_outputs.insert(
                            id,
                            FusedOutput::QKV { q: q_naive, k: k_naive, v: v_naive },
                        );
                    } else {
                        self.nodes[q_id].set_output(q_fused.clone());
                        self.nodes[k_id].set_output(k_fused.clone());
                        self.nodes[v_id].set_output(v_fused.clone());
                        self.fused_outputs.insert(
                            id,
                            FusedOutput::QKV { q: q_fused, k: k_fused, v: v_fused },
                        );
                    }
                } else {
                    // Original 4.14 behavior when APX < 6.9.
                    let q = nn_linear::linear(&x_t, &wq_t, bq_t.as_ref());
                    let k = nn_linear::linear(&x_t, &wk_t, bk_t.as_ref());
                    let v = nn_linear::linear(&x_t, &wv_t, bv_t.as_ref());

                    self.nodes[q_id].set_output(q.clone());
                    self.nodes[k_id].set_output(k.clone());
                    self.nodes[v_id].set_output(v.clone());
                    self.fused_outputs.insert(id, FusedOutput::QKV { q, k, v });
                }

                // APX 4.16: record a single fused BackOp for QKV when
                // recording the backward tape.
                if crate::apx_mode() == "4.16" && record_tape {
                    let has_bq = bq.is_some();
                    let has_bk = bk.is_some();
                    let has_bv = bv.is_some();

                    let bq_id = bq;
                    let bk_id = bk;
                    let bv_id = bv;

                    self.tape.push(BackOp {
                        inputs: vec![x, wq, wk, wv],
                        output: id,
                        backward: Box::new(move |store, forward_inputs, _out_grad| {
                            let x_f = forward_inputs[0];
                            let wq_f = forward_inputs[1];
                            let wk_f = forward_inputs[2];
                            let wv_f = forward_inputs[3];

                            // gQ, gK, gV accumulated in GradStore for Q, K, V nodes.
                            let mut gq_data = store.get(q_id);
                            let mut gk_data = store.get(k_id);
                            let mut gv_data = store.get(v_id);

                            if gq_data.is_empty() && gk_data.is_empty() && gv_data.is_empty() {
                                return;
                            }

                            // Shapes derived from X and weights: Q, K, V have shape [m, n].
                            let m = x_f.shape[0];
                            let n = wq_f.shape[1];

                            // Ensure grad buffers are not empty to avoid out-of-bounds
                            // accesses in matmul. If any grad is empty, interpret it
                            // as all zeros.
                            let expected_len = m * n;
                            if gq_data.is_empty() {
                                gq_data = vec![0.0; expected_len];
                            }
                            if gk_data.is_empty() {
                                gk_data = vec![0.0; expected_len];
                            }
                            if gv_data.is_empty() {
                                gv_data = vec![0.0; expected_len];
                            }

                            let make_grad_tensor = |data: Vec<f32>, proto: &Tensor| -> Tensor {
                                let mut t = Tensor::new_cpu_with_layout(
                                    vec![m, n],
                                    data,
                                    proto.device,
                                    proto.dtype,
                                    proto.layout,
                                );
                                t.strides = proto.strides.clone();
                                t
                            };

                            let gq = make_grad_tensor(gq_data, x_f);
                            let gk = make_grad_tensor(gk_data, x_f);
                            let gv = make_grad_tensor(gv_data, x_f);

                            // dX = gQ·Wq^T + gK·Wk^T + gV·Wv^T
                            let wq_t = transpose_2d(wq_f);
                            let wk_t = transpose_2d(wk_f);
                            let wv_t = transpose_2d(wv_f);

                            let dx_q = nn_linear::matmul(&gq, &wq_t);
                            let mut dx_k = nn_linear::matmul(&gk, &wk_t);
                            let mut dx_v = nn_linear::matmul(&gv, &wv_t);
                            dx_k.ensure_cpu().expect(
                                "backward intermediate: GPU->CPU transfer failed \
                                 (this indicates a CUDA driver issue during backward; \
                                 see StorageTransferError variants)",
                            );
                            dx_v.ensure_cpu().expect(
                                "backward intermediate: GPU->CPU transfer failed \
                                 (this indicates a CUDA driver issue during backward; \
                                 see StorageTransferError variants)",
                            );

                            let mut dx_total = dx_q.copy_to_cpu_vec();
                            {
                                let dx_k_slice = dx_k.as_cpu_slice();
                                let dx_v_slice = dx_v.as_cpu_slice();
                                for i in 0..dx_total.len() {
                                    dx_total[i] += dx_k_slice[i] + dx_v_slice[i];
                                }
                            }
                            add_to_grad_slice(store, x, &dx_total);

                            // dWq, dWk, dWv: X^T · g
                            let x_t = transpose_2d(x_f);

                            let mut dwq = nn_linear::matmul(&x_t, &gq);
                            dwq.ensure_cpu().expect(
                                "backward intermediate: GPU->CPU transfer failed \
                                 (this indicates a CUDA driver issue during backward; \
                                 see StorageTransferError variants)",
                            );
                            add_to_grad_slice(store, wq, dwq.as_cpu_slice());

                            let mut dwk = nn_linear::matmul(&x_t, &gk);
                            dwk.ensure_cpu().expect(
                                "backward intermediate: GPU->CPU transfer failed \
                                 (this indicates a CUDA driver issue during backward; \
                                 see StorageTransferError variants)",
                            );
                            add_to_grad_slice(store, wk, dwk.as_cpu_slice());

                            let mut dwv = nn_linear::matmul(&x_t, &gv);
                            dwv.ensure_cpu().expect(
                                "backward intermediate: GPU->CPU transfer failed \
                                 (this indicates a CUDA driver issue during backward; \
                                 see StorageTransferError variants)",
                            );
                            add_to_grad_slice(store, wv, dwv.as_cpu_slice());

                            // Biases opcionales: sumar filas de gQ,gK,gV
                            if has_bq {
                                if let Some(bq_node) = bq_id {
                                    let bgrad_q = sum_rows(&gq);
                                    add_to_grad_slice(store, bq_node, &bgrad_q);
                                }
                            }
                            if has_bk {
                                if let Some(bk_node) = bk_id {
                                    let bgrad_k = sum_rows(&gk);
                                    add_to_grad_slice(store, bk_node, &bgrad_k);
                                }
                            }
                            if has_bv {
                                if let Some(bv_node) = bv_id {
                                    let bgrad_v = sum_rows(&gv);
                                    add_to_grad_slice(store, bv_node, &bgrad_v);
                                }
                            }
                        }),
                    });
                }
            }
            FusedOp::FusedSelfAttention { x, wq, wk, wv, bq, bk, bv, q, k, v, out_id: _ } => {
                if crate::apx_debug_enabled() {
                    eprintln!("[APX 4.17] executing fused SelfAttention at node {}", id);
                }

                // Retrieve Q, K, V tensors already materialized by the previous linear nodes.
                let q_t = self.nodes[q]
                    .output
                    .as_ref()
                    .expect("FusedSelfAttention: missing q output")
                    .clone();
                let k_t = self.nodes[k]
                    .output
                    .as_ref()
                    .expect("FusedSelfAttention: missing k output")
                    .clone();
                let v_t = self.nodes[v]
                    .output
                    .as_ref()
                    .expect("FusedSelfAttention: missing v output")
                    .clone();

                // MatMul(Q, K^T) followed by Softmax over the last dimension,
                // exactly the same as the naive graph (without scaling factor).
                let k_t_t = transpose_2d(&k_t);
                let scores = nn_linear::matmul(&q_t, &k_t_t);
                let att = nn_softmax::softmax_last_dim(&scores);

                // MatMul(A, V)
                let t0 = Instant::now();
                let out = nn_linear::matmul(&att, &v_t);
                let baseline_us = t0.elapsed().as_micros() as u64;

                // Materialize output in the MatMul A·V node (id) so that the rest
                // of the graph and the naive backward work unchanged.
                self.nodes[id].set_output(out.clone());

                self.fused_outputs.insert(
                    id,
                    FusedOutput::SelfAttention {
                        q: q_t.clone(),
                        k: k_t.clone(),
                        v: v_t.clone(),
                        att: att.clone(),
                        out: out.clone(),
                    },
                );

                // APX 6.10: auxiliary full-attention measurement without changing real forward/backward.
                if is_610_or_higher {
                    // Retrieve X, weights, and bias as seen by the graph.
                    let x_t = self.nodes[x]
                        .output
                        .as_ref()
                        .expect("FusedSelfAttention: missing x output")
                        .clone();
                    let wq_t = self.nodes[wq]
                        .output
                        .as_ref()
                        .expect("FusedSelfAttention: missing wq output")
                        .clone();
                    let wk_t = self.nodes[wk]
                        .output
                        .as_ref()
                        .expect("FusedSelfAttention: missing wk output")
                        .clone();
                    let wv_t = self.nodes[wv]
                        .output
                        .as_ref()
                        .expect("FusedSelfAttention: missing wv output")
                        .clone();

                    let bq_t = bq.and_then(|i| self.nodes[i].output.as_ref()).cloned();
                    let bk_t = bk.and_then(|i| self.nodes[i].output.as_ref()).cloned();
                    let bv_t = bv.and_then(|i| self.nodes[i].output.as_ref()).cloned();

                    // Look for a projection Linear that consumes out (baseline), if it exists.
                    let mut wproj_t: Option<crate::tensor::Tensor> = None;
                    let mut bias_proj_t: Option<crate::tensor::Tensor> = None;

                    for node in &self.nodes {
                        if let super::nodes::NodeType::Linear = node.node_type {
                            if !node.inputs.is_empty() && node.inputs[0] == id {
                                // inputs: [x, w, (b)]
                                let w_id = node.inputs[1];
                                wproj_t = self.nodes[w_id].output.clone();
                                if node.inputs.len() == 3 {
                                    let b_id = node.inputs[2];
                                    bias_proj_t = self.nodes[b_id].output.clone();
                                }
                                break;
                            }
                        }
                    }

                    if let Some(wproj_t) = wproj_t {
                        use crate::amg::fusions::execute_fused_attention_full;
                        use crate::apx6_10::{FusionProfile, global_fusion_selector};

                        let t_full = Instant::now();
                        let _y_full = execute_fused_attention_full(
                            &x_t,
                            &wq_t,
                            &wk_t,
                            &wv_t,
                            bq_t.as_ref(),
                            bk_t.as_ref(),
                            bv_t.as_ref(),
                            &wproj_t,
                            bias_proj_t.as_ref(),
                        );
                        let fused_full_us = t_full.elapsed().as_micros() as u64;

                        // For now we do not have a separate time for FusedQKV here; use 0.
                        let fused_qkv_us = 0u64;

                        if let Ok(mut sel) = global_fusion_selector().lock() {
                            sel.record_profile(FusionProfile {
                                op_name: "FusedAttention".to_string(),
                                baseline_us,
                                fused_qkv_us,
                                fused_full_us,
                            });
                        }
                    }
                }

                // APX 4.18: fused Self-Attention backward is disabled for now.
                // Forward remains fused, but backward uses the naive chain of
                // individual BackOps.
            }
        }
    }

    /// CPU forward for Linear nodes, reusing nn::linear without GPU.
    pub fn exec_cpu_linear_fallback(&mut self, id: usize) {
        let inputs = self.nodes[id].inputs.clone();
        assert!(
            inputs.len() == 2 || inputs.len() == 3,
            "Linear node expects 2 or 3 inputs",
        );

        let x_id = inputs[0];
        let w_id = inputs[1];

        // Ensure inputs have their outputs materialized before reading them.
        if self.nodes[x_id].output.is_none() {
            self.execute_single(x_id, false);
        }
        if self.nodes[w_id].output.is_none() {
            self.execute_single(w_id, false);
        }

        let x = self.nodes[x_id]
            .output
            .as_ref()
            .expect("Linear missing x")
            .clone();
        let w = self.nodes[w_id]
            .output
            .as_ref()
            .expect("Linear missing weight")
            .clone();

        let out = if inputs.len() == 3 {
            let b_id = inputs[2];
            if self.nodes[b_id].output.is_none() {
                self.execute_single(b_id, false);
            }
            let b = self.nodes[b_id]
                .output
                .as_ref()
                .expect("Linear missing bias")
                .clone();
            nn_linear::linear(&x, &w, Some(&b))
        } else {
            nn_linear::linear(&x, &w, None)
        };

        self.nodes[id].output = Some(out);
    }

    /// Executes a single node without consulting guards.
    ///
    /// Guard evaluation is performed by the scheduler (see
    /// `apx7::*`, `run_plan`) via `check_guard_before_node` before
    /// calling this method. Internal recursive calls from
    /// `execute_single_inner` for dependency materialization
    /// intentionally bypass guard evaluation: the scheduler already
    /// evaluated runtime state before the parent node, and
    /// re-evaluating during sub-microsecond materialization adds
    /// probe latency without providing new information.
    pub(crate) fn execute_single(&mut self, node_id: usize, record_tape: bool) {
        if crate::apx_mode_at_least("8.2") {
            let op = match self.nodes[node_id].node_type {
                NodeType::MatMul => "MatMul",
                NodeType::Linear => "Linear",
                _ => "Other",
            };
            return crate::apx8::hybrid_dispatcher::HybridDispatcher::dispatch(self, node_id, op, record_tape);
        }

        self.execute_single_inner(node_id, record_tape);
    }

    /// M3 debt cleanup: centralized dispatch for
    /// `NodeType::FusedLinearActivationChain`. Before this helper
    /// existed, the logic lived duplicated in two places in
    /// `execute_single_inner` — a pre-match early-return handler
    /// (accepting 3/4/5 inputs) and a match-arm (stricter, accepting
    /// only 4/5). The duplicates had drifted on input validation;
    /// the match-arm was dead code (the pre-match always returned
    /// first) but invited accidental reactivation on future refactors
    /// that might try to gate the pre-match by `!record_tape`.
    ///
    /// The only caller is the pre-match early-return site in
    /// `execute_single_inner`. The match-arm was removed and
    /// replaced with an `unreachable!()` guard for exhaustivity.
    ///
    /// Accepts 3 (no biases), 4 (one bias), or 5 (both biases)
    /// inputs — matching what the APX 4.9 fusion detector
    /// legitimately produces in `apx4_9::patterns::fuse_linear_activation_linear`.
    ///
    /// **Backward** (Debt #8 closure): when `record_tape` is true
    /// this helper **also registers a `BackOp`** that implements the
    /// analytical backward for the full chain (grad for `x`, `W1`,
    /// `b1?`, `W2`, `b2?`). The activation derivatives for `ReLU`,
    /// `SiLU` and `GELU` are inlined inside the closure; a
    /// `Vec<ActType>` of length N > 1 is handled by iterating the
    /// chain in reverse, multiplying the incoming grad by each
    /// activation derivative evaluated at the cached pre-activation
    /// input.
    ///
    /// The intermediates needed by backward (`y1` = input to act[0],
    /// `a_i` = input to act[i+1] for i in 0..N-1, and `a_last` =
    /// input to the second linear) are **captured by move into the
    /// closure**. They are NOT stored in `self.nodes[*].output` and
    /// therefore escape `migrate_all_to_disk` / `migrate_all_cuda_to_cpu`.
    /// To keep migrations correct, each captured intermediate is forced
    /// to CPU via `ensure_cpu()` before the clone — any subsequent
    /// memory-pressure reaction that migrates the rest of the graph
    /// leaves the captured tensors on CPU where they are needed at
    /// backward time.
    fn exec_fused_linear_activation_chain(
        &mut self,
        node_id: usize,
        acts: Vec<super::nodes::ActType>,
        record_tape: bool,
    ) {
        if crate::apx_debug_enabled() && !crate::apx_is_silent() {
            eprintln!(
                "[APX4.9 DEBUG] Executing FusedLinearActivationChain node_id={} | inputs={:?}",
                node_id,
                self.nodes[node_id].inputs
            );
        }

        let inputs = self.nodes[node_id].inputs.clone();
        assert!(
            inputs.len() == 3 || inputs.len() == 4 || inputs.len() == 5,
            "FusedLinearActivationChain expects 3, 4 or 5 inputs",
        );

        let x = self.nodes[inputs[0]]
            .output
            .as_ref()
            .expect("FusedLinearActivationChain missing x")
            .clone();
        let w1 = self.nodes[inputs[1]]
            .output
            .as_ref()
            .expect("FusedLinearActivationChain missing w1")
            .clone();

        // Capture node ids for the BackOp closure up front, alongside
        // the forward disambiguation. The `op_inputs` layout for the
        // closure follows the canonical fused_inputs order:
        //   [x, w1, (b1?), w2, (b2?)]
        // so `b1_id` and `b2_id` are captured as `Option<usize>` and
        // drive both the grad-accumulation paths in backward and the
        // `forward_inputs` indexing for `w2`.
        let x_id = inputs[0];
        let w1_id = inputs[1];

        // M3 debt cleanup — parser disambiguation fix:
        //
        // `apx4_9::patterns::fuse_linear_activation_linear` produces
        // `fused_inputs` in the order `[x, w1, (b1?), w2, (b2?)]`, so
        // the four valid layouts are:
        //   len == 3: [x, w1, w2]            (no biases)
        //   len == 4: [x, w1, b1, w2]        (bias on first linear only)
        //   len == 4: [x, w1, w2, b2]        (bias on second linear only)
        //   len == 5: [x, w1, b1, w2, b2]    (both biases)
        //
        // Pre-cleanup the parser only inspected `inputs.len()` and
        // blindly assumed that a 4-input chain was always
        // `[x, w1, w2, b2]` (bias on second), which caused a panic
        // `"weight must be 2D"` whenever a 4-input chain actually
        // had the bias on the *first* linear — a common
        // transformer feed-forward pattern.
        //
        // The fix distinguishes via the tensor's shape: a bias is
        // 1D (`shape.len() == 1`); a weight is 2D
        // (`shape.len() == 2`). We apply the shape check only when
        // `inputs.len() == 4` — the ambiguous case. Lengths 3 and
        // 5 remain unambiguous and keep their pre-fix handling.
        let mut idx = 2;
        let (b1_opt, b1_id) = if inputs.len() == 5 {
            // Unambiguous: [x, w1, b1, w2, b2].
            let id = inputs[idx];
            let b1 = self.nodes[id]
                .output
                .as_ref()
                .expect("FusedLinearActivationChain missing b1")
                .clone();
            idx += 1;
            (Some(b1), Some(id))
        } else if inputs.len() == 4
            && self.nodes[inputs[idx]]
                .output
                .as_ref()
                .map(|t| t.shape.len() == 1)
                .unwrap_or(false)
        {
            // 4-input layout with bias on first linear:
            // [x, w1, b1, w2]. The tensor at idx=2 is 1D → it's b1.
            let id = inputs[idx];
            let b1 = self.nodes[id]
                .output
                .as_ref()
                .expect("FusedLinearActivationChain missing b1")
                .clone();
            idx += 1;
            (Some(b1), Some(id))
        } else {
            // 3-input [x, w1, w2] or 4-input [x, w1, w2, b2] —
            // the tensor at idx=2 is the 2D weight w2.
            (None, None)
        };

        let w2_id = inputs[idx];
        let w2 = self.nodes[w2_id]
            .output
            .as_ref()
            .expect("FusedLinearActivationChain missing w2")
            .clone();
        idx += 1;

        let (b2_opt, b2_id) = if idx < inputs.len() {
            let id = inputs[idx];
            let b2 = self.nodes[id]
                .output
                .as_ref()
                .expect("FusedLinearActivationChain missing b2")
                .clone();
            (Some(b2), Some(id))
        } else {
            (None, None)
        };

        // Forward path. When `record_tape` is true, we also capture
        // all intermediates needed by the BackOp closure. Each
        // captured tensor is forced to CPU via `ensure_cpu()` before
        // the clone so that a subsequent `migrate_all_to_disk` /
        // `migrate_all_cuda_to_cpu` on the rest of the graph does
        // not leave these closure-owned tensors on GPU storage that
        // the backward pre-pass cannot reach.
        let mut h = match b1_opt.as_ref() {
            Some(b1) => nn_linear::linear(&x, &w1, Some(b1)),
            None => nn_linear::linear(&x, &w1, None),
        };

        // `pre_act_chain[i]` is the tensor fed into `acts[i]` during
        // forward. For the first activation it is `y1`. For later
        // activations (Vec<ActType> with N > 1) it is the output of
        // the previous activation. The backward loop reads it to
        // compute the pointwise activation derivative.
        let mut pre_act_chain: Vec<Tensor> = Vec::new();
        if record_tape {
            h.ensure_cpu().expect(
                "FusedLinearActivationChain: y1 ensure_cpu before BackOp capture \
                 (see StorageTransferError variants)",
            );
            pre_act_chain.push(h.clone());
        }

        for (i, act) in acts.iter().enumerate() {
            h = match act {
                super::nodes::ActType::ReLU => nn_act::relu(&h),
                super::nodes::ActType::SiLU => nn_act::silu(&h),
                super::nodes::ActType::GELU => nn_act::gelu(&h),
            };
            if record_tape && i < acts.len() - 1 {
                h.ensure_cpu().expect(
                    "FusedLinearActivationChain: intermediate ensure_cpu before BackOp capture",
                );
                pre_act_chain.push(h.clone());
            }
        }

        // `a_last` is the input to the second linear — also the output
        // of the last activation. Needed by backward for `grad_W2`.
        let a_last_for_capture = if record_tape {
            h.ensure_cpu().expect(
                "FusedLinearActivationChain: a_last ensure_cpu before BackOp capture",
            );
            Some(h.clone())
        } else {
            None
        };

        let out = match b2_opt.as_ref() {
            Some(b2) => nn_linear::linear(&h, &w2, Some(b2)),
            None => nn_linear::linear(&h, &w2, None),
        };

        if crate::apx_debug_enabled() && !crate::apx_is_silent() {
            eprintln!(
                "[APX4.9 DEBUG] FusedLinearActivationChain node_id={} produced output len={}",
                node_id,
                out.numel()
            );
        }

        self.nodes[node_id].set_output(out);

        // --- BackOp registration (Debt #8) ---
        //
        // Captured by move:
        //   - `acts_capture`: the Vec<ActType> (owned copy).
        //   - `pre_act_capture`: inputs to each activation, in order.
        //   - `a_last_capture`: input to the second linear.
        //   - `x_id, w1_id, w2_id, b1_id, b2_id`: node ids for grad
        //     accumulation. `b1_id.is_some()` == (b1 was present);
        //     same for b2.
        //
        // The closure does NOT rely on `forward_inputs[*]` for any
        // intermediate — only for `x`, `w1`, `w2`, whose tensors live
        // in `self.nodes[*].output` and are reachable via the
        // standard pre-pass migration in `backward_checked`.
        if record_tape {
            let acts_capture: Vec<super::nodes::ActType> = acts.clone();
            let pre_act_capture: Vec<Tensor> = pre_act_chain;
            let a_last_capture: Tensor = a_last_for_capture
                .expect("record_tape implies a_last was captured");
            let op_inputs = inputs.clone();

            self.tape.push(BackOp {
                inputs: op_inputs,
                output: node_id,
                backward: Box::new(move |store, forward_inputs, out_grad| {
                    // Pointwise activation derivatives. Inlined to
                    // keep the closure self-contained.
                    fn act_derivative(
                        act: &super::nodes::ActType,
                        x_slice: &[f32],
                    ) -> Vec<f32> {
                        match act {
                            super::nodes::ActType::ReLU => x_slice
                                .iter()
                                .map(|&v| if v > 0.0 { 1.0 } else { 0.0 })
                                .collect(),
                            super::nodes::ActType::SiLU => x_slice
                                .iter()
                                .map(|&v| {
                                    let s = 1.0f32 / (1.0f32 + (-v).exp());
                                    s + v * s * (1.0 - s)
                                })
                                .collect(),
                            super::nodes::ActType::GELU => x_slice
                                .iter()
                                .map(|&v| {
                                    let c = 0.79788456_f32; // sqrt(2/pi)
                                    let x3 = v * v * v;
                                    let inner = c * (v + 0.044715_f32 * x3);
                                    let t = inner.tanh();
                                    let sech2 = 1.0 - t * t;
                                    let d_inner =
                                        c * (1.0 + 3.0 * 0.044715_f32 * v * v);
                                    0.5 * (1.0 + t) + 0.5 * v * sech2 * d_inner
                                })
                                .collect(),
                        }
                    }

                    let x = forward_inputs[0];
                    let w1 = forward_inputs[1];
                    let w2_fi_idx = 2 + if b1_id.is_some() { 1 } else { 0 };
                    let w2 = forward_inputs[w2_fi_idx];

                    // grad_b2 = sum_rows(grad_out)
                    if let Some(b2id) = b2_id {
                        let grad_b2 = sum_rows(out_grad);
                        add_to_grad_slice(store, b2id, &grad_b2);
                    }

                    // grad_W2 = a_last^T @ grad_out
                    let a_last_t = transpose_2d(&a_last_capture);
                    let mut grad_w2 = nn_linear::matmul(&a_last_t, out_grad);
                    grad_w2.ensure_cpu().expect(
                        "FusedLinearActivationChain backward: grad_W2 ensure_cpu failed",
                    );
                    add_to_grad_slice(store, w2_id, grad_w2.as_cpu_slice());

                    // grad_a_last = grad_out @ W2^T
                    let w2_t = transpose_2d(w2);
                    let mut grad_a = nn_linear::matmul(out_grad, &w2_t);
                    grad_a.ensure_cpu().expect(
                        "FusedLinearActivationChain backward: grad_a_last ensure_cpu failed",
                    );

                    // Loop in reverse through activations. After the
                    // loop, `grad_a` is `grad_y1` (grad wrt output of
                    // the first linear, before any activation).
                    for i in (0..acts_capture.len()).rev() {
                        let input_to_act = &pre_act_capture[i];
                        let d = act_derivative(
                            &acts_capture[i],
                            input_to_act.as_cpu_slice(),
                        );
                        let grad_a_slice = grad_a.as_cpu_slice();
                        debug_assert_eq!(
                            d.len(),
                            grad_a_slice.len(),
                            "activation derivative length mismatch"
                        );
                        let mut new_data = vec![0.0f32; grad_a_slice.len()];
                        for j in 0..grad_a_slice.len() {
                            new_data[j] = grad_a_slice[j] * d[j];
                        }
                        grad_a = Tensor::new_cpu_with_layout(
                            grad_a.shape.clone(),
                            new_data,
                            grad_a.device,
                            grad_a.dtype,
                            Layout::Contiguous,
                        );
                    }
                    let grad_y1 = grad_a;

                    // grad_b1 = sum_rows(grad_y1)
                    if let Some(b1id) = b1_id {
                        let grad_b1 = sum_rows(&grad_y1);
                        add_to_grad_slice(store, b1id, &grad_b1);
                    }

                    // grad_W1 = x^T @ grad_y1
                    let x_t = transpose_2d(x);
                    let mut grad_w1 = nn_linear::matmul(&x_t, &grad_y1);
                    grad_w1.ensure_cpu().expect(
                        "FusedLinearActivationChain backward: grad_W1 ensure_cpu failed",
                    );
                    add_to_grad_slice(store, w1_id, grad_w1.as_cpu_slice());

                    // grad_x = grad_y1 @ W1^T
                    let w1_t = transpose_2d(w1);
                    let mut grad_x = nn_linear::matmul(&grad_y1, &w1_t);
                    grad_x.ensure_cpu().expect(
                        "FusedLinearActivationChain backward: grad_x ensure_cpu failed",
                    );
                    add_to_grad_slice(store, x_id, grad_x.as_cpu_slice());
                }),
            });
        }
    }

    pub(crate) fn execute_single_inner(&mut self, node_id: usize, record_tape: bool) {
        // M3-e.10: single timer covers every exit path via Drop. See
        // the `NodeTimingRecorder` docstring for the unification
        // rationale. Pre-M3-e.10 the hpge timing lived as three
        // scattered `if use_timing { ... record_node_time(...) }`
        // blocks at specific returns; those are gone, the Drop does
        // it all.
        let _timer = NodeTimingRecorder {
            start: Instant::now(),
            node_id,
            node_count: self.nodes.len(),
            bus: self
                .reactive_context
                .as_ref()
                .map(|c| std::sync::Arc::clone(&c.signal_bus)),
            record_hpge: crate::apx_mode_at_least("7.6"),
        };

        // Backward tape invariants for the fused dispatch paths below:
        //
        // The `fused_ops` dispatch and the `fusion_plan.fused_pairs`
        // dispatch skip when `record_tape` is true (except
        // `FusedOp::FusedQKV`, which has a real BackOp registered
        // inside `exec_fused`). This routes training to the per-node
        // `match node_type` below, which registers one BackOp per node
        // and gives backward a complete chain to walk. Same pattern as
        // the `gpu_plan` intercept further down.
        //
        // `FusedLinearActivationChain` now registers a real
        // analytical BackOp inside `exec_fused_linear_activation_chain`
        // (Debt #8 closure). The helper receives `record_tape` and
        // dispatches through the pre-match early return below —
        // training mode gets a complete backward chain from the
        // fused node; inference mode skips the capture / tape
        // registration entirely.

        // APX 4.13: if there is a fused op associated with this node, delegate
        // to the fused executor and return. We clone the FusedOp to avoid
        // holding an immutable borrow of self while using it.
        if let Some(fused) = self.fused_ops.get(&node_id).cloned() {
            let has_fused_backward = matches!(fused, FusedOp::FusedQKV { .. });
            if has_fused_backward || !record_tape {
                // Timing handled by `NodeTimingRecorder` at function
                // exit (M3-e.10). Pre-M3-e.10 a pre-call record here
                // measured dispatch time rather than execution time.
                return self.exec_fused(node_id, fused, record_tape);
            }
        }

        // APX 4.9: directly execute fused Linear→[Act...]→Linear chains,
        // without going through APX 4.7 hooks nor GPU for this node.
        //
        // The helper `exec_fused_linear_activation_chain` centralizes
        // dispatch for this NodeType. It accepts 3, 4, or 5 inputs
        // depending on whether biases are present — matching what the
        // APX 4.9 fusion detector legitimately produces. This early-
        // return intercepts the NodeType BEFORE the `fused_ops`,
        // `fusion_plan`, `gpu_plan`, and generic `match node_type`
        // dispatches below. The match-arm copy that previously
        // duplicated this logic was removed as part of M3 debt
        // cleanup — it was dead code (always intercepted by this
        // early-return) and had stricter input validation (4/5 only)
        // that would break `apx_2_5_fused_kernels_test` if ever
        // reached. The helper registers a real analytical BackOp
        // when `record_tape` is true (Debt #8 closure), so the
        // backward chain now flows correctly through the fused node.
        if let super::nodes::NodeType::FusedLinearActivationChain(acts) =
            self.nodes[node_id].node_type.clone()
        {
            self.exec_fused_linear_activation_chain(node_id, acts, record_tape);
            // Timing handled by `NodeTimingRecorder` at function exit (M3-e.10).
            return;
        }

        // APX 4.7: if this node is the second of a fused Linear→Linear pair, execute the fusion.
        // Skip when recording tape: `exec_fused_linear_linear` does not
        // register a BackOp (its `_record_tape` parameter is unused).
        // See the "Backward tape invariants" note above.
        if !record_tape {
            if let Some(fplan) = &self.fusion_plan {
                if let Some((a, b)) = fplan
                    .fused_pairs
                    .iter()
                    .find(|(_, b)| *b == node_id)
                    .cloned()
                {
                    crate::apx4_7::exec_fused_linear_linear(self, a, b, record_tape);
                    return;
                }
            }
        }

        // APX 4.3: if this node is the start of a GPU segment, execute the
        // whole segment — but only when no backward tape is being recorded.
        //
        // exec_gpu_segment is a forward-only optimization: it runs the
        // kernels on GPU and does not register BackOp entries for the nodes
        // it covers. When record_tape is true (training mode), skipping the
        // intercept lets each node fall through to the regular NodeType
        // match below, which computes forward on CPU and registers the
        // appropriate tape entry. When record_tape is false (inference or
        // forward-only execution), the segment runs for the GPU speedup.
        if !record_tape {
            if let Some(plan) = &self.gpu_plan {
                if let Some(seg) = plan.segments.iter().find(|s| s.start == node_id).cloned() {
                    self.exec_gpu_segment(&seg);
                    // Timing handled by `NodeTimingRecorder` at function exit (M3-e.10).
                    return;
                }
            }
        }

        // APX 4.11: detect whether this node belongs to any already-planned GPU segment.
        let in_gpu_segment = self
            .gpu_plan
            .as_ref()
            .map(|plan| {
                plan
                    .segments
                    .iter()
                    .any(|s| s.start <= node_id && node_id <= s.end)
            })
            .unwrap_or(false);

        let node_type = self.nodes[node_id].node_type.clone();

        // APX 5.x: auxiliary variables for advanced planning.
        let apx_mode = crate::apx_mode();
        let is_5x = apx_mode.starts_with("5.");
        let is_54 = apx_mode.starts_with("5.4");
        let mut node_exec_info_5x: Option<NodeExecInfo> = None;

        // APX 5.1 / 5.2: kernel planner (for now, logs + CPU/GPU decision
        // used for MatMul in APX 5.2).
        let shape_hint: Vec<usize> = self
            .nodes[node_id]
            .inputs
            .get(0)
            .and_then(|&inp_id| self.nodes[inp_id].output.as_ref())
            .map(|t| t.shape.clone())
            .unwrap_or_default();

        let planner = KernelPlanner::new();

        // APX 5.4: obtain adaptive device preference for this node (for now we
        // use generic information, not op-specific).
        let mut adaptive_pref = None;
        if is_54 {
            if let Some(ref info) = node_exec_info_5x {
                let sel_mutex = crate::global_adaptive_selector();
                if let Ok(sel) = sel_mutex.lock() {
                    adaptive_pref = sel.device_preference_for(info);
                }
            }
        }

        let plan = planner.select_kernel(&format!("{:?}", node_type), &shape_hint, adaptive_pref);
        if crate::apx_debug_enabled() {
            eprintln!(
                "[APX 5] Kernel for node_id={} ({:?}) = {:?} ({})",
                node_id,
                node_type,
                plan.target,
                plan.reason,
            );
        }

        // APX 3.9: route node execution target based on type and approximate shape
        // (we use the first input's shape as a proxy when available).
        let exec_target = route(&node_type, &shape_hint);

        if !crate::apx_is_silent() && std::env::var("APX_TRACE").is_ok() {
            eprintln!(
                "[APX 3.9 ROUTER] node_id={} | {:?} | target={:?}",
                node_id, node_type, exec_target
            );
        }

        // APX 5.3: additional advanced planning layer. Here we only build an
        // execution plan and optionally log it, without modifying the real
        // execution path nor the node's math yet.
        if is_5x {
            // APX 5.3: try to use real Tensor information when available to
            // enrich NodeExecInfo.
            let tensor_opt = self.nodes[node_id]
                .output
                .as_ref()
                .or_else(|| {
                    self.nodes[node_id]
                        .inputs
                        .get(0)
                        .and_then(|&inp_id| self.nodes[inp_id].output.as_ref())
                });

            let (dtype_str, contiguous, estimated_bytes) = if let Some(t) = tensor_opt {
                (
                    format!("{:?}", t.dtype),
                    t.layout == Layout::Contiguous,
                    t.estimated_bytes(),
                )
            } else {
                // Approximate fallback if no tensor is available yet.
                let num_elems: usize = shape_hint.iter().product();
                (
                    "f32".to_string(),
                    true,
                    num_elems.saturating_mul(4),
                )
            };

            let num_elems: usize = shape_hint.iter().product();
            let estimated_flops = num_elems; // symbolic placeholder

            let device_52_str = match plan.target {
                KernelTarget::Cpu => "CPU".to_string(),
                KernelTarget::Gpu | KernelTarget::HybridCpuGpu => "GPU".to_string(),
                KernelTarget::CpuFastAvx2 => "CPU".to_string(),
            };

            let mut node_info = NodeExecInfo {
                node_id,
                op_name: format!("{:?}", node_type),
                shape: shape_hint.clone(),
                dtype: dtype_str,
                contiguous,
                device_52: device_52_str,
                estimated_bytes,
                estimated_flops,
                vram_free: 0,
                kernel_time_avg: 0.0,
                preferred_kernel_size: None,
                tile_override: None,
                scheduling_bias: None,
                qkv_bias: None,
                attention_bias: None,
                exec_priority: None,
                prefetch_hint: None,
            };

            // APX 6.11: apply the global fusion policy only as a planning bias
            // (does not alter real forward nor backward).
            if crate::apx_mode_at_least("6.11") {
                use crate::apx6_11::runtime_policy::{get_runtime_policy, FusionRuntimePolicy};

                match get_runtime_policy() {
                    FusionRuntimePolicy::Baseline => {}
                    FusionRuntimePolicy::PreferQKV => node_info.apply_qkv_bias(),
                    FusionRuntimePolicy::PreferFull => node_info.apply_attention_bias(),
                }
            }

            // APX 6.12: additional scheduling hints (ordering/priority,
            // prefetch) based on AdaptiveScheduleBias. Does not touch the
            // math nor the backward tape.
            if crate::apx_mode_at_least("6.12") {
                use crate::apx6_12::adaptive_scheduler::{
                    AdaptiveScheduleBias,
                    get_schedule_bias,
                };

                match get_schedule_bias() {
                    AdaptiveScheduleBias::None => {}
                    AdaptiveScheduleBias::QKVHeavy => node_info.bias_qkv_schedule(),
                    AdaptiveScheduleBias::AttentionHeavy => node_info.bias_attention_schedule(),
                }
            }

            node_exec_info_5x = Some(node_info.clone());

            let planner_5_3 = Planner5_3::new();
            let mut plan_5_3 = planner_5_3.select_plan(&node_info);

            // APX 5.4: optional adaptive selector. It only modifies the plan
            // at the preference level, not the real execution, and only in
            // 5.4 or higher.
            if is_54 {
                let sel_mutex = crate::global_adaptive_selector();
                if let Ok(sel) = sel_mutex.lock() {
                    sel.merge_into_plan(&node_info, &mut plan_5_3);
                }
            }

            if crate::apx_debug_enabled() {
                eprintln!(
                    "[APX 5.3] plan for node_id={} ({:?}): kernel={} layout={:?} chunking={:?} fp8={}",
                    node_id,
                    node_type,
                    plan_5_3.kernel_name,
                    plan_5_3.layout,
                    plan_5_3.chunking.as_ref().map(|c| c.chunks),
                    plan_5_3.use_fp8,
                );
            }
        }

        match exec_target {
            ExecTarget::GPU => {
                panic!("GPU execution not fully integrated yet (APX 4.1 / CUDA vec_add only)");
            }
            ExecTarget::CPU | ExecTarget::CpuOptimized | ExecTarget::Auto => {
                // For now all non-GPU targets fall through to existing CPU implementation.
            }
        }

        match node_type.clone() {
            NodeType::Input | NodeType::Parameter => {}
            NodeType::Add => {
                let inputs = self.nodes[node_id].inputs.clone();
                if inputs.len() < 2 {
                    // Inconsistent graph (e.g., artificial trace tests): do not execute Add.
                    return;
                }

                let a_opt = self.nodes[inputs[0]].output.as_ref();
                let b_opt = self.nodes[inputs[1]].output.as_ref();

                if let (Some(a), Some(b)) = (a_opt.cloned(), b_opt.cloned()) {
                    self.nodes[node_id].set_output(a.add(&b));

                    if record_tape {
                        let op_inputs = self.nodes[node_id].inputs.clone();
                        let ids = op_inputs.clone();
                        self.tape.push(BackOp {
                            inputs: op_inputs,
                            output: node_id,
                            backward: Box::new(move |store, _forward_inputs, out_grad| {
                                add_to_grad_slice(store, ids[0], out_grad.as_cpu_slice());
                                add_to_grad_slice(store, ids[1], out_grad.as_cpu_slice());
                            }),
                        });
                    }
                }
            }
            NodeType::NoOp => {
                // Logically removed node: ensures its single input has been
                // executed and forwards its result.
                if let Some(&inp) = self.nodes[node_id].inputs.get(0) {
                    if crate::apx_debug_enabled() && !crate::apx_is_silent() {
                        eprintln!(
                            "[APX4.9 DEBUG] Executing NoOp node_id={} -> input_id={} ({:?}) | input_has_output={}",
                            node_id,
                            inp,
                            self.nodes[inp].node_type,
                            self.nodes[inp].output.is_some(),
                        );
                    }
                    // If the input node has not produced output yet, execute it now.
                    if self.nodes[inp].output.is_none() {
                        self.execute_single(inp, record_tape);
                    }

                    if let Some(out) = self.nodes[inp].output.clone() {
                        self.nodes[node_id].set_output(out);
                    }
                }
            }
            NodeType::Sub => {
                let a_id = self.nodes[node_id].inputs[0];
                let b_id = self.nodes[node_id].inputs[1];

                let a = self.nodes[a_id]
                    .output
                    .as_ref()
                    .expect("Sub missing input A")
                    .clone();
                let b = self.nodes[b_id]
                    .output
                    .as_ref()
                    .expect("Sub missing input B")
                    .clone();

                self.nodes[node_id].set_output(a.sub(&b));

                if record_tape {
                    let op_inputs = self.nodes[node_id].inputs.clone();
                    let ids = op_inputs.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs,
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            add_to_grad_slice(store, ids[0], out_grad.as_cpu_slice());

                            let neg: Vec<f32> = out_grad.as_cpu_slice().iter().map(|v| -*v).collect();
                            add_to_grad_slice(store, ids[1], &neg);
                        }),
                    });
                }
            }
            NodeType::Mul => {
                let inputs = self.nodes[node_id].inputs.clone();
                if inputs.len() < 2 {
                    // Inconsistent graph (e.g., artificial trace tests): do not execute Mul.
                    return;
                }

                let a_opt = self.nodes[inputs[0]].output.as_ref();
                let b_opt = self.nodes[inputs[1]].output.as_ref();

                if let (Some(a), Some(b)) = (a_opt, b_opt) {
                    let a = a.clone();
                    let b = b.clone();

                    self.nodes[node_id].set_output(a.mul(&b));

                    if record_tape {
                        let op_inputs = self.nodes[node_id].inputs.clone();
                        let ids = op_inputs.clone();
                        self.tape.push(BackOp {
                            inputs: op_inputs,
                            output: node_id,
                            backward: Box::new(move |store, forward_inputs, out_grad| {
                                let lhs = forward_inputs[0];
                                let rhs = forward_inputs[1];
                                let out_grad_slice = out_grad.as_cpu_slice();
                                let lhs_slice = lhs.as_cpu_slice();
                                let rhs_slice = rhs.as_cpu_slice();
                                let n = out_grad.numel();
                                let mut grad_a = vec![0.0; n];
                                let mut grad_b = vec![0.0; n];
                                for i in 0..n {
                                    grad_a[i] = out_grad_slice[i] * rhs_slice[i];
                                    grad_b[i] = out_grad_slice[i] * lhs_slice[i];
                                }
                                add_to_grad_slice(store, ids[0], &grad_a);
                                add_to_grad_slice(store, ids[1], &grad_b);
                            }),
                        });
                    }
                }
            }
            NodeType::MatMul => {
                let a_id = self.nodes[node_id].inputs[0];
                let b_id = self.nodes[node_id].inputs[1];

                let a = self.nodes[a_id]
                    .output
                    .as_ref()
                    .expect("MatMul missing input A")
                    .clone();
                let b = self.nodes[b_id]
                    .output
                    .as_ref()
                    .expect("MatMul missing input B")
                    .clone();

                assert_eq!(a.shape.len(), 2, "MatMul expects 2D lhs tensor");
                assert_eq!(b.shape.len(), 2, "MatMul expects 2D rhs tensor");
                let m = a.shape[0];
                let k = a.shape[1];
                assert_eq!(b.shape[0], k, "MatMul inner dimension mismatch");
                let n = b.shape[1];

                let mut out = Tensor::with_layout(
                    vec![m, n],
                    0.0,
                    a.device,
                    Layout::Contiguous,
                    a.dtype,
                );

                // APX 5.4: timing measurement for MatMul (CPU/GPU) for
                // collecting adaptive statistics.
                let start_time = Instant::now();
                let mut device_chosen = DeviceTarget::CPU;

                let apx_mode = crate::apx_mode();

                // APX 6.6: Auto-Tiling Optimizer (ATO) only for CPU forward
                // with contiguous FP32 tensors. Does not modify backward nor
                // fusions; it only decides which existing kernel to use.
                let is_66_or_higher = crate::apx_mode_at_least("6.6");
                if is_66_or_higher
                    && a.device == crate::tensor::Device::CPU
                    && a.layout == Layout::Contiguous
                    && a.dtype == crate::tensor::DType::F32
                    && b.device == crate::tensor::Device::CPU
                    && b.layout == Layout::Contiguous
                    && b.dtype == crate::tensor::DType::F32
                {
                    use crate::apx6_6_auto_tiling::{AutoTilingSelector, KernelKind};

                    let kernel_choice = AutoTilingSelector::choose_kernel(m, n, k);
                    let tile_config = AutoTilingSelector::choose_tile_sizes(m, n, k);

                    if crate::apx_debug_enabled() && !crate::apx_is_silent() {
                        eprintln!(
                            "[APX 6.6] kernel={:?} tiles={:?} for MxKxN = {}x{}x{}",
                            kernel_choice, tile_config, m, k, n
                        );
                    }

                    match kernel_choice {
                        KernelKind::Baseline38 => {
                            use crate::apx3_8::{
                                device_context::DeviceContext,
                                kernel_dispatch::dispatch_matmul as dispatch_matmul_apx3_8,
                            };
                            let ctx = DeviceContext::new(crate::tensor::Device::CPU);
                            dispatch_matmul_apx3_8(a.as_cpu_slice(), b.as_cpu_slice(), out.as_cpu_slice_mut(), m, k, n, &ctx);
                        }
                        KernelKind::Tiled63B | KernelKind::Micro64 => {
                            // Reuse the CPU dispatcher that already integrates 6.3B,
                            // 6.4, and a safe fallback to 3.8/6.1.
                            crate::matmul_dispatcher::matmul_dispatch(
                                a.as_cpu_slice(),
                                b.as_cpu_slice(),
                                out.as_cpu_slice_mut(),
                                m,
                                k,
                                n,
                            );
                        }
                    }

                    // In 6.6 mode we do not register explicit 5.4 samples here
                    // to avoid coupling ATO with the statistics system.
                    self.nodes[node_id].set_output(out);
                    if record_tape {
                        let op_inputs = self.nodes[node_id].inputs.clone();
                        let ids = op_inputs.clone();
                        self.tape.push(BackOp {
                            inputs: op_inputs,
                            output: node_id,
                            backward: Box::new(move |store, forward_inputs, out_grad| {
                                let a = forward_inputs[0];
                                let b = forward_inputs[1];
                                let b_t = transpose_2d(b);
                                let mut grad_a = nn_linear::matmul(out_grad, &b_t);
                                grad_a.ensure_cpu().expect(
                                    "backward intermediate: GPU->CPU transfer failed \
                                     (this indicates a CUDA driver issue during backward; \
                                     see StorageTransferError variants)",
                                );
                                add_to_grad_slice(store, ids[0], grad_a.as_cpu_slice());

                                let a_t = transpose_2d(a);
                                let mut grad_b = nn_linear::matmul(&a_t, out_grad);
                                grad_b.ensure_cpu().expect(
                                    "backward intermediate: GPU->CPU transfer failed \
                                     (this indicates a CUDA driver issue during backward; \
                                     see StorageTransferError variants)",
                                );
                                add_to_grad_slice(store, ids[1], grad_b.as_cpu_slice());
                            }),
                        });
                    }

                    return;
                }

                // APX 4.11: per-operator direct GPU execution attempt only for
                // nodes outside planned GPU segments. This path is kept as-is
                // to guarantee compatibility.
                if !in_gpu_segment
                    && gpu_hooks::gpu_can_run_matmul(m, k, n)
                    && gpu_hooks::try_gpu_matmul(&a, &b, &mut out)
                {
                    device_chosen = DeviceTarget::GPU;

                    // Register a sample only in 5.4 mode, without changing
                    // semantics nor execution path.
                    if is_54 {
                        let duration_us = start_time.elapsed().as_micros() as u64;
                        if let Some(ref info) = node_exec_info_5x {
                            let sample = Sample {
                                op_name: info.op_name.clone(),
                                shape: info.shape.clone(),
                                dtype: match info.dtype.as_str() {
                                    "F16" => crate::tensor::DType::F16,
                                    "BF16" => crate::tensor::DType::BF16,
                                    "FP8" => crate::tensor::DType::FP8,
                                    _ => crate::tensor::DType::F32,
                                },
                                device_chosen: device_chosen.clone(),
                                duration_us,
                                vram_before: 0,
                                vram_after: 0,
                                fallback: false,
                            };

                            let sel_mutex = crate::global_adaptive_selector();
                            if let Ok(mut sel) = sel_mutex.lock() {
                                sel.register_sample(sample);
                            }
                        }
                    }

                    self.nodes[node_id].set_output(out);
                    return;
                }

                // APX 5.2 / 6.2: use KernelPlanner to decide the matmul target.
                // In modes >= 6.2 we can force the CPU AVX2 path (CpuFastAvx2);
                // otherwise we use the usual APX 4 dispatcher with CPU/GPU/Auto.
                let is_62_or_higher = apx_mode.starts_with("6.2") || apx_mode > "6.2".to_string();

                let target = if is_62_or_higher && matches!(plan.target, KernelTarget::CpuFastAvx2) {
                    // Optional AVX2 path. The APX 6.2 dispatcher is responsible
                    // for safely falling back to the APX 3.8 dispatcher when
                    // AVX2 is not available.
                    crate::apx6_2::dispatch::dispatch_matmul_avx2(
                        a.as_cpu_slice(),
                        b.as_cpu_slice(),
                        out.as_cpu_slice_mut(),
                        m,
                        k,
                        n,
                    );
                    Apx4ExecTarget::CPU
                } else {
                    // APX 5.2: in 5.x modes we use the plan to decide between
                    // CPU/GPU/Auto. In other modes, we keep Auto.
                    let mapped = if apx_mode.starts_with("5.") {
                        match plan.target {
                            KernelTarget::Cpu => Apx4ExecTarget::CPU,
                            KernelTarget::Gpu => Apx4ExecTarget::GPU,
                            KernelTarget::HybridCpuGpu => Apx4ExecTarget::Auto,
                            KernelTarget::CpuFastAvx2 => Apx4ExecTarget::CPU,
                        }
                    } else {
                        Apx4ExecTarget::Auto
                    };

                    dispatch_matmul_gpu(
                        a.as_cpu_slice(),
                        b.as_cpu_slice(),
                        m,
                        k,
                        n,
                        out.as_cpu_slice_mut(),
                        mapped,
                    );
                    mapped
                };

                // Heuristically infer the device used: if the target was GPU,
                // assume GPU path; otherwise CPU. This inference is approximate
                // but sufficient for initial statistics.
                if matches!(target, Apx4ExecTarget::GPU) {
                    device_chosen = DeviceTarget::GPU;
                }

                // Register a sample only in 5.4 mode, after MatMul completes.
                if is_54 {
                    let duration_us = start_time.elapsed().as_micros() as u64;
                    if let Some(ref info) = node_exec_info_5x {
                        let sample = Sample {
                            op_name: info.op_name.clone(),
                            shape: info.shape.clone(),
                            dtype: match info.dtype.as_str() {
                                "F16" => crate::tensor::DType::F16,
                                "BF16" => crate::tensor::DType::BF16,
                                "FP8" => crate::tensor::DType::FP8,
                                _ => crate::tensor::DType::F32,
                            },
                            device_chosen,
                            duration_us,
                            vram_before: 0,
                            vram_after: 0,
                            fallback: false,
                        };

                        let sel_mutex = crate::global_adaptive_selector();
                        if let Ok(mut sel) = sel_mutex.lock() {
                            sel.register_sample(sample);
                        }
                    }
                }

                self.nodes[node_id].set_output(out);

                if record_tape {
                    let op_inputs = self.nodes[node_id].inputs.clone();
                    let ids = op_inputs.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs,
                        output: node_id,
                        backward: Box::new(move |store, forward_inputs, out_grad| {
                            let a = forward_inputs[0];
                            let b = forward_inputs[1];
                            let b_t = transpose_2d(b);
                            let mut grad_a = nn_linear::matmul(out_grad, &b_t);
                            grad_a.ensure_cpu().expect(
                                "backward intermediate: GPU->CPU transfer failed \
                                 (this indicates a CUDA driver issue during backward; \
                                 see StorageTransferError variants)",
                            );
                            add_to_grad_slice(store, ids[0], grad_a.as_cpu_slice());

                            let a_t = transpose_2d(a);
                            let mut grad_b = nn_linear::matmul(&a_t, out_grad);
                            grad_b.ensure_cpu().expect(
                                "backward intermediate: GPU->CPU transfer failed \
                                 (this indicates a CUDA driver issue during backward; \
                                 see StorageTransferError variants)",
                            );
                            add_to_grad_slice(store, ids[1], grad_b.as_cpu_slice());
                        }),
                    });
                }
            }
            NodeType::Transpose2D => {
                let src = self.nodes[node_id].inputs[0];
                let x = self.nodes[src]
                    .output
                    .as_ref()
                    .expect("Transpose missing input")
                    .clone();
                let out = transpose_2d(&x);
                self.nodes[node_id].set_output(out.clone());

                if record_tape {
                    let op_inputs = self.nodes[node_id].inputs.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            let mut grad_x = transpose_2d(out_grad);
                            grad_x.ensure_cpu().expect(
                                "backward intermediate: GPU->CPU transfer failed \
                                 (this indicates a CUDA driver issue during backward; \
                                 see StorageTransferError variants)",
                            );
                            add_to_grad_slice(store, op_inputs[0], grad_x.as_cpu_slice());
                        }),
                    });
                }
            }
            NodeType::IndexSelect => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert_eq!(inputs.len(), 2, "IndexSelect expects table and indices inputs");

                let table = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("IndexSelect missing table")
                    .clone();
                let indices = self.nodes[inputs[1]]
                    .output
                    .as_ref()
                    .expect("IndexSelect missing indices")
                    .clone();

                let out = index_select_rows(&table, &indices);
                self.nodes[node_id].set_output(out.clone());

                if record_tape {
                    let op_inputs = self.nodes[node_id].inputs.clone();
                    let ids = op_inputs.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs,
                        output: node_id,
                        backward: Box::new(move |store, forward_inputs, out_grad| {
                            let table = forward_inputs[0];
                            let indices = forward_inputs[1];
                            let cols = *table
                                .shape
                                .get(1)
                                .expect("IndexSelect table must be 2D");
                            let mut grad_table = vec![0.0; table.numel()];
                            scatter_add_rows(&mut grad_table, indices, out_grad.as_cpu_slice(), cols);
                            add_to_grad_slice(store, ids[0], &grad_table);
                            // No gradient for indices (integer gather)
                        }),
                    });
                }
            }
            NodeType::Reshape { target } => {
                let inputs = self.nodes[node_id].inputs.clone();
                if inputs.is_empty() {
                    // Inconsistent graph (e.g., artificial trace tests): do not execute Reshape.
                    return;
                }

                let src = inputs[0];
                let x_opt = self.nodes[src].output.as_ref();
                if let Some(x) = x_opt.cloned() {
                    let out = reshape_tensor(&x, &target);
                    self.nodes[node_id].set_output(out.clone());

                    if record_tape {
                        let op_inputs = self.nodes[node_id].inputs.clone();
                        let original_shape = x.shape.clone();
                        self.tape.push(BackOp {
                            inputs: op_inputs.clone(),
                            output: node_id,
                            backward: Box::new(move |store, _forward_inputs, out_grad| {
                                let mut reshaped_back = reshape_back(out_grad, &original_shape);
                                reshaped_back.ensure_cpu().expect(
                                    "backward intermediate: GPU->CPU transfer failed \
                                     (this indicates a CUDA driver issue during backward; \
                                     see StorageTransferError variants)",
                                );
                                add_to_grad_slice(store, op_inputs[0], reshaped_back.as_cpu_slice());
                            }),
                        });
                    }
                }
            }
            NodeType::TransposeLastTwo => {
                let src = self.nodes[node_id].inputs[0];
                let x = self.nodes[src]
                    .output
                    .as_ref()
                    .expect("TransposeLastTwo missing input")
                    .clone();
                let out = transpose_last_two(&x);
                self.nodes[node_id].set_output(out.clone());

                if record_tape {
                    let op_inputs = self.nodes[node_id].inputs.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            let mut back = transpose_last_two(out_grad);
                            back.ensure_cpu().expect(
                                "backward intermediate: GPU->CPU transfer failed \
                                 (this indicates a CUDA driver issue during backward; \
                                 see StorageTransferError variants)",
                            );
                            add_to_grad_slice(store, op_inputs[0], back.as_cpu_slice());
                        }),
                    });
                }
            }
            NodeType::BatchMatMul => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert_eq!(inputs.len(), 2, "BatchMatMul expects two inputs");
                let a = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("BatchMatMul missing input A")
                    .clone();
                let b = self.nodes[inputs[1]]
                    .output
                    .as_ref()
                    .expect("BatchMatMul missing input B")
                    .clone();

                assert!(
                    a.shape.len() == 3 && b.shape.len() == 3,
                    "BatchMatMul expects 3D tensors",
                );
                let batch = a.shape[0];
                let m = a.shape[1];
                let k = a.shape[2];
                let n = b.shape[2];

                assert_eq!(b.shape[0], batch, "BatchMatMul batch mismatch");
                assert_eq!(b.shape[1], k, "BatchMatMul inner dim mismatch");

                let mut out = Tensor::with_layout(
                    vec![batch, m, n],
                    0.0,
                    a.device,
                    Layout::Contiguous,
                    a.dtype,
                );

                // APX 5.4: timing measurement for BatchMatMul (CPU/GPU) for
                // collecting adaptive statistics.
                let start_time = Instant::now();
                let device_chosen;

                // Target selection via APX 5.2 (KernelPlanner) already computed
                // in 'plan'. We use the same mapping logic as in MatMul.
                let apx_mode = crate::apx_mode();
                let target = if apx_mode.starts_with("5.") {
                    match plan.target {
                        KernelTarget::Cpu => Apx4ExecTarget::CPU,
                        KernelTarget::Gpu => Apx4ExecTarget::GPU,
                        KernelTarget::HybridCpuGpu => Apx4ExecTarget::Auto,
                        KernelTarget::CpuFastAvx2 => Apx4ExecTarget::CPU,
                    }
                } else {
                    Apx4ExecTarget::Auto
                };

                match target {
                    Apx4ExecTarget::GPU => {
                        // GPU execution attempt via APX 4.5. If for any reason
                        // it fails or CUDA is not available, keep the previous
                        // CPU path as fallback.
                        if crate::cuda::cuda_available() {
                            dispatch_batch_matmul_cuda(
                                &a,
                                &b,
                                &mut out,
                                batch,
                                m,
                                k,
                                n,
                            );
                            device_chosen = DeviceTarget::GPU;
                        } else {
                            let cpu_out = batch_matmul(&a, &b);
                            out.as_cpu_slice_mut().copy_from_slice(cpu_out.as_cpu_slice());
                            device_chosen = DeviceTarget::CPU;
                        }
                    }
                    Apx4ExecTarget::CPU | Apx4ExecTarget::Auto => {
                        let cpu_out = batch_matmul(&a, &b);
                        out.as_cpu_slice_mut().copy_from_slice(cpu_out.as_cpu_slice());
                        device_chosen = DeviceTarget::CPU;
                    }
                }

                // Register a sample only in 5.4 mode, after BatchMatMul completes.
                if is_54 {
                    let duration_us = start_time.elapsed().as_micros() as u64;
                    if let Some(ref info) = node_exec_info_5x {
                        let sample = Sample {
                            op_name: info.op_name.clone(),
                            shape: info.shape.clone(),
                            dtype: match info.dtype.as_str() {
                                "F16" => crate::tensor::DType::F16,
                                "BF16" => crate::tensor::DType::BF16,
                                "FP8" => crate::tensor::DType::FP8,
                                _ => crate::tensor::DType::F32,
                            },
                            device_chosen,
                            duration_us,
                            vram_before: 0,
                            vram_after: 0,
                            fallback: false,
                        };

                        let sel_mutex = crate::global_adaptive_selector();
                        if let Ok(mut sel) = sel_mutex.lock() {
                            sel.register_sample(sample);
                        }
                    }
                }

                self.nodes[node_id].set_output(out.clone());

                if record_tape {
                    let op_inputs = inputs.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, forward_inputs, out_grad| {
                            let a = forward_inputs[0];
                            let b = forward_inputs[1];
                            let (grad_a, grad_b) = batch_matmul_backward(a, b, out_grad);
                            add_to_grad_slice(store, op_inputs[0], &grad_a);
                            add_to_grad_slice(store, op_inputs[1], &grad_b);
                        }),
                    });
                }
            }
            NodeType::BroadcastAdd => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert_eq!(inputs.len(), 2, "BroadcastAdd expects two inputs");
                let a = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("BroadcastAdd missing A")
                    .clone();
                let b = self.nodes[inputs[1]]
                    .output
                    .as_ref()
                    .expect("BroadcastAdd missing B")
                    .clone();
                let out = broadcast_add(&a, &b);
                self.nodes[node_id].set_output(out.clone());

                if record_tape {
                    let op_inputs = inputs.clone();
                    let shape_a = a.shape.clone();
                    let shape_b = b.shape.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            let grad_a = reduce_broadcast_grad(out_grad, &shape_a);
                            let grad_b = reduce_broadcast_grad(out_grad, &shape_b);
                            add_to_grad_slice(store, op_inputs[0], &grad_a);
                            add_to_grad_slice(store, op_inputs[1], &grad_b);
                        }),
                    });
                }
            }
            NodeType::LogSoftmax => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert_eq!(inputs.len(), 1, "LogSoftmax expects a single input");

                let x = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("LogSoftmax missing input")
                    .clone();

                let out = log_softmax_last_dim(&x);
                let out_clone = out.clone();
                self.nodes[node_id].set_output(out);

                if record_tape {
                    let op_inputs = inputs.clone();
                    let output_shape = out_clone.shape.clone();
                    let output_data = out_clone.copy_to_cpu_vec();
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            let cols = *output_shape
                                .last()
                                .expect("LogSoftmax requires rank >= 1");
                            let rows = if output_shape.len() == 1 {
                                1
                            } else {
                                output_shape[..output_shape.len() - 1]
                                    .iter()
                                    .product()
                            };
                            let out_grad_slice = out_grad.as_cpu_slice();
                            let mut grad_x = vec![0.0; out_grad.numel()];
                            for row in 0..rows {
                                let start = row * cols;
                                let end = start + cols;
                                let row_grad = &out_grad_slice[start..end];
                                let row_logp = &output_data[start..end];
                                let sum_grad: f32 = row_grad.iter().copied().sum();
                                for i in 0..cols {
                                    let prob = row_logp[i].exp();
                                    grad_x[start + i] = row_grad[i] - prob * sum_grad;
                                }
                            }
                            add_to_grad_slice(store, op_inputs[0], &grad_x);
                        }),
                    });
                }
            }
            NodeType::Gather => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert_eq!(inputs.len(), 2, "Gather expects data and indices inputs");

                let data = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("Gather missing data input")
                    .clone();
                let indices = self.nodes[inputs[1]]
                    .output
                    .as_ref()
                    .expect("Gather missing indices input")
                    .clone();

                let out = gather_last_dim(&data, &indices);
                self.nodes[node_id].set_output(out);

                if record_tape {
                    let op_inputs = inputs.clone();
                    let data_shape = data.shape.clone();
                    let indices_values = indices.copy_to_cpu_vec();
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            let last_dim = *data_shape
                                .last()
                                .expect("Gather data must have rank >= 1");
                            let rows = indices_values.len();
                            assert_eq!(rows, out_grad.numel(), "Gather grad mismatch");
                            let mut grad_data = vec![0.0; data_shape.iter().product()];
                            let out_grad_slice = out_grad.as_cpu_slice();
                            for row in 0..rows {
                                let idx = indices_values[row].round() as isize;
                                assert!(idx >= 0 && (idx as usize) < last_dim, "Gather index out of bounds");
                                let dst = row * last_dim + idx as usize;
                                grad_data[dst] += out_grad_slice[row];
                            }
                            add_to_grad_slice(store, op_inputs[0], &grad_data);
                        }),
                    });
                }
            }
            NodeType::CrossEntropyLoss => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert_eq!(inputs.len(), 2, "CrossEntropyLoss expects log_probs and targets");

                let log_probs = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("CrossEntropyLoss missing log probs")
                    .clone();
                let targets = self.nodes[inputs[1]]
                    .output
                    .as_ref()
                    .expect("CrossEntropyLoss missing targets")
                    .clone();

                let last_dim = *log_probs
                    .shape
                    .last()
                    .expect("CrossEntropyLoss log probs require rank >= 1");
                let rows = log_probs.numel() / last_dim;
                assert_eq!(
                    targets.numel(),
                    rows,
                    "CrossEntropyLoss targets mismatch"
                );

                let targets_slice = targets.as_cpu_slice();
                let log_probs_slice = log_probs.as_cpu_slice();
                let mut total = 0.0f32;
                let mut target_indices = Vec::with_capacity(rows);
                for row in 0..rows {
                    let idx = targets_slice[row].round() as isize;
                    assert!(idx >= 0 && (idx as usize) < last_dim, "CrossEntropyLoss target out of bounds");
                    let idx_usize = idx as usize;
                    target_indices.push(idx_usize);
                    let pos = row * last_dim + idx_usize;
                    total += log_probs_slice[pos];
                }
                let loss_val = -total / rows as f32;

                let mut out = Tensor::with_layout(
                    vec![1, 1],
                    0.0,
                    log_probs.device,
                    Layout::Contiguous,
                    log_probs.dtype,
                );
                out.as_cpu_slice_mut()[0] = loss_val;
                self.nodes[node_id].set_output(out);

                if record_tape {
                    let op_inputs = inputs.clone();
                    let total_rows = rows;
                    let vocab = last_dim;
                    let log_prob_len = log_probs.numel();
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            assert_eq!(out_grad.numel(), 1, "CrossEntropyLoss grad must be scalar");
                            let scale = -out_grad.as_cpu_slice()[0] / total_rows as f32;
                            let mut grad_log_probs = vec![0.0; log_prob_len];
                            for (row, &idx) in target_indices.iter().enumerate() {
                                let pos = row * vocab + idx;
                                grad_log_probs[pos] += scale;
                            }
                            add_to_grad_slice(store, op_inputs[0], &grad_log_probs);
                        }),
                    });
                }
            }
            NodeType::Linear => {
                if crate::apx_debug_enabled() {
                    eprintln!(
                        "[DBG] ENTER Linear node_id={} | inputs_len={}",
                        node_id,
                        self.nodes[node_id].inputs.len()
                    );
                }
                let inputs = self.nodes[node_id].inputs.clone();
                assert!(
                    inputs.len() == 2 || inputs.len() == 3,
                    "Linear node expects 2 or 3 inputs"
                );

                let x = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("Linear missing x")
                    .clone();
                let w = self.nodes[inputs[1]]
                    .output
                    .as_ref()
                    .expect("Linear missing weight")
                    .clone();
                // APX 4.11: per-operator direct GPU execution attempt only for
                // nodes outside planned GPU segments.
                let use_bias = inputs.len() == 3;
                let b_opt = if use_bias {
                    Some(
                        self.nodes[inputs[2]]
                            .output
                            .as_ref()
                            .expect("Linear missing bias")
                            .clone(),
                    )
                } else {
                    None
                };

                // APX 5.4: optional timing measurement for adaptive Linear statistics.
                let apx_mode_local = crate::apx_mode();
                let is_54_local = apx_mode_local.starts_with("5.4");
                let start_time = if is_54_local {
                    Some(Instant::now())
                } else {
                    None
                };

                // APX 5.2 + 5.4: only in 5.x modes we use the planner to decide
                // whether to attempt GPU via gpu_hooks::try_gpu_linear.
                if x.shape.len() == 2 && w.shape.len() == 2 {
                    let m = x.shape[0];
                    let k = x.shape[1];
                    if w.shape[0] == k {
                        let n = w.shape[1];

                        if apx_mode_local.starts_with("5.") && !in_gpu_segment {
                            // We use the previously computed plan for this node.
                            // If the planner suggests GPU, we try the GPU path;
                            // otherwise we keep CPU.
                            if let KernelTarget::Gpu = plan.target {
                                let mut tmp = Tensor::zeros_new(&[m, n], x.device);

                                if gpu_hooks::try_gpu_linear(
                                    &x,
                                    &w,
                                    b_opt.as_ref(),
                                    &mut tmp,
                                ) {
                                    // Register a Linear sample only if 5.4 mode is active.
                                    if let (true, Some(t0)) = (is_54_local, start_time) {
                                        let duration_us = t0.elapsed().as_micros() as u64;
                                        if let Some(ref info) = node_exec_info_5x {
                                            let sample = Sample {
                                                op_name: info.op_name.clone(),
                                                shape: info.shape.clone(),
                                                dtype: match info.dtype.as_str() {
                                                    "F16" => crate::tensor::DType::F16,
                                                    "BF16" => crate::tensor::DType::BF16,
                                                    "FP8" => crate::tensor::DType::FP8,
                                                    _ => crate::tensor::DType::F32,
                                                },
                                                device_chosen: DeviceTarget::GPU,
                                                duration_us,
                                                vram_before: 0,
                                                vram_after: 0,
                                                fallback: false,
                                            };

                                            let sel_mutex = crate::global_adaptive_selector();
                                            if let Ok(mut sel) = sel_mutex.lock() {
                                                sel.register_sample(sample);
                                            }
                                        }
                                    }

                                    self.nodes[node_id].set_output(tmp);
                                    if crate::apx_debug_enabled() {
                                        eprintln!(
                                            "[DBG] EARLY RETURN in Linear node_id={} via gpu_hooks::try_gpu_linear (APX 5.x)",
                                            node_id
                                        );
                                    }
                                    return;
                                }
                            }
                        } else if !in_gpu_segment
                            && gpu_hooks::try_gpu_linear(
                                &x,
                                &w,
                                b_opt.as_ref(),
                                &mut Tensor::zeros_new(&[m, n], x.device),
                            )
                        {
                            // Pre-5.x mode: original behavior. We do not register
                            // samples here to avoid mixing statistics between modes.
                            let mut tmp = Tensor::zeros_new(&[m, n], x.device);
                            gpu_hooks::try_gpu_linear(&x, &w, b_opt.as_ref(), &mut tmp);
                            self.nodes[node_id].set_output(tmp);
                            if crate::apx_debug_enabled() {
                                eprintln!(
                                    "[DBG] EARLY RETURN in Linear node_id={} via gpu_hooks::try_gpu_linear",
                                    node_id
                                );
                            }
                            return;
                        }
                    }
                }

                let out = if use_bias {
                    let b = b_opt.as_ref().expect("Linear missing bias clone");
                    nn_linear::linear(&x, &w, Some(b))
                } else {
                    nn_linear::linear(&x, &w, None)
                };

                // Register a CPU Linear sample only in 5.4 mode and only if we
                // did not return via the GPU path.
                if let (true, Some(t0)) = (is_54_local, start_time) {
                    let duration_us = t0.elapsed().as_micros() as u64;
                    if let Some(ref info) = node_exec_info_5x {
                        let sample = Sample {
                            op_name: info.op_name.clone(),
                            shape: info.shape.clone(),
                            dtype: match info.dtype.as_str() {
                                "F16" => crate::tensor::DType::F16,
                                "BF16" => crate::tensor::DType::BF16,
                                "FP8" => crate::tensor::DType::FP8,
                                _ => crate::tensor::DType::F32,
                            },
                            device_chosen: DeviceTarget::CPU,
                            duration_us,
                            vram_before: 0,
                            vram_after: 0,
                            fallback: false,
                        };

                        let sel_mutex = crate::global_adaptive_selector();
                        if let Ok(mut sel) = sel_mutex.lock() {
                            sel.register_sample(sample);
                        }
                    }
                }

                self.nodes[node_id].set_output(out);

                if record_tape {
                    // APX 4.16: if this Linear is part of a fused QKV pattern,
                    // delegate backward to the fused BackOp and do not record
                    // the normal Linear BackOp.
                    let mode = crate::apx_mode();
                    if mode == "4.16" {
                        if self.fused_ops.contains_key(&node_id) || self.is_qkv_secondary(node_id) {
                            return;
                        }
                    }

                    // APX 4.18 formerly skipped BackOp registration on Q/K/V
                    // Linears of a FusedSelfAttention pattern, expecting a
                    // fused BackOp to cover them. That fused BackOp was never
                    // implemented (see the "disabled for now" comment on the
                    // FusedSelfAttention arm of `exec_fused`), so the skip
                    // left the tape empty on the params feeding attention and
                    // backward produced no grads for dX/dWq/dWk/dWv. The skip
                    // is removed: APX 4.18 fuses the forward only, and
                    // backward flows through the individual BackOps, same as
                    // naive mode.

                    let op_inputs = self.nodes[node_id].inputs.clone();
                    let ids = op_inputs.clone();
                    let has_bias = ids.len() == 3;
                    self.tape.push(BackOp {
                        inputs: op_inputs,
                        output: node_id,
                        backward: Box::new(move |store, forward_inputs, out_grad| {
                            // APX 3.0: fused Linear backward (dX + dW) with chunking.
                            if crate::apx_mode() == "3.0" && !has_bias {
                                use crate::apx3::fused_backward::fused_linear_backward;

                                fused_linear_backward(
                                    store,
                                    forward_inputs[0],
                                    forward_inputs[1],
                                    out_grad,
                                    ids[0],
                                    ids[1],
                                );
                                return;
                            }

                            let x = forward_inputs[0];
                            let w = forward_inputs[1];

                            let w_t = transpose_2d(w);
                            let mut grad_x = nn_linear::matmul(out_grad, &w_t);
                            grad_x.ensure_cpu().expect(
                                "backward intermediate: GPU->CPU transfer failed \
                                 (this indicates a CUDA driver issue during backward; \
                                 see StorageTransferError variants)",
                            );
                            add_to_grad_slice(store, ids[0], grad_x.as_cpu_slice());

                            let x_t = transpose_2d(x);
                            let mut grad_w = nn_linear::matmul(&x_t, out_grad);
                            grad_w.ensure_cpu().expect(
                                "backward intermediate: GPU->CPU transfer failed \
                                 (this indicates a CUDA driver issue during backward; \
                                 see StorageTransferError variants)",
                            );
                            add_to_grad_slice(store, ids[1], grad_w.as_cpu_slice());

                            if has_bias {
                                let bias_grad = sum_rows(out_grad);
                                add_to_grad_slice(store, ids[2], &bias_grad);
                            }
                        }),
                    });
                }
            }
            NodeType::Activation(act) => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert_eq!(inputs.len(), 1, "Activation expects a single input");

                let x = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("Activation missing input")
                    .clone();

                let out = match act {
                    crate::amg::nodes::ActType::ReLU => nn_act::relu(&x),
                    crate::amg::nodes::ActType::SiLU => nn_act::silu(&x),
                    crate::amg::nodes::ActType::GELU => nn_act::gelu(&x),
                };

                self.nodes[node_id].set_output(out);

                // Note: for now we do not record ActType-specific backward;
                // the training path for these nodes is not supported in APX 4.8.
            }
            NodeType::FusedLinearActivation(act) => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert!(
                    inputs.len() == 2 || inputs.len() == 3,
                    "FusedLinearActivation expects 2 or 3 inputs",
                );

                let x = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("FusedLinearActivation missing x")
                    .clone();
                let w = self.nodes[inputs[1]]
                    .output
                    .as_ref()
                    .expect("FusedLinearActivation missing weight")
                    .clone();

                let b_opt = if inputs.len() == 3 {
                    Some(
                        self.nodes[inputs[2]]
                            .output
                            .as_ref()
                            .expect("FusedLinearActivation missing bias"),
                    )
                } else {
                    None
                };

                let lin = nn_linear::linear(&x, &w, b_opt);

                let out = match act {
                    crate::amg::nodes::ActType::ReLU => nn_act::relu(&lin),
                    crate::amg::nodes::ActType::SiLU => {
                        // Use the SiLU-specific fused implementation.
                        crate::apx4_8::fused_linear_activation::exec_fused_linear_silu(
                            &x, &w, b_opt,
                        )
                    }
                    crate::amg::nodes::ActType::GELU => nn_act::gelu(&lin),
                };

                self.nodes[node_id].set_output(out);

                // Fused-node-specific backward is not implemented in APX 4.8.
            }
            // M3 debt cleanup: `FusedLinearActivationChain` is
            // intercepted by the early-return handler higher up in
            // this function (see the
            // `exec_fused_linear_activation_chain` call site), so
            // reaching this arm is a control-flow bug. The arm that
            // previously duplicated the dispatch here was removed —
            // it was dead code with stricter-than-necessary input
            // validation (4/5 only), inconsistent with the APX 4.9
            // fusion detector's 3/4/5 output. Kept as
            // `unreachable!()` only to satisfy the compiler's
            // exhaustivity check on `NodeType`.
            NodeType::FusedLinearActivationChain(_) => {
                unreachable!(
                    "FusedLinearActivationChain must be handled by the \
                     early-return helper `exec_fused_linear_activation_chain`; \
                     reaching the match arm implies the early-return was \
                     accidentally gated or skipped"
                )
            }
            NodeType::RmsNorm => {
                let inputs = self.nodes[node_id].inputs.clone();
                if inputs.len() != 1 {
                    // Inconsistent graph (e.g., artificial trace tests): do not execute RmsNorm.
                    return;
                }

                let x_opt = self.nodes[inputs[0]].output.as_ref();
                if let Some(x) = x_opt.cloned() {
                    let out = nn_norm::rms_norm(&x, 1e-5);
                    self.nodes[node_id].set_output(out);
                }
            }
            NodeType::SiLU => {
                let inputs = self.nodes[node_id].inputs.clone();
                if inputs.len() != 1 {
                    // Inconsistent graph (e.g., artificial trace tests): do not execute SiLU.
                    return;
                }

                let x_opt = self.nodes[inputs[0]].output.as_ref();
                if let Some(x) = x_opt.cloned() {
                    let out = nn_act::silu(&x);
                    self.nodes[node_id].set_output(out);

                    if record_tape {
                        let op_inputs = self.nodes[node_id].inputs.clone();
                        let ids = op_inputs.clone();
                        self.tape.push(BackOp {
                            inputs: op_inputs,
                            output: node_id,
                            backward: Box::new(move |store, forward_inputs, out_grad| {
                                let x = forward_inputs[0];
                                let x_slice = x.as_cpu_slice();
                                let out_grad_slice = out_grad.as_cpu_slice();
                                let n = x.numel();
                                let mut grad_x = vec![0.0; n];
                                for i in 0..n {
                                    let v = x_slice[i];
                                    let sig = 1.0f32 / (1.0f32 + (-v).exp());
                                    let deriv = sig + v * sig * (1.0 - sig);
                                    grad_x[i] = out_grad_slice[i] * deriv;
                                }
                                add_to_grad_slice(store, ids[0], &grad_x);
                            }),
                        });
                    }
                }
            }
            NodeType::Softmax => {
                let inputs = self.nodes[node_id].inputs.clone();
                if inputs.len() != 1 {
                    // Inconsistent graph (e.g., artificial trace tests): do not execute Softmax.
                    return;
                }

                let x_opt = self.nodes[inputs[0]].output.as_ref();
                if let Some(x) = x_opt.cloned() {
                    let out = nn_softmax::softmax_last_dim(&x);
                    self.nodes[node_id].set_output(out);

                    if record_tape {
                        let op_inputs = self.nodes[node_id].inputs.clone();
                        let ids = op_inputs.clone();
                        let softmax_out = self.nodes[node_id]
                            .output
                            .as_ref()
                            .cloned()
                            .expect("Softmax output just computed");
                        let serial_shape = softmax_out.shape.clone();
                        let serial_data = softmax_out.copy_to_cpu_vec();

                        self.tape.push(BackOp {
                            inputs: op_inputs,
                            output: node_id,
                            backward: Box::new(move |store, _forward_inputs, out_grad| {
                                if cpu_features().avx2 {
                                    let mut grad_x = nn_softmax::softmax_backward_parallel(&softmax_out, out_grad);
                                    grad_x.ensure_cpu().expect(
                                        "backward intermediate: GPU->CPU transfer failed \
                                         (this indicates a CUDA driver issue during backward; \
                                         see StorageTransferError variants)",
                                    );
                                    add_to_grad_slice(store, ids[0], grad_x.as_cpu_slice());
                                } else {
                                    let cols = *serial_shape
                                        .last()
                                        .expect("Softmax requires at least one dimension");
                                    let rows = if serial_shape.len() == 1 {
                                        1
                                    } else {
                                        serial_shape[..serial_shape.len() - 1]
                                            .iter()
                                            .product()
                                    };
                                    let out_grad_slice = out_grad.as_cpu_slice();
                                    let mut grad_x = vec![0.0; out_grad.numel()];
                                    for row in 0..rows {
                                        let start = row * cols;
                                        let end = start + cols;
                                        let row_grad = &out_grad_slice[start..end];
                                        let row_y = &serial_data[start..end];
                                        let dot: f32 = row_grad
                                            .iter()
                                            .zip(row_y.iter())
                                            .map(|(g, y)| g * y)
                                            .sum();
                                        for i in 0..cols {
                                            grad_x[start + i] = row_y[i] * (row_grad[i] - dot);
                                        }
                                    }
                                    add_to_grad_slice(store, ids[0], &grad_x);
                                }
                            }),
                        });
                    }
                }
            }
            NodeType::Output => {
                let src = self.nodes[node_id].inputs[0];
                if self.nodes[src].output.is_none() {
                    self.execute_single(src, record_tape);
                }

                if let Some(out) = self.nodes[src].output.clone() {
                    self.nodes[node_id].set_output(out);
                    let op_inputs = vec![src];
                    self.tape.push(BackOp {
                        inputs: op_inputs,
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            add_to_grad_slice(store, src, out_grad.as_cpu_slice());
                        }),
                    });
                }
            }
            NodeType::Conv2D(cfg) => {
                let inputs = self.nodes[node_id].inputs.clone();
                let input_t = self.nodes[inputs[0]]
                    .output
                    .clone()
                    .expect("Conv2D: input tensor missing");
                let weight_t = self.nodes[inputs[1]]
                    .output
                    .clone()
                    .expect("Conv2D: weight tensor missing");
                let bias_t = if inputs.len() == 3 {
                    Some(
                        self.nodes[inputs[2]]
                            .output
                            .clone()
                            .expect("Conv2D: bias tensor missing"),
                    )
                } else {
                    None
                };
                let out = crate::amg::ops::conv2d::execute_conv2d(
                    &input_t,
                    &weight_t,
                    bias_t.as_ref(),
                    &cfg,
                );
                self.nodes[node_id].set_output(out);

                if record_tape {
                    let op_inputs = inputs.clone();
                    let cfg_captured = cfg;
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, forward_inputs, out_grad| {
                            let input_t = forward_inputs[0];
                            let weight_t = forward_inputs[1];
                            let bias_t = if forward_inputs.len() == 3 {
                                Some(forward_inputs[2])
                            } else {
                                None
                            };
                            let grads = crate::amg::ops::conv2d::execute_conv2d_backward(
                                input_t,
                                weight_t,
                                bias_t,
                                out_grad,
                                &cfg_captured,
                            );
                            add_to_grad_slice(store, op_inputs[0], &grads.grad_input);
                            add_to_grad_slice(store, op_inputs[1], &grads.grad_weight);
                            if let (Some(gb), Some(&bias_id)) =
                                (grads.grad_bias, op_inputs.get(2))
                            {
                                add_to_grad_slice(store, bias_id, &gb);
                            }
                        }),
                    });
                }
            }
            NodeType::MaxPool2D(cfg) => {
                let inputs = self.nodes[node_id].inputs.clone();
                let input_t = self.nodes[inputs[0]]
                    .output
                    .clone()
                    .expect("MaxPool2D: input tensor missing");
                let out = crate::amg::ops::maxpool2d::execute_maxpool2d(&input_t, &cfg);
                self.nodes[node_id].set_output(out);

                if record_tape {
                    let op_inputs = inputs.clone();
                    let cfg_captured = cfg;
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, forward_inputs, out_grad| {
                            let input_t = forward_inputs[0];
                            let grad_input =
                                crate::amg::ops::maxpool2d::execute_maxpool2d_backward(
                                    input_t,
                                    out_grad,
                                    &cfg_captured,
                                );
                            add_to_grad_slice(store, op_inputs[0], &grad_input);
                        }),
                    });
                }
            }
        }

        if crate::apx_debug_enabled() {
            if let Some(out) = self.nodes[node_id].output.as_ref() {
                if !crate::apx_is_silent() {
                    let input_lens: Vec<usize> = self.nodes[node_id]
                        .inputs
                        .iter()
                        .map(|i| {
                            self.nodes[*i]
                                .output
                                .as_ref()
                                .map(|t| t.numel())
                                .unwrap_or(0)
                        })
                        .collect();
                    eprintln!(
                        "[TRACE] node_id={} | node={:?} | shape={:?} | len={} | input_lens={:?}",
                        node_id,
                        node_type,
                        out.shape,
                        out.numel(),
                        input_lens
                    );
                }
            }
        }
    }

    /// Parallel backward pass.
    ///
    /// Panics if a GPU→CPU pre-transfer fails. For structured error
    /// handling, use [`Graph::backward_checked`].
    pub fn backward(&mut self, loss_node_id: usize) {
        self.backward_checked(loss_node_id).expect(
            "Graph::backward: GPU->CPU pre-pass failed (call backward_checked \
             for structured error handling)",
        );
    }

    /// Parallel backward pass with structured error propagation.
    ///
    /// Before running any backward closure, migrates every `node.output`
    /// tensor to CPU storage via [`Tensor::ensure_cpu`]. This guarantees
    /// that the closures — which consume tensors through `as_cpu_slice`
    /// and friends — never encounter a GPU-resident input. Any transfer
    /// failure is returned as [`StorageTransferError`] rather than panicking.
    pub fn backward_checked(
        &mut self,
        loss_node_id: usize,
    ) -> Result<(), StorageTransferError> {
        // Reset gradient store for this backward pass.
        self.grad_store = GradStore::new();

        // Ensure every cached forward output is CPU-resident before any
        // backward closure runs. Backward closures assume CPU storage and
        // call `as_cpu_slice()` without checks, which panics on
        // `TensorStorage::Cuda`.
        for node in &mut self.nodes {
            if let Some(ref mut output) = node.output {
                output.ensure_cpu()?;
            }
        }

        // Seed gradient at the loss node.
        let loss = self.nodes[loss_node_id]
            .output
            .as_ref()
            .expect("Loss node missing output");
        self.grad_store
            .set(loss_node_id, vec![1.0; loss.numel()]);

        // Build topological levels starting from the loss and run per-level parallel backward.
        let levels = self.build_backward_levels(loss_node_id);
        for level in levels {
            if std::env::var("ATENIA_TRACE").unwrap_or_default() == "1" {
                eprintln!(
                    "[APX TRACE] Executing backward level with {} ops in parallel",
                    level.len()
                );
            }
            level.par_iter().for_each(|&node_id| {
                self.execute_backward_single(node_id);
            });
        }

        // Materialize gradients from GradStore into node.output.grad
        for (node_id, node) in self.nodes.iter_mut().enumerate() {
            let buffer = self.grad_store.take(node_id);
            if buffer.is_empty() {
                continue;
            }
            if let Some(output) = &mut node.output {
                assert_eq!(
                    buffer.len(),
                    output.numel(),
                    "gradient length mismatch for node {} ({:?})",
                    node_id,
                    node.node_type
                );
                output.grad = Some(buffer);
            }
        }

        Ok(())
    }

    /// Sequential backward variant used for APX 2.0 regression tests.
    /// Executes the same level order as `backward` but without rayon parallelism.
    ///
    /// Panics if a GPU→CPU pre-transfer fails. For structured error
    /// handling, use [`Graph::backward_sequential_checked`].
    pub fn backward_sequential(&mut self, loss_node_id: usize) {
        self.backward_sequential_checked(loss_node_id).expect(
            "Graph::backward_sequential: GPU->CPU pre-pass failed (call \
             backward_sequential_checked for structured error handling)",
        );
    }

    /// Sequential analog of [`Graph::backward_checked`].
    pub fn backward_sequential_checked(
        &mut self,
        loss_node_id: usize,
    ) -> Result<(), StorageTransferError> {
        // Reset gradient store for this backward pass.
        self.grad_store = GradStore::new();

        // Same pre-pass as `backward_checked`; see the comment there.
        for node in &mut self.nodes {
            if let Some(ref mut output) = node.output {
                output.ensure_cpu()?;
            }
        }

        // Seed gradient at the loss node.
        let loss = self.nodes[loss_node_id]
            .output
            .as_ref()
            .expect("Loss node missing output");
        self.grad_store
            .set(loss_node_id, vec![1.0; loss.numel()]);

        // Build topological levels starting from the loss and run backward sequentially.
        let levels = self.build_backward_levels(loss_node_id);
        for level in levels {
            for node_id in level {
                self.execute_backward_single(node_id);
            }
        }

        // Materialize gradients from GradStore into node.output.grad
        for (node_id, node) in self.nodes.iter_mut().enumerate() {
            let buffer = self.grad_store.take(node_id);
            if buffer.is_empty() {
                continue;
            }
            if let Some(output) = &mut node.output {
                assert_eq!(
                    buffer.len(),
                    output.numel(),
                    "gradient length mismatch for node {} ({:?}) (sequential)",
                    node_id,
                    node.node_type
                );
                output.grad = Some(buffer);
            }
        }

        Ok(())
    }

    fn execute_backward_single(&self, node_id: usize) {
        let op = match self.tape.get(node_id) {
            Some(op) => op,
            None => return,
        };

        if std::env::var("ATENIA_TRACE").unwrap_or_default() == "1" {
            eprintln!(
                "[APX TRACE] Backward op for node {} executed on thread {:?}",
                node_id,
                std::thread::current().id()
            );
        }

        // Take gradient for this node's output from the store.
        let grad_output = self.grad_store.take(op.output);
        if grad_output.is_empty() {
            return;
        }

        let output_template = self.nodes[op.output]
            .output
            .as_ref()
            .expect("Missing output tensor during backward");
        assert_eq!(
            grad_output.len(),
            output_template.numel(),
            "gradient length mismatch for node {} ({:?})",
            op.output,
            self.nodes[op.output].node_type
        );

        let mut out_grad_tensor = output_template.clone();
        out_grad_tensor.set_cpu_data(grad_output);
        out_grad_tensor.grad = None;

        let input_refs: Vec<&Tensor> = op
            .inputs
            .iter()
            .map(|&id| {
                self.nodes[id]
                    .output
                    .as_ref()
                    .expect("Missing input tensor during backward")
            })
            .collect();

        (op.backward)(&self.grad_store, &input_refs, &out_grad_tensor);
    }

    fn build_backward_levels(&self, loss_id: usize) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.nodes.len()];
        let mut current = Vec::new();

        if self.tape.has_op(loss_id) {
            current.push(loss_id);
            visited[loss_id] = true;
        }

        let mut levels: Vec<Vec<usize>> = Vec::new();
        while !current.is_empty() {
            levels.push(current.clone());
            let mut next = Vec::new();
            for &node_id in levels.last().unwrap() {
                for &parent in &self.nodes[node_id].inputs {
                    if !visited[parent] && self.tape.has_op(parent) {
                        visited[parent] = true;
                        next.push(parent);
                    }
                }
            }
            current = next;
        }

        levels
    }

    fn collect_outputs(&self) -> Vec<Tensor> {
        self
            .nodes
            .iter()
            .filter_map(|node| match node.node_type {
                NodeType::Output => node.output.clone(),
                _ => None,
            })
            .collect()
    }

    fn execute_fused_add_mul(&mut self, add_node: usize, mul_node: usize) {
        execute_fused_add_mul_impl(&mut self.nodes, add_node, mul_node);
    }

    pub fn get_params_mut<'a>(&'a mut self, param_ids: &[usize]) -> Vec<&'a mut Tensor> {
        #[cfg(debug_assertions)]
        {
            let mut seen = HashSet::new();
            for &pid in param_ids {
                assert!(seen.insert(pid), "duplicate parameter id {pid}");
            }
        }

        let len = self.nodes.len();
        let base_ptr = self.nodes.as_mut_ptr();
        let mut tensors = Vec::with_capacity(param_ids.len());
        for &pid in param_ids {
            assert!(pid < len, "parameter id {pid} out of bounds");
            let node = unsafe { &mut *base_ptr.add(pid) };
            let tensor_ptr = node
                .output
                .as_mut()
                .expect("parameter node missing tensor output") as *mut Tensor;
            tensors.push(unsafe { &mut *tensor_ptr });
        }
        tensors
    }
}

fn execute_fused_add_mul_impl(nodes: &mut [Node], add_node: usize, mul_node: usize) {
    let add_inputs = nodes[add_node].inputs.clone();
    assert!(
        add_inputs.len() == 2,
        "Fused Add node must have exactly 2 inputs"
    );

    let a_id = add_inputs[0];
    let b_id = add_inputs[1];
    let c_id = nodes[mul_node].inputs[1];

    let a = nodes[a_id]
        .output
        .as_ref()
        .expect("Fused/Add missing A")
        .clone();
    let b = nodes[b_id]
        .output
        .as_ref()
        .expect("Fused/Add missing B")
        .clone();
    let c = nodes[c_id]
        .output
        .as_ref()
        .expect("Fused/Mul missing C")
        .clone();

    let tmp = a.add(&b);
    let out = tmp.mul(&c);

    nodes[add_node].set_output(tmp);
    nodes[mul_node].set_output(out);
}

fn add_to_grad_slice(store: &GradStore, node_id: usize, values: &[f32]) {
    store.add(node_id, values);
}

fn transpose_2d(t: &Tensor) -> Tensor {
    assert_eq!(t.shape.len(), 2, "transpose_2d expects a 2D tensor");
    let rows = t.shape[0];
    let cols = t.shape[1];
    let t_data = t.as_cpu_slice();
    let mut data = vec![0.0; t.numel()];
    for r in 0..rows {
        for c in 0..cols {
            data[c * rows + r] = t_data[r * cols + c];
        }
    }
    let new_shape = vec![cols, rows];
    Tensor::new_cpu_with_layout(
        new_shape,
        data,
        t.device,
        t.dtype,
        Layout::Contiguous,
    )
}

fn sum_rows(t: &Tensor) -> Vec<f32> {
    assert!(t.shape.len() >= 1, "sum_rows expects at least 1D tensor");
    let cols = *t.shape.last().unwrap();
    let rows = if t.shape.len() == 1 {
        1
    } else {
        t.shape[..t.shape.len() - 1].iter().product()
    };
    let mut result = vec![0.0; cols];
    let t_data = t.as_cpu_slice();
    for row in 0..rows {
        let start = row * cols;
        for i in 0..cols {
            result[i] += t_data[start + i];
        }
    }
    result
}

fn reshape_tensor(x: &Tensor, target: &Vec<isize>) -> Tensor {
    let mut new_shape = Vec::with_capacity(target.len());
    let mut inferred = None;
    let total: usize = x.shape.iter().product();
    let mut known = 1usize;
    for &dim in target {
        if dim == -1 {
            assert!(inferred.is_none(), "only one inferred dimension allowed");
            inferred = Some(new_shape.len());
            new_shape.push(1);
        } else {
            let d = dim as usize;
            known *= d.max(1);
            new_shape.push(d);
        }
    }
    if let Some(idx) = inferred {
        let inferred_dim = total / known.max(1);
        new_shape[idx] = inferred_dim;
    }
    assert_eq!(new_shape.iter().product::<usize>(), total, "reshape must preserve elements");
    Tensor::new_cpu_with_layout(
        new_shape,
        x.copy_to_cpu_vec(),
        x.device,
        x.dtype,
        Layout::Contiguous,
    )
}

fn reshape_back(x: &Tensor, original_shape: &Vec<usize>) -> Tensor {
    let mut t = x.clone();
    t.shape = original_shape.clone();
    t.strides = Tensor::compute_strides(original_shape, &Layout::Contiguous);
    t
}

fn transpose_last_two(x: &Tensor) -> Tensor {
    assert!(x.shape.len() >= 2, "TransposeLastTwo expects rank >= 2");
    let mut new_shape = x.shape.clone();
    let rank = new_shape.len();
    new_shape.swap(rank - 1, rank - 2);
    let mut out = Tensor::with_layout(new_shape.clone(), 0.0, x.device, Layout::Contiguous, x.dtype);
    let outer: usize = new_shape[..rank - 2].iter().product();
    let rows = new_shape[rank - 2];
    let cols = new_shape[rank - 1];
    let x_data = x.as_cpu_slice();
    let out_data = out.as_cpu_slice_mut();
    for outer_idx in 0..outer {
        let offset_in = outer_idx * rows * cols;
        let offset_out = offset_in;
        for r in 0..rows {
            for c in 0..cols {
                let src = offset_in + c * rows + r;
                let dst = offset_out + r * cols + c;
                out_data[dst] = x_data[src];
            }
        }
    }
    out
}

fn batch_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.shape.len() == 3 && b.shape.len() == 3, "BatchMatMul expects 3D tensors");
    let batch = a.shape[0];
    let m = a.shape[1];
    let k = a.shape[2];
    let n = b.shape[2];

    assert_eq!(b.shape[0], batch, "BatchMatMul batch mismatch");
    assert_eq!(b.shape[1], k, "BatchMatMul inner dim mismatch");

    let mut out = Tensor::with_layout(
        vec![batch, m, n],
        0.0,
        a.device,
        Layout::Contiguous,
        a.dtype,
    );

    crate::matmul_dispatcher::batch_matmul_dispatch(
        a.as_cpu_slice(),
        b.as_cpu_slice(),
        out.as_cpu_slice_mut(),
        batch,
        m,
        k,
        n,
    );

    out
}

fn batch_matmul_backward(a: &Tensor, b: &Tensor, out_grad: &Tensor) -> (Vec<f32>, Vec<f32>) {
    let batch = a.shape[0];
    let m = a.shape[1];
    let k = a.shape[2];
    let n = b.shape[2];
    let a_slice = a.as_cpu_slice();
    let b_slice = b.as_cpu_slice();
    let out_grad_slice = out_grad.as_cpu_slice();
    let mut grad_a = vec![0.0; a.numel()];
    let mut grad_b = vec![0.0; b.numel()];
    for batch_idx in 0..batch {
        let a_offset = batch_idx * m * k;
        let b_offset = batch_idx * k * n;
        let out_offset = batch_idx * m * n;
        for i in 0..m {
            for kk in 0..k {
                let mut sum = 0.0;
                for j in 0..n {
                    let grad_idx = out_offset + i * n + j;
                    let b_idx = b_offset + kk * n + j;
                    sum += out_grad_slice[grad_idx] * b_slice[b_idx];
                }
                grad_a[a_offset + i * k + kk] = sum;
            }
        }
        for kk in 0..k {
            for j in 0..n {
                let mut sum = 0.0;
                for i in 0..m {
                    let grad_idx = out_offset + i * n + j;
                    let a_idx = a_offset + i * k + kk;
                    sum += out_grad_slice[grad_idx] * a_slice[a_idx];
                }
                grad_b[b_offset + kk * n + j] = sum;
            }
        }
    }
    (grad_a, grad_b)
}

fn broadcast_add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape.len(), b.shape.len(), "BroadcastAdd ranks must match");
    let mut out = a.clone();
    add_broadcast_inplace(&mut out, b);
    out
}

fn add_broadcast_inplace(out: &mut Tensor, other: &Tensor) {
    let rank = out.shape.len();
    let mut index = vec![0usize; rank];
    loop {
        let out_offset = linear_index(&index, &out.shape);
        let mut other_index = vec![0usize; rank];
        for d in 0..rank {
            if other.shape[d] == 1 {
                other_index[d] = 0;
            } else {
                other_index[d] = index[d];
            }
        }
        let other_offset = linear_index(&other_index, &other.shape);
        out.as_cpu_slice_mut()[out_offset] += other.as_cpu_slice()[other_offset];
        if !increment_multi_index(&mut index, &out.shape) {
            break;
        }
    }
}

fn reduce_broadcast_grad(out_grad: &Tensor, target_shape: &Vec<usize>) -> Vec<f32> {
    let mut grad = vec![0.0; target_shape.iter().product()];
    let rank = target_shape.len();
    let mut index = vec![0usize; rank];
    loop {
        let out_offset = linear_index(&index, &out_grad.shape);
        let mut target_index = vec![0usize; rank];
        for d in 0..rank {
            if target_shape[d] == 1 {
                target_index[d] = 0;
            } else {
                target_index[d] = index[d];
            }
        }
        let target_offset = linear_index(&target_index, target_shape);
        grad[target_offset] += out_grad.as_cpu_slice()[out_offset];
        if !increment_multi_index(&mut index, &out_grad.shape) {
            break;
        }
    }
    grad
}

fn linear_index(index: &[usize], shape: &[usize]) -> usize {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (&i, &dim) in index.iter().zip(shape.iter()).rev() {
        offset += i * stride;
        stride *= dim.max(1);
    }
    offset
}

fn increment_multi_index(index: &mut [usize], shape: &[usize]) -> bool {
    for axis in (0..index.len()).rev() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return true;
        }
        index[axis] = 0;
    }
    false
}

fn index_select_rows(table: &Tensor, indices: &Tensor) -> Tensor {
    assert_eq!(table.shape.len(), 2, "IndexSelect table must be 2D");
    let rows = table.shape[0];
    let cols = table.shape[1];

    let mut out_shape = indices.shape.clone();
    out_shape.push(cols);
    let mut out = Tensor::with_layout(
        out_shape,
        0.0,
        table.device,
        Layout::Contiguous,
        table.dtype,
    );

    let indices_slice = indices.as_cpu_slice();
    let table_slice = table.as_cpu_slice();
    let out_slice = out.as_cpu_slice_mut();
    for (slot, &raw_idx) in indices_slice.iter().enumerate() {
        let idx = raw_idx.round() as isize;
        assert!(idx >= 0 && (idx as usize) < rows, "IndexSelect index out of bounds");
        let idx = idx as usize;
        let src_start = idx * cols;
        let dst_start = slot * cols;
        out_slice[dst_start..dst_start + cols]
            .copy_from_slice(&table_slice[src_start..src_start + cols]);
    }

    out
}

fn scatter_add_rows(grad_table: &mut [f32], indices: &Tensor, grad_out: &[f32], cols: usize) {
    assert_eq!(grad_out.len(), indices.numel() * cols);
    for (slot, &raw_idx) in indices.as_cpu_slice().iter().enumerate() {
        let idx = raw_idx.round() as isize;
        assert!(idx >= 0, "IndexSelect gradient index negative");
        let idx = idx as usize;
        let dst_start = idx * cols;
        let src_start = slot * cols;
        for i in 0..cols {
            grad_table[dst_start + i] += grad_out[src_start + i];
        }
    }
}
