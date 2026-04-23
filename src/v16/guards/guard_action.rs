#![allow(dead_code)]

/// Action a guard can emit. Severity ordering used by
/// `GuardManager::evaluate`:
///
///   Abort > DeepDegrade > Degrade > Continue
///
/// where each level dominates the ones to its right. A `DeepDegrade`
/// from one guard combined with a `Degrade` from another yields
/// `DeepDegrade`; a single `Abort` from anywhere yields `Abort`.
///
/// Variants:
///
/// - `Continue`: no reaction, execution proceeds.
/// - `Degrade` (M3-e.1): VRAM pressure — migrate Cuda tensors back
///   to Cpu. Requires Cpu headroom; the reaction site may veto
///   this via the CPU-availability precondition introduced in
///   M3-e.6.
/// - `DeepDegrade` (M3-e.11.5): dual-pressure (VRAM + RAM both
///   saturated) — spill Cpu tensors to disk after the Cuda → Cpu
///   step. The reaction site usually reaches this via **promotion**
///   (a `Degrade` verdict upgraded to `DeepDegrade` when the
///   `dual_memory_pressure` helper sees both vram_pressure and
///   ram_pressure above the disk-spill thresholds). Guards are
///   also allowed to emit it directly, in which case dominance
///   above applies.
/// - `Abort`: irreversible condition — stop execution; the
///   checked-execution path surfaces this as
///   `ExecutionAbortReason::GuardAborted`.
#[derive(Debug, Clone, PartialEq)]
pub enum GuardAction {
    Continue,
    Degrade,
    DeepDegrade,
    Abort,
}
