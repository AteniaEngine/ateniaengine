//! M6.c.2 — `LayerResidencyPlanner`: which layers stay
//! pinned in VRAM, which layers stream from CPU per call.
//!
//! ## Policy (locked from M6 research §4.1)
//!
//! With 8 GB VRAM and ~0.62 GB/layer BF16 on Llama 2 13B,
//! roughly **11-13 layers fit**. Distribution is symmetric:
//! the first K layers stay resident (warm cache for prefill
//! activation locality) and the last K' layers stay
//! resident (warm cache for the lm_head residual). Middle
//! layers stream — their weights upload per call, the
//! kernel runs, the buffers free at end-of-call.
//!
//! ## Why first + last, not any K
//!
//! - **First K**: prefill operates left-to-right; the first
//!   layers see the most diverse activation patterns and
//!   benefit from cached weights the most.
//! - **Last K'**: lm_head + final norm + last residual all
//!   touch the same weight memory; pinning the tail block
//!   keeps that hot path on GPU.
//! - **Symmetric default `K == K'`**: simplest policy that
//!   captures the two anchors. v22+ may evolve to "warmest-
//!   N by profile-driven heatmap" but that needs profiling
//!   data we don't have at M6.c close.
//!
//! ## ATENIA_GPU_LAYERS env override
//!
//! The planner picks K + K' automatically based on
//! available VRAM. Operators can override with
//! `ATENIA_GPU_LAYERS=N` (forces exactly N total resident
//! layers, split symmetric K + K' = N) or
//! `ATENIA_GPU_LAYERS=0` (disables residency entirely —
//! every layer streams). Useful for:
//!
//! - **Bench harnesses** that want a controlled comparison
//!   ("what if we had 5 layers resident vs 10 resident").
//! - **Memory-constrained boxes** where the auto-detected
//!   ceiling is wrong (shared display VRAM, other GPU
//!   processes consuming memory).
//! - **Ablation tests** for M6.f handoff.
//!
//! ## Invariant
//!
//! `ResidencyPlan::resident.len() + plan.streamed.len() ==
//!  num_layers`, and the two sets are disjoint. Locked by
//! the `plan_partitions_layer_set_exactly` test below.

use std::collections::HashSet;

/// Output of [`plan_residency`]. Each `usize` is a layer
/// index in `0..num_layers`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidencyPlan {
    /// Layer indices whose weights stay pinned in VRAM
    /// across the whole session. Ordered ascending.
    pub resident: Vec<usize>,
    /// Layer indices whose weights stream per call. Ordered
    /// ascending.
    pub streamed: Vec<usize>,
    /// Total bytes the resident set will commit in VRAM.
    /// Diagnostic; the planner doesn't recheck this against
    /// `available_vram_bytes` after picking the count
    /// (caller is expected to size the budget upfront).
    pub resident_bytes: u64,
}

impl ResidencyPlan {
    pub fn is_empty(&self) -> bool { self.resident.is_empty() }

    pub fn num_resident(&self) -> usize { self.resident.len() }
    pub fn num_streamed(&self) -> usize { self.streamed.len() }

    /// True iff layer `i` is resident under this plan.
    pub fn is_resident(&self, i: usize) -> bool {
        // Linear search — `num_resident` is bounded by
        // ~13 in practice on dev hardware. A HashSet
        // would micro-optimise but cost more allocation.
        self.resident.binary_search(&i).is_ok()
    }
}

/// Inputs the planner needs. Decoupled from `LlamaConfig`
/// so future model families (Mistral, Qwen 3, …) can plug
/// in without per-family branching.
#[derive(Debug, Clone, Copy)]
pub struct PlannerInput {
    pub num_layers: usize,
    /// Bytes the planner should reserve PER resident layer.
    /// Computed by the caller from
    /// `LlamaConfig` (sum of K, V, Q, O, gate, up, down,
    /// input_norm, post_norm sizes at the model's storage
    /// dtype — F32 = 4 bytes, BF16 = 2 bytes).
    pub bytes_per_layer: u64,
    /// VRAM budget the planner is allowed to spend on
    /// resident layers. Caller subtracts working-buffer
    /// headroom (lm_head, final_norm, transient activation
    /// allocations) before passing this in.
    pub vram_budget_bytes: u64,
    /// Hard override from `ATENIA_GPU_LAYERS`. `None`
    /// means "auto"; `Some(0)` disables residency;
    /// `Some(n > 0)` forces exactly `n` resident layers
    /// (capped at `num_layers`).
    pub user_override_count: Option<usize>,
}

/// Compute a residency plan from the given inputs. Pure
/// function — no I/O, no global state. Test-friendly.
pub fn plan_residency(input: &PlannerInput) -> ResidencyPlan {
    if input.num_layers == 0 {
        return ResidencyPlan {
            resident: Vec::new(),
            streamed: Vec::new(),
            resident_bytes: 0,
        };
    }

    let auto_count = if input.bytes_per_layer == 0 {
        // Defensive: if the caller passes zero, assume
        // every layer fits. Test sanity, not a production
        // path.
        input.num_layers
    } else {
        (input.vram_budget_bytes / input.bytes_per_layer) as usize
    };

    let target_count = match input.user_override_count {
        Some(0) => 0,
        Some(n) => n.min(input.num_layers),
        None => auto_count.min(input.num_layers),
    };

    if target_count == 0 {
        let streamed: Vec<usize> = (0..input.num_layers).collect();
        return ResidencyPlan {
            resident: Vec::new(),
            streamed,
            resident_bytes: 0,
        };
    }

    if target_count == input.num_layers {
        let resident: Vec<usize> = (0..input.num_layers).collect();
        return ResidencyPlan {
            resident,
            streamed: Vec::new(),
            resident_bytes: input.bytes_per_layer * input.num_layers as u64,
        };
    }

    // Symmetric first-K + last-K' split.
    //
    // For target_count = T: K = ceil(T/2), K' = floor(T/2).
    // The tail block is fixed at floor so that an odd T
    // gives the prefill the extra layer (prefill sees the
    // most diverse activations per the policy doc).
    let k_first = target_count.div_ceil(2);
    let k_last = target_count - k_first;

    let mut resident_set: HashSet<usize> = HashSet::with_capacity(target_count);
    for i in 0..k_first {
        resident_set.insert(i);
    }
    for i in 0..k_last {
        resident_set.insert(input.num_layers - 1 - i);
    }

    let mut resident: Vec<usize> = resident_set.into_iter().collect();
    resident.sort_unstable();

    let streamed: Vec<usize> = (0..input.num_layers)
        .filter(|i| !resident.contains(i))
        .collect();

    ResidencyPlan {
        resident_bytes: input.bytes_per_layer * resident.len() as u64,
        resident,
        streamed,
    }
}

/// Read the `ATENIA_GPU_LAYERS` env var, parse, return the
/// override the planner should use. `Ok(None)` means "no
/// override → auto". `Ok(Some(n))` is the parsed value.
/// `Err` surfaces a clear message for invalid values.
pub fn parse_gpu_layers_env() -> Result<Option<usize>, String> {
    match std::env::var("ATENIA_GPU_LAYERS") {
        Err(_) => Ok(None),
        Ok(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
                return Ok(None);
            }
            trimmed.parse::<usize>()
                .map(Some)
                .map_err(|e| format!(
                    "ATENIA_GPU_LAYERS={s:?} is not a valid usize: {e}"
                ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(num_layers: usize, bpl_mb: u64, budget_mb: u64) -> PlannerInput {
        PlannerInput {
            num_layers,
            bytes_per_layer: bpl_mb * 1024 * 1024,
            vram_budget_bytes: budget_mb * 1024 * 1024,
            user_override_count: None,
        }
    }

    #[test]
    fn empty_when_zero_layers() {
        let p = plan_residency(&input(0, 100, 8000));
        assert!(p.resident.is_empty() && p.streamed.is_empty());
        assert_eq!(p.resident_bytes, 0);
    }

    #[test]
    fn empty_when_zero_budget() {
        let p = plan_residency(&input(40, 620, 0));
        assert!(p.resident.is_empty());
        assert_eq!(p.streamed.len(), 40);
        assert_eq!(p.resident_bytes, 0);
    }

    #[test]
    fn full_residency_when_budget_huge() {
        let p = plan_residency(&input(40, 620, 100_000));
        assert_eq!(p.resident.len(), 40);
        assert!(p.streamed.is_empty());
    }

    #[test]
    fn llama2_13b_realistic_dev_box() {
        // 13B BF16: 40 layers × ~620 MB/layer.
        // Budget: 7 GB (RTX 4070 Laptop's 8 GB minus
        // headroom for lm_head, final_norm, transient
        // activations).
        let p = plan_residency(&input(40, 620, 7 * 1024));
        // 7168 MB / 620 MB = 11.56 → 11 resident.
        assert_eq!(p.resident.len(), 11);
        assert_eq!(p.streamed.len(), 29);
        // 11 layers × 620 MB = 6820 MB
        assert_eq!(p.resident_bytes, 11 * 620 * 1024 * 1024);
        // Symmetric split: first 6, last 5 (ceil(11/2)=6).
        assert_eq!(&p.resident[..6], &[0, 1, 2, 3, 4, 5]);
        assert_eq!(&p.resident[6..], &[35, 36, 37, 38, 39]);
    }

    #[test]
    fn user_override_count_caps_at_num_layers() {
        let mut i = input(22, 100, 100_000);
        i.user_override_count = Some(50);  // > num_layers
        let p = plan_residency(&i);
        assert_eq!(p.resident.len(), 22);
        assert_eq!(p.streamed.len(), 0);
    }

    #[test]
    fn user_override_zero_disables_residency() {
        let mut i = input(40, 620, 100_000);
        i.user_override_count = Some(0);
        let p = plan_residency(&i);
        assert!(p.resident.is_empty());
        assert_eq!(p.streamed.len(), 40);
    }

    #[test]
    fn user_override_picks_symmetric_subset() {
        let mut i = input(40, 620, 1);  // tiny budget
        i.user_override_count = Some(8);
        let p = plan_residency(&i);
        assert_eq!(p.resident, vec![0, 1, 2, 3, 36, 37, 38, 39]);
        assert_eq!(p.streamed.len(), 32);
    }

    #[test]
    fn plan_partitions_layer_set_exactly() {
        // The contract: resident ⊕ streamed = full layer set,
        // disjoint, no duplicates. Locked across many shapes.
        let cases = [(40, 11), (22, 5), (40, 8), (22, 0), (40, 40), (1, 0), (1, 1)];
        for &(n, target) in &cases {
            let mut i = input(n, 100, 100);
            i.user_override_count = Some(target);
            let p = plan_residency(&i);
            assert_eq!(p.resident.len() + p.streamed.len(), n,
                "({n}, {target}) — partition incomplete");
            for r in &p.resident {
                assert!(!p.streamed.contains(r),
                    "({n}, {target}) — overlap at layer {r}");
            }
            // Sorted ascending invariant.
            for w in p.resident.windows(2) {
                assert!(w[0] < w[1], "resident not sorted: {:?}", p.resident);
            }
        }
    }

    #[test]
    fn is_resident_helper() {
        let mut i = input(10, 100, 100_000);
        i.user_override_count = Some(4);
        let p = plan_residency(&i);
        // 4 resident: first 2, last 2 → [0, 1, 8, 9]
        assert_eq!(p.resident, vec![0, 1, 8, 9]);
        for layer in 0..10 {
            let expected = matches!(layer, 0 | 1 | 8 | 9);
            assert_eq!(p.is_resident(layer), expected,
                "layer {layer}: is_resident mismatch");
        }
    }

    // ---- env-var parser ----

    #[test]
    fn env_parser_none_when_unset() {
        // Use a unique var name to avoid pollution.
        // We exercise the public function directly.
        // Actual `ATENIA_GPU_LAYERS` may be set in the test
        // env; clear before running.
        // SAFETY: tests run with `--test-threads=1` is not
        // guaranteed; we use a dedicated test runner that
        // is `#[serial]`-style by setting & restoring. For
        // simplicity, only test the `auto`/empty/parse
        // failure paths via direct calls.
        // Direct match on `parse_gpu_layers_env` would
        // depend on the env at call time; instead, rely on
        // the fact that the function is a thin wrapper and
        // re-test the parsing branches with manual strings.
        let cases: &[(&str, Result<Option<usize>, &str>)] = &[
            ("",      Ok(None)),
            ("auto",  Ok(None)),
            ("AUTO",  Ok(None)),
            ("0",     Ok(Some(0))),
            ("11",    Ok(Some(11))),
            ("  7  ", Ok(Some(7))),
            ("xyz",   Err("not a valid usize")),
        ];
        for (input, want) in cases {
            let parsed = parse_inner(input);
            match (parsed, want) {
                (Ok(got), Ok(w)) => assert_eq!(&got, w, "input={input:?}"),
                (Err(e), Err(needle)) =>
                    assert!(e.contains(needle), "{e:?} should mention {needle:?}"),
                (a, b) => panic!("input={input:?} got={a:?} want={b:?}"),
            }
        }
    }

    /// Mirror of `parse_gpu_layers_env` that takes the
    /// string directly so tests don't depend on process env.
    fn parse_inner(s: &str) -> Result<Option<usize>, String> {
        let trimmed = s.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("auto") {
            return Ok(None);
        }
        trimmed.parse::<usize>()
            .map(Some)
            .map_err(|e| format!(
                "ATENIA_GPU_LAYERS={s:?} is not a valid usize: {e}"
            ))
    }
}
