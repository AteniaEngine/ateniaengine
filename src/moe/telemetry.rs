//! **MOE-PERF-5** — MoE-generation telemetry (observability parity with the
//! dense generator).
//!
//! The dense path (`cli_generate.rs`) already reports load time, generation
//! wall, tokens, and tok/s plus loader/matmul counters. The controlled MoE path
//! (`controlled_moe_generate` → `MoeRuntime::generate`) returned only token ids.
//! This module adds an **additive, read-only** telemetry struct that the
//! instrumented generate entry points fill, so a real MoE workload can be
//! measured end-to-end (timing) and — for the disk-tier graph families — its
//! expert-cache / prefetch / tier I/O can be inspected.
//!
//! **Instrumentation only.** Nothing here changes numerics, routing, MLA, the
//! cache, or the loader. The metrics are gathered from the existing
//! [`CacheStats`](super::residency::CacheStats) (via
//! [`graph_op::aggregate_resident_cache_stats`](super::graph_op)) and from
//! wall-clock timers around the unchanged generation.
//!
//! ## Coverage (honest)
//!
//! * **Timing** (load / prefill / decode / first-token / total / tok-s):
//!   available for **all** families.
//! * **Expert-cache / prefetch / tier** metrics: available for the **graph
//!   families (Mixtral, Qwen-MoE) on the disk tier**, where experts stream
//!   through the registered [`ExpertCache`](super::residency::ExpertCache). On
//!   the RAM tier the experts are resident (zero tier reads — the metrics are a
//!   true zero). **DeepSeek (MLA)** streams its experts through the *uncached*
//!   `ResidentExpertLayer::forward`, so it is **not** in the cache registry and
//!   reports no cache metrics — flagged by [`MoeGenTelemetry::cache_telemetry_available`].

use std::time::Duration;

use super::residency::CacheStats;

/// Per-stage wall-clock timings captured by the instrumented generate entry
/// points. `first_token` is the time from the start of generation until the
/// first token is produced (i.e. the prefill cost). `prefill` and `decode` are
/// the cumulative build+exec time of the prefill and of all decode steps.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StageTimings {
    pub prefill: Duration,
    pub decode: Duration,
    pub first_token: Duration,
}

/// End-to-end telemetry for one MoE generation. Timing is always populated;
/// cache/prefetch/tier fields are populated for the disk-tier graph families and
/// are otherwise a true zero (RAM tier) or unavailable (DeepSeek — see
/// [`Self::cache_telemetry_available`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
pub struct MoeGenTelemetry {
    // ---- Generation (all families) ----
    /// Model load (`MoeRuntime::load_from_dir`) wall time.
    pub load_ms: f64,
    /// Prefill (prompt) build+exec wall time.
    pub prefill_ms: f64,
    /// Cumulative decode-step build+exec wall time.
    pub decode_ms: f64,
    /// Time to the first generated token (the prefill cost).
    pub first_token_ms: f64,
    /// Total generation wall time (prefill + all decode steps), excluding load.
    pub total_generation_ms: f64,
    /// Number of tokens produced.
    pub generated_tokens: usize,
    /// Decode throughput (`generated_tokens / total_generation_ms`).
    pub tokens_per_second: f64,

    // ---- Expert cache (graph families, disk tier) ----
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub evictions: usize,
    /// Host-RAM bytes the resident expert caches occupy at the end of generation.
    pub resident_bytes: usize,

    // ---- Prefetch (MOE-PERF-3) ----
    pub parallel_prefetches: usize,
    /// Tier read latency hidden by overlapping (`Σ read − overlapped wall`).
    /// Reported only when `parallel_prefetches > 0`; otherwise `0.0`.
    pub overlap_saved_ms: f64,
    /// Cumulative tier `resolve()` time (NVMe read + bf16→f32 decode).
    pub resolve_time_ms: f64,

    // ---- Tier I/O ----
    /// Total expert weight bytes read from the tier during generation.
    pub materialized_bytes: usize,
    /// Number of tier reads (cache misses + explicit prefetches).
    pub tier_reads: usize,

    /// `true` when expert-cache/prefetch/tier metrics reflect the real path
    /// (graph families). `false` for DeepSeek (uncached MLA expert stream) — its
    /// cache fields are left at zero and must not be read as a measurement.
    pub cache_telemetry_available: bool,
}

impl MoeGenTelemetry {
    /// Assemble telemetry from the timed generation and the per-generation
    /// cache-stats **delta** (after − before, so concurrent/earlier loads do not
    /// leak in). `cache_available` is `false` for DeepSeek (no cache path).
    pub fn assemble(
        load: Duration,
        timings: StageTimings,
        total: Duration,
        generated_tokens: usize,
        cache_delta: CacheStats,
        resident_bytes: usize,
        cache_available: bool,
    ) -> Self {
        let total_ms = total.as_secs_f64() * 1e3;
        let tps = if total_ms > 0.0 {
            generated_tokens as f64 / total.as_secs_f64()
        } else {
            0.0
        };
        // Overlap is only meaningful when the prefetch path actually ran a
        // parallel batch; otherwise prefetch_wall_nanos is 0 and the subtraction
        // would misreport the serial read sum as "saved".
        let overlap_saved_ms = if cache_delta.parallel_prefetches > 0 {
            cache_delta.resolve_nanos.saturating_sub(cache_delta.prefetch_wall_nanos) as f64 / 1e6
        } else {
            0.0
        };
        Self {
            load_ms: load.as_secs_f64() * 1e3,
            prefill_ms: timings.prefill.as_secs_f64() * 1e3,
            decode_ms: timings.decode.as_secs_f64() * 1e3,
            first_token_ms: timings.first_token.as_secs_f64() * 1e3,
            total_generation_ms: total_ms,
            generated_tokens,
            tokens_per_second: tps,
            cache_hits: cache_delta.hits,
            cache_misses: cache_delta.misses,
            evictions: cache_delta.evictions,
            resident_bytes,
            parallel_prefetches: cache_delta.parallel_prefetches,
            overlap_saved_ms,
            resolve_time_ms: cache_delta.resolve_nanos as f64 / 1e6,
            materialized_bytes: cache_delta.tier_bytes_read,
            tier_reads: cache_delta.misses + cache_delta.prefetched,
            cache_telemetry_available: cache_available,
        }
    }

    /// Human-readable multi-line block (stderr), parity with the dense path's
    /// "Generated: N tokens in Xs (Y tok/s)" plus the MoE-specific cache/tier
    /// lines. Mirrors the dense layout so the two paths read alike.
    pub fn render(&self) -> String {
        let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
        let mut s = String::new();
        s.push_str("[ATENIA] MoE generation telemetry\n");
        s.push_str(&format!(
            "  load        : {:>9.1} ms\n  prefill     : {:>9.1} ms (first token {:.1} ms)\n  \
             decode      : {:>9.1} ms\n  total gen   : {:>9.1} ms\n  tokens      : {:>9} ({:.2} tok/s)\n",
            self.load_ms,
            self.prefill_ms,
            self.first_token_ms,
            self.decode_ms,
            self.total_generation_ms,
            self.generated_tokens,
            self.tokens_per_second,
        ));
        if self.cache_telemetry_available {
            let total_lookups = self.cache_hits + self.cache_misses;
            let hit_rate = if total_lookups > 0 {
                100.0 * self.cache_hits as f64 / total_lookups as f64
            } else {
                0.0
            };
            s.push_str(&format!(
                "  expert cache: hits {} / misses {} ({:.1}% hit), evictions {}, resident {:.1} MiB\n  \
                 tier I/O    : {} reads, {:.1} MiB materialized, resolve {:.1} ms\n  \
                 prefetch    : {} parallel, {:.1} ms overlapped\n",
                self.cache_hits,
                self.cache_misses,
                hit_rate,
                self.evictions,
                mib(self.resident_bytes),
                self.tier_reads,
                mib(self.materialized_bytes),
                self.resolve_time_ms,
                self.parallel_prefetches,
                self.overlap_saved_ms,
            ));
        } else {
            s.push_str(
                "  expert cache: n/a (DeepSeek/MLA streams experts uncached — timing only)\n",
            );
        }
        s
    }
}

/// **MOE-PERF-5 (instrumentation)** — field-wise `after − before` of two
/// [`CacheStats`] snapshots, so a single generation's deltas can be isolated
/// from the process-global, cumulative registry counters.
pub fn cache_stats_delta(after: CacheStats, before: CacheStats) -> CacheStats {
    CacheStats {
        hits: after.hits.saturating_sub(before.hits),
        misses: after.misses.saturating_sub(before.misses),
        evictions: after.evictions.saturating_sub(before.evictions),
        prefetched: after.prefetched.saturating_sub(before.prefetched),
        tier_bytes_read: after.tier_bytes_read.saturating_sub(before.tier_bytes_read),
        shared_hits: after.shared_hits.saturating_sub(before.shared_hits),
        shared_misses: after.shared_misses.saturating_sub(before.shared_misses),
        shared_fwd_nanos: after.shared_fwd_nanos.saturating_sub(before.shared_fwd_nanos),
        routed_fwd_nanos: after.routed_fwd_nanos.saturating_sub(before.routed_fwd_nanos),
        resolve_nanos: after.resolve_nanos.saturating_sub(before.resolve_nanos),
        parallel_prefetches: after
            .parallel_prefetches
            .saturating_sub(before.parallel_prefetches),
        prefetch_wall_nanos: after
            .prefetch_wall_nanos
            .saturating_sub(before.prefetch_wall_nanos),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(hits: usize, misses: usize, prefetch: usize, resolve_ns: u128, wall_ns: u128) -> CacheStats {
        CacheStats {
            hits,
            misses,
            parallel_prefetches: prefetch,
            resolve_nanos: resolve_ns,
            prefetch_wall_nanos: wall_ns,
            tier_bytes_read: misses * 1000,
            ..CacheStats::default()
        }
    }

    #[test]
    fn delta_is_field_wise_and_saturating() {
        let before = stats(10, 5, 2, 100, 40);
        let after = stats(15, 9, 5, 250, 90);
        let d = cache_stats_delta(after, before);
        assert_eq!(d.hits, 5);
        assert_eq!(d.misses, 4);
        assert_eq!(d.parallel_prefetches, 3);
        assert_eq!(d.resolve_nanos, 150);
        assert_eq!(d.prefetch_wall_nanos, 50);
        // saturating: an "after" smaller than "before" never underflows.
        let d2 = cache_stats_delta(before, after);
        assert_eq!(d2.hits, 0);
        assert_eq!(d2.misses, 0);
    }

    #[test]
    fn assemble_computes_tps_and_overlap() {
        let timings = StageTimings {
            prefill: Duration::from_millis(20),
            decode: Duration::from_millis(80),
            first_token: Duration::from_millis(20),
        };
        let delta = stats(3, 7, 7, 30_000_000, 12_000_000); // 30ms read, 12ms wall
        let t = MoeGenTelemetry::assemble(
            Duration::from_millis(500),
            timings,
            Duration::from_millis(100),
            4,
            delta,
            2 * 1024 * 1024,
            true,
        );
        assert_eq!(t.generated_tokens, 4);
        assert!((t.tokens_per_second - 40.0).abs() < 1e-6); // 4 tok / 0.1 s
        assert_eq!(t.cache_hits, 3);
        assert_eq!(t.cache_misses, 7);
        assert_eq!(t.tier_reads, 7);
        // overlap = 30ms − 12ms = 18ms (parallel_prefetches > 0).
        assert!((t.overlap_saved_ms - 18.0).abs() < 1e-6);
        assert!(t.cache_telemetry_available);
        assert!(t.render().contains("MoE generation telemetry"));
    }

    #[test]
    fn overlap_zero_without_prefetch() {
        let timings = StageTimings::default();
        let delta = stats(0, 5, 0, 50_000_000, 0); // reads but no parallel batch
        let t = MoeGenTelemetry::assemble(
            Duration::ZERO,
            timings,
            Duration::from_millis(10),
            1,
            delta,
            0,
            true,
        );
        // No parallel prefetch => overlap must be 0 (not the serial read sum).
        assert_eq!(t.overlap_saved_ms, 0.0);
        assert!((t.resolve_time_ms - 50.0).abs() < 1e-6);
    }

    #[test]
    fn deepseek_flag_renders_uncached_note() {
        let t = MoeGenTelemetry::assemble(
            Duration::from_millis(1),
            StageTimings::default(),
            Duration::from_millis(5),
            2,
            CacheStats::default(),
            0,
            false,
        );
        assert!(!t.cache_telemetry_available);
        assert!(t.render().contains("uncached"));
    }
}
