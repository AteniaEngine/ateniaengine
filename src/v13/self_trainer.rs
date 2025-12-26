use std::collections::HashMap;

use crate::v13::checkpoint::{WarmStartAction, WarmStartPlan};
use crate::v13::learning_explanation::DecisionExplanation;
use crate::v13::learning_factors::{DecisionFactor, DecisionFactorKind, StructuredDecisionExplanation};
use crate::v13::learning_snapshot::{BackendKind, LearningContextSnapshot, LearningEntrySnapshot, LearningSnapshot, LearningStatsSnapshot, LearningSummary};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BackendChoice {
    Cpu,
    Gpu,
}

#[derive(Debug, Clone, Copy)]
pub struct ExecutionContext {
    pub gpu_available: bool,
    pub vram_pressure: f32,
    pub ram_pressure: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct DecisionSummary {
    pub backend: BackendChoice,
    pub promote_count: usize,
    pub degrade_count: usize,
    pub keep_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct EpisodeOutcome {
    pub success: bool,
    pub score: i32,
    pub had_drift: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct TrainingEpisode {
    pub ctx: ExecutionContext,
    pub decision: DecisionSummary,
    pub outcome: EpisodeOutcome,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContextBucket {
    pub gpu_available: bool,
    pub vram_band: u8,
    pub ram_band: u8,
}

#[derive(Debug, Clone, Copy)]
pub struct ChoiceStats {
    pub count: u32,
    pub success_count: u32,
    pub score_sum: i64,
    pub drift_count: u32,
}

impl ChoiceStats {
    pub fn new() -> Self {
        ChoiceStats {
            count: 0,
            success_count: 0,
            score_sum: 0,
            drift_count: 0,
        }
    }
}

fn band_for_pressure(p: f32) -> u8 {
    if p < 0.5 {
        0
    } else if p < 0.75 {
        1
    } else if p < 0.9 {
        2
    } else {
        3
    }
}

fn bucket_from_ctx(ctx: ExecutionContext) -> ContextBucket {
    ContextBucket {
        gpu_available: ctx.gpu_available,
        vram_band: band_for_pressure(ctx.vram_pressure),
        ram_band: band_for_pressure(ctx.ram_pressure),
    }
}

pub struct SelfTrainer {
    table: HashMap<(ContextBucket, BackendChoice), ChoiceStats>,
}

impl SelfTrainer {
    pub fn new() -> Self {
        SelfTrainer {
            table: HashMap::new(),
        }
    }

    pub fn record_episode(&mut self, ep: TrainingEpisode) {
        let bucket = bucket_from_ctx(ep.ctx);
        let key = (bucket, ep.decision.backend);

        let entry = match self.table.get_mut(&key) {
            Some(e) => e,
            None => {
                self.table.insert(key, ChoiceStats::new());
                match self.table.get_mut(&key) {
                    Some(e) => e,
                    None => return,
                }
            }
        };

        entry.count = entry.count.saturating_add(1);
        if ep.outcome.success {
            entry.success_count = entry.success_count.saturating_add(1);
        }
        entry.score_sum = entry.score_sum.saturating_add(ep.outcome.score as i64);
        if ep.outcome.had_drift {
            entry.drift_count = entry.drift_count.saturating_add(1);
        }
    }

    pub fn all_stats(&self) -> Vec<(ContextBucket, BackendChoice, ChoiceStats)> {
        let mut out = Vec::new();
        for ((bucket, backend), stats) in &self.table {
            out.push((*bucket, *backend, *stats));
        }
        out
    }

    pub fn set_stats_entry(
        &mut self,
        bucket: ContextBucket,
        backend: BackendChoice,
        stats: ChoiceStats,
    ) {
        self.table.insert((bucket, backend), stats);
    }

    pub fn stats_for(
        &self,
        ctx: ExecutionContext,
        backend: BackendChoice,
    ) -> Option<ChoiceStats> {
        let bucket = bucket_from_ctx(ctx);
        self.table.get(&(bucket, backend)).cloned()
    }

    fn value_for(&self, ctx: ExecutionContext, backend: BackendChoice) -> i64 {
        let bucket = bucket_from_ctx(ctx);
        let key = (bucket, backend);
        let stats = match self.table.get(&key) {
            Some(s) => *s,
            None => {
                // No data: neutral value 0.
                return 0;
            }
        };

        if stats.count == 0 {
            return 0;
        }

        let avg_score = stats.score_sum / (stats.count as i64);
        let drift_penalty: i64 = 5;
        let drift_rate_numer = stats.drift_count as i64;
        let drift_rate_denom = stats.count as i64;
        let drift_penalty_term = (drift_rate_numer * drift_penalty) / drift_rate_denom;

        avg_score - drift_penalty_term
    }

    pub fn recommend_backend(&self, ctx: ExecutionContext) -> BackendChoice {
        if !ctx.gpu_available {
            return BackendChoice::Cpu;
        }

        let cpu_value = self.value_for(ctx, BackendChoice::Cpu);
        let gpu_value = self.value_for(ctx, BackendChoice::Gpu);

        if gpu_value > cpu_value {
            BackendChoice::Gpu
        } else {
            // On tie or CPU better, prefer CPU for stability.
            BackendChoice::Cpu
        }
    }

    pub fn snapshot(&self) -> LearningSnapshot {
        let mut entries: Vec<LearningEntrySnapshot> = Vec::new();

        for (bucket, backend, stats) in self.all_stats() {
            let context = LearningContextSnapshot {
                gpu_available: bucket.gpu_available,
                vram_band: bucket.vram_band,
                ram_band: bucket.ram_band,
            };

            let recommended_backend = match backend {
                BackendChoice::Cpu => BackendKind::Cpu,
                BackendChoice::Gpu => BackendKind::Gpu,
            };

            let avg_score = if stats.count == 0 {
                0.0
            } else {
                (stats.score_sum as f32) / (stats.count as f32)
            };

            let stats_snapshot = LearningStatsSnapshot {
                episodes: stats.count,
                successes: stats.success_count,
                drift_events: stats.drift_count,
                average_score: avg_score,
            };

            entries.push(LearningEntrySnapshot {
                context,
                recommended_backend,
                stats: stats_snapshot,
            });
        }

        // Deterministic ordering: (gpu_available, vram_band, ram_band).
        entries.sort_by(|a, b| {
            let ag = if a.context.gpu_available { 1u8 } else { 0u8 };
            let bg = if b.context.gpu_available { 1u8 } else { 0u8 };
            ag.cmp(&bg)
                .then_with(|| a.context.vram_band.cmp(&b.context.vram_band))
                .then_with(|| a.context.ram_band.cmp(&b.context.ram_band))
        });

        let total_entries = entries.len();
        let mut total_episodes: u64 = 0;
        let mut gpu_pref_entries: u64 = 0;

        for entry in &entries {
            total_episodes = total_episodes.saturating_add(entry.stats.episodes as u64);
            if let BackendKind::Gpu = entry.recommended_backend {
                gpu_pref_entries = gpu_pref_entries.saturating_add(1);
            }
        }

        let gpu_preference_ratio = if total_entries == 0 {
            0.0
        } else {
            (gpu_pref_entries as f32) / (total_entries as f32)
        };

        let summary = LearningSummary {
            total_entries,
            total_episodes,
            gpu_preference_ratio,
        };

        LearningSnapshot { entries, summary }
    }

    pub fn explain_decision(
        &self,
        context: LearningContextSnapshot,
    ) -> Option<DecisionExplanation> {
        let snap = self.snapshot();

        // Filter entries that match the provided context.
        let mut best_entry: Option<&LearningEntrySnapshot> = None;

        for entry in &snap.entries {
            if entry.context.gpu_available != context.gpu_available {
                continue;
            }
            if entry.context.vram_band != context.vram_band {
                continue;
            }
            if entry.context.ram_band != context.ram_band {
                continue;
            }

            match best_entry {
                None => best_entry = Some(entry),
                Some(current) => {
                    if entry.stats.episodes > current.stats.episodes {
                        best_entry = Some(entry);
                    }
                }
            }
        }

        let chosen = match best_entry {
            Some(e) => e,
            None => return None,
        };

        let episodes = chosen.stats.episodes;
        let successes = chosen.stats.successes;
        let drift_events = chosen.stats.drift_events;

        let raw_confidence = if episodes == 0 {
            0.0
        } else {
            successes as f32 / episodes as f32
        };

        let confidence = DecisionExplanation::clamp_confidence(raw_confidence);

        // Derive an approximate execution context from the snapshot bands so we
        // can reuse the real recommendation logic without mutating state.
        fn band_to_pressure(band: u8) -> f32 {
            match band {
                0 => 0.2,
                1 => 0.5,
                2 => 0.8,
                _ => 0.5,
            }
        }

        let exec_ctx = ExecutionContext {
            gpu_available: context.gpu_available,
            vram_pressure: band_to_pressure(context.vram_band),
            ram_pressure: band_to_pressure(context.ram_band),
        };

        let backend_choice = self.recommend_backend(exec_ctx);
        let recommended_backend = match backend_choice {
            BackendChoice::Cpu => BackendKind::Cpu,
            BackendChoice::Gpu => BackendKind::Gpu,
        };

        let explanation = match recommended_backend {
            BackendKind::Gpu => {
                let stability = if drift_events == 0 {
                    "low drift"
                } else {
                    "some drift observed"
                };
                format!(
                    "GPU was selected because this context has been observed {} times, with {} successful executions and {}. Historical performance favors GPU execution under similar memory conditions.",
                    episodes,
                    successes,
                    stability,
                )
            }
            BackendKind::Cpu => {
                let instability_phrase = if drift_events > 0 {
                    "GPU execution in this context showed instability or drift"
                } else {
                    "historical data does not strongly favor GPU execution in this context"
                };
                format!(
                    "CPU was selected because {}. Historical data indicates CPU is more reliable under the current memory pressure, based on {} observations.",
                    instability_phrase,
                    episodes,
                )
            }
        };

        Some(DecisionExplanation {
            context,
            recommended_backend,
            confidence,
            explanation,
        })
    }

    pub fn explain_decision_structured(
        &self,
        context: LearningContextSnapshot,
    ) -> Option<StructuredDecisionExplanation> {
        let snap = self.snapshot();

        let mut best_entry: Option<&LearningEntrySnapshot> = None;

        for entry in &snap.entries {
            if entry.context.gpu_available != context.gpu_available {
                continue;
            }
            if entry.context.vram_band != context.vram_band {
                continue;
            }
            if entry.context.ram_band != context.ram_band {
                continue;
            }

            match best_entry {
                None => best_entry = Some(entry),
                Some(current) => {
                    if entry.stats.episodes > current.stats.episodes {
                        best_entry = Some(entry);
                    }
                }
            }
        }

        let chosen = match best_entry {
            Some(e) => e,
            None => return None,
        };

        let episodes = chosen.stats.episodes;
        let successes = chosen.stats.successes;
        let drift_events = chosen.stats.drift_events;

        let success_weight = if episodes == 0 {
            0.0
        } else {
            successes as f32 / episodes as f32
        };

        let drift_weight_raw = if episodes == 0 {
            0.0
        } else {
            drift_events as f32 / episodes as f32
        };

        let observation_weight_raw = episodes as f32 / 50.0;

        let drift_weight = if drift_weight_raw > 1.0 { 1.0 } else if drift_weight_raw < 0.0 { 0.0 } else { drift_weight_raw };
        let observation_weight = if observation_weight_raw > 1.0 { 1.0 } else if observation_weight_raw < 0.0 { 0.0 } else { observation_weight_raw };
        let memory_stability_weight = 1.0 - drift_weight;

        let success_factor = DecisionFactor::new(
            DecisionFactorKind::HistoricalSuccessRate,
            success_weight,
            format!(
                "Historical success rate for this context is {:.2} ({} successes over {} episodes).",
                if episodes == 0 { 0.0 } else { successes as f32 / episodes as f32 },
                successes,
                episodes,
            ),
        );

        let drift_factor = DecisionFactor::new(
            DecisionFactorKind::DriftPenalty,
            drift_weight,
            if drift_events > 0 {
                format!(
                    "Drift was observed {} times out of {} episodes, indicating potential instability.",
                    drift_events,
                    episodes,
                )
            } else {
                "No drift has been observed for this context.".to_string()
            },
        );

        let observation_factor = DecisionFactor::new(
            DecisionFactorKind::ObservationCount,
            observation_weight,
            format!(
                "Decision is based on {} total observations; more data increases confidence.",
                episodes,
            ),
        );

        let memory_stability_factor = DecisionFactor::new(
            DecisionFactorKind::MemoryStability,
            memory_stability_weight,
            if drift_events == 0 {
                "Observed executions were stable under the current memory pressure.".to_string()
            } else {
                "Some instability was observed under the current memory pressure due to drift.".to_string()
            },
        );

        // Factors must appear in a fixed, deterministic order.
        let factors = vec![
            success_factor,
            drift_factor,
            observation_factor,
            memory_stability_factor,
        ];

        let raw_confidence = if episodes == 0 {
            0.0
        } else {
            successes as f32 / episodes as f32
        };

        let confidence = StructuredDecisionExplanation::clamp_confidence(raw_confidence);

        // Reuse the same backend decision that the human-readable explanation
        // would describe, to avoid divergence between layers.
        fn band_to_pressure(band: u8) -> f32 {
            match band {
                0 => 0.2,
                1 => 0.5,
                2 => 0.8,
                _ => 0.5,
            }
        }

        let exec_ctx = ExecutionContext {
            gpu_available: context.gpu_available,
            vram_pressure: band_to_pressure(context.vram_band),
            ram_pressure: band_to_pressure(context.ram_band),
        };

        let backend_choice = self.recommend_backend(exec_ctx);
        let recommended_backend = match backend_choice {
            BackendChoice::Cpu => BackendKind::Cpu,
            BackendChoice::Gpu => BackendKind::Gpu,
        };

        Some(StructuredDecisionExplanation {
            context,
            recommended_backend,
            confidence,
            factors,
        })
    }
}

pub fn summarize_warm_start_plan(plan: &WarmStartPlan) -> DecisionSummary {
    let mut promote = 0usize;
    let mut degrade = 0usize;
    let mut keep = 0usize;

    for d in &plan.decisions {
        match d.action {
            WarmStartAction::Keep => keep = keep.saturating_add(1),
            WarmStartAction::HintPromote { .. } => promote = promote.saturating_add(1),
            WarmStartAction::DegradeSafe { .. } => degrade = degrade.saturating_add(1),
        }
    }

    let mut backend = BackendChoice::Cpu;

    for d in &plan.decisions {
        let reason_upper = d.reason.to_uppercase();
        if reason_upper.contains("GPU") {
            backend = BackendChoice::Gpu;
            break;
        }
    }

    DecisionSummary {
        backend,
        promote_count: promote,
        degrade_count: degrade,
        keep_count: keep,
    }
}
