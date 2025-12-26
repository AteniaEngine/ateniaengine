use std::collections::{HashMap, HashSet};

use super::hybrid_memory::HybridMemoryManager;
use super::memory_types::{MemorySnapshot, MemoryTier, MoveError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OffloadAction {
    MoveToRam { tensor_id: String },
    MoveToSsd { tensor_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OffloadPlan {
    pub actions: Vec<OffloadAction>,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SmartOffloadEngine {
    pub vram_high: f32,
    pub vram_low: f32,
    pub ram_high: f32,
    pub ram_low: f32,
    pub cooldown_ticks: u64,
    pub max_actions_per_tick: usize,
    last_moved: HashMap<String, u64>,
}

impl SmartOffloadEngine {
    pub fn default() -> Self {
        SmartOffloadEngine {
            vram_high: 0.95,
            vram_low: 0.85,
            ram_high: 0.95,
            ram_low: 0.85,
            cooldown_ticks: 5,
            max_actions_per_tick: 4,
            last_moved: HashMap::new(),
        }
    }

    fn score_tensor(&self, mem: &HybridMemoryManager, id: &str) -> u64 {
        match mem.tensor_len_bytes(id) {
            Some(len) => len as u64,
            None => 0,
        }
    }

    // Legacy planning API (13.5.0 baseline semantics, no hysteresis/cooldown).
    pub fn plan(
        &self,
        snapshot: &MemorySnapshot,
        tensor_ids: &[&str],
        mem: &HybridMemoryManager,
    ) -> OffloadPlan {
        let vram_pressure = snapshot.vram.pressure.unwrap_or(0.0);
        let ram_pressure = snapshot.ram.pressure.unwrap_or(0.0);

        let vram_high = vram_pressure >= self.vram_high;
        let ram_high = ram_pressure >= self.ram_high;

        let mut actions: Vec<OffloadAction> = Vec::new();

        if vram_high && ram_high {
            // Prefer offloading everything to SSD when both VRAM and RAM are under pressure.
            for id in tensor_ids {
                let tier = match mem.get_tier(id) {
                    Some(t) => t,
                    None => continue,
                };
                if tier == MemoryTier::Vram || tier == MemoryTier::Ram {
                    // Avoid duplicates by checking if we already have this action.
                    let tid = (*id).to_string();
                    let already = actions.iter().any(|a| match a {
                        OffloadAction::MoveToSsd { tensor_id } => tensor_id == &tid,
                        _ => false,
                    });
                    if !already {
                        actions.push(OffloadAction::MoveToSsd { tensor_id: tid });
                    }
                }
            }

            let reason = "VRAM and RAM pressure high".to_string();
            return OffloadPlan { actions, reason };
        }

        if vram_high {
            for id in tensor_ids {
                let tier = match mem.get_tier(id) {
                    Some(t) => t,
                    None => continue,
                };
                if tier == MemoryTier::Vram {
                    let tid = (*id).to_string();
                    let already = actions.iter().any(|a| match a {
                        OffloadAction::MoveToRam { tensor_id } => tensor_id == &tid,
                        _ => false,
                    });
                    if !already {
                        actions.push(OffloadAction::MoveToRam { tensor_id: tid });
                    }
                }
            }

            let reason = if actions.is_empty() {
                "VRAM pressure high but no offloadable tensors in VRAM".to_string()
            } else {
                "VRAM pressure high".to_string()
            };
            return OffloadPlan { actions, reason };
        }

        if ram_high {
            for id in tensor_ids {
                let tier = match mem.get_tier(id) {
                    Some(t) => t,
                    None => continue,
                };
                if tier == MemoryTier::Ram {
                    let tid = (*id).to_string();
                    let already = actions.iter().any(|a| match a {
                        OffloadAction::MoveToSsd { tensor_id } => tensor_id == &tid,
                        _ => false,
                    });
                    if !already {
                        actions.push(OffloadAction::MoveToSsd { tensor_id: tid });
                    }
                }
            }

            let reason = if actions.is_empty() {
                "RAM pressure high but no offloadable tensors in RAM".to_string()
            } else {
                "RAM pressure high".to_string()
            };
            return OffloadPlan { actions, reason };
        }

        OffloadPlan {
            actions,
            reason: "No offloading needed".to_string(),
        }
    }

    pub fn plan_with_tick(
        &mut self,
        snapshot: &MemorySnapshot,
        tensor_ids: &[&str],
        mem: &HybridMemoryManager,
        tick: u64,
    ) -> OffloadPlan {
        let vram_pressure = snapshot.vram.pressure.unwrap_or(0.0);
        let ram_pressure = snapshot.ram.pressure.unwrap_or(0.0);

        let vram_high = vram_pressure >= self.vram_high;
        let vram_low = vram_pressure <= self.vram_low;
        let ram_high = ram_pressure >= self.ram_high;
        let ram_low = ram_pressure <= self.ram_low;

        let mut actions: Vec<OffloadAction> = Vec::new();
        let mut skipped_due_to_cooldown = false;

        // Helper closure to check cooldown for a tensor id.
        let can_schedule = |id: &str,
                            last_moved: &HashMap<String, u64>,
                            cooldown_ticks: u64|
         -> bool {
            match last_moved.get(id) {
                Some(last_tick) => {
                    if tick <= *last_tick {
                        false
                    } else if tick - *last_tick < cooldown_ticks {
                        false
                    } else {
                        true
                    }
                }
                None => true,
            }
        };

        // Helper to build sorted, budgeted candidates given a predicate and action kind.
        let mut build_priority_actions = |predicate: &dyn Fn(MemoryTier) -> bool,
                                          make_action: &dyn Fn(String) -> OffloadAction,
                                          base_reason: &str,
                                          empty_reason: &str,
                                          actions_out: &mut Vec<OffloadAction>|
         -> OffloadPlan {
            let mut seen: HashSet<String> = HashSet::new();
            let mut candidates: Vec<(String, u64)> = Vec::new();

            for id in tensor_ids {
                let tier = match mem.get_tier(id) {
                    Some(t) => t,
                    None => continue,
                };

                if !predicate(tier) {
                    continue;
                }

                if !can_schedule(id, &self.last_moved, self.cooldown_ticks) {
                    skipped_due_to_cooldown = true;
                    continue;
                }

                let tid = (*id).to_string();
                if seen.contains(&tid) {
                    continue;
                }
                seen.insert(tid.clone());

                let score = self.score_tensor(mem, &tid);
                candidates.push((tid, score));
            }

            // Sort by score desc, then id asc.
            candidates.sort_by(|a, b| {
                match b.1.cmp(&a.1) {
                    std::cmp::Ordering::Equal => a.0.cmp(&b.0),
                    other => other,
                }
            });

            let total_candidates = candidates.len();
            let mut selected = 0usize;

            for (tid, _score) in candidates.into_iter() {
                if selected >= self.max_actions_per_tick {
                    break;
                }
                actions_out.push(make_action(tid.clone()));
                self.last_moved.insert(tid, tick);
                selected += 1;
            }

            let mut reason = if actions_out.is_empty() {
                empty_reason.to_string()
            } else {
                base_reason.to_string()
            };

            if skipped_due_to_cooldown {
                reason.push_str("; some tensors skipped due to cooldown");
            }

            if total_candidates > 0 {
                let summary = format!(
                    "; Priority offloading enabled; selected {}/{}",
                    selected, total_candidates
                );
                reason.push_str(&summary);
            }

            OffloadPlan { actions: actions_out.clone(), reason }
        };

        // Both VRAM and RAM high: prefer offloading to SSD for both tiers.
        if vram_high && ram_high {
            return build_priority_actions(
                &|tier| tier == MemoryTier::Vram || tier == MemoryTier::Ram,
                &|tid| OffloadAction::MoveToSsd { tensor_id: tid },
                "VRAM and RAM pressure high",
                "VRAM and RAM pressure high but no offloadable tensors in RAM or VRAM",
                &mut actions,
            );
        }

        // VRAM-only high: move VRAM tensors to RAM.
        if vram_high && !ram_high {
            return build_priority_actions(
                &|tier| tier == MemoryTier::Vram,
                &|tid| OffloadAction::MoveToRam { tensor_id: tid },
                "VRAM pressure high",
                "VRAM pressure high but no offloadable tensors in VRAM",
                &mut actions,
            );
        }

        // RAM-only high: move RAM tensors to SSD.
        if ram_high && !vram_high {
            return build_priority_actions(
                &|tier| tier == MemoryTier::Ram,
                &|tid| OffloadAction::MoveToSsd { tensor_id: tid },
                "RAM pressure high",
                "RAM pressure high but no offloadable tensors in RAM",
                &mut actions,
            );
        }

        // Stable band or below-low pressure: do not plan new offloads.
        let reason = if vram_low && ram_low {
            "No offloading needed".to_string()
        } else {
            "No new offloading due to hysteresis band".to_string()
        };

        OffloadPlan { actions, reason }
    }

    pub fn apply(
        &self,
        snapshot: &MemorySnapshot,
        plan: &OffloadPlan,
        mem: &mut HybridMemoryManager,
    ) -> Result<(), MoveError> {
        for action in &plan.actions {
            match action {
                OffloadAction::MoveToRam { tensor_id } => {
                    let move_plan = mem.plan_move(tensor_id, MemoryTier::Ram, snapshot)?;
                    mem.apply_move(tensor_id, &move_plan)?;
                }
                OffloadAction::MoveToSsd { tensor_id } => {
                    // If tensor is in VRAM, move it to RAM first, then to SSD.
                    let current_tier = mem.get_tier(tensor_id).unwrap_or(MemoryTier::Ram);
                    if current_tier == MemoryTier::Vram {
                        let to_ram = mem.plan_move(tensor_id, MemoryTier::Ram, snapshot)?;
                        mem.apply_move(tensor_id, &to_ram)?;
                    }

                    let to_ssd = mem.plan_move(tensor_id, MemoryTier::Ssd, snapshot)?;
                    mem.apply_move(tensor_id, &to_ssd)?;
                }
            }
        }

        Ok(())
    }
}
