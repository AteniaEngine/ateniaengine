use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};
use crate::apx7::hpfa::FusionAffinity;

pub struct FusionProfile {
    pub op_name: String,
    pub baseline_us: u64,
    pub fused_qkv_us: u64,
    pub fused_full_us: u64,
}

pub struct FusionDecision {
    pub use_full_fusion: Option<bool>,
}

#[derive(Debug, Clone, Copy)]
pub enum GlobalDecision {
    PreferFull,
    PreferQKV,
    NoPreference,
}

pub struct FusionSelector {
    pub history: Vec<FusionProfile>,
    // APX 7.7: información auxiliar de afinidad de fusión. En esta
    // implementación inicial se mantiene vacía salvo que los tests la
    // pueblen explícitamente; no afecta a 6.9/6.10.
    pub qkv_candidates: HashSet<usize>,
    pub attn_candidates: HashSet<usize>,
    pub proj_candidates: HashSet<usize>,
    pub hist_profile: HashMap<usize, f64>,
}

impl FusionSelector {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            qkv_candidates: HashSet::new(),
            attn_candidates: HashSet::new(),
            proj_candidates: HashSet::new(),
            hist_profile: HashMap::new(),
        }
    }

    pub fn record_profile(&mut self, profile: FusionProfile) {
        self.history.push(profile);
    }

    /// APX 7.7: obtener afinidad de fusión para un nodo concreto. En esta
    /// versión inicial, la información proviene de estructuras auxiliares
    /// que pueden poblarse desde tests o futuras extensiones del profiler.
    pub fn get_fusion_affinity(&self, node_id: usize) -> FusionAffinity {
        FusionAffinity {
            qkv_chain: self.qkv_candidates.contains(&node_id),
            attn_fusable: self.attn_candidates.contains(&node_id),
            proj_fusable: self.proj_candidates.contains(&node_id),
            hot_factor: *self.hist_profile.get(&node_id).unwrap_or(&0.0),
        }
    }

    pub fn decide(&self) -> FusionDecision {
        if self.history.is_empty() {
            return FusionDecision { use_full_fusion: None };
        }

        let mut sum_qkv = 0.0f64;
        let mut sum_full = 0.0f64;
        let mut count = 0.0f64;

        for p in &self.history {
            sum_qkv += p.fused_qkv_us as f64;
            sum_full += p.fused_full_us as f64;
            count += 1.0;
        }

        let mean_qkv = sum_qkv / count;
        let mean_full = sum_full / count;

        if mean_full < 0.85 * mean_qkv {
            FusionDecision { use_full_fusion: Some(true) }
        } else if mean_qkv < 0.85 * mean_full {
            FusionDecision { use_full_fusion: Some(false) }
        } else {
            FusionDecision { use_full_fusion: None }
        }
    }

    pub fn best_decision(&self) -> Option<GlobalDecision> {
        if self.history.is_empty() {
            return None;
        }

        let d = self.decide();
        let gd = match d.use_full_fusion {
            Some(true) => GlobalDecision::PreferFull,
            Some(false) => GlobalDecision::PreferQKV,
            None => GlobalDecision::NoPreference,
        };
        Some(gd)
    }

    /// APX 6.13: obtener scores normalizados para full / qkv / baseline
    /// a partir de tiempos medios. Menor tiempo => mayor score.
    pub fn normalized_scores(&self) -> (f32, f32, f32) {
        assert!(
            !self.history.is_empty(),
            "normalized_scores requires at least one FusionProfile in history",
        );

        let mut sum_base = 0.0f64;
        let mut sum_qkv = 0.0f64;
        let mut sum_full = 0.0f64;
        let mut count = 0.0f64;

        for p in &self.history {
            sum_base += p.baseline_us as f64;
            sum_qkv += p.fused_qkv_us as f64;
            sum_full += p.fused_full_us as f64;
            count += 1.0;
        }

        let avg_base = sum_base / count;
        let avg_qkv = sum_qkv / count;
        let avg_full = sum_full / count;

        let full_s = 1.0f32 / (avg_full as f32 + 1.0);
        let qkv_s  = 1.0f32 / (avg_qkv as f32 + 1.0);
        let base_s = 1.0f32 / (avg_base as f32 + 1.0);

        (full_s, qkv_s, base_s)
    }
}

static GLOBAL_FUSION_SELECTOR: OnceLock<Mutex<FusionSelector>> = OnceLock::new();

pub fn global_fusion_selector() -> &'static Mutex<FusionSelector> {
    GLOBAL_FUSION_SELECTOR.get_or_init(|| Mutex::new(FusionSelector::new()))
}
