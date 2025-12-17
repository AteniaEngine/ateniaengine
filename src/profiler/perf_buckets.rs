use crate::profiler::GpuHeatmap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerfTier {
    T0,
    T1,
    T2,
    T3,
    T4,
}

pub fn classify(avg_ms: f32) -> PerfTier {
    if avg_ms < 0.05 {
        PerfTier::T0
    } else if avg_ms < 0.5 {
        PerfTier::T1
    } else if avg_ms < 2.0 {
        PerfTier::T2
    } else if avg_ms < 10.0 {
        PerfTier::T3
    } else {
        PerfTier::T4
    }
}

#[derive(Debug, Clone)]
pub struct PerfBuckets {
    pub tiers: Vec<(String, PerfTier)>,
}

impl PerfBuckets {
    pub fn from_heatmap(hm: &GpuHeatmap) -> Self {
        let mut v = Vec::new();

        for (k, entry) in hm.entries.iter() {
            let tier = classify(entry.avg_ms);
            v.push((k.clone(), tier));
        }

        Self { tiers: v }
    }

    pub fn get(&self, name: &str) -> Option<PerfTier> {
        for (k, t) in &self.tiers {
            if k == name {
                return Some(*t);
            }
        }
        None
    }
}
