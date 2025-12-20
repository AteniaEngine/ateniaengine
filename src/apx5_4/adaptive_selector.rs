use crate::apx5::apx_5_3_planner::{ExecutionPlan5_3, NodeExecInfo, LayoutDecision};

use super::{OpStats, Sample, DeviceTarget};

#[derive(Clone, Debug)]
pub struct AdaptiveDecision {
    pub prefer_device: Option<DeviceTarget>,
    pub prefer_layout: Option<LayoutDecision>,
}

#[derive(Clone, Debug)]
pub struct AdaptiveSelector {
    pub cpu_stats: OpStats,
    pub gpu_stats: OpStats,
    pub history: Vec<Sample>,
}

impl AdaptiveSelector {
    pub fn new() -> Self {
        Self {
            cpu_stats: OpStats::new(),
            gpu_stats: OpStats::new(),
            history: Vec::new(),
        }
    }

    pub fn register_sample(&mut self, sample: Sample) {
        match sample.device_chosen {
            DeviceTarget::CPU => self.cpu_stats.record(sample.duration_us, sample.fallback),
            DeviceTarget::GPU => self.gpu_stats.record(sample.duration_us, sample.fallback),
        }
        self.history.push(sample);
    }

    pub fn decide(&self, info: &NodeExecInfo) -> AdaptiveDecision {
        // Minimal heuristic based on history for the same op/shape/dtype.
        let relevant: Vec<&Sample> = self
            .history
            .iter()
            .filter(|s| {
                s.op_name == info.op_name
                    && s.shape == info.shape
                    && format!("{:?}", s.dtype) == info.dtype
            })
            .collect();

        let mut prefer_device = None;
        let prefer_layout = None;

        if !relevant.is_empty() {
            let (mut cpu_durs, mut gpu_durs) = (Vec::new(), Vec::new());
            let mut gpu_fallbacks = 0u32;

            for s in relevant {
                match s.device_chosen {
                    DeviceTarget::CPU => cpu_durs.push(s.duration_us),
                    DeviceTarget::GPU => {
                        gpu_durs.push(s.duration_us);
                        if s.fallback {
                            gpu_fallbacks += 1;
                        }
                    }
                }
            }

            let avg = |v: &Vec<u64>| {
                if v.is_empty() { None } else { Some(v.iter().copied().sum::<u64>() as f64 / v.len() as f64) }
            };

            let cpu_avg = avg(&cpu_durs);
            let gpu_avg = avg(&gpu_durs);

            // 1) Penalize GPU if there were many fallbacks.
            let gpu_reliable = gpu_fallbacks == 0;

            // 2) Choose the faster device when data is available.
            match (cpu_avg, gpu_avg) {
                (Some(c), Some(g)) if c < g && g > 0.0 => {
                    prefer_device = Some(DeviceTarget::CPU);
                }
                (Some(c), Some(g)) if g < c && gpu_reliable => {
                    prefer_device = Some(DeviceTarget::GPU);
                }
                _ => {}
            }

            // 3) Layout: if historically we never marked a contiguity benefit,
            // we can leave prefer_layout as None. For this minimal version, we
            // only suggest ForceContiguous if plan 5.3 already indicated it.
            if !info.contiguous {
                // If plan 5.3 already suggested contiguity (via static heuristic),
                // we respect that suggestion.
                // Real layout decision will be taken in merge_into_plan.
            }
        }

        AdaptiveDecision {
            prefer_device,
            prefer_layout,
        }
    }

    /// Return a device preference (CPU/GPU) based on aggregated statistics.
    /// If both averages are available, choose the one with lower duration.
    /// If there is not enough data, return None.
    pub fn device_preference_for(&self, _info: &NodeExecInfo) -> Option<DeviceTarget> {
        match (self.cpu_stats.avg(), self.gpu_stats.avg()) {
            (Some(cpu_avg), Some(gpu_avg)) => {
                if cpu_avg < gpu_avg {
                    Some(DeviceTarget::CPU)
                } else {
                    Some(DeviceTarget::GPU)
                }
            }
            _ => None,
        }
    }

    pub fn merge_into_plan(&self, info: &NodeExecInfo, plan: &mut ExecutionPlan5_3) {
        let decision = self.decide(info);

        if let Some(layout) = decision.prefer_layout {
            plan.layout = layout;
        }

        // Note: prefer_device is still not applied to real execution; it is
        // kept only as diagnostic information until the engine explicitly
        // connects this preference with APX 5.2.
    }
}
