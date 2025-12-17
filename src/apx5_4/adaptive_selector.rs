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
        // Heurística mínima basada en historial para la misma op/shape/dtype.
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

            // 1) Penalizar GPU si hubo muchos fallbacks.
            let gpu_reliable = gpu_fallbacks == 0;

            // 2) Elegir dispositivo más rápido cuando haya datos.
            match (cpu_avg, gpu_avg) {
                (Some(c), Some(g)) if c < g && g > 0.0 => {
                    prefer_device = Some(DeviceTarget::CPU);
                }
                (Some(c), Some(g)) if g < c && gpu_reliable => {
                    prefer_device = Some(DeviceTarget::GPU);
                }
                _ => {}
            }

            // 3) Layout: si históricamente nunca marcamos beneficio de contiguidad,
            // podemos dejar prefer_layout en None. Para esta versión mínima, sólo
            // sugerimos ForceContiguous si ya lo indicó el plan 5.3.
            if !info.contiguous {
                // Si el plan 5.3 ya sugería contiguidad (por heurística estática),
                // respetamos esa sugerencia.
                // Decisión real sobre layout se hará en merge_into_plan.
            }
        }

        AdaptiveDecision {
            prefer_device,
            prefer_layout,
        }
    }

    /// Devuelve una preferencia de dispositivo (CPU/GPU) basada en
    /// estadísticas agregadas. Si ambas medias están disponibles,
    /// elegimos la de menor duración. Si no hay datos suficientes,
    /// devolvemos None.
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

        // Nota: prefer_device todavía no se aplica a la ejecución real; se
        // mantiene sólo como información de diagnóstico hasta que el engine
        // conecte explícitamente esta preferencia con APX 5.2.
    }
}
