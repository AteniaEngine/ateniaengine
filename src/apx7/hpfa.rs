/// Fusion affinity for a specific node according to APX 6.9 / 6.10.
#[derive(Clone, Debug, Default)]
pub struct FusionAffinity {
    pub qkv_chain: bool,
    pub attn_fusable: bool,
    pub proj_fusable: bool,
    pub hot_factor: f64,
}

// In this APX 7.7 version, HPFA only provides priority signals to HPGE v2.
// It does not modify kernels, the graph, or backward.
impl FusionAffinity {
    pub fn fusion_bonus(&self) -> f64 {
        let mut bonus = 0.0;
        if self.qkv_chain {
            bonus += 20.0;
        }
        if self.attn_fusable {
            bonus += 30.0;
        }
        if self.proj_fusable {
            bonus += 10.0;
        }
        bonus + 0.1 * self.hot_factor
    }
}

/// Debug helper used only in tests to verify HPFA wiring inside the
/// priority scheduler.
#[allow(dead_code)]
pub fn apply_hpfa_bonus(base: f64, affinity: &FusionAffinity) -> f64 {
    base + 0.2 * affinity.fusion_bonus()
}
