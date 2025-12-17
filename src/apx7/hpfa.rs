/// Afinidad de fusión para un nodo concreto según APX 6.9 / 6.10.
#[derive(Clone, Debug, Default)]
pub struct FusionAffinity {
    pub qkv_chain: bool,
    pub attn_fusable: bool,
    pub proj_fusable: bool,
    pub hot_factor: f64,
}

// En esta versión APX 7.7, HPFA sólo aporta señales de prioridad a HPGE v2.
// No modifica kernels, ni el grafo, ni el backward.
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

/// Helper de depuración usado sólo en tests para verificar el wiring de HPFA
/// dentro del scheduler de prioridades.
#[allow(dead_code)]
pub fn apply_hpfa_bonus(base: f64, affinity: &FusionAffinity) -> f64 {
    base + 0.2 * affinity.fusion_bonus()
}
