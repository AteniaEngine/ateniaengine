#[derive(Clone, Copy, Debug)]
pub struct TemperedDecision {
    pub p_full: f32,
    pub p_qkv: f32,
    pub p_base: f32,
}

pub fn softmax3(full: f32, qkv: f32, base: f32, temperature: f32) -> TemperedDecision {
    let t = temperature.max(0.05);

    let ef = (full / t).exp();
    let eq = (qkv / t).exp();
    let eb = (base / t).exp();

    let sum = ef + eq + eb;

    TemperedDecision {
        p_full: ef / sum,
        p_qkv: eq / sum,
        p_base: eb / sum,
    }
}

pub fn sample_decision(td: &TemperedDecision) -> &'static str {
    let r: f32 = rand::random();

    if r < td.p_full {
        "full"
    } else if r < td.p_full + td.p_qkv {
        "qkv"
    } else {
        "base"
    }
}
