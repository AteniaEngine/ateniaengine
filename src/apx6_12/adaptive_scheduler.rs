use std::sync::RwLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdaptiveScheduleBias {
    None,
    QKVHeavy,
    AttentionHeavy,
}

static SCHED_BIAS: RwLock<AdaptiveScheduleBias> = RwLock::new(AdaptiveScheduleBias::None);

pub fn set_schedule_bias(bias: AdaptiveScheduleBias) {
    *SCHED_BIAS.write().unwrap() = bias;
}

pub fn get_schedule_bias() -> AdaptiveScheduleBias {
    *SCHED_BIAS.read().unwrap()
}
