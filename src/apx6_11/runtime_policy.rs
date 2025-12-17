use std::sync::{RwLock, OnceLock};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionRuntimePolicy {
    Baseline,
    PreferQKV,
    PreferFull,
}

static RUNTIME_POLICY: OnceLock<RwLock<FusionRuntimePolicy>> = OnceLock::new();

fn global_runtime_policy() -> &'static RwLock<FusionRuntimePolicy> {
    RUNTIME_POLICY.get_or_init(|| RwLock::new(FusionRuntimePolicy::Baseline))
}

pub fn set_runtime_policy(policy: FusionRuntimePolicy) {
    if let Ok(mut guard) = global_runtime_policy().write() {
        *guard = policy;
    }
}

pub fn get_runtime_policy() -> FusionRuntimePolicy {
    global_runtime_policy()
        .read()
        .map(|g| *g)
        .unwrap_or(FusionRuntimePolicy::Baseline)
}
