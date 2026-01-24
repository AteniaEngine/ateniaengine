#![allow(dead_code)]

/// Explicit model of user preferences. All preferences are soft and never
/// interpreted as hard orders or hardware-specific directives.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct UserPreferences {
    pub prefer_latency: bool,
    pub avoid_ssd: bool,
    pub prioritize_stability: bool,
    pub minimize_power: bool,
    pub prefer_gpu: bool,
}
