#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintSeverity {
    Hard,
    Soft,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintKind {
    ForbidOffload,
    LimitAggressiveness { max: f32 },
    RequireStability,
    RequireFallback,
    MemoryHeadroom { min: f32 },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    pub kind: ConstraintKind,
    pub severity: ConstraintSeverity,
}

impl Constraint {
    pub fn hard(kind: ConstraintKind) -> Self {
        Self { kind, severity: ConstraintSeverity::Hard }
    }

    pub fn soft(kind: ConstraintKind) -> Self {
        Self { kind, severity: ConstraintSeverity::Soft }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Constraints {
    pub items: Vec<Constraint>,
}

/// Passive snapshot of relevant runtime state for contract resolution.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeState {
    /// Normalized memory headroom in [0.0, 1.0].
    pub memory_headroom: f32,
    /// Whether the system is currently considered stable.
    pub is_stable: bool,
    /// Whether a recovery happened recently.
    pub recent_recovery: bool,
    /// Whether offload-style backends are available at all.
    pub offload_supported: bool,
}
