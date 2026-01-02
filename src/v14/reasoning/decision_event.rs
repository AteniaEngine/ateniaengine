#![allow(dead_code)]

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionEventKind {
    DeviceSelection,
    TensorMovement,
    KernelPlacement,
    FallbackAvoided,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionEvent {
    pub id: String,
    pub kind: DecisionEventKind,
    pub object_id: String,
    pub timestamp: u64,
}
