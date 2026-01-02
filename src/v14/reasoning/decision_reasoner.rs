#![allow(dead_code)]

use super::decision_event::{DecisionEvent, DecisionEventKind};
use super::decision_record::DecisionRecord;
use super::reasoning_factors::ReasoningFactors;

#[derive(Debug, Clone)]
pub struct DecisionReasoner {
    records: Vec<DecisionRecord>,
    next_timestamp: u64,
}

impl DecisionReasoner {
    pub fn new() -> Self {
        DecisionReasoner {
            records: Vec::new(),
            next_timestamp: 0,
        }
    }

    pub fn record_decision(
        &mut self,
        event_kind: DecisionEventKind,
        decision_id: String,
        object_id: String,
        factors: ReasoningFactors,
        avoided_alternative: Option<String>,
        justification_code: u32,
    ) {
        let ts = self.next_timestamp;
        self.next_timestamp = self.next_timestamp.saturating_add(1);

        let event = DecisionEvent {
            id: decision_id,
            kind: event_kind,
            object_id,
            timestamp: ts,
        };

        let record = DecisionRecord::new(event, factors, avoided_alternative, justification_code, ts);
        self.records.push(record);
    }

    pub fn records(&self) -> &[DecisionRecord] {
        &self.records
    }

    pub fn reset(&mut self) {
        self.records.clear();
        self.next_timestamp = 0;
    }

    pub fn export_json(&self) -> String {
        let mut out = String::new();
        out.push('[');
        for (i, rec) in self.records.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            serialize_record(rec, &mut out);
        }
        out.push(']');
        out
    }
}

fn serialize_record(rec: &DecisionRecord, out: &mut String) {
    out.push('{');

    // Top-level fields
    out.push_str("\"timestamp\":");
    out.push_str(&rec.timestamp.to_string());
    out.push_str(",\"event\":");
    serialize_event(&rec.event, out);
    out.push_str(",\"factors\":");
    serialize_factors(&rec.factors, out);

    out.push_str(",\"avoided_alternative\":");
    match &rec.avoided_alternative {
        Some(s) => {
            out.push('"');
            escape_json_string(s, out);
            out.push('"');
        }
        None => out.push_str("null"),
    }

    out.push_str(",\"justification_code\":");
    out.push_str(&rec.justification_code.to_string());

    out.push('}');
}

fn serialize_event(ev: &DecisionEvent, out: &mut String) {
    out.push('{');
    out.push_str("\"id\":\"");
    escape_json_string(&ev.id, out);
    out.push_str("\",\"kind\":\"");
    let kind_str = match ev.kind {
        DecisionEventKind::DeviceSelection => "DeviceSelection",
        DecisionEventKind::TensorMovement => "TensorMovement",
        DecisionEventKind::KernelPlacement => "KernelPlacement",
        DecisionEventKind::FallbackAvoided => "FallbackAvoided",
    };
    out.push_str(kind_str);
    out.push_str("\",\"object_id\":\"");
    escape_json_string(&ev.object_id, out);
    out.push_str("\",\"timestamp\":");
    out.push_str(&ev.timestamp.to_string());
    out.push('}');
}

fn serialize_factors(f: &ReasoningFactors, out: &mut String) {
    out.push('{');

    // memory_snapshot
    out.push_str("\"memory_snapshot\":");
    if let Some(snap) = f.memory_snapshot {
        serialize_snapshot(&snap, out);
    } else {
        out.push_str("null");
    }

    // memory_risk
    out.push_str(",\"memory_risk\":");
    if let Some(risk) = f.memory_risk {
        let risk_str = match risk {
            crate::v14::memory::pressure_snapshot::MemoryRiskLevel::Safe => "Safe",
            crate::v14::memory::pressure_snapshot::MemoryRiskLevel::Warning => "Warning",
            crate::v14::memory::pressure_snapshot::MemoryRiskLevel::Critical => "Critical",
            crate::v14::memory::pressure_snapshot::MemoryRiskLevel::PreOOM => "PreOOM",
        };
        out.push('"');
        out.push_str(risk_str);
        out.push('"');
    } else {
        out.push_str("null");
    }

    // fragmentation_ratio
    out.push_str(",\"fragmentation_ratio\":");
    if let Some(r) = f.fragmentation_ratio {
        out.push_str(&r.to_string());
    } else {
        out.push_str("null");
    }

    // device_available
    out.push_str(",\"device_available\":");
    match f.device_available {
        Some(true) => out.push_str("true"),
        Some(false) => out.push_str("false"),
        None => out.push_str("null"),
    }

    // recent_decisions_count
    out.push_str(",\"recent_decisions_count\":");
    if let Some(c) = f.recent_decisions_count {
        out.push_str(&c.to_string());
    } else {
        out.push_str("null");
    }

    out.push('}');
}

fn serialize_snapshot(snap: &crate::v14::memory::pressure_snapshot::PressureSnapshot, out: &mut String) {
    out.push('{');
    out.push_str("\"layer\":\"");
    let layer_str = match snap.layer {
        crate::v14::memory::memory_layer::MemoryLayer::VRAM => "VRAM",
        crate::v14::memory::memory_layer::MemoryLayer::RAM => "RAM",
        crate::v14::memory::memory_layer::MemoryLayer::SSD => "SSD",
    };
    out.push_str(layer_str);
    out.push_str("\",\"used_bytes\":");
    out.push_str(&snap.used_bytes.to_string());
    out.push_str(",\"capacity_bytes\":");
    out.push_str(&snap.capacity_bytes.to_string());
    out.push_str(",\"pressure_ratio\":");
    out.push_str(&snap.pressure_ratio.to_string());
    out.push_str(",\"fragmentation_ratio\":");
    out.push_str(&snap.fragmentation_ratio.to_string());
    out.push_str(",\"risk_level\":\"");
    let risk_str = match snap.risk_level {
        crate::v14::memory::pressure_snapshot::MemoryRiskLevel::Safe => "Safe",
        crate::v14::memory::pressure_snapshot::MemoryRiskLevel::Warning => "Warning",
        crate::v14::memory::pressure_snapshot::MemoryRiskLevel::Critical => "Critical",
        crate::v14::memory::pressure_snapshot::MemoryRiskLevel::PreOOM => "PreOOM",
    };
    out.push_str(risk_str);
    out.push_str("\",\"timestamp\":");
    out.push_str(&snap.timestamp.to_string());
    out.push('}');
}

fn escape_json_string(input: &str, out: &mut String) {
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                use core::fmt::Write as _;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}
