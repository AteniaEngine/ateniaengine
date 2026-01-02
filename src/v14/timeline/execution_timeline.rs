#![allow(dead_code)]

use super::timeline_event::TimelineEvent;

#[derive(Debug, Default)]
pub struct ExecutionTimeline {
    events: Vec<RecordedEvent>,
    next_timestamp: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedEvent {
    pub timestamp: u64,
    pub event: TimelineEvent,
}

impl ExecutionTimeline {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            next_timestamp: 0,
        }
    }

    pub fn record(&mut self, event: TimelineEvent) {
        let ts = self.next_timestamp;
        self.next_timestamp = self.next_timestamp.saturating_add(1);
        self.events.push(RecordedEvent { timestamp: ts, event });
    }

    pub fn reset(&mut self) {
        self.events.clear();
        self.next_timestamp = 0;
    }

    pub fn events(&self) -> &[RecordedEvent] {
        &self.events
    }

    pub fn export_json(&self) -> String {
        let mut out = String::new();
        out.push('[');
        for (i, rec) in self.events.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push('{');
            // common fields
            out.push_str("\"timestamp\":");
            out.push_str(&rec.timestamp.to_string());
            out.push_str(",\"kind\":");
            match &rec.event {
                TimelineEvent::KernelStart { kernel_id, device } => {
                    out.push('"');
                    out.push_str("KernelStart");
                    out.push_str("\",");
                    out.push_str("\"kernel_id\":\"");
                    escape_json_string(kernel_id, &mut out);
                    out.push_str("\",\"device\":\"");
                    escape_json_string(device, &mut out);
                    out.push_str("\"");
                }
                TimelineEvent::KernelEnd { kernel_id, device } => {
                    out.push('"');
                    out.push_str("KernelEnd");
                    out.push_str("\",");
                    out.push_str("\"kernel_id\":\"");
                    escape_json_string(kernel_id, &mut out);
                    out.push_str("\",\"device\":\"");
                    escape_json_string(device, &mut out);
                    out.push_str("\"");
                }
                TimelineEvent::DeviceSelected { device } => {
                    out.push('"');
                    out.push_str("DeviceSelected");
                    out.push_str("\",");
                    out.push_str("\"device\":\"");
                    escape_json_string(device, &mut out);
                    out.push_str("\"");
                }
                TimelineEvent::MemoryTransfer { src_device, dst_device, bytes } => {
                    out.push('"');
                    out.push_str("MemoryTransfer");
                    out.push_str("\",");
                    out.push_str("\"src_device\":\"");
                    escape_json_string(src_device, &mut out);
                    out.push_str("\",\"dst_device\":\"");
                    escape_json_string(dst_device, &mut out);
                    out.push_str("\",\"bytes\":");
                    out.push_str(&bytes.to_string());
                }
            }
            out.push('}');
        }
        out.push(']');
        out
    }
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
