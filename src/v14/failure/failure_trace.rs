#![allow(dead_code)]

use super::failure_event::{FailureEvent, FailureSeverity};
use super::failure_kind::FailureKind;
use super::recovery_action::RecoveryAction;
use super::recovery_record::{RecoveryRecord, RecoveryResult};

#[derive(Debug, Default)]
pub struct FailureTrace {
    records: Vec<RecoveryRecord>,
    next_timestamp: u64,
}

impl FailureTrace {
    pub fn new() -> Self {
        FailureTrace {
            records: Vec::new(),
            next_timestamp: 0,
        }
    }

    pub fn record_failure_with_recovery(
        &mut self,
        kind: FailureKind,
        message: String,
        device: Option<String>,
        tensor_id: Option<String>,
        kernel_id: Option<String>,
        severity: FailureSeverity,
        action_taken: RecoveryAction,
        action_reason: String,
        result: RecoveryResult,
    ) {
        let ts = self.next_timestamp;
        self.next_timestamp = self.next_timestamp.saturating_add(1);

        let failure_event = FailureEvent::new(
            kind,
            ts,
            message,
            device,
            tensor_id,
            kernel_id,
            severity,
        );

        let record = RecoveryRecord::new(failure_event, action_taken, action_reason, result, ts);
        self.records.push(record);
    }

    pub fn records(&self) -> &[RecoveryRecord] {
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

fn serialize_record(rec: &RecoveryRecord, out: &mut String) {
    out.push('{');

    out.push_str("\"timestamp\":");
    out.push_str(&rec.timestamp.to_string());

    out.push_str(",\"failure_event\":");
    serialize_failure_event(&rec.failure_event, out);

    out.push_str(",\"action_taken\":\"");
    let action_str = match rec.action_taken {
        RecoveryAction::Retry => "Retry",
        RecoveryAction::FallbackToCPU => "FallbackToCPU",
        RecoveryAction::MoveTensorToRAM => "MoveTensorToRAM",
        RecoveryAction::MoveTensorToSSD => "MoveTensorToSSD",
        RecoveryAction::ReduceBatch => "ReduceBatch",
        RecoveryAction::SkipKernel => "SkipKernel",
        RecoveryAction::Abort => "Abort",
        RecoveryAction::None => "None",
    };
    out.push_str(action_str);
    out.push('"');

    out.push_str(",\"action_reason\":\"");
    escape_json_string(&rec.action_reason, out);
    out.push('"');

    out.push_str(",\"result\":\"");
    let result_str = match rec.result {
        RecoveryResult::Recovered => "Recovered",
        RecoveryResult::Degraded => "Degraded",
        RecoveryResult::Failed => "Failed",
        RecoveryResult::Avoided => "Avoided",
    };
    out.push_str(result_str);
    out.push('"');

    out.push('}');
}

fn serialize_failure_event(ev: &FailureEvent, out: &mut String) {
    out.push('{');

    out.push_str("\"kind\":\"");
    let kind_str = match ev.kind {
        FailureKind::OutOfMemoryRisk => "OutOfMemoryRisk",
        FailureKind::OutOfMemory => "OutOfMemory",
        FailureKind::KernelLaunchFailure => "KernelLaunchFailure",
        FailureKind::DeviceUnavailable => "DeviceUnavailable",
        FailureKind::TransferFailure => "TransferFailure",
        FailureKind::Unknown => "Unknown",
    };
    out.push_str(kind_str);
    out.push('"');

    out.push_str(",\"timestamp\":");
    out.push_str(&ev.timestamp.to_string());

    out.push_str(",\"message\":\"");
    escape_json_string(&ev.message, out);
    out.push('"');

    out.push_str(",\"device\":");
    match &ev.device {
        Some(d) => {
            out.push('"');
            escape_json_string(d, out);
            out.push('"');
        }
        None => out.push_str("null"),
    }

    out.push_str(",\"tensor_id\":");
    match &ev.tensor_id {
        Some(t) => {
            out.push('"');
            escape_json_string(t, out);
            out.push('"');
        }
        None => out.push_str("null"),
    }

    out.push_str(",\"kernel_id\":");
    match &ev.kernel_id {
        Some(k) => {
            out.push('"');
            escape_json_string(k, out);
            out.push('"');
        }
        None => out.push_str("null"),
    }

    out.push_str(",\"severity\":\"");
    let sev_str = match ev.severity {
        FailureSeverity::Info => "Info",
        FailureSeverity::Warning => "Warning",
        FailureSeverity::Critical => "Critical",
    };
    out.push_str(sev_str);
    out.push('"');

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
