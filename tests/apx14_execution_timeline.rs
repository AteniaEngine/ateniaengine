#[path = "../src/v14/mod.rs"]
mod v14;

use v14::timeline::execution_timeline::ExecutionTimeline;
use v14::timeline::timeline_event::TimelineEvent;

#[test]
fn records_events_in_order_with_monotonic_timestamps() {
    let mut timeline = ExecutionTimeline::new();

    timeline.record(TimelineEvent::DeviceSelected { device: "cpu0".to_string() });
    timeline.record(TimelineEvent::KernelStart { kernel_id: "k1".to_string(), device: "cpu0".to_string() });
    timeline.record(TimelineEvent::MemoryTransfer {
        src_device: "cpu0".to_string(),
        dst_device: "gpu0".to_string(),
        bytes: 1024,
    });
    timeline.record(TimelineEvent::KernelEnd { kernel_id: "k1".to_string(), device: "gpu0".to_string() });

    let events = timeline.events();
    assert_eq!(events.len(), 4);
    for window in events.windows(2) {
        assert!(window[0].timestamp < window[1].timestamp);
    }
}

#[test]
fn export_json_is_stable_and_structurally_correct() {
    let mut timeline = ExecutionTimeline::new();

    timeline.record(TimelineEvent::DeviceSelected { device: "cpu0".to_string() });
    timeline.record(TimelineEvent::KernelStart { kernel_id: "k1".to_string(), device: "cpu0".to_string() });
    timeline.record(TimelineEvent::MemoryTransfer {
        src_device: "cpu0".to_string(),
        dst_device: "gpu0".to_string(),
        bytes: 1024,
    });

    let json1 = timeline.export_json();
    let json2 = timeline.export_json();
    assert_eq!(json1, json2);

    let expected = "[{".to_string()
        + "\"timestamp\":0,\"kind\":\"DeviceSelected\",\"device\":\"cpu0\"},{"
        + "\"timestamp\":1,\"kind\":\"KernelStart\",\"kernel_id\":\"k1\",\"device\":\"cpu0\"},{"
        + "\"timestamp\":2,\"kind\":\"MemoryTransfer\",\"src_device\":\"cpu0\",\"dst_device\":\"gpu0\",\"bytes\":1024} ]";

    // Minimal structural validation: starts with '[' and ends with ']'.
    assert!(json1.starts_with('['));
    assert!(json1.ends_with(']'));

    // Ensure deterministic ordering of timestamps and kinds by simple substring checks.
    assert!(json1.contains("\"timestamp\":0,\"kind\":\"DeviceSelected"));
    assert!(json1.contains("\"timestamp\":1,\"kind\":\"KernelStart"));
    assert!(json1.contains("\"timestamp\":2,\"kind\":\"MemoryTransfer"));

    // The expected string is only used to ensure our format does not accidentally change
    // in field order or naming.
    let normalized_actual = json1.replace(" ", "");
    let normalized_expected = expected.replace(" ", "");
    assert_eq!(normalized_actual, normalized_expected);
}

#[test]
fn reset_clears_all_state() {
    let mut timeline = ExecutionTimeline::new();

    timeline.record(TimelineEvent::DeviceSelected { device: "cpu0".to_string() });
    assert_eq!(timeline.events().len(), 1);

    timeline.reset();
    assert_eq!(timeline.events().len(), 0);

    // After reset, timestamps should start again from zero.
    timeline.record(TimelineEvent::DeviceSelected { device: "cpu1".to_string() });
    let events = timeline.events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].timestamp, 0);
}
