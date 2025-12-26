use atenia_engine::v13::async_executor::AsyncExecutor;
use atenia_engine::v13::streams::{StreamConfig, StreamKind, TaskKind};

fn make_executor(advanced: bool) -> AsyncExecutor {
    let cfg = StreamConfig {
        advanced_streams_supported: advanced,
    };
    AsyncExecutor::new(cfg)
}

#[test]
fn advanced_streams_round_robin_order() {
    let mut ex = make_executor(true);

    let _ = ex.submit(
        StreamKind::Cpu,
        TaskKind::Compute {
            name: "cpu_task1".to_string(),
        },
        1,
    );
    let _ = ex.submit(
        StreamKind::Gpu,
        TaskKind::Compute {
            name: "gpu_task1".to_string(),
        },
        2,
    );
    let _ = ex.submit(
        StreamKind::SsdPrefetch,
        TaskKind::Io {
            name: "ssd_task1".to_string(),
        },
        3,
    );
    let _ = ex.submit(
        StreamKind::Cpu,
        TaskKind::Compute {
            name: "cpu_task2".to_string(),
        },
        4,
    );

    ex.run_to_completion();

    // We expect RUN entries to follow a round-robin pattern per stream.
    let runs: Vec<&String> = ex
        .timeline
        .iter()
        .filter(|l| l.starts_with("RUN"))
        .collect();

    assert!(runs.len() >= 4);

    // First three runs should be Cpu, Gpu, SsdPrefetch in order.
    assert!(runs[0].contains("RUN stream=Cpu"));
    assert!(runs[1].contains("RUN stream=Gpu"));
    assert!(runs[2].contains("RUN stream=SsdPrefetch"));

    // Per-stream FIFO ordering: cpu_task1 before cpu_task2.
    let cpu1 = runs
        .iter()
        .position(|l| l.contains("name=cpu_task1"));
    let cpu2 = runs
        .iter()
        .position(|l| l.contains("name=cpu_task2"));

    match (cpu1, cpu2) {
        (Some(i1), Some(i2)) => {
            assert!(i1 < i2);
        }
        _ => panic!("expected both cpu_task1 and cpu_task2 to be present"),
    }
}

#[test]
fn fallback_serializes_all_to_cpu() {
    let mut ex = make_executor(false);

    let _ = ex.submit(
        StreamKind::Gpu,
        TaskKind::Compute {
            name: "gpu_task1".to_string(),
        },
        10,
    );
    let _ = ex.submit(
        StreamKind::SsdPrefetch,
        TaskKind::Io {
            name: "ssd_task1".to_string(),
        },
        20,
    );

    ex.run_to_completion();

    let mut saw_fallback_gpu = false;
    let mut saw_fallback_ssd = false;

    for entry in &ex.timeline {
        if entry.starts_with("FALLBACK") {
            if entry.contains("stream=Gpu") {
                saw_fallback_gpu = true;
            }
            if entry.contains("stream=SsdPrefetch") {
                saw_fallback_ssd = true;
            }
        }
        if entry.starts_with("RUN") {
            assert!(entry.contains("stream=Cpu"));
        }
    }

    assert!(saw_fallback_gpu);
    assert!(saw_fallback_ssd);
}

#[test]
fn ids_are_monotonic() {
    let mut ex = make_executor(true);

    let id1 = ex.submit(
        StreamKind::Cpu,
        TaskKind::Compute {
            name: "t1".to_string(),
        },
        1,
    );
    let id2 = ex.submit(
        StreamKind::Gpu,
        TaskKind::Transfer {
            name: "t2".to_string(),
        },
        1,
    );
    let id3 = ex.submit(
        StreamKind::SsdPrefetch,
        TaskKind::Io {
            name: "t3".to_string(),
        },
        1,
    );

    assert!(id1 < id2);
    assert!(id2 < id3);
}
