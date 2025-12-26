use std::collections::VecDeque;

use super::streams::{StreamConfig, StreamKind, StreamTask, TaskKind};

pub struct AsyncExecutor {
    cfg: StreamConfig,
    cpu_q: VecDeque<StreamTask>,
    gpu_q: VecDeque<StreamTask>,
    ssd_q: VecDeque<StreamTask>,
    pub timeline: Vec<String>,
    next_id: u64,
}

impl AsyncExecutor {
    pub fn new(cfg: StreamConfig) -> Self {
        AsyncExecutor {
            cfg,
            cpu_q: VecDeque::new(),
            gpu_q: VecDeque::new(),
            ssd_q: VecDeque::new(),
            timeline: Vec::new(),
            next_id: 1,
        }
    }

    pub fn submit(&mut self, stream: StreamKind, kind: TaskKind, cost: u64) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);

        let task = StreamTask {
            id,
            stream,
            kind: kind.clone(),
            estimated_cost: cost,
        };

        match stream {
            StreamKind::Cpu => self.cpu_q.push_back(task),
            StreamKind::Gpu => self.gpu_q.push_back(task),
            StreamKind::SsdPrefetch => self.ssd_q.push_back(task),
        }

        let (kind_label, name) = match kind {
            TaskKind::Compute { name } => ("Compute", name),
            TaskKind::Transfer { name } => ("Transfer", name),
            TaskKind::Io { name } => ("Io", name),
        };

        let stream_label = match stream {
            StreamKind::Cpu => "Cpu",
            StreamKind::Gpu => "Gpu",
            StreamKind::SsdPrefetch => "SsdPrefetch",
        };

        self.timeline.push(format!(
            "ENQUEUE stream={} id={} kind={} name={} cost={}",
            stream_label, id, kind_label, name, cost
        ));

        id
    }

    pub fn run_to_completion(&mut self) {
        if self.cfg.advanced_streams_supported {
            self.run_advanced();
        } else {
            self.run_fallback();
        }
    }

    fn run_advanced(&mut self) {
        let mut index = 0usize;
        while !(self.cpu_q.is_empty() && self.gpu_q.is_empty() && self.ssd_q.is_empty()) {
            let stream = match index % 3 {
                0 => StreamKind::Cpu,
                1 => StreamKind::Gpu,
                _ => StreamKind::SsdPrefetch,
            };
            index += 1;

            match stream {
                StreamKind::Cpu => {
                    if let Some(task) = self.cpu_q.pop_front() {
                        self.record_run(task, StreamKind::Cpu);
                    }
                }
                StreamKind::Gpu => {
                    if let Some(task) = self.gpu_q.pop_front() {
                        self.record_run(task, StreamKind::Gpu);
                    }
                }
                StreamKind::SsdPrefetch => {
                    if let Some(task) = self.ssd_q.pop_front() {
                        self.record_run(task, StreamKind::SsdPrefetch);
                    }
                }
            }
        }
    }

    fn run_fallback(&mut self) {
        // First drain CPU tasks as-is.
        while let Some(task) = self.cpu_q.pop_front() {
            self.record_run(task, StreamKind::Cpu);
        }

        // Then GPU tasks fall back to CPU.
        while let Some(task) = self.gpu_q.pop_front() {
            self.record_fallback_and_run(task, StreamKind::Gpu);
        }

        // Then SSD prefetch tasks fall back to CPU.
        while let Some(task) = self.ssd_q.pop_front() {
            self.record_fallback_and_run(task, StreamKind::SsdPrefetch);
        }
    }

    fn record_run(&mut self, task: StreamTask, stream: StreamKind) {
        let (kind_label, name) = match task.kind {
            TaskKind::Compute { name } => ("Compute", name),
            TaskKind::Transfer { name } => ("Transfer", name),
            TaskKind::Io { name } => ("Io", name),
        };

        let stream_label = match stream {
            StreamKind::Cpu => "Cpu",
            StreamKind::Gpu => "Gpu",
            StreamKind::SsdPrefetch => "SsdPrefetch",
        };

        self.timeline.push(format!(
            "RUN stream={} id={} kind={} name={} cost={}",
            stream_label, task.id, kind_label, name, task.estimated_cost
        ));
    }

    fn record_fallback_and_run(&mut self, task: StreamTask, original_stream: StreamKind) {
        let original_label = match original_stream {
            StreamKind::Cpu => "Cpu",
            StreamKind::Gpu => "Gpu",
            StreamKind::SsdPrefetch => "SsdPrefetch",
        };

        self.timeline.push(format!(
            "FALLBACK stream={} id={} -> Cpu",
            original_label, task.id
        ));

        // Execute as CPU in-order.
        self.record_run(task, StreamKind::Cpu);
    }
}
