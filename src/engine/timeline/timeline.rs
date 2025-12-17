// src/engine/timeline/timeline.rs

use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use crate::engine::timeline::buffer::TimelineBuffer;
use crate::engine::timeline::event::TimelineEvent;
use crate::engine::fingerprint::ExecFingerprint;

#[derive(Clone)]
pub struct LaunchTimeline {
    buffer: Arc<Mutex<TimelineBuffer>>,
}

impl LaunchTimeline {
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(TimelineBuffer::new(2048))),
        }
    }

    pub fn record(
        &self,
        name: impl Into<String>,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        params: usize,
    ) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_micros();

        let evt = TimelineEvent::new(name, grid, block, shared_mem, params, now);

        if let Ok(mut guard) = self.buffer.lock() {
            guard.push(evt);
        }
    }

    pub fn last(&self) -> Option<TimelineEvent> {
        self.buffer.lock().ok()?.last().cloned()
    }

    pub fn len(&self) -> usize {
        self.buffer.lock().map(|b| b.len()).unwrap_or(0)
    }

    pub fn last_fingerprint(&self) -> Option<u64> {
        let evt = self.last()?;
        let fp = ExecFingerprint::new(
            evt.name,
            evt.grid,
            evt.block,
            evt.shared_mem,
            evt.params,
        );
        Some(fp.hash64())
    }
}
