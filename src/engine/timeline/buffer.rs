// src/engine/timeline/buffer.rs
// Circular buffer for timeline

use super::TimelineEvent;

pub struct TimelineBuffer {
    events: Vec<TimelineEvent>,
    capacity: usize,
}

impl TimelineBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, evt: TimelineEvent) {
        if self.events.len() == self.capacity {
            self.events.remove(0);
        }
        self.events.push(evt);
    }

    pub fn last(&self) -> Option<&TimelineEvent> {
        self.events.last()
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn all(&self) -> &[TimelineEvent] {
        &self.events
    }
}
