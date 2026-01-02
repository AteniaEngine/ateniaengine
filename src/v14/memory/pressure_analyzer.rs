#![allow(dead_code)]

use super::pressure_snapshot::PressureSnapshot;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PressureTrend {
    Up,
    Stable,
    Down,
}

#[derive(Debug, Clone)]
pub struct AnalyzerResult {
    pub latest: Option<PressureSnapshot>,
    pub trend: Option<PressureTrend>,
}

#[derive(Debug, Clone)]
pub struct MemoryPressureAnalyzer {
    history: Vec<PressureSnapshot>,
    max_history: usize,
}

impl MemoryPressureAnalyzer {
    pub fn new(max_history: usize) -> Self {
        MemoryPressureAnalyzer {
            history: Vec::new(),
            max_history: max_history.max(1),
        }
    }

    pub fn record(&mut self, snapshot: PressureSnapshot) {
        self.history.push(snapshot);
        if self.history.len() > self.max_history {
            let overflow = self.history.len() - self.max_history;
            self.history.drain(0..overflow);
        }
    }

    pub fn reset(&mut self) {
        self.history.clear();
    }

    pub fn history(&self) -> &[PressureSnapshot] {
        &self.history
    }

    pub fn analyze(&self) -> AnalyzerResult {
        let latest = self.history.last().cloned();
        let trend = compute_trend(&self.history);
        AnalyzerResult { latest, trend }
    }
}

fn compute_trend(history: &[PressureSnapshot]) -> Option<PressureTrend> {
    if history.len() < 2 {
        return None;
    }

    let mut up = 0usize;
    let mut down = 0usize;

    for window in history.windows(2) {
        let prev = window[0].pressure_ratio;
        let next = window[1].pressure_ratio;

        if next > prev {
            up += 1;
        } else if next < prev {
            down += 1;
        }
    }

    if up == 0 && down == 0 {
        Some(PressureTrend::Stable)
    } else if up > down {
        Some(PressureTrend::Up)
    } else if down > up {
        Some(PressureTrend::Down)
    } else {
        Some(PressureTrend::Stable)
    }
}
