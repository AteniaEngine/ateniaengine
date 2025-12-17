#[derive(Clone, Debug)]
pub struct OpStats {
    pub durations: Vec<u64>,
    pub fallbacks: u32,
}

impl OpStats {
    pub fn new() -> Self {
        Self {
            durations: Vec::new(),
            fallbacks: 0,
        }
    }

    pub fn record(&mut self, duration_us: u64, fallback: bool) {
        self.durations.push(duration_us);
        if fallback {
            self.fallbacks += 1;
        }
    }

    pub fn avg(&self) -> Option<f64> {
        if self.durations.is_empty() {
            None
        } else {
            let sum: u64 = self.durations.iter().copied().sum();
            Some(sum as f64 / self.durations.len() as f64)
        }
    }
}
