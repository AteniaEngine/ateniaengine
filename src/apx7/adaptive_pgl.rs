// APX 7.3: Adaptive PGL (runtime self-tuning)

use std::sync::RwLock;

#[derive(Debug, Clone, Copy)]
pub struct TimingStats {
    pub count: u32,
    pub sum_seq: f64,
    pub sum_pex: f64,
    pub sum_ws: f64,
}

impl TimingStats {
    pub const fn new() -> Self {
        Self {
            count: 0,
            sum_seq: 0.0,
            sum_pex: 0.0,
            sum_ws: 0.0,
        }
    }

    pub fn record(&mut self, t_seq: f64, t_pex: f64, t_ws: f64) {
        self.count = self.count.saturating_add(1);
        self.sum_seq += t_seq;
        self.sum_pex += t_pex;
        self.sum_ws += t_ws;
    }

    pub fn avg(&self) -> Option<(f64, f64, f64)> {
        if self.count < 5 {
            return None; // minimal stability
        }
        let c = self.count as f64;
        Some((self.sum_seq / c, self.sum_pex / c, self.sum_ws / c))
    }
}

// Buckets by approximate size
pub static ADAPTIVE_BUCKETS: RwLock<[TimingStats; 3]> =
    RwLock::new([TimingStats::new(), TimingStats::new(), TimingStats::new()]);

pub fn bucket_for(n: usize) -> usize {
    if n <= 512 {
        0
    } else if n <= 1536 {
        1
    } else {
        2
    }
}
