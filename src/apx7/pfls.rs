use std::sync::{Mutex, OnceLock};

#[derive(Clone, Default)]
pub struct PFLSHistory {
    pub sl_durations: Vec<f64>,     // tiempo acumulado por SuperLevel
    pub sl_congestion: Vec<usize>,  // nodos activos por SuperLevel
    pub samples: usize,
}

impl PFLSHistory {
    pub fn record(&mut self, sl: usize, dur: f64, cong: usize) {
        if sl >= self.sl_durations.len() {
            self.sl_durations.resize(sl + 1, 0.0);
            self.sl_congestion.resize(sl + 1, 0);
        }
        self.sl_durations[sl] += dur;
        self.sl_congestion[sl] += cong;
        self.samples += 1;
    }

    pub fn predict_next_hotspot(&self) -> Option<usize> {
        if self.samples == 0 {
            return None;
        }
        let mut best = None;
        let mut best_score = 0.0;
        for sl in 0..self.sl_durations.len() {
            let score = self.sl_durations[sl] + self.sl_congestion[sl] as f64;
            if score > best_score {
                best_score = score;
                best = Some(sl);
            }
        }
        best
    }
}

static GLOBAL_PFLS: OnceLock<Mutex<PFLSHistory>> = OnceLock::new();

pub fn global_pfls() -> &'static Mutex<PFLSHistory> {
    GLOBAL_PFLS.get_or_init(|| Mutex::new(PFLSHistory::default()))
}
