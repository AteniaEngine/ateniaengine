#[derive(Clone, Copy, Debug)]
pub struct TemperatureSchedule {
    pub t0: f32,
    pub t_min: f32,
    pub decay: f32,
}

impl TemperatureSchedule {
    pub fn value_at(&self, step: usize) -> f32 {
        let t = self.t0 * (-self.decay * step as f32).exp();
        t.max(self.t_min)
    }
}

use std::sync::RwLock;

static TEMP_STATE: RwLock<f32> = RwLock::new(1.0);

pub fn set_current_temperature(t: f32) {
    *TEMP_STATE.write().unwrap() = t;
}

pub fn get_current_temperature() -> f32 {
    *TEMP_STATE.read().unwrap()
}
