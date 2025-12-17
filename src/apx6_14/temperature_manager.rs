use crate::apx6_14::temperature_schedule::TemperatureSchedule;
use crate::apx6_14::temperature_schedule::set_current_temperature;

static SCHED: TemperatureSchedule = TemperatureSchedule {
    t0: 1.2,
    t_min: 0.25,
    decay: 0.0005,
};

pub fn update_temperature(step: usize) {
    let t = SCHED.value_at(step);
    set_current_temperature(t);
}
