use std::sync::RwLock;
use sysinfo::System;

#[derive(Debug, Clone, Copy)]
pub struct LoadSnapshot {
    pub cpu_load: f32,       // porcentaje
    pub threads_available: usize,
}

pub static LAST_SNAPSHOT: RwLock<LoadSnapshot> = RwLock::new(LoadSnapshot {
    cpu_load: 0.0,
    threads_available: 1,
});

pub fn sample_system_load() -> LoadSnapshot {
    let mut sys = System::new();
    sys.refresh_cpu();

    let mut load = 0.0f32;
    let cpus = sys.cpus();
    if !cpus.is_empty() {
        for cpu in cpus {
            load += cpu.cpu_usage() as f32;
        }
        load /= cpus.len() as f32;
    }

    let total_threads = num_cpus::get_physical().max(1);
    let mut threads_available = ((1.0 - (load / 100.0)) * total_threads as f32).round() as usize;
    if threads_available < 1 {
        threads_available = 1;
    }

    let snap = LoadSnapshot {
        cpu_load: load,
        threads_available,
    };

    *LAST_SNAPSHOT.write().unwrap() = snap;

    snap
}

pub fn get_last_snapshot() -> LoadSnapshot {
    *LAST_SNAPSHOT.read().unwrap()
}

/// Estrategia de scheduling sugerida según el snapshot actual.
/// Devuelve "seq", "pex", "ws" o "pgl" (fallback heurístico).
pub fn choose_strategy(snap: &LoadSnapshot) -> &'static str {
    if snap.cpu_load > 85.0 {
        return "seq";
    }

    if snap.threads_available <= 4 {
        return "pex";
    }

    if snap.threads_available >= 12 {
        return "ws";
    }

    "pgl"
}
