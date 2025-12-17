//! APX 7.x — Parallel Execution Engine (PEX)
//! Scheduler paralelo para tiles y kernels de MatMul.

use std::sync::Arc;
use std::thread;

use crossbeam_deque::{Injector, Steal};
use crate::apx7::dynamic_load::get_last_snapshot;

/// Ejecutor simple de tareas en paralelo (PEX v1).
pub struct PEXExecutor {
    _threads: usize,
}

impl PEXExecutor {
    pub fn new(threads: usize) -> Self {
        Self { _threads: threads }
    }

    pub fn execute_parallel<F>(&self, tasks: Vec<F>)
    where
        F: Fn() + Send + 'static + Clone,
    {
        let mut handles = Vec::new();
        for task in tasks {
            handles.push(thread::spawn(task));
        }
        for h in handles {
            let _ = h.join();
        }
    }
}

/// Tarea ejecutable por el scheduler PEX.
pub enum PEXTask {
    Tile(Box<dyn FnOnce() + Send>),
}

/// Work-stealing sencillo basado en una cola global compartida.
pub struct PEXWorkStealing {
    global: Arc<Injector<PEXTask>>,
    threads: usize,
}

impl PEXWorkStealing {
    pub fn new(threads: usize) -> Self {
        let global = Arc::new(Injector::new());
        Self { global, threads }
    }

    pub fn execute_parallel_ws(&self, tasks: Vec<PEXTask>) {
        // Sembrar las tareas iniciales en la cola global.
        for t in tasks {
            self.global.push(t);
        }

        let global = self.global.clone();

        // APX 7.4: adaptar el número de hilos usados según la carga
        // observada del sistema para no saturar la máquina cuando hay
        // carga externa significativa.
        let snap = get_last_snapshot();
        let threads_to_use = snap.threads_available.min(self.threads).max(1);

        thread::scope(|scope| {
            for _tid in 0..threads_to_use {
                let g = global.clone();

                scope.spawn(move || loop {
                    match g.steal() {
                        Steal::Success(task) => match task {
                            PEXTask::Tile(job) => job(),
                        },
                        Steal::Retry => continue,
                        Steal::Empty => break,
                    }
                });
            }
        });
    }
}
