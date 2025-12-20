//! APX 7.x — Parallel Execution Engine (PEX)
//! Parallel scheduler for MatMul tiles and kernels.

use std::sync::Arc;
use std::thread;

use crossbeam_deque::{Injector, Steal};
use crate::apx7::dynamic_load::get_last_snapshot;

/// Simple parallel task executor (PEX v1).
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

/// Executable task for the PEX scheduler.
pub enum PEXTask {
    Tile(Box<dyn FnOnce() + Send>),
}

/// Simple work-stealing based on a shared global queue.
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
        // Seed initial tasks into the global queue.
        for t in tasks {
            self.global.push(t);
        }

        let global = self.global.clone();

        // APX 7.4: adapt the number of threads used based on observed system
        // load to avoid saturating the machine when there is significant
        // external load.
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
