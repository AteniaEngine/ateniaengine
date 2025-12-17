use std::sync::{Mutex, OnceLock};

pub mod runtime_flags;

pub use runtime_flags::RuntimeFlags;

static RUNTIME_FLAGS: OnceLock<Mutex<RuntimeFlags>> = OnceLock::new();

fn global_runtime_flags() -> &'static Mutex<RuntimeFlags> {
    RUNTIME_FLAGS.get_or_init(|| Mutex::new(RuntimeFlags::default()))
}

pub fn get_runtime_flags() -> std::sync::MutexGuard<'static, RuntimeFlags> {
    global_runtime_flags().lock().unwrap()
}
