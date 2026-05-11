//! Adaptive Memory Manager: forecasting, offloading, checkpointing, and batching.

pub mod batch_manager;
pub mod battery_probe;
pub mod checkpointing;
pub mod cpu_probe;
pub mod failure_counter;
pub mod forecaster;
pub mod foreground_probe;
pub mod fp8_manager;
pub mod gpu_util_probe;
pub mod latency_monitor;
pub mod memory_manager;
pub mod offloading;
pub mod ram_probe;
pub mod signal_bus;
pub mod vram_probe;
