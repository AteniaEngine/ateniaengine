use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::{HybridDispatcher, ExecDevice};
use atenia_engine::apx8::persistent::current_global_step;

#[test]
fn apx_8_5_persistence_basic() {
    let mut t = Tensor::randn(&[4, 4], Device::CPU);
    t.enable_gpu_persistence();
    assert!(t.persistence.is_some());
}

#[test]
fn apx_8_5_persistence_reuse_score() {
    let mut t = Tensor::randn(&[4, 4], Device::CPU);
    t.enable_gpu_persistence();

    // Simular múltiples usos GPU sin mover datos reales.
    t.note_gpu_use();
    t.note_gpu_use();

    let p = t.persistence.as_ref().unwrap();
    assert!(p.reuse_score >= 1);
}

#[test]
fn apx_8_5_persistence_mirror_survives_op() {
    unsafe { std::env::set_var("ATENIA_APX_MODE", "8.5"); }

    let mut t = Tensor::randn(&[8, 8], Device::CPU);
    t.ensure_gpu_mirror();
    t.enable_gpu_persistence();
    t.note_gpu_use();

    // Simular una "op" GPU vía HybridDispatcher
    let choice = HybridDispatcher::choose_device_for(&t);
    if choice == ExecDevice::GPU {
        assert!(t.gpu.is_some());
    }
}

#[test]
fn apx_8_5_persistence_cpu_equivalence() {
    let mut t = Tensor::randn(&[4, 4], Device::CPU);
    let before = t.data.clone();

    t.ensure_gpu_mirror();
    t.enable_gpu_persistence();
    t.note_gpu_use();
    t.sync_cpu();

    let after = t.data.clone();
    assert_eq!(before, after);
}

#[test]
fn apx_8_5_persistence_maybe_drop_gpu() {
    let mut t = Tensor::randn(&[4, 4], Device::CPU);
    t.ensure_gpu_mirror();
    t.enable_gpu_persistence();

    // Simular que el tensor casi no se reutilizó y que pasó mucho tiempo.
    if let Some(ref mut p) = t.persistence {
        p.reuse_score = 0;
        p.last_used_step = 0;
    }

    let current = current_global_step() + 100;
    t.maybe_drop_gpu(current, 10);

    assert!(t.gpu.is_none());
}
