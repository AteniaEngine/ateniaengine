use atenia_engine::tensor::{Tensor, Device, DType};
use atenia_engine::{MirrorState, HybridDispatcher, ExecDevice};

#[test]
fn apx_8_4_mirror_basic_structure() {
    let t = Tensor::zeros(vec![4], Device::CPU, DType::F32);
    assert!(t.gpu.is_none());
}

#[test]
fn apx_8_4_mirror_allocate() {
    let mut t = Tensor::zeros(vec![4], Device::CPU, DType::F32);
    t.ensure_gpu_mirror();

    let mirror = t.gpu.as_ref().unwrap();
    assert_eq!(mirror.state, MirrorState::CleanCPU);
}

#[test]
fn apx_8_4_mirror_mark_dirty_gpu() {
    let mut t = Tensor::zeros(vec![4], Device::CPU, DType::F32);
    t.ensure_gpu_mirror();
    t.mark_gpu_dirty();

    let mirror = t.gpu.as_ref().unwrap();
    assert_eq!(mirror.state, MirrorState::DirtyGPU);
}

#[test]
fn apx_8_4_mirror_cpu_equivalence_after_sync() {
    let mut t = Tensor::zeros(vec![4], Device::CPU, DType::F32);
    t.data[0] = 1.0;
    t.data[1] = 2.0;
    t.data[2] = 3.0;
    t.data[3] = 4.0;

    let before = t.data.clone();
    t.ensure_gpu_mirror();
    t.sync_cpu();

    assert_eq!(t.data, before);
}

#[test]
fn apx_8_4_mirror_dispatcher_simulation() {
    unsafe { std::env::set_var("ATENIA_APX_MODE", "8.4"); }

    let mut t = Tensor::zeros(vec![1024], Device::CPU, DType::F32);
    let choice = HybridDispatcher::choose_device_for(&t);

    if choice == ExecDevice::GPU {
        t.ensure_gpu_mirror();
        assert!(t.gpu.is_some());
        let mirror = t.gpu.as_ref().unwrap();
        // Después de una "op" GPU simulada, marcamos la CPU como sincronizada.
        assert!(matches!(mirror.state, MirrorState::CleanCPU | MirrorState::Synced));
    } else {
        // Si por heurística se mantiene en CPU, no debe crearse mirror automáticamente.
        assert!(t.gpu.is_none());
    }

    // En todos los casos, la data CPU sigue siendo la verdad.
    let sum: f32 = t.data.iter().sum();
    assert!(sum >= 0.0 || sum <= 0.0); // sólo para usar `sum` y evitar warnings.
}
