use atenia_engine::apx9::vgpu_memory::VGpuMemory;
use atenia_engine::apx9::vgpu_tensor_core::{VGPUTensorCore, TENSOR_CORE_TILE};
use atenia_engine::apx9::vgpu_instr::VGPUInstr;
use atenia_engine::{tensor::Tensor, tensor::DType, tensor::Device};

#[test]
fn apx_9_24_tensor_core_basic_structure() {
    assert!(TENSOR_CORE_TILE >= 2);
}

#[test]
fn apx_9_24_hmma_matches_cpu_matmul_2x2() {
    let mut mem = VGpuMemory::new(1024, 0, 1, 1);

    // A (2x2): [1 2; 3 4]
    let a_ptr = 0;
    mem.store_f32(a_ptr + 0, 1.0);
    mem.store_f32(a_ptr + 1, 2.0);
    mem.store_f32(a_ptr + 2, 3.0);
    mem.store_f32(a_ptr + 3, 4.0);

    // B (2x2): [5 6; 7 8]
    let b_ptr = 16;
    mem.store_f32(b_ptr + 0, 5.0);
    mem.store_f32(b_ptr + 1, 6.0);
    mem.store_f32(b_ptr + 2, 7.0);
    mem.store_f32(b_ptr + 3, 8.0);

    let c_ptr = 32;

    let instr = VGPUInstr::HMMA {
        a_ptr,
        b_ptr,
        c_ptr,
        m: 2,
        k: 2,
        n: 2,
    };

    let used = VGPUTensorCore::try_execute_ir(&mut mem, &instr);
    assert!(used);

    // C = A*B = [19 22; 43 50]
    let c00 = mem.load_f32(c_ptr + 0);
    let c01 = mem.load_f32(c_ptr + 1);
    let c10 = mem.load_f32(c_ptr + 2);
    let c11 = mem.load_f32(c_ptr + 3);

    assert!((c00 - 19.0).abs() < 1e-5);
    assert!((c01 - 22.0).abs() < 1e-5);
    assert!((c10 - 43.0).abs() < 1e-5);
    assert!((c11 - 50.0).abs() < 1e-5);
}

#[test]
fn apx_9_24_no_numeric_change() {
    let t = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let sum: f32 = t.data.iter().sum();
    assert!((sum - 4.0).abs() < 1e-6);
}
