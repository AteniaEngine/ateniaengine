use atenia_engine::{
    VirtualGpuExecutor, VirtualKernel,
    tensor::{Tensor, Device, DType}
};

#[test]
fn apx_9_12_structure() {
    let _exec = VirtualGpuExecutor::new();
    assert!(true); // smoke
}

#[test]
fn apx_9_12_no_numeric_change() {
    let a = Tensor::ones(vec![8], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![8], Device::CPU, DType::F32);
    let c = Tensor::zeros(vec![8], Device::CPU, DType::F32);

    // Pseudo-PTX que marca el kernel como VECADD
    let ptx = String::from("// VECADD kernel");

    let mut k = VirtualKernel {
        ptx,
        threads_per_block: 32,
        blocks: 1,
        args: vec![a.clone(), b.clone(), c.clone()],
    };

    let exec = VirtualGpuExecutor::new();
    exec.launch(&mut k);

    for i in 0..8 {
        assert!((k.args[2].data[i] - 2.0).abs() < 1e-6);
    }
}

#[test]
fn apx_9_12_launch_thread_indexing() {
    let ptx = String::from("// VECADD kernel");
    let out = Tensor::zeros(vec![16], Device::CPU, DType::F32);

    // Construir tensores de entrada [0, 1, ..., 15]
    let mut a = Tensor::zeros(vec![16], Device::CPU, DType::F32);
    let mut b = Tensor::zeros(vec![16], Device::CPU, DType::F32);
    for i in 0..16 {
        a.data[i] = i as f32;
        b.data[i] = i as f32;
    }

    let mut k = VirtualKernel {
        ptx,
        threads_per_block: 8,
        blocks: 2,
        args: vec![
            a,
            b,
            out.clone(),
        ],
    };

    let exec = VirtualGpuExecutor::new();
    exec.launch(&mut k);

    for i in 0..16 {
        assert!((k.args[2].data[i] - (i as f32 * 2.0)).abs() < 1e-6);
    }
}
