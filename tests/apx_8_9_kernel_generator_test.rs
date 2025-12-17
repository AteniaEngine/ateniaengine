use atenia_engine::apx8::kernel_generator::*;
use atenia_engine::apx8::kernel_registry::KERNEL_REGISTRY;
use atenia_engine::tensor::{Tensor, Device, DType};

#[test]
fn apx_8_9_basic_structure() {
    let tpl = GpuKernelTemplate {
        op: GpuKernelOp::VecAdd,
        block: (32, 1),
        tile_k: 32,
        use_shared: false,
        vectorize: false,
        unroll: 1,
    };

    assert_eq!(tpl.op, GpuKernelOp::VecAdd);
}

#[test]
fn apx_8_9_to_ir_conversion() {
    let tpl = GpuKernelTemplate {
        op: GpuKernelOp::MatMul,
        block: (32, 32),
        tile_k: 32,
        use_shared: true,
        vectorize: true,
        unroll: 2,
    };

    let ir = tpl.to_ir();
    assert_eq!(ir.params.get("tile_k").unwrap(), "32");
}

#[test]
fn apx_8_9_template_registry() {
    let tpl = GpuKernelTemplate {
        op: GpuKernelOp::VecAdd,
        block: (32, 1),
        tile_k: 32,
        use_shared: false,
        vectorize: false,
        unroll: 1,
    };

    KERNEL_REGISTRY.register_template(GpuKernelOp::VecAdd, tpl.clone());
    let retrieved = KERNEL_REGISTRY.get_template(&GpuKernelOp::VecAdd);
    assert!(retrieved.is_some());
}

#[test]
fn apx_8_9_no_math_change() {
    let a = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let b = Tensor::ones(vec![4], Device::CPU, DType::F32);

    let res = a.add(&b);

    for v in res.data.iter() {
        assert!((*v - 2.0).abs() < 1e-6);
    }
}
