use atenia_engine::apx8::kernel_generator::KernelIR;
use atenia_engine::apx8::gpu_compiler_stub::*;

#[test]
fn apx_8_11_compiler_basic() {
    let ir = KernelIR::mock_add();
    let mut comp = GpuCompilerStub::new();
    let c = comp.compile(&ir, GpuTarget::NvidiaPTX);
    assert!(c.binary_stub.contains("NvidiaPTX") || c.binary_stub.contains("PTX"));
}

#[test]
fn apx_8_11_compiler_cache() {
    let ir = KernelIR::mock_add();
    let mut comp = GpuCompilerStub::new();

    let c1 = comp.compile(&ir, GpuTarget::NvidiaPTX);
    let c2 = comp.compile(&ir, GpuTarget::NvidiaPTX);

    // Son valores clonados pero deben salir del mismo cache lógico.
    assert_eq!(c1.ir_hash, c2.ir_hash);
    assert!(comp.has_cache(&ir, GpuTarget::NvidiaPTX));
}

// Dispatcher/logging se valida de forma indirecta: no toca rutas de ejecución,
// así que no añadimos aserciones numéricas aquí.
