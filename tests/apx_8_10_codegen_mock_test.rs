use atenia_engine::apx8::codegen_mock::*;

#[test]
fn apx_8_10_codegen_structure() {
    let tpl = KernelTemplate {
        op_name: "vec_add",
        arity: 2,
        body_ir: "ADD a b".into(),
    };
    let cuda = CudaCodegen.generate(&tpl);
    assert!(cuda.contains("vec_add"));
    assert!(cuda.contains("__global__"));
}

#[test]
fn apx_8_10_codegen_semantic_equivalence() {
    let tpl = KernelTemplate {
        op_name: "vec_add",
        arity: 2,
        body_ir: "ADD a b".into(),
    };

    let cuda = CudaCodegen.generate(&tpl);
    let metal = MetalCodegen.generate(&tpl);

    assert!(cuda.contains("vec_add"));
    assert!(cuda.contains("ADD a b"));
    assert!(metal.contains("vec_add"));
    assert!(metal.contains("ADD a b"));
}

#[test]
fn apx_8_10_codegen_is_mock() {
    let tpl = KernelTemplate {
        op_name: "vec_add",
        arity: 2,
        body_ir: "ADD a b".into(),
    };

    let cuda = CudaCodegen.generate(&tpl);
    let metal = MetalCodegen.generate(&tpl);

    assert!(cuda.contains("mock"));
    assert!(metal.contains("mock"));
}
