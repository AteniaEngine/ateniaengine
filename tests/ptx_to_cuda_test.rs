use atenia_engine::gpu::translator::PtxToCuda;

#[test]
fn test_translation_basic() {
    let ptx = r#"
        .reg .f32 r1;
        r1 = x[i] + 1;
        x[i] = r1;
    "#;

    let cuda = PtxToCuda::translate(ptx, "test_kernel").unwrap();

    assert!(cuda.contains("extern \"C\" __global__ void test_kernel"));
    assert!(cuda.contains("float r1;"));
    assert!(cuda.contains("x[i] = r1;"));
}
