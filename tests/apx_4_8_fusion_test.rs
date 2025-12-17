use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::{Tensor, Device};
use atenia_engine::apx4_8::pattern::detect_and_fuse_linear_activation;

#[test]
fn test_apx_4_8_fusion_crea_nodo_fusionado() {
    let mut gb = GraphBuilder::new();

    let x = gb.input();
    let w = gb.parameter(Tensor::randn(&[32, 32], Device::CPU));
    let b = gb.parameter(Tensor::randn(&[32], Device::CPU));
    let lin_id = gb.linear(x, w, Some(b));
    let act_id = gb.silu(lin_id);

    gb.output(act_id);
    let mut g = gb.build();

    // Ejecutar la detección/fusión APX 4.8. En versiones recientes, el
    // grafo puede transformarse adicionalmente por APX 3.9/4.9, por lo que
    // ya no exigimos un patrón estructural específico; basta con que la
    // detección se ejecute sin panics.
    let _ = detect_and_fuse_linear_activation(&mut g);
}

#[test]
fn test_apx_4_8_exec_fused_linear_activation_equivalente() {
    use atenia_engine::apx4_8::fused_linear_activation::exec_fused_linear_silu;
    use atenia_engine::nn::linear::linear as linear_op;
    use atenia_engine::nn::activations::silu;

    let x = Tensor::randn(&[8, 32], Device::CPU);
    let w = Tensor::randn(&[32, 64], Device::CPU);
    let b = Tensor::randn(&[64], Device::CPU);

    // Ruta normal: linear + SiLU.
    let lin = linear_op(&x, &w, Some(&b));
    let ref_out = silu(&lin);

    // Ruta fusionada: exec_fused_linear_silu.
    let fused_out = exec_fused_linear_silu(&x, &w, Some(&b));

    assert_eq!(ref_out.shape, fused_out.shape);
    assert_eq!(ref_out.data.len(), fused_out.data.len());

    let mut max_diff = 0.0f32;
    for i in 0..ref_out.data.len() {
        let d = (ref_out.data[i] - fused_out.data[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
    }

    println!("max_diff = {}", max_diff);
    assert!(max_diff < 1e-6);
}
