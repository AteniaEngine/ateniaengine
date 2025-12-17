use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::{Tensor, Device};

#[test]
fn test_apx_4_7_fuses_linear_linear() {
    let mut gb = GraphBuilder::new();

    let x = gb.input();
    let w1 = gb.parameter(Tensor::randn(&[32, 32], Device::CPU));
    let b1 = gb.parameter(Tensor::randn(&[32], Device::CPU));
    let l1 = gb.linear(x, w1, Some(b1));

    let w2 = gb.parameter(Tensor::randn(&[32, 32], Device::CPU));
    let b2 = gb.parameter(Tensor::randn(&[32], Device::CPU));
    let l2 = gb.linear(l1, w2, Some(b2));

    gb.output(l2);
    let mut g = gb.build();

    let plan = g.fusion_plan.as_ref().unwrap();
    assert_eq!(plan.fused_pairs.len(), 1);

    let c = g.execute(vec![Tensor::randn(&[1, 32], Device::CPU)]);
    assert_eq!(c[0].shape, vec![1, 32]);
}
