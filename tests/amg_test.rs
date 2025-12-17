use atenia_engine::tensor::{Device, DType, Layout, Tensor};
use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::scheduler::{build_execution_plan, ExecStep};

#[test]
fn simple_add_graph() {
    let mut gb = GraphBuilder::new();
    let a = gb.input();
    let b = gb.input();
    let c = gb.add(a, b);
    let _out = gb.output(c);

    let mut g = gb.build();

    let t1 = Tensor::ones(vec![4], Device::CPU, DType::F32);
    let t2 = Tensor::ones(vec![4], Device::CPU, DType::F32);

    let results = g.execute(vec![t1, t2]);

    assert_eq!(results.len(), 1);
    for v in &results[0].data {
        assert_eq!(*v, 2.0);
    }
}

#[test]
fn fusion_builds_fused_add_mul_step() {
    let mut gb = GraphBuilder::new();
    let a = gb.input();
    let b = gb.input();
    let c = gb.input();

    let add = gb.add(a, b);
    let mul = gb.mul(add, c);
    let _out = gb.output(mul);

    let g = gb.build();
    let (plan, _) = build_execution_plan(&g.nodes);

    let has_fused = plan.steps.iter().any(|s| matches!(s, ExecStep::FusedAddMul { .. }));
    assert!(has_fused, "Expected a fused Add→Mul step in execution plan");
}

#[test]
fn chunked_execution_produces_same_result_as_normal() {
    let mut gb = GraphBuilder::new();
    let a = gb.input();
    let b = gb.input();
    let out_idx = gb.add(a, b);
    let _out = gb.output(out_idx);

    let mut g1 = gb.build();
    let mut g2 = g1.clone();

    let len = 10_000;
    let mut t1 = Tensor::with_layout(
        vec![len],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );
    let mut t2 = Tensor::with_layout(
        vec![len],
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );

    for (i, v) in t1.data.iter_mut().enumerate() {
        *v = i as f32;
    }
    for (i, v) in t2.data.iter_mut().enumerate() {
        *v = (len - i) as f32;
    }

    let normal = g1.execute(vec![t1.clone(), t2.clone()]);
    let chunked = g2.execute_chunked(vec![t1, t2], 1024);

    assert_eq!(normal.len(), 1);
    assert_eq!(chunked.len(), 1);
    assert_eq!(normal[0].data.len(), chunked[0].data.len());

    for (a, b) in normal[0].data.iter().zip(chunked[0].data.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn chained_ops_graph() {
    let mut gb = GraphBuilder::new();
    let a = gb.input();
    let b = gb.input();
    let c = gb.add(a, b);
    let d = gb.mul(c, b);
    let _out = gb.output(d);

    let mut g = gb.build();

    let t1 = Tensor::ones(vec![3], Device::CPU, DType::F32);
    let t2 = Tensor::ones(vec![3], Device::CPU, DType::F32);

    let results = g.execute(vec![t1, t2]);

    for v in &results[0].data {
        assert_eq!(*v, 2.0);
    }
}
