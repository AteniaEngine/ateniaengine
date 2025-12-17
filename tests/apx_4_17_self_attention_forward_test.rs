use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::tensor::{Tensor, Device, DType, Layout};

fn assert_close(a: &Tensor, b: &Tensor, tol: f32) {
    assert_eq!(a.shape, b.shape, "shape mismatch: {:?} vs {:?}", a.shape, b.shape);
    assert_eq!(a.data.len(), b.data.len(), "len mismatch");
    for (i, (va, vb)) in a.data.iter().zip(b.data.iter()).enumerate() {
        let diff = (va - vb).abs();
        assert!(
            diff <= tol,
            "mismatch at idx {}: a={} b={} diff={} tol={}",
            i,
            va,
            vb,
            diff,
            tol
        );
    }
}

fn build_self_attention_graph() -> (GraphBuilder, usize, usize, usize, usize) {
    let mut gb = GraphBuilder::new();
    let x = gb.input();
    let wq = gb.input();
    let wk = gb.input();
    let wv = gb.input();

    let q = gb.linear(x, wq, None);
    let k = gb.linear(x, wk, None);
    let v = gb.linear(x, wv, None);

    let k_t = gb.transpose_last_two(k);
    let qk = gb.matmul(q, k_t);
    let att = gb.softmax(qk);
    let out = gb.matmul(att, v);
    let _ = gb.output(out);

    (gb, x, wq, wk, wv)
}

#[test]
fn self_attention_forward_fused_matches_naive() {
    // Configurar modo 4.17 para activar la detección de Self-Attention fusionado.
    unsafe { std::env::set_var("ATENIA_APX_MODE", "4.17"); }

    // Construir grafo base y clonar para obtener versiones naive y fusionada.
    let (gb, _x_id, _wq_id, _wk_id, _wv_id) = build_self_attention_graph();
    let graph_naive = gb.build();
    let graph_fused = graph_naive.clone();

    let mut g_naive = graph_naive;
    let mut g_fused = graph_fused;

    // Entradas determinísticas pequeñas.
    let m = 2usize;
    let d = 4usize;

    let x = Tensor::with_layout(vec![m, d], 0.5, Device::CPU, Layout::Contiguous, DType::F32);
    let wq = Tensor::with_layout(vec![d, d], 0.1, Device::CPU, Layout::Contiguous, DType::F32);
    let wk = Tensor::with_layout(vec![d, d], -0.2, Device::CPU, Layout::Contiguous, DType::F32);
    let wv = Tensor::with_layout(vec![d, d], 0.3, Device::CPU, Layout::Contiguous, DType::F32);

    let inputs = vec![x.clone(), wq.clone(), wk.clone(), wv.clone()];

    let out_naive = {
        let outs = g_naive.execute(inputs.clone());
        assert_eq!(outs.len(), 1);
        outs[0].clone()
    };

    let out_fused = {
        let outs = g_fused.execute(inputs);
        assert_eq!(outs.len(), 1);
        outs[0].clone()
    };

    let tol = 1e-5;
    assert_close(&out_naive, &out_fused, tol);
}
