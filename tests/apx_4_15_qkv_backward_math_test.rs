use atenia_engine::amg::builder::GraphBuilder;
use atenia_engine::amg::graph::Graph;
use atenia_engine::nn::linear as nn_linear;
use atenia_engine::tensor::{Tensor, Device, DType, Layout};

fn build_qkv_sum_graph() -> (Graph, usize, usize, usize, usize, usize, usize, usize) {
    let mut gb = GraphBuilder::new();
    let x = gb.input();
    let wq = gb.input();
    let wk = gb.input();
    let wv = gb.input();

    let q = gb.linear(x, wq, None);
    let k = gb.linear(x, wk, None);
    let v = gb.linear(x, wv, None);

    let sum1 = gb.add(q, k);
    let sum2 = gb.add(sum1, v);
    let _out = gb.output(sum2);

    let g = gb.build();

    (g, x, wq, wk, wv, q, k, v)
}

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

/// Transpuesta 2D simple para tests: [m, n] -> [n, m].
fn transpose_2d_2d(t: &Tensor) -> Tensor {
    assert_eq!(t.shape.len(), 2, "transpose_2d_2d expects a 2D tensor");
    let m = t.shape[0];
    let n = t.shape[1];
    let mut out = Tensor::with_layout(vec![n, m], 0.0, t.device, Layout::Contiguous, t.dtype);
    for i in 0..m {
        for j in 0..n {
            let src = i * n + j;
            let dst = j * m + i;
            out.data[dst] = t.data[src];
        }
    }
    out
}

#[test]
#[ignore]
fn test_qkv_backward_math_matches_naive() {
    // Grafo sintético Q,K,V con suma: out = q + k + v
    let (mut graph, x_id, wq_id, wk_id, wv_id, q_id, k_id, v_id) = build_qkv_sum_graph();

    // Entradas determinísticas pequeñas
    let m = 2usize;
    let k = 4usize;
    let n = 3usize;

    let x = Tensor::with_layout(vec![m, k], 0.5, Device::CPU, Layout::Contiguous, DType::F32);
    let wq = Tensor::with_layout(vec![k, n], 0.1, Device::CPU, Layout::Contiguous, DType::F32);
    let wk = Tensor::with_layout(vec![k, n], -0.2, Device::CPU, Layout::Contiguous, DType::F32);
    let wv = Tensor::with_layout(vec![k, n], 0.3, Device::CPU, Layout::Contiguous, DType::F32);

    let inputs = vec![x.clone(), wq.clone(), wk.clone(), wv.clone()];

    // Forward + backward naive usando el engine.
    // Usamos execute() para forward y luego backward() con el último Output.
    let out_id = graph.last_output_id();
    let _outs = graph.execute(inputs);
    graph.backward(out_id);

    // Tensors forward
    let x_t = graph.nodes[x_id]
        .output
        .as_ref()
        .expect("x output missing")
        .clone();
    let wq_t = graph.nodes[wq_id]
        .output
        .as_ref()
        .expect("wq output missing")
        .clone();
    let wk_t = graph.nodes[wk_id]
        .output
        .as_ref()
        .expect("wk output missing")
        .clone();
    let wv_t = graph.nodes[wv_id]
        .output
        .as_ref()
        .expect("wv output missing")
        .clone();

    // Helper para obtener gradientes "naive" combinando output.grad y grad_store.
    let make_grad = |node_id: usize, proto: &Tensor| -> Tensor {
        // 1) Intentar leer el grad directamente del tensor de salida.
        let from_tensor = graph.nodes[node_id]
            .output
            .as_ref()
            .and_then(|t| t.grad.as_ref())
            .cloned();

        // 2) Si no hay grad en el tensor, intentar GradStore.
        let data = if let Some(g) = from_tensor {
            g
        } else {
            let g = graph.grad_store.get(node_id);
            if g.is_empty() {
                // 3) Como último recurso, devolver gradiente cero del mismo tamaño.
                vec![0.0; proto.data.len()]
            } else {
                g
            }
        };

        Tensor {
            shape: proto.shape.clone(),
            data,
            device: proto.device,
            dtype: proto.dtype,
            layout: proto.layout,
            strides: proto.strides.clone(),
            grad: None,
            gpu: None,
            persistence: None,
            op: None,
        }
    };

    // Gradientes naive obtenidos vía helper (equivalente lógico al caso 4.16).
    let dx_naive  = make_grad(x_id,  &x_t);
    let dwq_naive = make_grad(wq_id, &wq_t);
    let dwk_naive = make_grad(wk_id, &wk_t);
    let dwv_naive = make_grad(wv_id, &wv_t);

    // gQ, gK, gV analíticos: out = q + k + v, dL/dout = 1 => gq = gk = gv = 1
    let q_out = graph.nodes[q_id].output.as_ref().expect("q output missing");
    let gq = Tensor::with_layout(
        q_out.shape.clone(),
        1.0,
        q_out.device,
        Layout::Contiguous,
        q_out.dtype,
    );

    let k_out = graph.nodes[k_id].output.as_ref().expect("k output missing");
    let gk = Tensor::with_layout(
        k_out.shape.clone(),
        1.0,
        k_out.device,
        Layout::Contiguous,
        k_out.dtype,
    );

    let v_out = graph.nodes[v_id].output.as_ref().expect("v output missing");
    let gv = Tensor::with_layout(
        v_out.shape.clone(),
        1.0,
        v_out.device,
        Layout::Contiguous,
        v_out.dtype,
    );

    // --- backward QKV fusionado (matemático) ---
    let wq_t_t = transpose_2d_2d(&wq_t);
    let wk_t_t = transpose_2d_2d(&wk_t);
    let wv_t_t = transpose_2d_2d(&wv_t);

    let dx_q = nn_linear::matmul(&gq, &wq_t_t);
    let dx_k = nn_linear::matmul(&gk, &wk_t_t);
    let dx_v = nn_linear::matmul(&gv, &wv_t_t);

    let dx_tmp = dx_q.add(&dx_k);
    let dx_fused = dx_tmp.add(&dx_v);

    let x_t_t = transpose_2d_2d(&x_t);

    let dwq_fused = nn_linear::matmul(&x_t_t, &gq);
    let dwk_fused = nn_linear::matmul(&x_t_t, &gk);
    let dwv_fused = nn_linear::matmul(&x_t_t, &gv);

    // --- comparaciones ---
    let tol = 1e-6;
    assert_close(&dx_fused, &dx_naive, tol);
    assert_close(&dwq_fused, &dwq_naive, tol);
    assert_close(&dwk_fused, &dwk_naive, tol);
    assert_close(&dwv_fused, &dwv_naive, tol);
}
