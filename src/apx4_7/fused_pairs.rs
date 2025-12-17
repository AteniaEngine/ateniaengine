use crate::amg::graph::Graph;

fn node_in_gpu_segment(graph: &Graph, id: usize) -> bool {
    if let Some(plan) = &graph.gpu_plan {
        plan.segments
            .iter()
            .any(|seg| seg.start <= id && id <= seg.end)
    } else {
        false
    }
}

/// Ejecuta una pareja Linear→Linear (A → B) usando los ejecutores existentes
/// de Linear (GPU si aplica, CPU en fallback), sin tocar métodos privados.
pub fn exec_fused_linear_linear(
    graph: &mut Graph,
    id_a: usize,
    id_b: usize,
    _record_tape: bool,
) {
    // 1) Ejecutar Linear A
    if node_in_gpu_segment(graph, id_a) {
        graph.exec_gpu_linear(id_a);
    } else {
        graph.exec_cpu_linear_fallback(id_a);
    }

    // 2) Ejecutar Linear B: sus inputs ya apuntan al output de A vía GraphBuilder,
    // así que basta con ejecutar B usando el mismo criterio GPU/CPU.
    if node_in_gpu_segment(graph, id_b) {
        graph.exec_gpu_linear(id_b);
    } else {
        graph.exec_cpu_linear_fallback(id_b);
    }

    graph.fusions_applied += 1;
}
