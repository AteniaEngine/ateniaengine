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

/// Execute a Linear->Linear pair (A -> B) using existing Linear executors
/// (GPU if applicable, CPU as fallback), without touching private methods.
pub fn exec_fused_linear_linear(
    graph: &mut Graph,
    id_a: usize,
    id_b: usize,
    _record_tape: bool,
) {
    // 1) Execute Linear A
    if node_in_gpu_segment(graph, id_a) {
        graph.exec_gpu_linear(id_a);
    } else {
        graph.exec_cpu_linear_fallback(id_a);
    }

    // 2) Execute Linear B: its inputs already point to A's output via GraphBuilder,
    // so it is enough to execute B using the same GPU/CPU criterion.
    if node_in_gpu_segment(graph, id_b) {
        graph.exec_gpu_linear(id_b);
    } else {
        graph.exec_cpu_linear_fallback(id_b);
    }

    graph.fusions_applied += 1;
}
