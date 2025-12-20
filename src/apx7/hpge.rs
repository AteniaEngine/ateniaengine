use crate::amg::graph::Graph;

/// APX 7.5 - Hierarchical Parallel Graph Executor (HPGE)
///
/// Node-level hierarchical scheduler that sits above `execute_single` and
/// strictly respects graph dependencies. If it detects any inconsistency, it
/// falls back to `run_plan(true)`.
pub fn execute_graph_parallel(graph: &mut Graph) {
    let node_count = graph.nodes.len();
    if node_count == 0 {
        return;
    }

    // Build children list and remaining-parent counter per node.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); node_count];
    let mut parents_left: Vec<usize> = vec![0; node_count];

    for (id, node) in graph.nodes.iter().enumerate() {
        parents_left[id] = node.inputs.len();
        for &inp in &node.inputs {
            if inp < node_count {
                children[inp].push(id);
            }
        }
    }

    // Initialize ready queue (no pending parents).
    let mut ready: Vec<usize> = Vec::new();
    for id in 0..node_count {
        if parents_left[id] == 0 {
            ready.push(id);
        }
    }

    // If for some reason there is no ready node but the graph is not empty,
    // immediately fall back to sequential execution.
    if ready.is_empty() {
        if crate::apx_debug_enabled() {
            eprintln!("[APX 7.5 HPGE] no ready nodes found, falling back to run_plan()");
        }
        graph.run_plan(true);
        return;
    }

    let mut executed = 0usize;

    // Simple topological loop: always take all ready nodes in a batch and
    // execute them safely by calling `execute_single`.
    while !ready.is_empty() {
        // APX 7.8: reorder ready nodes by temporal locality (TLO). This only
        // affects ordering within the set of independent nodes; it does not
        // change math nor dependencies.
        if crate::apx_mode_at_least("7.8") {
            crate::apx7::tlo::reorder_ready_by_locality(&mut ready);
        }

        let batch: Vec<usize> = ready.drain(..).collect();

        // Execute the batch. Conservative implementation: sequential, but
        // ready for future parallel extensions.
        for node_id in &batch {
            graph.execute_single(*node_id, true);
            executed += 1;
        }

        // Update parent counters and populate the next wave of ready nodes.
        for node_id in &batch {
            for &child in &children[*node_id] {
                if parents_left[child] > 0 {
                    parents_left[child] -= 1;
                    if parents_left[child] == 0 {
                        ready.push(child);
                    }
                }
            }
        }
    }

    // If for any reason we did not execute all nodes, protect correctness by
    // falling back to the classic sequential path.
    if executed != node_count {
        if crate::apx_debug_enabled() {
            eprintln!(
                "[APX 7.5 HPGE] executed {} of {} nodes, falling back to run_plan()",
                executed, node_count
            );
        }
        graph.run_plan(true);
    }
}
