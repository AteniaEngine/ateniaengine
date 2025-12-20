use crate::amg::graph::Graph;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct HLSDPLevel {
    pub nodes: Vec<usize>,
    pub depth: usize,
}

const W_THRESHOLD: usize = 6;
const D_THRESHOLD: usize = 3;

/// Compute the topological depth of each node (distance from inputs).
pub fn compute_depth(graph: &Graph) -> Vec<usize> {
    let n = graph.nodes.len();
    if n == 0 {
        return Vec::new();
    }

    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut in_deg = vec![0usize; n];

    for (id, node) in graph.nodes.iter().enumerate() {
        in_deg[id] = node.inputs.len();
        for &inp in &node.inputs {
            if inp < n {
                children[inp].push(id);
            }
        }
    }

    let mut depth = vec![0usize; n];
    let mut ready: Vec<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();
    let mut idx = 0usize;

    while idx < ready.len() {
        let u = ready[idx];
        idx += 1;
        for &v in &children[u] {
            if depth[v] < depth[u] + 1 {
                depth[v] = depth[u] + 1;
            }
            if in_deg[v] > 0 {
                in_deg[v] -= 1;
                if in_deg[v] == 0 {
                    ready.push(v);
                }
            }
        }
    }

    depth
}

/// Build superlevels by grouping consecutive levels based on width and
/// accumulated depth.
pub fn build_superlevels(depths: &[usize]) -> Vec<HLSDPLevel> {
    if depths.is_empty() {
        return Vec::new();
    }
    let max_depth = *depths.iter().max().unwrap_or(&0);

    // Nodes per level.
    let mut levels: Vec<Vec<usize>> = vec![Vec::new(); max_depth + 1];
    for (id, &d) in depths.iter().enumerate() {
        if d < levels.len() {
            levels[d].push(id);
        }
    }

    let mut superlevels: Vec<HLSDPLevel> = Vec::new();
    let mut cur_nodes: Vec<usize> = Vec::new();
    let mut cur_start = 0usize;
    let mut cur_width = 0usize;

    for level in 0..=max_depth {
        let lvl_nodes = &levels[level];
        if lvl_nodes.is_empty() {
            continue;
        }

        if cur_nodes.is_empty() {
            cur_start = level;
            cur_width = lvl_nodes.len();
            cur_nodes.extend_from_slice(lvl_nodes);
            continue;
        }

        let new_width = cur_width + lvl_nodes.len();
        let depth_span = level - cur_start;
        if new_width > W_THRESHOLD || depth_span > D_THRESHOLD {
            superlevels.push(HLSDPLevel {
                nodes: cur_nodes.clone(),
                depth: cur_start,
            });
            cur_nodes.clear();
            cur_start = level;
            cur_width = lvl_nodes.len();
            cur_nodes.extend_from_slice(lvl_nodes);
        } else {
            cur_width = new_width;
            cur_nodes.extend_from_slice(lvl_nodes);
        }
    }

    if !cur_nodes.is_empty() {
        superlevels.push(HLSDPLevel {
            nodes: cur_nodes,
            depth: cur_start,
        });
    }

    superlevels
}

/// APX 7.10: execution via deep superlevels. Does not modify kernels nor
/// backward, only the order in which execute_single is called while respecting
/// dependencies.
pub fn execute_graph_hls_deep(graph: &mut Graph) {
    let n = graph.nodes.len();
    if n == 0 {
        return;
    }

    // Build children and parents_left the same way as HPGE.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut parents_left: Vec<usize> = vec![0; n];

    for (id, node) in graph.nodes.iter().enumerate() {
        parents_left[id] = node.inputs.len();
        for &inp in &node.inputs {
            if inp < n {
                children[inp].push(id);
            }
        }
    }

    let depths = compute_depth(graph);
    let superlevels = build_superlevels(&depths);
    if superlevels.is_empty() {
        // Safe fallback.
        graph.run_plan(true);
        return;
    }

    let mut executed = vec![false; n];
    let mut executed_count = 0usize;

    for (sl_index, level) in superlevels.iter().enumerate() {
        let t0 = Instant::now();
        let mut remaining: Vec<usize> = level.nodes.iter().copied().collect();

        while !remaining.is_empty() {
            // Nodes in this superlevel that are already ready.
            let mut ready: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|&id| parents_left[id] == 0 && !executed[id])
                .collect();

            if ready.is_empty() {
                // If there are still nodes in this superlevel not executed,
                // but none is ready, something is inconsistent.
                if remaining.iter().any(|&id| !executed[id]) {
                    graph.run_plan(true);
                    return;
                } else {
                    break;
                }
            }

            // Strategy depending on the width of the current superlevel.
            let width = ready.len();
            if width > 4 {
                // Use TLO (temporal locality) as an ordering heuristic.
                crate::apx7::tlo::reorder_ready_by_locality(&mut ready);
            }

            // APX 7.11: predictive reordering based on PFLS if a future hotspot
            // has been observed ("hot" SuperLevel).
            if crate::apx_mode_at_least("7.11") {
                if let Ok(hist) = crate::apx7::pfls::global_pfls().lock() {
                    if let Some(hot) = hist.predict_next_hotspot() {
                        // If this SL is immediately before the predicted hotspot
                        // (X-1 or X-2), prioritize nodes that release many
                        // children and are on deep paths.
                        if hot == sl_index + 1 || hot == sl_index + 2 {
                            let children_ref = &children;
                            let depths_ref = &depths;
                            ready.sort_by_key(|&nid| {
                                let out_degree = children_ref[nid].len() as i32;
                                let depth = depths_ref[nid] as i32;
                                -(out_degree + depth)
                            });
                        }
                    }
                }
            }

            for node_id in ready {
                if executed[node_id] {
                    continue;
                }
                graph.execute_single(node_id, true);
                executed[node_id] = true;
                executed_count += 1;

                // Update remaining parents for children.
                for &child in &children[node_id] {
                    if parents_left[child] > 0 {
                        parents_left[child] -= 1;
                    }
                }
            }

            remaining.retain(|&id| !executed[id]);
        }

        // PFLS measurement for this SuperLevel (time + congestion).
        if crate::apx_mode_at_least("7.11") {
            let dt = t0.elapsed().as_secs_f64();
            let cong = level.nodes.len();
            if let Ok(mut hist) = crate::apx7::pfls::global_pfls().lock() {
                hist.record(sl_index, dt, cong);
            }
        }
    }

    if executed_count != n {
        // Fallback to protect correctness.
        graph.run_plan(true);
    }
}
