use crate::amg::graph::Graph;
use crate::amg::nodes::{Node, NodeType};
use std::sync::{OnceLock, RwLock};
use crate::apx7::hpfa::FusionAffinity;

/// Per-node priority information for APX 7.6 HPGE v2.
#[derive(Clone, Debug, Default)]
pub struct NodePriorityInfo {
    pub est_cost: f64,
    pub subtree_size: usize,
    pub hist_time_us: f64,
    pub fusion_bonus: f64,
}

/// Priority scheduler for a specific graph.
#[derive(Clone, Debug)]
pub struct PriorityScheduler {
    pub priorities: Vec<NodePriorityInfo>,
}

// Global per-node time history (in microseconds).
// Indexed by `node_id` within the current graph. We use OnceLock+RwLock
// to avoid `static mut` while remaining thread-safe.
static GLOBAL_HIST_TIMES: OnceLock<RwLock<Vec<f64>>> = OnceLock::new();

fn get_global_hist_times(len: usize) -> &'static RwLock<Vec<f64>> {
    let lock = GLOBAL_HIST_TIMES.get_or_init(|| RwLock::new(vec![0.0; len]));
    {
        if let Ok(guard) = lock.read() {
            if guard.len() != len {
                drop(guard);
                if let Ok(mut w) = lock.write() {
                    *w = vec![0.0; len];
                }
            }
        }
    }
    lock
}

/// Called from `Graph::execute_single` to record historical time per node
/// when APX >= 7.6.
pub fn record_node_time(node_id: usize, dt_us: f64, node_count: usize) {
    let hist = get_global_hist_times(node_count);
    if let Ok(mut guard) = hist.write() {
        if node_id < guard.len() {
            let prev = guard[node_id];
            guard[node_id] = 0.8 * prev + 0.2 * dt_us;
        }
    }
}

fn compute_subtree_sizes(children: &Vec<Vec<usize>>) -> Vec<usize> {
    fn dfs(u: usize, ch: &Vec<Vec<usize>>, memo: &mut Vec<Option<usize>>) -> usize {
        if let Some(v) = memo[u] {
            return v;
        }
        if ch[u].is_empty() {
            memo[u] = Some(1);
            return 1;
        }
        let mut s = 1;
        for &c in &ch[u] {
            s += dfs(c, ch, memo);
        }
        memo[u] = Some(s);
        s
    }

    let n = children.len();
    let mut memo = vec![None; n];
    let mut out = vec![0; n];
    for i in 0..n {
        out[i] = dfs(i, children, &mut memo);
    }
    out
}

fn estimate_cost(node: &Node) -> f64 {
    match node.node_type {
        NodeType::MatMul => {
            // Simple heuristic: approximate element count (if shape exists).
            let elems = node
                .output
                .as_ref()
                .map(|t| t.shape.iter().product::<usize>() as f64)
                .unwrap_or(1.0);
            elems
        }
        NodeType::Linear => {
            let elems = node
                .output
                .as_ref()
                .map(|t| t.shape.iter().product::<usize>() as f64)
                .unwrap_or(1.0);
            elems * 0.5
        }
        NodeType::SiLU | NodeType::Activation(_) | NodeType::Softmax | NodeType::LogSoftmax => 1.0,
        _ => 10.0,
    }
}

fn compute_priority(p: &NodePriorityInfo) -> f64 {
    (p.est_cost * 0.4)
        + ((p.subtree_size as f64) * 0.3)
        + (p.hist_time_us * 0.1)
        + (p.fusion_bonus * 0.2)
}

/// Version 7.6 of the priority graph executor (Critical-Path Optimizer).
///
/// Keeps the same guarantees as HPGE v1 and never alters kernels nor backward.
pub fn execute_graph_parallel_priority(graph: &mut Graph) {
    let node_count = graph.nodes.len();
    if node_count == 0 {
        return;
    }

    // Build children and parents_left (same as HPGE v1).
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

    let mut ready: Vec<usize> = Vec::new();
    for id in 0..node_count {
        if parents_left[id] == 0 {
            ready.push(id);
        }
    }

    if ready.is_empty() {
        if crate::apx_debug_enabled() {
            eprintln!("[APX 7.6 HPGE] no ready nodes found, falling back to run_plan()");
        }
        graph.run_plan(true);
        return;
    }

    // Prepare priority scheduler.
    let subtree_sizes = compute_subtree_sizes(&children);
    let hist = get_global_hist_times(node_count)
        .read()
        .ok()
        .map(|g| g.clone());

    let mut priorities: Vec<NodePriorityInfo> = (0..node_count)
        .map(|i| NodePriorityInfo {
            est_cost: estimate_cost(&graph.nodes[i]),
            subtree_size: subtree_sizes[i],
            hist_time_us: hist.as_ref().and_then(|v| v.get(i).cloned()).unwrap_or(0.0),
            fusion_bonus: 0.0,
        })
        .collect();

    // APX 7.7: incorporate Hot-Path Fusion Awareness (HPFA) signals.
    if crate::apx_mode_at_least("7.7") {
        if let Ok(sel) = crate::apx6_10::global_fusion_selector().lock() {
            for node_id in 0..node_count {
                let fa: FusionAffinity = sel.get_fusion_affinity(node_id);
                priorities[node_id].fusion_bonus = fa.fusion_bonus();
            }
        }
    }

    // APX 7.9: use HLS to obtain a hierarchical cluster order and derive a
    // per-node cluster priority index. This still does not modify math nor
    // dependencies; it only provides hints for ordering within ready.
    let cluster_order: Vec<usize> = if crate::apx_mode_at_least("7.9") {
        let mut order = vec![usize::MAX; node_count];
        let hls = crate::apx7::hls::HLSScheduler::new(graph);
        let clusters = hls.run();
        let mut _pos = 0usize;
        for (cidx, cluster) in clusters.iter().enumerate() {
            for &nid in &cluster.nodes {
                if nid < order.len() {
                    if order[nid] == usize::MAX {
                        order[nid] = cidx;
                        _pos += 1;
                    }
                }
            }
        }
        order
    } else {
        vec![usize::MAX; node_count]
    };

    let sched = PriorityScheduler { priorities };

    let mut executed = 0usize;

    while !ready.is_empty() {
        // Sort ready nodes by priority (highest first). In APX 7.9 we apply
        // HLS cluster hierarchical order first and then HPFA+TLO numeric
        // priority.
        ready.sort_by(|&a, &b| {
            if crate::apx_mode_at_least("7.9") {
                let ca = cluster_order.get(a).copied().unwrap_or(usize::MAX);
                let cb = cluster_order.get(b).copied().unwrap_or(usize::MAX);
                if ca != cb {
                    return ca.cmp(&cb);
                }
            }

            let pa = compute_priority(&sched.priorities[a]);
            let pb = compute_priority(&sched.priorities[b]);
            pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
        });

        let batch: Vec<usize> = ready.drain(..).collect();

        for node_id in &batch {
            graph.execute_single(*node_id, true);
            executed += 1;
        }

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

    if executed != node_count {
        if crate::apx_debug_enabled() {
            eprintln!(
                "[APX 7.6 HPGE] executed {} of {} nodes, falling back to run_plan()",
                executed, node_count
            );
        }
        graph.run_plan(true);
    }
}
