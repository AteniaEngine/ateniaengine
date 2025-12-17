use crate::amg::graph::Graph;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct HLSDPLevel {
    pub nodes: Vec<usize>,
    pub depth: usize,
}

const W_THRESHOLD: usize = 6;
const D_THRESHOLD: usize = 3;

/// Calcular la profundidad topológica de cada nodo (distancia desde inputs).
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

/// Construir superniveles agrupando niveles consecutivos según anchura y
/// profundidad acumulada.
pub fn build_superlevels(depths: &[usize]) -> Vec<HLSDPLevel> {
    if depths.is_empty() {
        return Vec::new();
    }
    let max_depth = *depths.iter().max().unwrap_or(&0);

    // nodos por nivel
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

/// APX 7.10: ejecución por superniveles profundos. No modifica kernels ni
/// backward, sólo el orden en que se llaman a execute_single respetando
/// dependencias.
pub fn execute_graph_hls_deep(graph: &mut Graph) {
    let n = graph.nodes.len();
    if n == 0 {
        return;
    }

    // Construir hijos y parents_left igual que HPGE.
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
        // Fallback seguro.
        graph.run_plan(true);
        return;
    }

    let mut executed = vec![false; n];
    let mut executed_count = 0usize;

    for (sl_index, level) in superlevels.iter().enumerate() {
        let t0 = Instant::now();
        let mut remaining: Vec<usize> = level.nodes.iter().copied().collect();

        while !remaining.is_empty() {
            // Nodos de este supernivel que ya están listos.
            let mut ready: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|&id| parents_left[id] == 0 && !executed[id])
                .collect();

            if ready.is_empty() {
                // Si aún quedan nodos de este supernivel sin ejecutar,
                // pero ninguno está ready, algo está inconsistente.
                if remaining.iter().any(|&id| !executed[id]) {
                    graph.run_plan(true);
                    return;
                } else {
                    break;
                }
            }

            // Estrategia según ancho del supernivel actual.
            let width = ready.len();
            if width > 4 {
                // Usar TLO (localidad temporal) como heurística de orden.
                crate::apx7::tlo::reorder_ready_by_locality(&mut ready);
            }

            // APX 7.11: reordenamiento predictivo basado en PFLS si se ha
            // observado un hotspot futuro (SuperLevel "caliente").
            if crate::apx_mode_at_least("7.11") {
                if let Ok(hist) = crate::apx7::pfls::global_pfls().lock() {
                    if let Some(hot) = hist.predict_next_hotspot() {
                        // Si este SL es inmediatamente previo al hotspot
                        // predicho (X-1 o X-2), priorizar nodos que liberen
                        // muchos hijos y estén en caminos profundos.
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

                // Actualizar padres restantes de los hijos.
                for &child in &children[node_id] {
                    if parents_left[child] > 0 {
                        parents_left[child] -= 1;
                    }
                }
            }

            remaining.retain(|&id| !executed[id]);
        }

        // Medición PFLS para este SuperLevel (tiempo + congestión).
        if crate::apx_mode_at_least("7.11") {
            let dt = t0.elapsed().as_secs_f64();
            let cong = level.nodes.len();
            if let Ok(mut hist) = crate::apx7::pfls::global_pfls().lock() {
                hist.record(sl_index, dt, cong);
            }
        }
    }

    if executed_count != n {
        // Fallback para proteger correctitud.
        graph.run_plan(true);
    }
}
