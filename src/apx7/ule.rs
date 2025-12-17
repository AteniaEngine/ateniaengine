use std::time::Instant;

use crate::amg::graph::Graph;
use crate::apx7::hls_deep::{compute_depth, build_superlevels};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ULEStrategy {
    Seq,
    Pex,
    WorkStealing,
}

pub fn choose_backend(sl_nodes: &[usize]) -> ULEStrategy {
    let n = sl_nodes.len();
    if n <= 1 {
        return ULEStrategy::Seq;
    }

    let threads = crate::cpu_features::cpu_features().threads as usize;

    if n < 4 {
        return ULEStrategy::Pex;
    }

    if n >= 8 && threads >= 16 {
        return ULEStrategy::WorkStealing;
    }

    ULEStrategy::Pex
}

/// APX 7.12: Unified Level Executor.
///
/// Unifica las heurísticas 7.5–7.11 sin tocar kernels ni backward. Sólo
/// reorganiza el orden en que se llama a `execute_single` respetando siempre
/// las dependencias. Si detecta inconsistencia, hace fallback a
/// `graph.run_plan(true)`.
pub fn ule_execute_graph(graph: &mut Graph) {
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
        graph.run_plan(true);
        return;
    }

    let mut executed = vec![false; n];
    let mut executed_count = 0usize;

    for (sl_index, level) in superlevels.iter().enumerate() {
        let t0 = Instant::now();
        let mut remaining: Vec<usize> = level.nodes.iter().copied().collect();

        while !remaining.is_empty() {
            let mut ready: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|&id| parents_left[id] == 0 && !executed[id])
                .collect();

            if ready.is_empty() {
                if remaining.iter().any(|&id| !executed[id]) {
                    graph.run_plan(true);
                    return;
                } else {
                    break;
                }
            }

            // APX 7.8: hints de localidad temporal (TLO).
            if crate::apx_mode_at_least("7.8") {
                crate::apx7::tlo::reorder_ready_by_locality(&mut ready);
            }

            // Prioridad estructural: nodos que liberan más hijos y están más
            // profundos en el grafo primero.
            let children_ref = &children;
            let depths_ref = &depths;
            ready.sort_by_key(|&nid| {
                let out_degree = children_ref[nid].len() as i32;
                let depth = depths_ref[nid] as i32;
                -(out_degree + depth)
            });

            // APX 7.11 PFLS: si hay hotspot futuro, reforzar esta prioridad en
            // SL inmediatamente previos.
            if crate::apx_mode_at_least("7.11") {
                if let Ok(hist) = crate::apx7::pfls::global_pfls().lock() {
                    if let Some(hot) = hist.predict_next_hotspot() {
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

            let backend = choose_backend(&ready);
            let batch: Vec<usize> = ready;

            // En esta implementación de referencia, todos los backends
            // ejecutan de forma secuencial en el hilo actual. La elección
            // de backend sigue existiendo para tests y para futuras
            // extensiones paralelas seguras, pero no pasamos `Graph` entre
            // hilos.
            match backend {
                ULEStrategy::Seq | ULEStrategy::Pex | ULEStrategy::WorkStealing => {
                    for node_id in &batch {
                        graph.execute_single(*node_id, true);
                        executed[*node_id] = true;
                        executed_count += 1;
                    }
                }
            }

            // Actualizar padres restantes tras ejecutar el batch.
            for node_id in &batch {
                for &child in &children[*node_id] {
                    if parents_left[child] > 0 {
                        parents_left[child] -= 1;
                    }
                }
            }

            remaining.retain(|&id| !executed[id]);
        }

        // Registrar tiempo y congestión del SuperLevel en PFLS.
        if crate::apx_mode_at_least("7.11") {
            let dt = t0.elapsed().as_secs_f64();
            let cong = level.nodes.len();
            if let Ok(mut hist) = crate::apx7::pfls::global_pfls().lock() {
                hist.record(sl_index, dt, cong);
            }
        }
    }

    if executed_count != n {
        graph.run_plan(true);
    }
}
