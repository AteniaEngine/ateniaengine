use crate::amg::graph::Graph;

/// APX 7.5 - Hierarchical Parallel Graph Executor (HPGE)
///
/// Scheduler jerárquico por nodos que se sitúa por encima de `execute_single`
/// y respeta estrictamente las dependencias del grafo. Si detecta cualquier
/// inconsistencia, hace fallback a `run_plan(true)`.
pub fn execute_graph_parallel(graph: &mut Graph) {
    let node_count = graph.nodes.len();
    if node_count == 0 {
        return;
    }

    // Construir lista de hijos y contador de padres restantes por nodo.
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

    // Inicializar cola de nodos listos (sin padres pendientes).
    let mut ready: Vec<usize> = Vec::new();
    for id in 0..node_count {
        if parents_left[id] == 0 {
            ready.push(id);
        }
    }

    // Si por alguna razón no hay ningún nodo listo pero el grafo no está vacío,
    // hacemos fallback inmediato a la ejecución secuencial.
    if ready.is_empty() {
        if crate::apx_debug_enabled() {
            eprintln!("[APX 7.5 HPGE] no ready nodes found, falling back to run_plan()");
        }
        graph.run_plan(true);
        return;
    }

    let mut executed = 0usize;

    // Bucle topológico sencillo: siempre tomamos todos los nodos ready en un
    // batch y los ejecutamos de forma segura llamando a `execute_single`.
    while !ready.is_empty() {
        // APX 7.8: ordenar nodos ready según localidad temporal (TLO). Esto
        // sólo afecta al orden dentro del conjunto de nodos independientes;
        // no modifica matemática ni dependencias.
        if crate::apx_mode_at_least("7.8") {
            crate::apx7::tlo::reorder_ready_by_locality(&mut ready);
        }

        let batch: Vec<usize> = ready.drain(..).collect();

        // Ejecutar el batch. Implementación conservadora: secuencial, pero
        // preparada para futuras extensiones paralelas.
        for node_id in &batch {
            graph.execute_single(*node_id, true);
            executed += 1;
        }

        // Actualizar contadores de padres y poblar la siguiente wave de nodos
        // listos para ejecutar.
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

    // Si por algún motivo no ejecutamos todos los nodos, protegemos la
    // correctitud haciendo fallback a la ruta secuencial clásica.
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
