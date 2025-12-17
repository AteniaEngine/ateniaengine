use std::sync::RwLock;

/// Hint de localidad temporal para un nodo del grafo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LocalityHint {
    pub branch_id: usize,
    pub depth: usize,
}

// Mapa global de hints indexado por node_id. No modifica matemática ni
// estructura del grafo; sólo guía el orden de scheduling en HPGE.
static LOCALITY_HINTS: RwLock<Vec<LocalityHint>> = RwLock::new(Vec::new());

/// Registrar los hints de localidad para el grafo actual.
pub fn set_locality_hints(hints: Vec<LocalityHint>) {
    if let Ok(mut h) = LOCALITY_HINTS.write() {
        *h = hints;
    }
}

/// Reordenar la cola de nodos listos priorizando localidad temporal.
///
/// - Nodos del mismo `branch_id` se agrupan.
/// - Dentro de cada rama se ordena por `depth`.
pub fn reorder_ready_by_locality(ready: &mut Vec<usize>) {
    let hints_guard = match LOCALITY_HINTS.read() {
        Ok(h) => h,
        Err(_) => return,
    };
    let hints = &*hints_guard;

    ready.sort_by_key(|&id| {
        if id < hints.len() {
            (hints[id].branch_id, hints[id].depth)
        } else {
            (usize::MAX, usize::MAX)
        }
    });
}
