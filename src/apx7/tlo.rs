use std::sync::RwLock;

/// Temporal locality hint for a graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LocalityHint {
    pub branch_id: usize,
    pub depth: usize,
}

// Global hint map indexed by node_id. Does not modify math nor graph
// structure; it only guides scheduling order in HPGE.
static LOCALITY_HINTS: RwLock<Vec<LocalityHint>> = RwLock::new(Vec::new());

/// Register locality hints for the current graph.
pub fn set_locality_hints(hints: Vec<LocalityHint>) {
    if let Ok(mut h) = LOCALITY_HINTS.write() {
        *h = hints;
    }
}

/// Reorder the ready queue prioritizing temporal locality.
///
/// - Nodes with the same `branch_id` are grouped.
/// - Within each branch, ordering is by `depth`.
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
