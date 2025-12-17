#[derive(Debug, Clone)]
pub struct KernelPerf {
    pub size: usize,          // tamaño típico, ej. 128, 256, 512, 1024
    pub baseline_us: u128,    // tiempo baseline APX 3.8
    pub micro64_us: u128,     // tiempo microkernel 6.4
    pub selected: String,     // "baseline" | "micro64"
}

#[derive(Debug, Clone)]
pub struct RuntimeProfile {
    pub entries: Vec<KernelPerf>,
    pub last_updated: u64,
}

impl RuntimeProfile {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            last_updated: 0,
        }
    }

    pub fn best_for(&self, size: usize) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        // Elegir la entrada con tamaño más cercano.
        let mut best = None;
        let mut best_diff = usize::MAX;
        for e in &self.entries {
            let diff = if e.size > size { e.size - size } else { size - e.size };
            if diff < best_diff {
                best_diff = diff;
                best = Some(e.selected.clone());
            }
        }
        best
    }

    pub fn record(&mut self, perf: KernelPerf) {
        // Reemplazar si ya había una entrada para ese tamaño.
        if let Some(existing) = self.entries.iter_mut().find(|e| e.size == perf.size) {
            *existing = perf;
        } else {
            self.entries.push(perf);
        }
        self.last_updated = self.last_updated.saturating_add(1);
    }
}
