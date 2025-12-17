use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct HeatEntry {
    pub count: u32,
    pub avg_ms: f32,
}

impl HeatEntry {
    pub fn new() -> Self {
        Self { count: 0, avg_ms: 0.0 }
    }

    pub fn update(&mut self, time_ms: f32) {
        self.count += 1;
        self.avg_ms = ((self.avg_ms * ((self.count - 1) as f32)) + time_ms) / (self.count as f32);
    }
}

#[derive(Debug, Clone, Default)]
pub struct GpuHeatmap {
    pub entries: HashMap<String, HeatEntry>,
}

impl GpuHeatmap {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    pub fn record(&mut self, kernel_name: &str, time_ms: f32) {
        let key = kernel_name.to_string();
        self.entries
            .entry(key)
            .or_insert_with(HeatEntry::new)
            .update(time_ms);
    }

    pub fn get(&self, kernel_name: &str) -> Option<&HeatEntry> {
        self.entries.get(kernel_name)
    }

    pub fn total_entries(&self) -> usize {
        self.entries.len()
    }
}
