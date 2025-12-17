use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct GradStore {
    slots: Arc<Mutex<Vec<Mutex<Vec<f32>>>>>,
}

impl GradStore {
    /// Create an empty GradStore. Slots grow dynamically on first use.
    pub fn new() -> Self {
        Self {
            slots: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Ensure a slot exists for `id`. Create it if missing.
    fn ensure_slot(&self, id: usize) {
        let mut slots_guard = self.slots.lock().expect("gradstore poisoned");

        if id >= slots_guard.len() {
            let missing = id + 1 - slots_guard.len();
            for _ in 0..missing {
                slots_guard.push(Mutex::new(Vec::new()));
            }
        }
    }

    /// Add a gradient to slot `id`, resizing if needed.
    pub fn add(&self, id: usize, grad: &[f32]) {
        self.ensure_slot(id);

        let slots_guard = self.slots.lock().expect("gradstore poisoned");
        let mut slot = slots_guard[id].lock().expect("slot poisoned");

        if slot.len() != grad.len() {
            *slot = vec![0.0; grad.len()];
        }

        for (dst, src) in slot.iter_mut().zip(grad.iter()) {
            *dst += *src;
        }
    }

    /// Read a gradient slot.
    /// Returns an empty vec if slot never used.
    pub fn get(&self, id: usize) -> Vec<f32> {
        self.ensure_slot(id);

        let slots_guard = self.slots.lock().expect("gradstore poisoned");
        slots_guard[id]
            .lock()
            .expect("slot poisoned")
            .clone()
    }

    /// Overwrite a gradient slot with an explicit vector.
    pub fn set(&self, id: usize, grad: Vec<f32>) {
        self.ensure_slot(id);
        let slots_guard = self.slots.lock().expect("gradstore poisoned");
        *slots_guard[id].lock().expect("slot poisoned") = grad;
    }

    /// Take slot contents, leaving an empty vec behind.
    pub fn take(&self, id: usize) -> Vec<f32> {
        self.ensure_slot(id);
        let slots_guard = self.slots.lock().expect("gradstore poisoned");
        let mut slot = slots_guard[id].lock().expect("slot poisoned");
        let mut out = Vec::new();
        std::mem::swap(&mut *slot, &mut out);
        out
    }

    pub fn len(&self) -> usize {
        self.slots.lock().unwrap().len()
    }
}
