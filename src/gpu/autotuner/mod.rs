use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

#[derive(Clone, Debug)]
pub struct TuningResult {
    pub block_x: u32,
    pub block_y: u32,
    pub grid_x: u32,
    pub grid_y: u32,
}

fn autotuner_cache() -> &'static Mutex<HashMap<u64, TuningResult>> {
    static CACHE: OnceLock<Mutex<HashMap<u64, TuningResult>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn hash_key(kernel: &str, n: usize, compute_cap: i32) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    kernel.hash(&mut h);
    n.hash(&mut h);
    compute_cap.hash(&mut h);
    h.finish()
}

pub fn autotune_matmul(
    n: usize,
    compute_cap: i32,
    runner: &dyn Fn((u32, u32, u32, u32)) -> f32,
    gpu_enabled: bool,
) -> TuningResult {
    // CPU fallback?  return safe default
    if !gpu_enabled {
        return TuningResult {
            block_x: 16,
            block_y: 16,
            grid_x: (n as u32 + 15) / 16,
            grid_y: (n as u32 + 15) / 16,
        };
    }

    let key = hash_key("matmul", n, compute_cap);
    if let Some(cached) = autotuner_cache().lock().unwrap().get(&key) {
        return cached.clone();
    }

    // Candidate layouts
    let candidates = [
        (8, 8),
        (16, 16),
        (32, 8),
        (8, 32),
        (32, 16),
    ];

    let mut best: Option<(f32, TuningResult)> = None;

    for &(bx, by) in &candidates {
        let gx = (n as u32 + bx - 1) / bx;
        let gy = (n as u32 + by - 1) / by;

        let t = runner((bx, by, gx, gy));

        match best {
            None => best = Some((t, TuningResult { block_x: bx, block_y: by, grid_x: gx, grid_y: gy })),
            Some((bt, _)) if t < bt =>
                best = Some((t, TuningResult { block_x: bx, block_y: by, grid_x: gx, grid_y: gy })),
            _ => {}
        }
    }

    let final_best = best.unwrap().1;
    autotuner_cache().lock().unwrap().insert(key, final_best.clone());
    final_best
}
