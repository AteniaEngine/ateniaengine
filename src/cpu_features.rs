use rayon::ThreadPoolBuilder;
use std::arch::is_x86_feature_detected;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    pub avx2: bool,
    pub avx512f: bool,
    pub fma: bool,
    pub threads: usize,
}

static FEATURES: OnceLock<CpuFeatures> = OnceLock::new();

pub fn cpu_features() -> &'static CpuFeatures {
    FEATURES.get_or_init(|| CpuFeatures {
        avx2: is_x86_feature_detected!("avx2"),
        avx512f: is_x86_feature_detected!("avx512f"),
        fma: is_x86_feature_detected!("fma"),
        threads: num_cpus::get().max(2),
    })
}

pub fn init_parallel_runtime() {
    let threads = num_cpus::get().max(2);
    let _ = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok();
}
