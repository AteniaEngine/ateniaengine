#[derive(Debug, Clone)]
pub struct RuntimeFlags {
    pub enable_pex: bool,
    pub enable_workstealing: bool,
    pub enable_adaptive_pgl: bool,
    /// APX 8.6: enables the use of GPU mini-kernels v0 (VecAdd) when true.
    pub enable_gpu_kernels: bool,
    /// APX 9.6: enables the simulated GPU memory planner.
    pub enable_gpu_memory_planner: bool,
    /// APX 9.7: enables the simulated GPU execution planner.
    pub enable_gpu_execution_planner: bool,
    /// APX 9.8: enables the simulated GPU executor.
    pub enable_gpu_executor_mock: bool,
}

impl RuntimeFlags {
    pub fn default() -> Self {
        Self {
            enable_pex: false,
            enable_workstealing: false,
            enable_adaptive_pgl: false,
            enable_gpu_kernels: false,
            enable_gpu_memory_planner: false,
            enable_gpu_execution_planner: false,
            enable_gpu_executor_mock: false,
        }
    }
}
