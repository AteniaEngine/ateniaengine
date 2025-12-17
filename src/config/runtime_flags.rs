#[derive(Debug, Clone)]
pub struct RuntimeFlags {
    pub enable_pex: bool,
    pub enable_workstealing: bool,
    pub enable_adaptive_pgl: bool,
    /// APX 8.6: habilita el uso de mini-kernels GPU v0 (VecAdd) cuando true.
    pub enable_gpu_kernels: bool,
    /// APX 9.6: habilita el planificador de memoria GPU simulado.
    pub enable_gpu_memory_planner: bool,
    /// APX 9.7: habilita el planificador de ejecución GPU simulado.
    pub enable_gpu_execution_planner: bool,
    /// APX 9.8: habilita el ejecutor GPU simulado.
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
