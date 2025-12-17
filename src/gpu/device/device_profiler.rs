use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub sm_count: i32,
    pub max_threads_per_sm: i32,
    pub max_threads_per_block: i32,
    pub warp_size: i32,
    pub shared_mem_per_sm: usize,
    pub max_registers_per_sm: i32,
}

static PROFILE: OnceLock<DeviceProfile> = OnceLock::new();

pub fn device_profile() -> &'static DeviceProfile {
    PROFILE.get_or_init(|| {
        // Implementación mínima sin dependencias externas.
        // En entornos sin GPU, devolvemos defaults "seguros".
        DeviceProfile {
            sm_count: 0,
            max_threads_per_sm: 2048,
            max_threads_per_block: 1024,
            warp_size: 32,
            shared_mem_per_sm: 0,
            max_registers_per_sm: 0,
        }
    })
}
