/// APX 12.1: AutoPlanner v3 para configurar grid/block/shared_mem
/// de kernels GPU de forma consistente.

pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
}

pub struct AutoPlanner;

impl AutoPlanner {
    /// Planeador sencillo para matmul cuadrado N x N.
    ///
    /// Versión inicial (12.1): heurística estática basada en N.
    /// Más adelante se puede extender para leer atributos reales de CUDA
    /// (threads/SM, warp size, etc.) y ajustar occupancy.
    pub fn plan_square_matmul(n: usize) -> LaunchConfig {
        // Selección de blockDim básica.
        // Para N >= 16 usamos 16x16, que es un layout razonable.
        // Este layout cumple con el test pedido para N=512.
        let (block_x, block_y) = if n >= 16 { (16u32, 16u32) } else { (8u32, 8u32) };

        // Calcular gridDim como techo de N / block.
        let grid_x = ((n as u32) + block_x - 1) / block_x;
        let grid_y = ((n as u32) + block_y - 1) / block_y;

        // Memoria compartida estimada para dos tiles de tamaño block_x * block_y.
        let tile_elems = (block_x as u32) * (block_y as u32);
        let shared_mem = tile_elems * 2 * 4; // 2 tiles * sizeof(f32)

        LaunchConfig {
            grid: (grid_x, grid_y, 1),
            block: (block_x, block_y, 1),
            shared_mem,
        }
    }
}
