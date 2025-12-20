/// APX 12.1: AutoPlanner v3 to configure grid/block/shared_mem
/// for GPU kernels in a consistent way.

pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem: u32,
}

pub struct AutoPlanner;

impl AutoPlanner {
    /// Simple planner for square matmul N x N.
    ///
    /// Initial version (12.1): static heuristic based on N.
    /// Later this can be extended to read real CUDA attributes
    /// (threads/SM, warp size, etc.) and adjust occupancy.
    pub fn plan_square_matmul(n: usize) -> LaunchConfig {
        // Basic blockDim selection.
        // For N >= 16 we use 16x16, which is a reasonable layout.
        // This layout satisfies the requested test for N=512.
        let (block_x, block_y) = if n >= 16 { (16u32, 16u32) } else { (8u32, 8u32) };

        // Compute gridDim as ceil(N / block).
        let grid_x = ((n as u32) + block_x - 1) / block_x;
        let grid_y = ((n as u32) + block_y - 1) / block_y;

        // Estimated shared memory for two tiles of size block_x * block_y.
        let tile_elems = (block_x as u32) * (block_y as u32);
        let shared_mem = tile_elems * 2 * 4; // 2 tiles * sizeof(f32)

        LaunchConfig {
            grid: (grid_x, grid_y, 1),
            block: (block_x, block_y, 1),
            shared_mem,
        }
    }
}
