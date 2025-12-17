#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    Baseline38,
    Tiled63B,
    Micro64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileConfig {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
}

pub struct AutoTilingSelector;

impl AutoTilingSelector {
    pub fn choose_kernel(m: usize, n: usize, _k: usize) -> KernelKind {
        let max_dim = m.max(n);

        if max_dim < 128 {
            KernelKind::Baseline38
        } else if max_dim <= 512 {
            KernelKind::Micro64
        } else {
            KernelKind::Micro64
        }
    }

    pub fn choose_tile_sizes(_m: usize, _n: usize, k: usize) -> TileConfig {
        let bk = if k >= 64 { 64 } else { 32 };

        TileConfig { bm: 64, bn: 64, bk }
    }
}
