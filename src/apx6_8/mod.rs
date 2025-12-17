pub struct BlockConfig {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub score: f64,
}

pub struct BlockSizePredictor {
    pub tested: Vec<BlockConfig>,
}

impl BlockSizePredictor {
    pub fn new() -> Self {
        Self { tested: Vec::new() }
    }

    pub fn candidate_block_sizes() -> Vec<(usize, usize, usize)> {
        vec![
            (32, 32, 32),
            (64, 64, 32),
            (64, 64, 64),
            (128, 64, 32),
            (64, 128, 32),
        ]
    }

    pub fn record_result(&mut self, bm: usize, bn: usize, bk: usize, us: u128) {
        self.tested.push(BlockConfig {
            bm,
            bn,
            bk,
            score: us as f64,
        });
    }

    pub fn best_block(&self) -> Option<(usize, usize, usize)> {
        if self.tested.is_empty() {
            None
        } else {
            let best = self
                .tested
                .iter()
                .min_by(|a, b| a.score.total_cmp(&b.score))
                .unwrap();
            Some((best.bm, best.bn, best.bk))
        }
    }
}
