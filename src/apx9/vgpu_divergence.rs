// APX 9.19 — Warp Divergence Solver (WDS)
// Modelo de máscaras y pila de reconvergencia SIMT, totalmente simulado en CPU.

#[derive(Debug, Clone)]
pub struct WarpMask {
    pub lanes: [bool; 32],
}

impl WarpMask {
    pub fn full() -> Self {
        Self { lanes: [true; 32] }
    }

    pub fn empty() -> Self {
        Self { lanes: [false; 32] }
    }

    pub fn from_predicate(pred: &[bool]) -> Self {
        let mut lanes = [false; 32];
        for (i, p) in pred.iter().enumerate().take(32) {
            lanes[i] = *p;
        }
        Self { lanes }
    }
}

#[derive(Debug, Clone)]
pub struct DivergenceFrame {
    pub mask: WarpMask,
    pub join_pc: usize,
}

#[derive(Debug, Clone)]
pub struct DivergenceStack {
    pub stack: Vec<DivergenceFrame>,
}

impl DivergenceStack {
    pub fn new() -> Self {
        Self { stack: vec![] }
    }

    pub fn push(&mut self, mask: WarpMask, join_pc: usize) {
        self.stack.push(DivergenceFrame { mask, join_pc });
    }

    pub fn pop(&mut self) -> Option<DivergenceFrame> {
        self.stack.pop()
    }
}
