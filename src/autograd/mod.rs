use crate::amg::grad_store::GradStore;
use crate::tensor::Tensor;
use std::collections::HashMap;

pub struct BackOp {
    pub inputs: Vec<usize>,
    pub output: usize,
    pub backward: Box<dyn Fn(&GradStore, &[&Tensor], &Tensor) + Send + Sync>,
}

pub struct BackwardTape {
    pub ops: Vec<BackOp>,
    index: HashMap<usize, usize>,
}

impl BackwardTape {
    pub fn new() -> Self {
        Self {
            ops: Vec::new(),
            index: HashMap::new(),
        }
    }

    pub fn push(&mut self, op: BackOp) -> usize {
        let node_id = op.output;
        let idx = self.ops.len();
        self.index.insert(node_id, idx);
        self.ops.push(op);
        idx
    }

    pub fn clear(&mut self) {
        self.ops.clear();
        self.index.clear();
    }

    pub fn get(&self, node_id: usize) -> Option<&BackOp> {
        self.index
            .get(&node_id)
            .and_then(|&idx| self.ops.get(idx))
    }

    pub fn has_op(&self, node_id: usize) -> bool {
        self.index.contains_key(&node_id)
    }
}
