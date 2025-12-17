use crate::amg::graph::Graph;
use crate::optim::adamw::AdamW;
use crate::profiler::profiler::Profiler;
use crate::tensor::Tensor;

pub struct TrainerV2 {
    pub graph: Graph,
    pub param_ids: Vec<usize>,
    pub optim: AdamW,
    step: usize,
}

impl TrainerV2 {
    pub fn new(graph: Graph, param_ids: Vec<usize>, optim: AdamW) -> Self {
        Self {
            graph,
            param_ids,
            optim,
            step: 0,
        }
    }

    pub fn train_step(&mut self, inputs: Vec<Tensor>) -> Vec<Tensor> {
        let mut profiler = Profiler::new();
        profiler.begin_step();

        profiler.begin_forward();
        let outputs = self.graph.execute(inputs);
        profiler.end_forward();

        let loss_id = self.graph.last_output_id();
        profiler.begin_backward();
        self.graph.backward(loss_id);
        profiler.end_backward();

        profiler.begin_optim();
        self.graph.apply_optimizer(&mut self.optim, &self.param_ids);
        profiler.end_optim();

        self.graph.clear_all_grads();

        let metrics = profiler.finalize_step();
        if self.step % 20 == 0 {
            profiler.print(self.step, &metrics);
        }
        self.step += 1;

        outputs
    }
}
