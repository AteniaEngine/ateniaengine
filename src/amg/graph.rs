#[allow(unused_imports)]
use crate::tensor::ops::*;
use crate::apx3_9::op_router::{route, ExecTarget};
use crate::apx4::gpu_dispatch::{dispatch_matmul as dispatch_matmul_gpu, ApxExecTarget as Apx4ExecTarget};
use crate::apx4_5::dispatcher::dispatch_batch_matmul_cuda;
use crate::apx4_3::GpuPlan;
use crate::apx4_11::gpu_hooks;
use crate::apx4_7::{PersistentPlan, FusionPlan};
use crate::apx4_13::fusion_engine::FusedOp;
use crate::apx5::kernel_planner::{KernelPlanner, KernelTarget};
use crate::apx5::apx_5_3_planner::{Planner5_3, NodeExecInfo};
use crate::apx5_4::{Sample, DeviceTarget};
use crate::tensor::{Layout, Tensor};
use crate::cpu_features::cpu_features;
use super::chunking::{chunk_tensor, merge_chunks};
use super::nodes::{Node, NodeType};
use super::scheduler::{build_execution_plan, ExecStep, ExecutionPlan};
use crate::amg::grad_store::GradStore;
use crate::autograd::{BackOp, BackwardTape};
use rayon::prelude::*;
use crate::nn::activations as nn_act;
use crate::nn::linear as nn_linear;
use std::time::Instant;
use crate::nn::normalization as nn_norm;
use crate::nn::softmax as nn_softmax;
use crate::optim::adamw::AdamW;
#[cfg(debug_assertions)]
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub enum FusedOutput {
    QKV { q: Tensor, k: Tensor, v: Tensor },
    /// APX 4.17: output of fused Self-Attention (forward-only).
    SelfAttention {
        q: Tensor,
        k: Tensor,
        v: Tensor,
        att: Tensor,
        out: Tensor,
    },
}

pub struct Graph {
    pub nodes: Vec<Node>,
    pub plan: ExecutionPlan,
    pub tape: BackwardTape,
    pub grad_store: GradStore,
    pub gpu_plan: Option<GpuPlan>,
    pub persistent_plan: Option<PersistentPlan>,
    pub fusion_plan: Option<FusionPlan>,
    pub fusions_applied: usize,
    pub fused_ops: std::collections::HashMap<usize, FusedOp>,
    pub fused_outputs: std::collections::HashMap<usize, FusedOutput>,
    /// APX 8.1: optional CPU+GPU dual graph. It is not used for execution yet;
    /// it is only built as structural metadata.
    pub dual_graph: Option<crate::apx8::dualgraph::DualGraph>,
}

impl Graph {
    /// Returns true if the given node is the secondary Linear (K or V) of
    /// some FusedQKV pattern detected in the graph.
    pub fn is_qkv_secondary(&self, id: usize) -> bool {
        for fused in self.fused_ops.values() {
            if let FusedOp::FusedQKV { q_id, k_id, v_id, .. } = fused {
                if id == *k_id || id == *v_id {
                    return true;
                }
                // The representative node (q_id) is handled via exec_fused.
                let _ = q_id; // avoid warnings in builds without direct usage
            }
        }
        false
    }

    /// Returns true if the given node is one of the Q/K/V linear nodes
    /// involved in a FusedSelfAttention pattern.
    pub fn is_sa_linear(&self, id: usize) -> bool {
        for fused in self.fused_ops.values() {
            if let FusedOp::FusedSelfAttention { q, k, v, .. } = fused {
                if id == *q || id == *k || id == *v {
                    return true;
                }
            }
        }
        false
    }
}

fn log_softmax_last_dim(x: &Tensor) -> Tensor {
    assert!(x.shape.len() >= 1, "LogSoftmax requires tensors with rank >= 1");
    let ndim = x.shape.len();
    let cols = *x.shape.last().expect("log_softmax needs last dim");
    let rows = if ndim == 1 {
        1
    } else {
        x.shape[..ndim - 1].iter().product()
    };

    let mut out = Tensor::with_layout(
        x.shape.clone(),
        0.0,
        x.device,
        Layout::Contiguous,
        x.dtype,
    );

    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let slice = &x.data[start..end];
        let mut max_val = f32::NEG_INFINITY;
        for &v in slice {
            if v > max_val {
                max_val = v;
            }
        }
        let mut sum_exp = 0.0f32;
        let mut temp = Vec::with_capacity(cols);
        for &v in slice {
            let e = (v - max_val).exp();
            temp.push(e);
            sum_exp += e;
        }
        let log_denom = max_val + sum_exp.ln();
        for i in 0..cols {
            out.data[start + i] = slice[i] - log_denom;
        }
    }

    out
}

fn gather_last_dim(data: &Tensor, indices: &Tensor) -> Tensor {
    assert!(
        data.shape.len() >= 1,
        "Gather data must have at least one dimension"
    );
    let last_dim = *data.shape.last().expect("gather last dim");
    let data_rows = data.data.len() / last_dim;
    assert_eq!(
        data_rows,
        indices.data.len(),
        "Gather indices must match data rows"
    );

    let mut out = Tensor::with_layout(
        indices.shape.clone(),
        0.0,
        data.device,
        Layout::Contiguous,
        data.dtype,
    );

    for row in 0..data_rows {
        let idx = indices.data[row].round() as isize;
        assert!(idx >= 0 && (idx as usize) < last_dim, "Gather index out of bounds");
        let src = row * last_dim + idx as usize;
        out.data[row] = data.data[src];
    }

    out
}

impl Clone for Graph {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            plan: self.plan.clone(),
            tape: BackwardTape::new(),
            grad_store: GradStore::new(),
            gpu_plan: self.gpu_plan.clone(),
            persistent_plan: self.persistent_plan.clone(),
            fusion_plan: self.fusion_plan.clone(),
            fusions_applied: self.fusions_applied,
            fused_ops: self.fused_ops.clone(),
            fused_outputs: self.fused_outputs.clone(),
            dual_graph: None,
        }
    }
}

impl Graph {
    pub fn new(nodes: Vec<Node>) -> Self {
        Self::build(nodes)
    }

    /// Build a graph from pre-constructed nodes.
    /// Initializes execution plan, backward tape and an empty GradStore.
    pub fn build(nodes: Vec<Node>) -> Self {
        let (plan, fused_ops) = build_execution_plan(&nodes);
        let node_types: Vec<_> = nodes.iter().map(|n| n.node_type.clone()).collect();
        let gpu_plan = Some(GpuPlan::build(&node_types));

        let mut graph = Self {
            nodes,
            plan,
            tape: BackwardTape::new(),
            grad_store: GradStore::new(),
            gpu_plan,
            persistent_plan: None,
            fusion_plan: None,
            fusions_applied: 0,
            fused_ops,
            fused_outputs: std::collections::HashMap::new(),
            dual_graph: None,
        };

        graph.persistent_plan = Some(PersistentPlan::analyze(&graph));
        graph.fusion_plan = Some(FusionPlan::analyze(&graph));

        // APX 7.8: generate temporal-locality hints per node. This does not
        // modify the graph nor its math; it only provides metadata so that
        // TLO can reorder independent nodes in HPGE.
        if !graph.nodes.is_empty() {
            use crate::apx7::tlo::{LocalityHint, set_locality_hints};

            let mut hints = vec![LocalityHint { branch_id: 0, depth: 0 }; graph.nodes.len()];
            for i in 0..graph.nodes.len() {
                let node = &graph.nodes[i];
                if node.inputs.is_empty() {
                    hints[i] = LocalityHint { branch_id: i, depth: 0 };
                } else {
                    let parent = node.inputs[0];
                    if parent < hints.len() {
                        hints[i].branch_id = hints[parent].branch_id;
                        hints[i].depth = hints[parent].depth.saturating_add(1);
                    } else {
                        hints[i] = LocalityHint { branch_id: i, depth: 0 };
                    }
                }
            }

            set_locality_hints(hints);
        }

        // APX 8.1: if the active mode requires it, also build the structural
        // DualGraph. This does not touch backward nor execute GPU.
        if crate::apx_mode_at_least("8.1") {
            graph.build_plan();
        }

        graph
    }

    /// Rebuild the execution plan from the current nodes. APX 8.1: when the
    /// mode allows it, also builds the structural CPU+GPU DualGraph.
    pub fn build_plan(&mut self) {
        let (plan, fused_ops) = build_execution_plan(&self.nodes);
        self.plan = plan;
        self.fused_ops = fused_ops;

        if crate::apx_mode_at_least("8.1") {
            self.dual_graph = Some(crate::apx8::dualgraph::DualGraphBuilder::build(self));
        }
    }

    /// Validate structural consistency of the graph.
    ///
    /// Checks:
    /// - Per-node inputs match what is expected for the NodeType.
    /// - All input indices reference existing nodes.
    /// - There are no cycles in the directed graph (from inputs to consumers).
    /// - All nodes are reachable from at least one Input node.
    pub fn validate(&self) -> Result<(), String> {
        // 1) Index range and expected number of inputs.
        for (id, node) in self.nodes.iter().enumerate() {
            for &inp in &node.inputs {
                if inp >= self.nodes.len() {
                    return Err(format!(
                        "Node {id} ({:?}) references a non-existent input: {inp}",
                        node.node_type
                    ));
                }
            }

            let in_len = node.inputs.len();
            let ok = match &node.node_type {
                NodeType::Input | NodeType::Parameter => in_len == 0,
                NodeType::Add
                | NodeType::Sub
                | NodeType::Mul
                | NodeType::MatMul
                | NodeType::BatchMatMul
                | NodeType::BroadcastAdd
                | NodeType::Gather
                | NodeType::CrossEntropyLoss => in_len == 2,
                NodeType::Reshape { .. }
                | NodeType::Transpose2D
                | NodeType::TransposeLastTwo
                | NodeType::RmsNorm
                | NodeType::SiLU
                | NodeType::Softmax
                | NodeType::LogSoftmax
                | NodeType::Output => in_len == 1,
                NodeType::IndexSelect => in_len == 2,
                NodeType::Linear => in_len == 2 || in_len == 3,
                NodeType::Activation(_) => in_len == 1,
                NodeType::FusedLinearActivation(_) => in_len == 2 || in_len == 3,
                NodeType::FusedLinearActivationChain(_) => in_len == 4 || in_len == 5,
                NodeType::NoOp => in_len == 1,
            };

            if !ok {
                return Err(format!(
                    "Node {id} ({:?}) has {} inputs, but a different number was expected",
                    node.node_type, in_len
                ));
            }
        }

        // 2) Build children list for cycle and reachability analysis.
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        for (id, node) in self.nodes.iter().enumerate() {
            for &inp in &node.inputs {
                children[inp].push(id);
            }
        }

        // 3) Cycle detection with DFS (colors: 0=white, 1=gray, 2=black).
        let mut color = vec![0u8; self.nodes.len()];
        fn dfs_cycle(
            u: usize,
            children: &Vec<Vec<usize>>,
            color: &mut [u8],
        ) -> Option<(usize, usize)> {
            color[u] = 1;
            for &v in &children[u] {
                match color[v] {
                    0 => {
                        if let Some(c) = dfs_cycle(v, children, color) {
                            return Some(c);
                        }
                    }
                    1 => {
                        return Some((u, v));
                    }
                    _ => {}
                }
            }
            color[u] = 2;
            None
        }

        for start in 0..self.nodes.len() {
            if color[start] == 0 {
                if let Some((u, v)) = dfs_cycle(start, &children, &mut color) {
                    return Err(format!(
                        "A cycle was detected in the graph between nodes {u} and {v}",
                    ));
                }
            }
        }

        // 4) Reachability from Input nodes.
        let mut reachable = vec![false; self.nodes.len()];
        let mut stack = Vec::new();
        for (id, node) in self.nodes.iter().enumerate() {
            if matches!(node.node_type, NodeType::Input) {
                reachable[id] = true;
                stack.push(id);
            }
        }

        while let Some(u) = stack.pop() {
            for &v in &children[u] {
                if !reachable[v] {
                    reachable[v] = true;
                    stack.push(v);
                }
            }
        }

        for (id, node) in self.nodes.iter().enumerate() {
            if !reachable[id]
                && !matches!(node.node_type, NodeType::Parameter)
            {
                return Err(format!(
                    "Node {id} ({:?}) is not reachable from any Input node",
                    node.node_type
                ));
            }
        }

        Ok(())
    }

    /// Dynamically append a node to the graph and rebuild the execution plan.
    pub fn add_node_of_type(&mut self, node_type: NodeType, inputs: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node::new(id, node_type, inputs));
        let (plan, fused_ops) = build_execution_plan(&self.nodes);
        self.plan = plan;
        self.fused_ops = fused_ops;
        id
    }

    /// Convenience helper for parameter (constant/trainable) tensors.
    pub fn add_parameter(&mut self, tensor: Tensor) -> usize {
        let id = self.add_node_of_type(NodeType::Parameter, vec![]);
        self.nodes[id].output = Some(tensor);
        id
    }

    /// Standard execution with automatic execution plan (including fusion).
    pub fn execute(&mut self, inputs: Vec<Tensor>) -> Vec<Tensor> {
        self.tape.clear();
        self.set_input_outputs(inputs);
        if crate::apx_mode_at_least("7.12") {
            crate::apx7::ule::ule_execute_graph(self);
        } else if crate::apx_mode_at_least("7.11") {
            crate::apx7::hls_deep::execute_graph_hls_deep(self);
        } else if crate::apx_mode_at_least("7.10") {
            crate::apx7::hls_deep::execute_graph_hls_deep(self);
        } else if crate::apx_mode_at_least("7.7") {
            crate::apx7::hpge_priority::execute_graph_parallel_priority(self);
        } else if crate::apx_mode_at_least("7.5") {
            crate::apx7::hpge::execute_graph_parallel(self);
        } else {
            self.run_plan(true);
        }

        // APX 6.11 / 6.12 / 6.13: update global execution policies based on
        // measurements accumulated by the FusionSelector 6.10.
        if crate::apx_mode_at_least("6.11") {
            if let Ok(sel) = crate::apx6_10::global_fusion_selector().lock() {
                if let Some(global_decision) = sel.best_decision() {
                    use crate::apx6_10::GlobalDecision;
                    use crate::apx6_11::runtime_policy::{set_runtime_policy, FusionRuntimePolicy};

                    // Deterministic 6.11 policy, also used as the base for 6.12
                    // scheduling hints.
                    match global_decision {
                        GlobalDecision::PreferFull => {
                            set_runtime_policy(FusionRuntimePolicy::PreferFull);
                        }
                        GlobalDecision::PreferQKV => {
                            set_runtime_policy(FusionRuntimePolicy::PreferQKV);
                        }
                        GlobalDecision::NoPreference => {
                            set_runtime_policy(FusionRuntimePolicy::Baseline);
                        }
                    }

                    // APX 6.12: also derive the pure scheduling bias.
                    if crate::apx_mode_at_least("6.12") {
                        use crate::apx6_12::adaptive_scheduler::{
                            AdaptiveScheduleBias,
                            set_schedule_bias,
                        };

                        match global_decision {
                            GlobalDecision::PreferQKV =>
                                set_schedule_bias(AdaptiveScheduleBias::QKVHeavy),
                            GlobalDecision::PreferFull =>
                                set_schedule_bias(AdaptiveScheduleBias::AttentionHeavy),
                            GlobalDecision::NoPreference =>
                                set_schedule_bias(AdaptiveScheduleBias::None),
                        }
                    }

                    // APX 6.13: probabilistic "tempered" policy that replaces
                    // the 6.11 global policy, but still only affects planning hints.
                    if crate::apx_mode_at_least("6.13") {
                        use crate::apx6_11::runtime_policy::{set_runtime_policy, FusionRuntimePolicy};

                        // APX 6.14: update the adaptive temperature before
                        // computing the tempered softmax. For now we use a
                        // synthetic integration-level step (0); the 6.14 tests
                        // exercise the real decay by explicitly calling
                        // update_temperature.
                        if crate::apx_mode_at_least("6.14") {
                            crate::apx6_14::temperature_manager::update_temperature(0);
                        }

                        let (full_s, qkv_s, base_s) = sel.normalized_scores();
                        let t = if crate::apx_mode_at_least("6.14") {
                            crate::apx6_14::temperature_schedule::get_current_temperature()
                        } else {
                            0.8 // fixed value in APX 6.13
                        };

                        let td = crate::softmax3(full_s, qkv_s, base_s, t);
                        let choice = crate::sample_decision(&td);

                        match choice {
                            "full" => set_runtime_policy(FusionRuntimePolicy::PreferFull),
                            "qkv"  => set_runtime_policy(FusionRuntimePolicy::PreferQKV),
                            _       => set_runtime_policy(FusionRuntimePolicy::Baseline),
                        }
                    }

                    // APX 6.15: stabilize the global decision using the
                    // current temperature as an exploration control, without
                    // touching forward/backward nor real tensors.
                    if crate::apx_mode_at_least("6.15") {
                        use crate::apx6_15::stabilizer::ApxTemperature;

                        let t_val = if crate::apx_mode_at_least("6.14") {
                            crate::apx6_14::temperature_schedule::get_current_temperature()
                        } else {
                            1.0
                        };

                        let temp = ApxTemperature::from_value(t_val);

                        if let Ok(mut stab) = crate::apx6_15::stabilizer::global_stabilizer().write() {
                            let decision = sel.best_decision();
                            let _final = stab.stabilize(decision, &temp);
                        }
                    }
                }
            }
        }

        self.collect_outputs()
    }

    /// APX 7.5: opt-in execution using the Hierarchical Parallel Graph
    /// Executor (HPGE). This API does not replace the standard path; it only
    /// forces mode 7.5 before delegating to `execute`.
    pub fn execute_hpge(&mut self, inputs: Vec<Tensor>) -> Vec<Tensor> {
        unsafe {
            std::env::set_var("ATENIA_APX_MODE", "7.5");
        }
        self.execute(inputs)
    }

    pub fn apply_optimizer(&mut self, optim: &mut AdamW, param_ids: &[usize]) {
        let mut params = self.get_params_mut(param_ids);
        optim.update(params.as_mut_slice());
    }

    pub fn last_output_id(&self) -> usize {
        self
            .nodes
            .iter()
            .rev()
            .find(|node| matches!(node.node_type, NodeType::Output))
            .map(|node| node.id)
            .expect("Graph must contain at least one Output node")
    }

    pub fn clear_all_grads(&mut self) {
        for node in &mut self.nodes {
            if let Some(out) = &mut node.output {
                out.clear_grad();
            }
        }
    }

    /// Execute graph in chunks for 1D element-wise style graphs.
    /// - `inputs`: same number of tensors as Input nodes
    /// - `max_chunk_elements`: maximum number of elements per chunk
    pub fn execute_chunked(
        &mut self,
        inputs: Vec<Tensor>,
        max_chunk_elements: usize,
    ) -> Vec<Tensor> {
        assert!(
            !inputs.is_empty(),
            "execute_chunked requires at least one input tensor"
        );

        let original_shape = inputs[0].shape.clone();

        let mut chunked_inputs: Vec<Vec<Tensor>> = Vec::new();
        for t in &inputs {
            chunked_inputs.push(chunk_tensor(t, max_chunk_elements));
        }

        let num_chunks = chunked_inputs[0].len();
        for ci in &chunked_inputs {
            assert_eq!(
                ci.len(),
                num_chunks,
                "all inputs must produce the same number of chunks"
            );
        }

        self.tape.clear();
        let mut output_chunks: Vec<Tensor> = Vec::new();

        for chunk_idx in 0..num_chunks {
            let mut per_chunk_inputs = Vec::new();
            for ci in &chunked_inputs {
                per_chunk_inputs.push(ci[chunk_idx].clone());
            }

            self.clear_intermediate_outputs();
            self.set_input_outputs(per_chunk_inputs);
            self.run_plan(false);

            let mut chunk_outputs = self.collect_outputs();
            assert_eq!(
                chunk_outputs.len(),
                1,
                "execute_chunked currently supports a single Output node"
            );

            output_chunks.push(chunk_outputs.remove(0));
        }

        vec![merge_chunks(output_chunks, original_shape)]
    }

    fn clear_intermediate_outputs(&mut self) {
        for node in &mut self.nodes {
            if matches!(node.node_type, NodeType::Input | NodeType::Parameter) {
                continue;
            }
            node.output = None;
        }
    }

    fn set_input_outputs(&mut self, inputs: Vec<Tensor>) {
        let mut provided = inputs.into_iter();
        for node in &mut self.nodes {
            if matches!(node.node_type, NodeType::Input) {
                let tensor = provided
                    .next()
                    .expect("not enough input tensors provided for graph execution");
                node.set_output(tensor);
            }
        }

        if provided.next().is_some() {
            panic!("too many input tensors provided for graph execution");
        }
    }

    pub(crate) fn run_plan(&mut self, record_tape: bool) {
        let steps = self.plan.steps.clone();
        for step in steps {
            match step {
                ExecStep::Single(node_id) => self.execute_single(node_id, record_tape),
                ExecStep::FusedAddMul { add_node, mul_node } => {
                    if record_tape {
                        self.execute_single(add_node, true);
                        self.execute_single(mul_node, true);
                    } else {
                        self.execute_fused_add_mul(add_node, mul_node);
                    }
                }
            }
        }

    }

    fn exec_fused(&mut self, id: usize, fused: FusedOp, record_tape: bool) {
        let apx_mode = crate::apx_mode();
        let is_69_or_higher = apx_mode.starts_with("6.9") || apx_mode > "6.9".to_string();
        let is_610_or_higher = apx_mode.starts_with("6.10") || apx_mode > "6.10".to_string();

        match fused {
            FusedOp::LinearSilu { x, w, b } => {
                unsafe {
                    crate::apx4_11::gpu_hooks::fused_linear_silu_gpu(
                        x,
                        w,
                        b,
                        id,
                        self,
                        record_tape,
                    );
                }
            }
            FusedOp::FusedQKV { x, wq, wk, wv, bq, bk, bv, q_id, k_id, v_id } => {
                use std::time::Instant;

                // Real forward (prototype) for Q/K/V sharing the same X.
                if crate::apx_debug_enabled() {
                    eprintln!("[APX 4.14] executing fused QKV at node {}", id);
                }

                let x_t = self.nodes[x]
                    .output
                    .as_ref()
                    .expect("FusedQKV: missing x output")
                    .clone();
                let wq_t = self.nodes[wq]
                    .output
                    .as_ref()
                    .expect("FusedQKV: missing wq output")
                    .clone();
                let wk_t = self.nodes[wk]
                    .output
                    .as_ref()
                    .expect("FusedQKV: missing wk output")
                    .clone();
                let wv_t = self.nodes[wv]
                    .output
                    .as_ref()
                    .expect("FusedQKV: missing wv output")
                    .clone();

                let bq_t = bq.and_then(|i| self.nodes[i].output.as_ref()).cloned();
                let bk_t = bk.and_then(|i| self.nodes[i].output.as_ref()).cloned();
                let bv_t = bv.and_then(|i| self.nodes[i].output.as_ref()).cloned();

                // In 6.9+ mode, we optionally measure naive vs fused timings.
                let mut use_fused = true;

                if is_69_or_higher {
                    // Naive (non-fused) measurement: execute Q, K, V separately.
                    let t0 = Instant::now();
                    let q_naive = nn_linear::linear(&x_t, &wq_t, bq_t.as_ref());
                    let k_naive = nn_linear::linear(&x_t, &wk_t, bk_t.as_ref());
                    let v_naive = nn_linear::linear(&x_t, &wv_t, bv_t.as_ref());
                    let unfused_time_us = t0.elapsed().as_micros() as u64;

                    // Fused measurement: reuse the current implementation.
                    let t1 = Instant::now();
                    let q_fused = q_naive.clone();
                    let k_fused = k_naive.clone();
                    let v_fused = v_naive.clone();
                    let fused_time_us = t1.elapsed().as_micros() as u64;

                    if let Ok(mut fp) = crate::apx6_9::fusion_profiler::fusion_profiler().lock() {
                        fp.record("FusedQKV", unfused_time_us, fused_time_us);
                        if let Some(decision) = fp.should_use_fused("FusedQKV") {
                            use_fused = decision;
                        }
                    }

                    // Depending on the decision, we use the already-computed
                    // naive tensors or let the fused path generate them. To avoid
                    // duplicate computation, if we decide not to use fused, we
                    // directly reuse q_naive/k_naive/v_naive.
                    if !use_fused {
                        self.nodes[q_id].set_output(q_naive.clone());
                        self.nodes[k_id].set_output(k_naive.clone());
                        self.nodes[v_id].set_output(v_naive.clone());
                        self.fused_outputs.insert(
                            id,
                            FusedOutput::QKV { q: q_naive, k: k_naive, v: v_naive },
                        );
                    } else {
                        self.nodes[q_id].set_output(q_fused.clone());
                        self.nodes[k_id].set_output(k_fused.clone());
                        self.nodes[v_id].set_output(v_fused.clone());
                        self.fused_outputs.insert(
                            id,
                            FusedOutput::QKV { q: q_fused, k: k_fused, v: v_fused },
                        );
                    }
                } else {
                    // Original 4.14 behavior when APX < 6.9.
                    let q = nn_linear::linear(&x_t, &wq_t, bq_t.as_ref());
                    let k = nn_linear::linear(&x_t, &wk_t, bk_t.as_ref());
                    let v = nn_linear::linear(&x_t, &wv_t, bv_t.as_ref());

                    self.nodes[q_id].set_output(q.clone());
                    self.nodes[k_id].set_output(k.clone());
                    self.nodes[v_id].set_output(v.clone());
                    self.fused_outputs.insert(id, FusedOutput::QKV { q, k, v });
                }

                // APX 4.16: record a single fused BackOp for QKV when
                // recording the backward tape.
                if crate::apx_mode() == "4.16" && record_tape {
                    let has_bq = bq.is_some();
                    let has_bk = bk.is_some();
                    let has_bv = bv.is_some();

                    let bq_id = bq;
                    let bk_id = bk;
                    let bv_id = bv;

                    self.tape.push(BackOp {
                        inputs: vec![x, wq, wk, wv],
                        output: id,
                        backward: Box::new(move |store, forward_inputs, _out_grad| {
                            let x_f = forward_inputs[0];
                            let wq_f = forward_inputs[1];
                            let wk_f = forward_inputs[2];
                            let wv_f = forward_inputs[3];

                            // gQ, gK, gV accumulated in GradStore for Q, K, V nodes.
                            let mut gq_data = store.get(q_id);
                            let mut gk_data = store.get(k_id);
                            let mut gv_data = store.get(v_id);

                            if gq_data.is_empty() && gk_data.is_empty() && gv_data.is_empty() {
                                return;
                            }

                            // Shapes derived from X and weights: Q, K, V have shape [m, n].
                            let m = x_f.shape[0];
                            let n = wq_f.shape[1];

                            // Ensure grad buffers are not empty to avoid out-of-bounds
                            // accesses in matmul. If any grad is empty, interpret it
                            // as all zeros.
                            let expected_len = m * n;
                            if gq_data.is_empty() {
                                gq_data = vec![0.0; expected_len];
                            }
                            if gk_data.is_empty() {
                                gk_data = vec![0.0; expected_len];
                            }
                            if gv_data.is_empty() {
                                gv_data = vec![0.0; expected_len];
                            }

                            let make_grad_tensor = |data: Vec<f32>, proto: &Tensor| -> Tensor {
                                Tensor {
                                    shape: vec![m, n],
                                    data,
                                    device: proto.device,
                                    dtype: proto.dtype,
                                    layout: proto.layout,
                                    strides: proto.strides.clone(),
                                    grad: None,
                                    gpu: None,
                                    persistence: None,
                                    op: None,
                                }
                            };

                            let gq = make_grad_tensor(gq_data, x_f);
                            let gk = make_grad_tensor(gk_data, x_f);
                            let gv = make_grad_tensor(gv_data, x_f);

                            // dX = gQ·Wq^T + gK·Wk^T + gV·Wv^T
                            let wq_t = transpose_2d(wq_f);
                            let wk_t = transpose_2d(wk_f);
                            let wv_t = transpose_2d(wv_f);

                            let dx_q = nn_linear::matmul(&gq, &wq_t);
                            let dx_k = nn_linear::matmul(&gk, &wk_t);
                            let dx_v = nn_linear::matmul(&gv, &wv_t);

                            let mut dx_total = dx_q.data.clone();
                            for i in 0..dx_total.len() {
                                dx_total[i] += dx_k.data[i] + dx_v.data[i];
                            }
                            add_to_grad_slice(store, x, &dx_total);

                            // dWq, dWk, dWv: X^T · g
                            let x_t = transpose_2d(x_f);

                            let dwq = nn_linear::matmul(&x_t, &gq);
                            add_to_grad_slice(store, wq, &dwq.data);

                            let dwk = nn_linear::matmul(&x_t, &gk);
                            add_to_grad_slice(store, wk, &dwk.data);

                            let dwv = nn_linear::matmul(&x_t, &gv);
                            add_to_grad_slice(store, wv, &dwv.data);

                            // Biases opcionales: sumar filas de gQ,gK,gV
                            if has_bq {
                                if let Some(bq_node) = bq_id {
                                    let bgrad_q = sum_rows(&gq);
                                    add_to_grad_slice(store, bq_node, &bgrad_q);
                                }
                            }
                            if has_bk {
                                if let Some(bk_node) = bk_id {
                                    let bgrad_k = sum_rows(&gk);
                                    add_to_grad_slice(store, bk_node, &bgrad_k);
                                }
                            }
                            if has_bv {
                                if let Some(bv_node) = bv_id {
                                    let bgrad_v = sum_rows(&gv);
                                    add_to_grad_slice(store, bv_node, &bgrad_v);
                                }
                            }
                        }),
                    });
                }
            }
            FusedOp::FusedSelfAttention { x, wq, wk, wv, bq, bk, bv, q, k, v, out_id: _ } => {
                if crate::apx_debug_enabled() {
                    eprintln!("[APX 4.17] executing fused SelfAttention at node {}", id);
                }

                // Retrieve Q, K, V tensors already materialized by the previous linear nodes.
                let q_t = self.nodes[q]
                    .output
                    .as_ref()
                    .expect("FusedSelfAttention: missing q output")
                    .clone();
                let k_t = self.nodes[k]
                    .output
                    .as_ref()
                    .expect("FusedSelfAttention: missing k output")
                    .clone();
                let v_t = self.nodes[v]
                    .output
                    .as_ref()
                    .expect("FusedSelfAttention: missing v output")
                    .clone();

                // MatMul(Q, K^T) followed by Softmax over the last dimension,
                // exactly the same as the naive graph (without scaling factor).
                let k_t_t = transpose_2d(&k_t);
                let scores = nn_linear::matmul(&q_t, &k_t_t);
                let att = nn_softmax::softmax_last_dim(&scores);

                // MatMul(A, V)
                let t0 = Instant::now();
                let out = nn_linear::matmul(&att, &v_t);
                let baseline_us = t0.elapsed().as_micros() as u64;

                // Materialize output in the MatMul A·V node (id) so that the rest
                // of the graph and the naive backward work unchanged.
                self.nodes[id].set_output(out.clone());

                self.fused_outputs.insert(
                    id,
                    FusedOutput::SelfAttention {
                        q: q_t.clone(),
                        k: k_t.clone(),
                        v: v_t.clone(),
                        att: att.clone(),
                        out: out.clone(),
                    },
                );

                // APX 6.10: auxiliary full-attention measurement without changing real forward/backward.
                if is_610_or_higher {
                    // Retrieve X, weights, and bias as seen by the graph.
                    let x_t = self.nodes[x]
                        .output
                        .as_ref()
                        .expect("FusedSelfAttention: missing x output")
                        .clone();
                    let wq_t = self.nodes[wq]
                        .output
                        .as_ref()
                        .expect("FusedSelfAttention: missing wq output")
                        .clone();
                    let wk_t = self.nodes[wk]
                        .output
                        .as_ref()
                        .expect("FusedSelfAttention: missing wk output")
                        .clone();
                    let wv_t = self.nodes[wv]
                        .output
                        .as_ref()
                        .expect("FusedSelfAttention: missing wv output")
                        .clone();

                    let bq_t = bq.and_then(|i| self.nodes[i].output.as_ref()).cloned();
                    let bk_t = bk.and_then(|i| self.nodes[i].output.as_ref()).cloned();
                    let bv_t = bv.and_then(|i| self.nodes[i].output.as_ref()).cloned();

                    // Look for a projection Linear that consumes out (baseline), if it exists.
                    let mut wproj_t: Option<crate::tensor::Tensor> = None;
                    let mut bias_proj_t: Option<crate::tensor::Tensor> = None;

                    for node in &self.nodes {
                        if let super::nodes::NodeType::Linear = node.node_type {
                            if !node.inputs.is_empty() && node.inputs[0] == id {
                                // inputs: [x, w, (b)]
                                let w_id = node.inputs[1];
                                wproj_t = self.nodes[w_id].output.clone();
                                if node.inputs.len() == 3 {
                                    let b_id = node.inputs[2];
                                    bias_proj_t = self.nodes[b_id].output.clone();
                                }
                                break;
                            }
                        }
                    }

                    if let Some(wproj_t) = wproj_t {
                        use crate::amg::fusions::execute_fused_attention_full;
                        use crate::apx6_10::{FusionProfile, global_fusion_selector};

                        let t_full = Instant::now();
                        let _y_full = execute_fused_attention_full(
                            &x_t,
                            &wq_t,
                            &wk_t,
                            &wv_t,
                            bq_t.as_ref(),
                            bk_t.as_ref(),
                            bv_t.as_ref(),
                            &wproj_t,
                            bias_proj_t.as_ref(),
                        );
                        let fused_full_us = t_full.elapsed().as_micros() as u64;

                        // For now we do not have a separate time for FusedQKV here; use 0.
                        let fused_qkv_us = 0u64;

                        if let Ok(mut sel) = global_fusion_selector().lock() {
                            sel.record_profile(FusionProfile {
                                op_name: "FusedAttention".to_string(),
                                baseline_us,
                                fused_qkv_us,
                                fused_full_us,
                            });
                        }
                    }
                }

                // APX 4.18: fused Self-Attention backward is disabled for now.
                // Forward remains fused, but backward uses the naive chain of
                // individual BackOps.
            }
        }
    }

    /// CPU forward for Linear nodes, reusing nn::linear without GPU.
    pub fn exec_cpu_linear_fallback(&mut self, id: usize) {
        let inputs = self.nodes[id].inputs.clone();
        assert!(
            inputs.len() == 2 || inputs.len() == 3,
            "Linear node expects 2 or 3 inputs",
        );

        let x_id = inputs[0];
        let w_id = inputs[1];

        // Ensure inputs have their outputs materialized before reading them.
        if self.nodes[x_id].output.is_none() {
            self.execute_single(x_id, false);
        }
        if self.nodes[w_id].output.is_none() {
            self.execute_single(w_id, false);
        }

        let x = self.nodes[x_id]
            .output
            .as_ref()
            .expect("Linear missing x")
            .clone();
        let w = self.nodes[w_id]
            .output
            .as_ref()
            .expect("Linear missing weight")
            .clone();

        let out = if inputs.len() == 3 {
            let b_id = inputs[2];
            if self.nodes[b_id].output.is_none() {
                self.execute_single(b_id, false);
            }
            let b = self.nodes[b_id]
                .output
                .as_ref()
                .expect("Linear missing bias")
                .clone();
            nn_linear::linear(&x, &w, Some(&b))
        } else {
            nn_linear::linear(&x, &w, None)
        };

        self.nodes[id].output = Some(out);
    }

    pub(crate) fn execute_single(&mut self, node_id: usize, record_tape: bool) {
        if crate::apx_mode_at_least("8.2") {
            let op = match self.nodes[node_id].node_type {
                NodeType::MatMul => "MatMul",
                NodeType::Linear => "Linear",
                _ => "Other",
            };
            return crate::apx8::hybrid_dispatcher::HybridDispatcher::dispatch(self, node_id, op, record_tape);
        }

        self.execute_single_inner(node_id, record_tape);
    }

    pub(crate) fn execute_single_inner(&mut self, node_id: usize, record_tape: bool) {
        let use_timing = crate::apx_mode_at_least("7.6");
        let t0 = if use_timing { Some(Instant::now()) } else { None };

        // APX 4.13: if there is a fused op associated with this node, delegate
        // to the fused executor and return. We clone the FusedOp to avoid
        // holding an immutable borrow of self while using it.
        if let Some(fused) = self.fused_ops.get(&node_id).cloned() {
            if use_timing {
                if let Some(start) = t0 {
                    let dt = start.elapsed().as_micros() as f64;
                    crate::apx7::hpge_priority::record_node_time(node_id, dt, self.nodes.len());
                }
            }
            return self.exec_fused(node_id, fused, record_tape);
        }

        // APX 4.9: directly execute fused Linear→[Act...]→Linear chains,
        // without going through APX 4.7 hooks nor GPU for this node.
        if let super::nodes::NodeType::FusedLinearActivationChain(acts) =
            self.nodes[node_id].node_type.clone()
        {
            if crate::apx_debug_enabled() && !crate::apx_is_silent() {
                eprintln!(
                    "[APX4.9 DEBUG] Executing FusedLinearActivationChain node_id={} | inputs={:?}",
                    node_id,
                    self.nodes[node_id].inputs
                );
            }

            let inputs = self.nodes[node_id].inputs.clone();
            assert!(
                inputs.len() == 3 || inputs.len() == 4 || inputs.len() == 5,
                "FusedLinearActivationChain expects 3, 4 or 5 inputs",
            );

            let x = self.nodes[inputs[0]]
                .output
                .as_ref()
                .expect("FusedLinearActivationChain missing x")
                .clone();
            let w1 = self.nodes[inputs[1]]
                .output
                .as_ref()
                .expect("FusedLinearActivationChain missing w1")
                .clone();

            let mut idx = 2;
            let b1_opt = if inputs.len() == 5 {
                let b1 = self.nodes[inputs[idx]]
                    .output
                    .as_ref()
                    .expect("FusedLinearActivationChain missing b1")
                    .clone();
                idx += 1;
                Some(b1)
            } else {
                None
            };

            let w2 = self.nodes[inputs[idx]]
                .output
                .as_ref()
                .expect("FusedLinearActivationChain missing w2")
                .clone();
            idx += 1;

            let b2_opt = if idx < inputs.len() {
                let b2 = self.nodes[inputs[idx]]
                    .output
                    .as_ref()
                    .expect("FusedLinearActivationChain missing b2")
                    .clone();
                Some(b2)
            } else {
                None
            };

            let mut h = match b1_opt.as_ref() {
                Some(b1) => nn_linear::linear(&x, &w1, Some(b1)),
                None => nn_linear::linear(&x, &w1, None),
            };

            for act in acts {
                h = match act {
                    super::nodes::ActType::ReLU => nn_act::relu(&h),
                    super::nodes::ActType::SiLU => nn_act::silu(&h),
                    super::nodes::ActType::GELU => nn_act::gelu(&h),
                };
            }

            let out = match b2_opt.as_ref() {
                Some(b2) => nn_linear::linear(&h, &w2, Some(b2)),
                None => nn_linear::linear(&h, &w2, None),
            };

            if crate::apx_debug_enabled() && !crate::apx_is_silent() {
                eprintln!(
                    "[APX4.9 DEBUG] FusedLinearActivationChain node_id={} produced output len={}",
                    node_id,
                    out.data.len()
                );
            }

            self.nodes[node_id].set_output(out);
            if use_timing {
                if let Some(start) = t0 {
                    let dt = start.elapsed().as_micros() as f64;
                    crate::apx7::hpge_priority::record_node_time(node_id, dt, self.nodes.len());
                }
            }
            return;
        }

        // APX 4.7: if this node is the second of a fused Linear→Linear pair, execute the fusion.
        if let Some(fplan) = &self.fusion_plan {
            if let Some((a, b)) = fplan
                .fused_pairs
                .iter()
                .find(|(_, b)| *b == node_id)
                .cloned()
            {
                crate::apx4_7::exec_fused_linear_linear(self, a, b, record_tape);
                return;
            }
        }

        // APX 4.3: if this node is the start of a GPU segment, execute the whole segment.
        if let Some(plan) = &self.gpu_plan {
            if let Some(seg) = plan.segments.iter().find(|s| s.start == node_id).cloned() {
                self.exec_gpu_segment(&seg);
                if use_timing {
                    if let Some(start) = t0 {
                        let dt = start.elapsed().as_micros() as f64;
                        crate::apx7::hpge_priority::record_node_time(node_id, dt, self.nodes.len());
                    }
                }
                return;
            }
        }

        // APX 4.11: detect whether this node belongs to any already-planned GPU segment.
        let in_gpu_segment = self
            .gpu_plan
            .as_ref()
            .map(|plan| {
                plan
                    .segments
                    .iter()
                    .any(|s| s.start <= node_id && node_id <= s.end)
            })
            .unwrap_or(false);

        let node_type = self.nodes[node_id].node_type.clone();

        // APX 5.x: auxiliary variables for advanced planning.
        let apx_mode = crate::apx_mode();
        let is_5x = apx_mode.starts_with("5.");
        let is_54 = apx_mode.starts_with("5.4");
        let mut node_exec_info_5x: Option<NodeExecInfo> = None;

        // APX 5.1 / 5.2: kernel planner (for now, logs + CPU/GPU decision
        // used for MatMul in APX 5.2).
        let shape_hint: Vec<usize> = self
            .nodes[node_id]
            .inputs
            .get(0)
            .and_then(|&inp_id| self.nodes[inp_id].output.as_ref())
            .map(|t| t.shape.clone())
            .unwrap_or_default();

        let planner = KernelPlanner::new();

        // APX 5.4: obtain adaptive device preference for this node (for now we
        // use generic information, not op-specific).
        let mut adaptive_pref = None;
        if is_54 {
            if let Some(ref info) = node_exec_info_5x {
                let sel_mutex = crate::global_adaptive_selector();
                if let Ok(sel) = sel_mutex.lock() {
                    adaptive_pref = sel.device_preference_for(info);
                }
            }
        }

        let plan = planner.select_kernel(&format!("{:?}", node_type), &shape_hint, adaptive_pref);
        if crate::apx_debug_enabled() {
            eprintln!(
                "[APX 5] Kernel for node_id={} ({:?}) = {:?} ({})",
                node_id,
                node_type,
                plan.target,
                plan.reason,
            );
        }

        // APX 3.9: route node execution target based on type and approximate shape
        // (we use the first input's shape as a proxy when available).
        let exec_target = route(&node_type, &shape_hint);

        if !crate::apx_is_silent() && std::env::var("APX_TRACE").is_ok() {
            eprintln!(
                "[APX 3.9 ROUTER] node_id={} | {:?} | target={:?}",
                node_id, node_type, exec_target
            );
        }

        // APX 5.3: additional advanced planning layer. Here we only build an
        // execution plan and optionally log it, without modifying the real
        // execution path nor the node's math yet.
        if is_5x {
            // APX 5.3: try to use real Tensor information when available to
            // enrich NodeExecInfo.
            let tensor_opt = self.nodes[node_id]
                .output
                .as_ref()
                .or_else(|| {
                    self.nodes[node_id]
                        .inputs
                        .get(0)
                        .and_then(|&inp_id| self.nodes[inp_id].output.as_ref())
                });

            let (dtype_str, contiguous, estimated_bytes) = if let Some(t) = tensor_opt {
                (
                    format!("{:?}", t.dtype),
                    t.layout == Layout::Contiguous,
                    t.estimated_bytes(),
                )
            } else {
                // Approximate fallback if no tensor is available yet.
                let num_elems: usize = shape_hint.iter().product();
                (
                    "f32".to_string(),
                    true,
                    num_elems.saturating_mul(4),
                )
            };

            let num_elems: usize = shape_hint.iter().product();
            let estimated_flops = num_elems; // symbolic placeholder

            let device_52_str = match plan.target {
                KernelTarget::Cpu => "CPU".to_string(),
                KernelTarget::Gpu | KernelTarget::HybridCpuGpu => "GPU".to_string(),
                KernelTarget::CpuFastAvx2 => "CPU".to_string(),
            };

            let mut node_info = NodeExecInfo {
                node_id,
                op_name: format!("{:?}", node_type),
                shape: shape_hint.clone(),
                dtype: dtype_str,
                contiguous,
                device_52: device_52_str,
                estimated_bytes,
                estimated_flops,
                vram_free: 0,
                kernel_time_avg: 0.0,
                preferred_kernel_size: None,
                tile_override: None,
                scheduling_bias: None,
                qkv_bias: None,
                attention_bias: None,
                exec_priority: None,
                prefetch_hint: None,
            };

            // APX 6.11: apply the global fusion policy only as a planning bias
            // (does not alter real forward nor backward).
            if crate::apx_mode_at_least("6.11") {
                use crate::apx6_11::runtime_policy::{get_runtime_policy, FusionRuntimePolicy};

                match get_runtime_policy() {
                    FusionRuntimePolicy::Baseline => {}
                    FusionRuntimePolicy::PreferQKV => node_info.apply_qkv_bias(),
                    FusionRuntimePolicy::PreferFull => node_info.apply_attention_bias(),
                }
            }

            // APX 6.12: additional scheduling hints (ordering/priority,
            // prefetch) based on AdaptiveScheduleBias. Does not touch the
            // math nor the backward tape.
            if crate::apx_mode_at_least("6.12") {
                use crate::apx6_12::adaptive_scheduler::{
                    AdaptiveScheduleBias,
                    get_schedule_bias,
                };

                match get_schedule_bias() {
                    AdaptiveScheduleBias::None => {}
                    AdaptiveScheduleBias::QKVHeavy => node_info.bias_qkv_schedule(),
                    AdaptiveScheduleBias::AttentionHeavy => node_info.bias_attention_schedule(),
                }
            }

            node_exec_info_5x = Some(node_info.clone());

            let planner_5_3 = Planner5_3::new();
            let mut plan_5_3 = planner_5_3.select_plan(&node_info);

            // APX 5.4: optional adaptive selector. It only modifies the plan
            // at the preference level, not the real execution, and only in
            // 5.4 or higher.
            if is_54 {
                let sel_mutex = crate::global_adaptive_selector();
                if let Ok(sel) = sel_mutex.lock() {
                    sel.merge_into_plan(&node_info, &mut plan_5_3);
                }
            }

            if crate::apx_debug_enabled() {
                eprintln!(
                    "[APX 5.3] plan for node_id={} ({:?}): kernel={} layout={:?} chunking={:?} fp8={}",
                    node_id,
                    node_type,
                    plan_5_3.kernel_name,
                    plan_5_3.layout,
                    plan_5_3.chunking.as_ref().map(|c| c.chunks),
                    plan_5_3.use_fp8,
                );
            }
        }

        match exec_target {
            ExecTarget::GPU => {
                panic!("GPU execution not fully integrated yet (APX 4.1 / CUDA vec_add only)");
            }
            ExecTarget::CPU | ExecTarget::CpuOptimized | ExecTarget::Auto => {
                // For now all non-GPU targets fall through to existing CPU implementation.
            }
        }

        match node_type.clone() {
            NodeType::Input | NodeType::Parameter => {}
            NodeType::Add => {
                let inputs = self.nodes[node_id].inputs.clone();
                if inputs.len() < 2 {
                    // Inconsistent graph (e.g., artificial trace tests): do not execute Add.
                    return;
                }

                let a_opt = self.nodes[inputs[0]].output.as_ref();
                let b_opt = self.nodes[inputs[1]].output.as_ref();

                if let (Some(a), Some(b)) = (a_opt.cloned(), b_opt.cloned()) {
                    self.nodes[node_id].set_output(a.add(&b));

                    if record_tape {
                        let op_inputs = self.nodes[node_id].inputs.clone();
                        let ids = op_inputs.clone();
                        self.tape.push(BackOp {
                            inputs: op_inputs,
                            output: node_id,
                            backward: Box::new(move |store, _forward_inputs, out_grad| {
                                add_to_grad_slice(store, ids[0], &out_grad.data);
                                add_to_grad_slice(store, ids[1], &out_grad.data);
                            }),
                        });
                    }
                }
            }
            NodeType::NoOp => {
                // Logically removed node: ensures its single input has been
                // executed and forwards its result.
                if let Some(&inp) = self.nodes[node_id].inputs.get(0) {
                    if crate::apx_debug_enabled() && !crate::apx_is_silent() {
                        eprintln!(
                            "[APX4.9 DEBUG] Executing NoOp node_id={} -> input_id={} ({:?}) | input_has_output={}",
                            node_id,
                            inp,
                            self.nodes[inp].node_type,
                            self.nodes[inp].output.is_some(),
                        );
                    }
                    // If the input node has not produced output yet, execute it now.
                    if self.nodes[inp].output.is_none() {
                        self.execute_single(inp, record_tape);
                    }

                    if let Some(out) = self.nodes[inp].output.clone() {
                        self.nodes[node_id].set_output(out);
                    }
                }
            }
            NodeType::Sub => {
                let a_id = self.nodes[node_id].inputs[0];
                let b_id = self.nodes[node_id].inputs[1];

                let a = self.nodes[a_id]
                    .output
                    .as_ref()
                    .expect("Sub missing input A")
                    .clone();
                let b = self.nodes[b_id]
                    .output
                    .as_ref()
                    .expect("Sub missing input B")
                    .clone();

                self.nodes[node_id].set_output(a.sub(&b));

                if record_tape {
                    let op_inputs = self.nodes[node_id].inputs.clone();
                    let ids = op_inputs.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs,
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            add_to_grad_slice(store, ids[0], &out_grad.data);

                            let neg: Vec<f32> = out_grad.data.iter().map(|v| -*v).collect();
                            add_to_grad_slice(store, ids[1], &neg);
                        }),
                    });
                }
            }
            NodeType::Mul => {
                let inputs = self.nodes[node_id].inputs.clone();
                if inputs.len() < 2 {
                    // Inconsistent graph (e.g., artificial trace tests): do not execute Mul.
                    return;
                }

                let a_opt = self.nodes[inputs[0]].output.as_ref();
                let b_opt = self.nodes[inputs[1]].output.as_ref();

                if let (Some(a), Some(b)) = (a_opt, b_opt) {
                    let a = a.clone();
                    let b = b.clone();

                    self.nodes[node_id].set_output(a.mul(&b));

                    if record_tape {
                        let op_inputs = self.nodes[node_id].inputs.clone();
                        let ids = op_inputs.clone();
                        self.tape.push(BackOp {
                            inputs: op_inputs,
                            output: node_id,
                            backward: Box::new(move |store, forward_inputs, out_grad| {
                                let lhs = forward_inputs[0];
                                let rhs = forward_inputs[1];
                                let mut grad_a = vec![0.0; out_grad.data.len()];
                                let mut grad_b = vec![0.0; out_grad.data.len()];
                                for i in 0..out_grad.data.len() {
                                    grad_a[i] = out_grad.data[i] * rhs.data[i];
                                    grad_b[i] = out_grad.data[i] * lhs.data[i];
                                }
                                add_to_grad_slice(store, ids[0], &grad_a);
                                add_to_grad_slice(store, ids[1], &grad_b);
                            }),
                        });
                    }
                }
            }
            NodeType::MatMul => {
                let a_id = self.nodes[node_id].inputs[0];
                let b_id = self.nodes[node_id].inputs[1];

                let a = self.nodes[a_id]
                    .output
                    .as_ref()
                    .expect("MatMul missing input A")
                    .clone();
                let b = self.nodes[b_id]
                    .output
                    .as_ref()
                    .expect("MatMul missing input B")
                    .clone();

                assert_eq!(a.shape.len(), 2, "MatMul expects 2D lhs tensor");
                assert_eq!(b.shape.len(), 2, "MatMul expects 2D rhs tensor");
                let m = a.shape[0];
                let k = a.shape[1];
                assert_eq!(b.shape[0], k, "MatMul inner dimension mismatch");
                let n = b.shape[1];

                let mut out = Tensor::with_layout(
                    vec![m, n],
                    0.0,
                    a.device,
                    Layout::Contiguous,
                    a.dtype,
                );

                // APX 5.4: timing measurement for MatMul (CPU/GPU) for
                // collecting adaptive statistics.
                let start_time = Instant::now();
                let mut device_chosen = DeviceTarget::CPU;

                let apx_mode = crate::apx_mode();

                // APX 6.6: Auto-Tiling Optimizer (ATO) only for CPU forward
                // with contiguous FP32 tensors. Does not modify backward nor
                // fusions; it only decides which existing kernel to use.
                let is_66_or_higher = crate::apx_mode_at_least("6.6");
                if is_66_or_higher
                    && a.device == crate::tensor::Device::CPU
                    && a.layout == Layout::Contiguous
                    && a.dtype == crate::tensor::DType::F32
                    && b.device == crate::tensor::Device::CPU
                    && b.layout == Layout::Contiguous
                    && b.dtype == crate::tensor::DType::F32
                {
                    use crate::apx6_6_auto_tiling::{AutoTilingSelector, KernelKind};

                    let kernel_choice = AutoTilingSelector::choose_kernel(m, n, k);
                    let tile_config = AutoTilingSelector::choose_tile_sizes(m, n, k);

                    if crate::apx_debug_enabled() && !crate::apx_is_silent() {
                        eprintln!(
                            "[APX 6.6] kernel={:?} tiles={:?} for MxKxN = {}x{}x{}",
                            kernel_choice, tile_config, m, k, n
                        );
                    }

                    match kernel_choice {
                        KernelKind::Baseline38 => {
                            use crate::apx3_8::{
                                device_context::DeviceContext,
                                kernel_dispatch::dispatch_matmul as dispatch_matmul_apx3_8,
                            };
                            let ctx = DeviceContext::new(crate::tensor::Device::CPU);
                            dispatch_matmul_apx3_8(&a.data, &b.data, &mut out.data, m, k, n, &ctx);
                        }
                        KernelKind::Tiled63B | KernelKind::Micro64 => {
                            // Reuse the CPU dispatcher that already integrates 6.3B,
                            // 6.4, and a safe fallback to 3.8/6.1.
                            crate::matmul_dispatcher::matmul_dispatch(
                                &a.data,
                                &b.data,
                                &mut out.data,
                                m,
                                k,
                                n,
                            );
                        }
                    }

                    // In 6.6 mode we do not register explicit 5.4 samples here
                    // to avoid coupling ATO with the statistics system.
                    self.nodes[node_id].set_output(out);
                    if record_tape {
                        let op_inputs = self.nodes[node_id].inputs.clone();
                        let ids = op_inputs.clone();
                        self.tape.push(BackOp {
                            inputs: op_inputs,
                            output: node_id,
                            backward: Box::new(move |store, forward_inputs, out_grad| {
                                let a = forward_inputs[0];
                                let b = forward_inputs[1];
                                let b_t = transpose_2d(b);
                                let grad_a = nn_linear::matmul(out_grad, &b_t);
                                add_to_grad_slice(store, ids[0], &grad_a.data);

                                let a_t = transpose_2d(a);
                                let grad_b = nn_linear::matmul(&a_t, out_grad);
                                add_to_grad_slice(store, ids[1], &grad_b.data);
                            }),
                        });
                    }

                    return;
                }

                // APX 4.11: per-operator direct GPU execution attempt only for
                // nodes outside planned GPU segments. This path is kept as-is
                // to guarantee compatibility.
                if !in_gpu_segment
                    && gpu_hooks::gpu_can_run_matmul(m, k, n)
                    && gpu_hooks::try_gpu_matmul(&a, &b, &mut out)
                {
                    device_chosen = DeviceTarget::GPU;

                    // Register a sample only in 5.4 mode, without changing
                    // semantics nor execution path.
                    if is_54 {
                        let duration_us = start_time.elapsed().as_micros() as u64;
                        if let Some(ref info) = node_exec_info_5x {
                            let sample = Sample {
                                op_name: info.op_name.clone(),
                                shape: info.shape.clone(),
                                dtype: match info.dtype.as_str() {
                                    "F16" => crate::tensor::DType::F16,
                                    "BF16" => crate::tensor::DType::BF16,
                                    "FP8" => crate::tensor::DType::FP8,
                                    _ => crate::tensor::DType::F32,
                                },
                                device_chosen: device_chosen.clone(),
                                duration_us,
                                vram_before: 0,
                                vram_after: 0,
                                fallback: false,
                            };

                            let sel_mutex = crate::global_adaptive_selector();
                            if let Ok(mut sel) = sel_mutex.lock() {
                                sel.register_sample(sample);
                            }
                        }
                    }

                    self.nodes[node_id].set_output(out);
                    return;
                }

                // APX 5.2 / 6.2: use KernelPlanner to decide the matmul target.
                // In modes >= 6.2 we can force the CPU AVX2 path (CpuFastAvx2);
                // otherwise we use the usual APX 4 dispatcher with CPU/GPU/Auto.
                let is_62_or_higher = apx_mode.starts_with("6.2") || apx_mode > "6.2".to_string();

                let target = if is_62_or_higher && matches!(plan.target, KernelTarget::CpuFastAvx2) {
                    // Optional AVX2 path. The APX 6.2 dispatcher is responsible
                    // for safely falling back to the APX 3.8 dispatcher when
                    // AVX2 is not available.
                    crate::apx6_2::dispatch::dispatch_matmul_avx2(
                        &a.data,
                        &b.data,
                        &mut out.data,
                        m,
                        k,
                        n,
                    );
                    Apx4ExecTarget::CPU
                } else {
                    // APX 5.2: in 5.x modes we use the plan to decide between
                    // CPU/GPU/Auto. In other modes, we keep Auto.
                    let mapped = if apx_mode.starts_with("5.") {
                        match plan.target {
                            KernelTarget::Cpu => Apx4ExecTarget::CPU,
                            KernelTarget::Gpu => Apx4ExecTarget::GPU,
                            KernelTarget::HybridCpuGpu => Apx4ExecTarget::Auto,
                            KernelTarget::CpuFastAvx2 => Apx4ExecTarget::CPU,
                        }
                    } else {
                        Apx4ExecTarget::Auto
                    };

                    dispatch_matmul_gpu(
                        &a.data,
                        &b.data,
                        m,
                        k,
                        n,
                        &mut out.data,
                        mapped,
                    );
                    mapped
                };

                // Heuristically infer the device used: if the target was GPU,
                // assume GPU path; otherwise CPU. This inference is approximate
                // but sufficient for initial statistics.
                if matches!(target, Apx4ExecTarget::GPU) {
                    device_chosen = DeviceTarget::GPU;
                }

                // Register a sample only in 5.4 mode, after MatMul completes.
                if is_54 {
                    let duration_us = start_time.elapsed().as_micros() as u64;
                    if let Some(ref info) = node_exec_info_5x {
                        let sample = Sample {
                            op_name: info.op_name.clone(),
                            shape: info.shape.clone(),
                            dtype: match info.dtype.as_str() {
                                "F16" => crate::tensor::DType::F16,
                                "BF16" => crate::tensor::DType::BF16,
                                "FP8" => crate::tensor::DType::FP8,
                                _ => crate::tensor::DType::F32,
                            },
                            device_chosen,
                            duration_us,
                            vram_before: 0,
                            vram_after: 0,
                            fallback: false,
                        };

                        let sel_mutex = crate::global_adaptive_selector();
                        if let Ok(mut sel) = sel_mutex.lock() {
                            sel.register_sample(sample);
                        }
                    }
                }

                self.nodes[node_id].set_output(out);

                if record_tape {
                    let op_inputs = self.nodes[node_id].inputs.clone();
                    let ids = op_inputs.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs,
                        output: node_id,
                        backward: Box::new(move |store, forward_inputs, out_grad| {
                            let a = forward_inputs[0];
                            let b = forward_inputs[1];
                            let b_t = transpose_2d(b);
                            let grad_a = nn_linear::matmul(out_grad, &b_t);
                            add_to_grad_slice(store, ids[0], &grad_a.data);

                            let a_t = transpose_2d(a);
                            let grad_b = nn_linear::matmul(&a_t, out_grad);
                            add_to_grad_slice(store, ids[1], &grad_b.data);
                        }),
                    });
                }
            }
            NodeType::Transpose2D => {
                let src = self.nodes[node_id].inputs[0];
                let x = self.nodes[src]
                    .output
                    .as_ref()
                    .expect("Transpose missing input")
                    .clone();
                let out = transpose_2d(&x);
                self.nodes[node_id].set_output(out.clone());

                if record_tape {
                    let op_inputs = self.nodes[node_id].inputs.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            let grad_x = transpose_2d(out_grad);
                            add_to_grad_slice(store, op_inputs[0], &grad_x.data);
                        }),
                    });
                }
            }
            NodeType::IndexSelect => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert_eq!(inputs.len(), 2, "IndexSelect expects table and indices inputs");

                let table = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("IndexSelect missing table")
                    .clone();
                let indices = self.nodes[inputs[1]]
                    .output
                    .as_ref()
                    .expect("IndexSelect missing indices")
                    .clone();

                let out = index_select_rows(&table, &indices);
                self.nodes[node_id].set_output(out.clone());

                if record_tape {
                    let op_inputs = self.nodes[node_id].inputs.clone();
                    let ids = op_inputs.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs,
                        output: node_id,
                        backward: Box::new(move |store, forward_inputs, out_grad| {
                            let table = forward_inputs[0];
                            let indices = forward_inputs[1];
                            let cols = *table
                                .shape
                                .get(1)
                                .expect("IndexSelect table must be 2D");
                            let mut grad_table = vec![0.0; table.data.len()];
                            scatter_add_rows(&mut grad_table, indices, &out_grad.data, cols);
                            add_to_grad_slice(store, ids[0], &grad_table);
                            // No gradient for indices (integer gather)
                        }),
                    });
                }
            }
            NodeType::Reshape { target } => {
                let inputs = self.nodes[node_id].inputs.clone();
                if inputs.is_empty() {
                    // Inconsistent graph (e.g., artificial trace tests): do not execute Reshape.
                    return;
                }

                let src = inputs[0];
                let x_opt = self.nodes[src].output.as_ref();
                if let Some(x) = x_opt.cloned() {
                    let out = reshape_tensor(&x, &target);
                    self.nodes[node_id].set_output(out.clone());

                    if record_tape {
                        let op_inputs = self.nodes[node_id].inputs.clone();
                        let original_shape = x.shape.clone();
                        self.tape.push(BackOp {
                            inputs: op_inputs.clone(),
                            output: node_id,
                            backward: Box::new(move |store, _forward_inputs, out_grad| {
                                let reshaped_back = reshape_back(out_grad, &original_shape);
                                add_to_grad_slice(store, op_inputs[0], &reshaped_back.data);
                            }),
                        });
                    }
                }
            }
            NodeType::TransposeLastTwo => {
                let src = self.nodes[node_id].inputs[0];
                let x = self.nodes[src]
                    .output
                    .as_ref()
                    .expect("TransposeLastTwo missing input")
                    .clone();
                let out = transpose_last_two(&x);
                self.nodes[node_id].set_output(out.clone());

                if record_tape {
                    let op_inputs = self.nodes[node_id].inputs.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            let back = transpose_last_two(out_grad);
                            add_to_grad_slice(store, op_inputs[0], &back.data);
                        }),
                    });
                }
            }
            NodeType::BatchMatMul => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert_eq!(inputs.len(), 2, "BatchMatMul expects two inputs");
                let a = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("BatchMatMul missing input A")
                    .clone();
                let b = self.nodes[inputs[1]]
                    .output
                    .as_ref()
                    .expect("BatchMatMul missing input B")
                    .clone();

                assert!(
                    a.shape.len() == 3 && b.shape.len() == 3,
                    "BatchMatMul expects 3D tensors",
                );
                let batch = a.shape[0];
                let m = a.shape[1];
                let k = a.shape[2];
                let n = b.shape[2];

                assert_eq!(b.shape[0], batch, "BatchMatMul batch mismatch");
                assert_eq!(b.shape[1], k, "BatchMatMul inner dim mismatch");

                let mut out = Tensor::with_layout(
                    vec![batch, m, n],
                    0.0,
                    a.device,
                    Layout::Contiguous,
                    a.dtype,
                );

                // APX 5.4: timing measurement for BatchMatMul (CPU/GPU) for
                // collecting adaptive statistics.
                let start_time = Instant::now();
                let device_chosen;

                // Target selection via APX 5.2 (KernelPlanner) already computed
                // in 'plan'. We use the same mapping logic as in MatMul.
                let apx_mode = crate::apx_mode();
                let target = if apx_mode.starts_with("5.") {
                    match plan.target {
                        KernelTarget::Cpu => Apx4ExecTarget::CPU,
                        KernelTarget::Gpu => Apx4ExecTarget::GPU,
                        KernelTarget::HybridCpuGpu => Apx4ExecTarget::Auto,
                        KernelTarget::CpuFastAvx2 => Apx4ExecTarget::CPU,
                    }
                } else {
                    Apx4ExecTarget::Auto
                };

                match target {
                    Apx4ExecTarget::GPU => {
                        // GPU execution attempt via APX 4.5. If for any reason
                        // it fails or CUDA is not available, keep the previous
                        // CPU path as fallback.
                        if crate::cuda::cuda_available() {
                            dispatch_batch_matmul_cuda(
                                &a.data,
                                &b.data,
                                &mut out.data,
                                batch,
                                m,
                                k,
                                n,
                            );
                            device_chosen = DeviceTarget::GPU;
                        } else {
                            let cpu_out = batch_matmul(&a, &b);
                            out.data.copy_from_slice(&cpu_out.data);
                            device_chosen = DeviceTarget::CPU;
                        }
                    }
                    Apx4ExecTarget::CPU | Apx4ExecTarget::Auto => {
                        let cpu_out = batch_matmul(&a, &b);
                        out.data.copy_from_slice(&cpu_out.data);
                        device_chosen = DeviceTarget::CPU;
                    }
                }

                // Register a sample only in 5.4 mode, after BatchMatMul completes.
                if is_54 {
                    let duration_us = start_time.elapsed().as_micros() as u64;
                    if let Some(ref info) = node_exec_info_5x {
                        let sample = Sample {
                            op_name: info.op_name.clone(),
                            shape: info.shape.clone(),
                            dtype: match info.dtype.as_str() {
                                "F16" => crate::tensor::DType::F16,
                                "BF16" => crate::tensor::DType::BF16,
                                "FP8" => crate::tensor::DType::FP8,
                                _ => crate::tensor::DType::F32,
                            },
                            device_chosen,
                            duration_us,
                            vram_before: 0,
                            vram_after: 0,
                            fallback: false,
                        };

                        let sel_mutex = crate::global_adaptive_selector();
                        if let Ok(mut sel) = sel_mutex.lock() {
                            sel.register_sample(sample);
                        }
                    }
                }

                self.nodes[node_id].set_output(out.clone());

                if record_tape {
                    let op_inputs = inputs.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, forward_inputs, out_grad| {
                            let a = forward_inputs[0];
                            let b = forward_inputs[1];
                            let (grad_a, grad_b) = batch_matmul_backward(a, b, out_grad);
                            add_to_grad_slice(store, op_inputs[0], &grad_a);
                            add_to_grad_slice(store, op_inputs[1], &grad_b);
                        }),
                    });
                }
            }
            NodeType::BroadcastAdd => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert_eq!(inputs.len(), 2, "BroadcastAdd expects two inputs");
                let a = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("BroadcastAdd missing A")
                    .clone();
                let b = self.nodes[inputs[1]]
                    .output
                    .as_ref()
                    .expect("BroadcastAdd missing B")
                    .clone();
                let out = broadcast_add(&a, &b);
                self.nodes[node_id].set_output(out.clone());

                if record_tape {
                    let op_inputs = inputs.clone();
                    let shape_a = a.shape.clone();
                    let shape_b = b.shape.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            let grad_a = reduce_broadcast_grad(out_grad, &shape_a);
                            let grad_b = reduce_broadcast_grad(out_grad, &shape_b);
                            add_to_grad_slice(store, op_inputs[0], &grad_a);
                            add_to_grad_slice(store, op_inputs[1], &grad_b);
                        }),
                    });
                }
            }
            NodeType::LogSoftmax => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert_eq!(inputs.len(), 1, "LogSoftmax expects a single input");

                let x = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("LogSoftmax missing input")
                    .clone();

                let out = log_softmax_last_dim(&x);
                let out_clone = out.clone();
                self.nodes[node_id].set_output(out);

                if record_tape {
                    let op_inputs = inputs.clone();
                    let output_shape = out_clone.shape.clone();
                    let output_data = out_clone.data.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            let cols = *output_shape
                                .last()
                                .expect("LogSoftmax requires rank >= 1");
                            let rows = if output_shape.len() == 1 {
                                1
                            } else {
                                output_shape[..output_shape.len() - 1]
                                    .iter()
                                    .product()
                            };
                            let mut grad_x = vec![0.0; out_grad.data.len()];
                            for row in 0..rows {
                                let start = row * cols;
                                let end = start + cols;
                                let row_grad = &out_grad.data[start..end];
                                let row_logp = &output_data[start..end];
                                let sum_grad: f32 = row_grad.iter().copied().sum();
                                for i in 0..cols {
                                    let prob = row_logp[i].exp();
                                    grad_x[start + i] = row_grad[i] - prob * sum_grad;
                                }
                            }
                            add_to_grad_slice(store, op_inputs[0], &grad_x);
                        }),
                    });
                }
            }
            NodeType::Gather => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert_eq!(inputs.len(), 2, "Gather expects data and indices inputs");

                let data = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("Gather missing data input")
                    .clone();
                let indices = self.nodes[inputs[1]]
                    .output
                    .as_ref()
                    .expect("Gather missing indices input")
                    .clone();

                let out = gather_last_dim(&data, &indices);
                self.nodes[node_id].set_output(out);

                if record_tape {
                    let op_inputs = inputs.clone();
                    let data_shape = data.shape.clone();
                    let indices_values = indices.data.clone();
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            let last_dim = *data_shape
                                .last()
                                .expect("Gather data must have rank >= 1");
                            let rows = indices_values.len();
                            assert_eq!(rows, out_grad.data.len(), "Gather grad mismatch");
                            let mut grad_data = vec![0.0; data_shape.iter().product()];
                            for row in 0..rows {
                                let idx = indices_values[row].round() as isize;
                                assert!(idx >= 0 && (idx as usize) < last_dim, "Gather index out of bounds");
                                let dst = row * last_dim + idx as usize;
                                grad_data[dst] += out_grad.data[row];
                            }
                            add_to_grad_slice(store, op_inputs[0], &grad_data);
                        }),
                    });
                }
            }
            NodeType::CrossEntropyLoss => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert_eq!(inputs.len(), 2, "CrossEntropyLoss expects log_probs and targets");

                let log_probs = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("CrossEntropyLoss missing log probs")
                    .clone();
                let targets = self.nodes[inputs[1]]
                    .output
                    .as_ref()
                    .expect("CrossEntropyLoss missing targets")
                    .clone();

                let last_dim = *log_probs
                    .shape
                    .last()
                    .expect("CrossEntropyLoss log probs require rank >= 1");
                let rows = log_probs.data.len() / last_dim;
                assert_eq!(
                    targets.data.len(),
                    rows,
                    "CrossEntropyLoss targets mismatch"
                );

                let mut total = 0.0f32;
                let mut target_indices = Vec::with_capacity(rows);
                for row in 0..rows {
                    let idx = targets.data[row].round() as isize;
                    assert!(idx >= 0 && (idx as usize) < last_dim, "CrossEntropyLoss target out of bounds");
                    let idx_usize = idx as usize;
                    target_indices.push(idx_usize);
                    let pos = row * last_dim + idx_usize;
                    total += log_probs.data[pos];
                }
                let loss_val = -total / rows as f32;

                let mut out = Tensor::with_layout(
                    vec![1, 1],
                    0.0,
                    log_probs.device,
                    Layout::Contiguous,
                    log_probs.dtype,
                );
                out.data[0] = loss_val;
                self.nodes[node_id].set_output(out);

                if record_tape {
                    let op_inputs = inputs.clone();
                    let total_rows = rows;
                    let vocab = last_dim;
                    let log_prob_len = log_probs.data.len();
                    self.tape.push(BackOp {
                        inputs: op_inputs.clone(),
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            assert_eq!(out_grad.data.len(), 1, "CrossEntropyLoss grad must be scalar");
                            let scale = -out_grad.data[0] / total_rows as f32;
                            let mut grad_log_probs = vec![0.0; log_prob_len];
                            for (row, &idx) in target_indices.iter().enumerate() {
                                let pos = row * vocab + idx;
                                grad_log_probs[pos] += scale;
                            }
                            add_to_grad_slice(store, op_inputs[0], &grad_log_probs);
                        }),
                    });
                }
            }
            NodeType::Linear => {
                if crate::apx_debug_enabled() {
                    eprintln!(
                        "[DBG] ENTER Linear node_id={} | inputs_len={}",
                        node_id,
                        self.nodes[node_id].inputs.len()
                    );
                }
                let inputs = self.nodes[node_id].inputs.clone();
                assert!(
                    inputs.len() == 2 || inputs.len() == 3,
                    "Linear node expects 2 or 3 inputs"
                );

                let x = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("Linear missing x")
                    .clone();
                let w = self.nodes[inputs[1]]
                    .output
                    .as_ref()
                    .expect("Linear missing weight")
                    .clone();
                // APX 4.11: per-operator direct GPU execution attempt only for
                // nodes outside planned GPU segments.
                let use_bias = inputs.len() == 3;
                let b_opt = if use_bias {
                    Some(
                        self.nodes[inputs[2]]
                            .output
                            .as_ref()
                            .expect("Linear missing bias")
                            .clone(),
                    )
                } else {
                    None
                };

                // APX 5.4: optional timing measurement for adaptive Linear statistics.
                let apx_mode_local = crate::apx_mode();
                let is_54_local = apx_mode_local.starts_with("5.4");
                let start_time = if is_54_local {
                    Some(Instant::now())
                } else {
                    None
                };

                // APX 5.2 + 5.4: only in 5.x modes we use the planner to decide
                // whether to attempt GPU via gpu_hooks::try_gpu_linear.
                if x.shape.len() == 2 && w.shape.len() == 2 {
                    let m = x.shape[0];
                    let k = x.shape[1];
                    if w.shape[0] == k {
                        let n = w.shape[1];

                        if apx_mode_local.starts_with("5.") && !in_gpu_segment {
                            // We use the previously computed plan for this node.
                            // If the planner suggests GPU, we try the GPU path;
                            // otherwise we keep CPU.
                            if let KernelTarget::Gpu = plan.target {
                                let mut tmp = Tensor::zeros_new(&[m, n], x.device);

                                if gpu_hooks::try_gpu_linear(
                                    &x,
                                    &w,
                                    b_opt.as_ref(),
                                    &mut tmp,
                                ) {
                                    // Register a Linear sample only if 5.4 mode is active.
                                    if let (true, Some(t0)) = (is_54_local, start_time) {
                                        let duration_us = t0.elapsed().as_micros() as u64;
                                        if let Some(ref info) = node_exec_info_5x {
                                            let sample = Sample {
                                                op_name: info.op_name.clone(),
                                                shape: info.shape.clone(),
                                                dtype: match info.dtype.as_str() {
                                                    "F16" => crate::tensor::DType::F16,
                                                    "BF16" => crate::tensor::DType::BF16,
                                                    "FP8" => crate::tensor::DType::FP8,
                                                    _ => crate::tensor::DType::F32,
                                                },
                                                device_chosen: DeviceTarget::GPU,
                                                duration_us,
                                                vram_before: 0,
                                                vram_after: 0,
                                                fallback: false,
                                            };

                                            let sel_mutex = crate::global_adaptive_selector();
                                            if let Ok(mut sel) = sel_mutex.lock() {
                                                sel.register_sample(sample);
                                            }
                                        }
                                    }

                                    self.nodes[node_id].set_output(tmp);
                                    if crate::apx_debug_enabled() {
                                        eprintln!(
                                            "[DBG] EARLY RETURN in Linear node_id={} via gpu_hooks::try_gpu_linear (APX 5.x)",
                                            node_id
                                        );
                                    }
                                    return;
                                }
                            }
                        } else if !in_gpu_segment
                            && gpu_hooks::try_gpu_linear(
                                &x,
                                &w,
                                b_opt.as_ref(),
                                &mut Tensor::zeros_new(&[m, n], x.device),
                            )
                        {
                            // Pre-5.x mode: original behavior. We do not register
                            // samples here to avoid mixing statistics between modes.
                            let mut tmp = Tensor::zeros_new(&[m, n], x.device);
                            gpu_hooks::try_gpu_linear(&x, &w, b_opt.as_ref(), &mut tmp);
                            self.nodes[node_id].set_output(tmp);
                            if crate::apx_debug_enabled() {
                                eprintln!(
                                    "[DBG] EARLY RETURN in Linear node_id={} via gpu_hooks::try_gpu_linear",
                                    node_id
                                );
                            }
                            return;
                        }
                    }
                }

                let out = if use_bias {
                    let b = b_opt.as_ref().expect("Linear missing bias clone");
                    nn_linear::linear(&x, &w, Some(b))
                } else {
                    nn_linear::linear(&x, &w, None)
                };

                // Register a CPU Linear sample only in 5.4 mode and only if we
                // did not return via the GPU path.
                if let (true, Some(t0)) = (is_54_local, start_time) {
                    let duration_us = t0.elapsed().as_micros() as u64;
                    if let Some(ref info) = node_exec_info_5x {
                        let sample = Sample {
                            op_name: info.op_name.clone(),
                            shape: info.shape.clone(),
                            dtype: match info.dtype.as_str() {
                                "F16" => crate::tensor::DType::F16,
                                "BF16" => crate::tensor::DType::BF16,
                                "FP8" => crate::tensor::DType::FP8,
                                _ => crate::tensor::DType::F32,
                            },
                            device_chosen: DeviceTarget::CPU,
                            duration_us,
                            vram_before: 0,
                            vram_after: 0,
                            fallback: false,
                        };

                        let sel_mutex = crate::global_adaptive_selector();
                        if let Ok(mut sel) = sel_mutex.lock() {
                            sel.register_sample(sample);
                        }
                    }
                }

                self.nodes[node_id].set_output(out);

                if record_tape {
                    // APX 4.16: if this Linear is part of a fused QKV pattern,
                    // delegate backward to the fused BackOp and do not record
                    // the normal Linear BackOp.
                    let mode = crate::apx_mode();
                    if mode == "4.16" {
                        if self.fused_ops.contains_key(&node_id) || self.is_qkv_secondary(node_id) {
                            return;
                        }
                    }

                    // APX 4.18: if this Linear is one of the Q/K/V of a fused
                    // Self-Attention pattern, also delegate backward to the
                    // fused BackOp.
                    if mode == "4.18" {
                        if self.is_sa_linear(node_id) {
                            return;
                        }
                    }

                    let op_inputs = self.nodes[node_id].inputs.clone();
                    let ids = op_inputs.clone();
                    let has_bias = ids.len() == 3;
                    self.tape.push(BackOp {
                        inputs: op_inputs,
                        output: node_id,
                        backward: Box::new(move |store, forward_inputs, out_grad| {
                            // APX 3.0: fused Linear backward (dX + dW) with chunking.
                            if crate::apx_mode() == "3.0" && !has_bias {
                                use crate::apx3::fused_backward::fused_linear_backward;

                                fused_linear_backward(
                                    store,
                                    forward_inputs[0],
                                    forward_inputs[1],
                                    out_grad,
                                    ids[0],
                                    ids[1],
                                );
                                return;
                            }

                            let x = forward_inputs[0];
                            let w = forward_inputs[1];

                            let w_t = transpose_2d(w);
                            let grad_x = nn_linear::matmul(out_grad, &w_t);
                            add_to_grad_slice(store, ids[0], &grad_x.data);

                            let x_t = transpose_2d(x);
                            let grad_w = nn_linear::matmul(&x_t, out_grad);
                            add_to_grad_slice(store, ids[1], &grad_w.data);

                            if has_bias {
                                let bias_grad = sum_rows(out_grad);
                                add_to_grad_slice(store, ids[2], &bias_grad);
                            }
                        }),
                    });
                }
            }
            NodeType::Activation(act) => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert_eq!(inputs.len(), 1, "Activation expects a single input");

                let x = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("Activation missing input")
                    .clone();

                let out = match act {
                    crate::amg::nodes::ActType::ReLU => nn_act::relu(&x),
                    crate::amg::nodes::ActType::SiLU => nn_act::silu(&x),
                    crate::amg::nodes::ActType::GELU => nn_act::gelu(&x),
                };

                self.nodes[node_id].set_output(out);

                // Note: for now we do not record ActType-specific backward;
                // the training path for these nodes is not supported in APX 4.8.
            }
            NodeType::FusedLinearActivation(act) => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert!(
                    inputs.len() == 2 || inputs.len() == 3,
                    "FusedLinearActivation expects 2 or 3 inputs",
                );

                let x = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("FusedLinearActivation missing x")
                    .clone();
                let w = self.nodes[inputs[1]]
                    .output
                    .as_ref()
                    .expect("FusedLinearActivation missing weight")
                    .clone();

                let b_opt = if inputs.len() == 3 {
                    Some(
                        self.nodes[inputs[2]]
                            .output
                            .as_ref()
                            .expect("FusedLinearActivation missing bias"),
                    )
                } else {
                    None
                };

                let lin = nn_linear::linear(&x, &w, b_opt);

                let out = match act {
                    crate::amg::nodes::ActType::ReLU => nn_act::relu(&lin),
                    crate::amg::nodes::ActType::SiLU => {
                        // Use the SiLU-specific fused implementation.
                        crate::apx4_8::fused_linear_activation::exec_fused_linear_silu(
                            &x, &w, b_opt,
                        )
                    }
                    crate::amg::nodes::ActType::GELU => nn_act::gelu(&lin),
                };

                self.nodes[node_id].set_output(out);

                // Fused-node-specific backward is not implemented in APX 4.8.
            }
            NodeType::FusedLinearActivationChain(acts) => {
                let inputs = self.nodes[node_id].inputs.clone();
                assert!(
                    inputs.len() == 5 || inputs.len() == 4,
                    "FusedLinearActivationChain expects 4 or 5 inputs",
                );

                let x = self.nodes[inputs[0]]
                    .output
                    .as_ref()
                    .expect("FusedLinearActivationChain missing x")
                    .clone();
                let w1 = self.nodes[inputs[1]]
                    .output
                    .as_ref()
                    .expect("FusedLinearActivationChain missing w1")
                    .clone();

                let mut idx = 2;
                let b1_opt = if inputs.len() == 5 {
                    let b1 = self.nodes[inputs[idx]]
                        .output
                        .as_ref()
                        .expect("FusedLinearActivationChain missing b1")
                        .clone();
                    idx += 1;
                    Some(b1)
                } else {
                    None
                };

                let w2 = self.nodes[inputs[idx]]
                    .output
                    .as_ref()
                    .expect("FusedLinearActivationChain missing w2")
                    .clone();
                idx += 1;

                let b2_opt = if idx < inputs.len() {
                    let b2 = self.nodes[inputs[idx]]
                        .output
                        .as_ref()
                        .expect("FusedLinearActivationChain missing b2")
                        .clone();
                    Some(b2)
                } else {
                    None
                };

                let mut h = match b1_opt.as_ref() {
                    Some(b1) => nn_linear::linear(&x, &w1, Some(b1)),
                    None => nn_linear::linear(&x, &w1, None),
                };

                for act in acts {
                    h = match act {
                        crate::amg::nodes::ActType::ReLU => nn_act::relu(&h),
                        crate::amg::nodes::ActType::SiLU => nn_act::silu(&h),
                        crate::amg::nodes::ActType::GELU => nn_act::gelu(&h),
                    };
                }

                let out = match b2_opt.as_ref() {
                    Some(b2) => nn_linear::linear(&h, &w2, Some(b2)),
                    None => nn_linear::linear(&h, &w2, None),
                };

                self.nodes[node_id].set_output(out);
            }
            NodeType::RmsNorm => {
                let inputs = self.nodes[node_id].inputs.clone();
                if inputs.len() != 1 {
                    // Inconsistent graph (e.g., artificial trace tests): do not execute RmsNorm.
                    return;
                }

                let x_opt = self.nodes[inputs[0]].output.as_ref();
                if let Some(x) = x_opt.cloned() {
                    let out = nn_norm::rms_norm(&x, 1e-5);
                    self.nodes[node_id].set_output(out);
                }
            }
            NodeType::SiLU => {
                let inputs = self.nodes[node_id].inputs.clone();
                if inputs.len() != 1 {
                    // Inconsistent graph (e.g., artificial trace tests): do not execute SiLU.
                    return;
                }

                let x_opt = self.nodes[inputs[0]].output.as_ref();
                if let Some(x) = x_opt.cloned() {
                    let out = nn_act::silu(&x);
                    self.nodes[node_id].set_output(out);

                    if record_tape {
                        let op_inputs = self.nodes[node_id].inputs.clone();
                        let ids = op_inputs.clone();
                        self.tape.push(BackOp {
                            inputs: op_inputs,
                            output: node_id,
                            backward: Box::new(move |store, forward_inputs, out_grad| {
                                let x = forward_inputs[0];
                                let mut grad_x = vec![0.0; x.data.len()];
                                for i in 0..x.data.len() {
                                    let v = x.data[i];
                                    let sig = 1.0f32 / (1.0f32 + (-v).exp());
                                    let deriv = sig + v * sig * (1.0 - sig);
                                    grad_x[i] = out_grad.data[i] * deriv;
                                }
                                add_to_grad_slice(store, ids[0], &grad_x);
                            }),
                        });
                    }
                }
            }
            NodeType::Softmax => {
                let inputs = self.nodes[node_id].inputs.clone();
                if inputs.len() != 1 {
                    // Inconsistent graph (e.g., artificial trace tests): do not execute Softmax.
                    return;
                }

                let x_opt = self.nodes[inputs[0]].output.as_ref();
                if let Some(x) = x_opt.cloned() {
                    let out = nn_softmax::softmax_last_dim(&x);
                    self.nodes[node_id].set_output(out);

                    if record_tape {
                        let op_inputs = self.nodes[node_id].inputs.clone();
                        let ids = op_inputs.clone();
                        let softmax_out = self.nodes[node_id]
                            .output
                            .as_ref()
                            .cloned()
                            .expect("Softmax output just computed");
                        let serial_shape = softmax_out.shape.clone();
                        let serial_data = softmax_out.data.clone();

                        self.tape.push(BackOp {
                            inputs: op_inputs,
                            output: node_id,
                            backward: Box::new(move |store, _forward_inputs, out_grad| {
                                if cpu_features().avx2 {
                                    let grad_x = nn_softmax::softmax_backward_parallel(&softmax_out, out_grad);
                                    add_to_grad_slice(store, ids[0], &grad_x.data);
                                } else {
                                    let cols = *serial_shape
                                        .last()
                                        .expect("Softmax requires at least one dimension");
                                    let rows = if serial_shape.len() == 1 {
                                        1
                                    } else {
                                        serial_shape[..serial_shape.len() - 1]
                                            .iter()
                                            .product()
                                    };
                                    let mut grad_x = vec![0.0; out_grad.data.len()];
                                    for row in 0..rows {
                                        let start = row * cols;
                                        let end = start + cols;
                                        let row_grad = &out_grad.data[start..end];
                                        let row_y = &serial_data[start..end];
                                        let dot: f32 = row_grad
                                            .iter()
                                            .zip(row_y.iter())
                                            .map(|(g, y)| g * y)
                                            .sum();
                                        for i in 0..cols {
                                            grad_x[start + i] = row_y[i] * (row_grad[i] - dot);
                                        }
                                    }
                                    add_to_grad_slice(store, ids[0], &grad_x);
                                }
                            }),
                        });
                    }
                }
            }
            NodeType::Output => {
                let src = self.nodes[node_id].inputs[0];
                if self.nodes[src].output.is_none() {
                    self.execute_single(src, record_tape);
                }

                if let Some(out) = self.nodes[src].output.clone() {
                    self.nodes[node_id].set_output(out);
                    let op_inputs = vec![src];
                    self.tape.push(BackOp {
                        inputs: op_inputs,
                        output: node_id,
                        backward: Box::new(move |store, _forward_inputs, out_grad| {
                            add_to_grad_slice(store, src, &out_grad.data);
                        }),
                    });
                }
            }
        }

        if crate::apx_debug_enabled() {
            if let Some(out) = self.nodes[node_id].output.as_ref() {
                if !crate::apx_is_silent() {
                    let input_lens: Vec<usize> = self.nodes[node_id]
                        .inputs
                        .iter()
                        .map(|i| {
                            self.nodes[*i]
                                .output
                                .as_ref()
                                .map(|t| t.data.len())
                                .unwrap_or(0)
                        })
                        .collect();
                    eprintln!(
                        "[TRACE] node_id={} | node={:?} | shape={:?} | len={} | input_lens={:?}",
                        node_id,
                        node_type,
                        out.shape,
                        out.data.len(),
                        input_lens
                    );
                }
            }
        }
    }

    pub fn backward(&mut self, loss_node_id: usize) {
        // Reset gradient store for this backward pass.
        self.grad_store = GradStore::new();

        // Seed gradient at the loss node.
        let loss = self.nodes[loss_node_id]
            .output
            .as_ref()
            .expect("Loss node missing output");
        self.grad_store
            .set(loss_node_id, vec![1.0; loss.data.len()]);

        // Build topological levels starting from the loss and run per-level parallel backward.
        let levels = self.build_backward_levels(loss_node_id);
        for level in levels {
            if std::env::var("ATENIA_TRACE").unwrap_or_default() == "1" {
                eprintln!(
                    "[APX TRACE] Executing backward level with {} ops in parallel",
                    level.len()
                );
            }
            level.par_iter().for_each(|&node_id| {
                self.execute_backward_single(node_id);
            });
        }

        // Materialize gradients from GradStore into node.output.grad
        for (node_id, node) in self.nodes.iter_mut().enumerate() {
            let buffer = self.grad_store.take(node_id);
            if buffer.is_empty() {
                continue;
            }
            if let Some(output) = &mut node.output {
                assert_eq!(
                    buffer.len(),
                    output.data.len(),
                    "gradient length mismatch for node {} ({:?})",
                    node_id,
                    node.node_type
                );
                output.grad = Some(buffer);
            }
        }
    }

    /// Sequential backward variant used for APX 2.0 regression tests.
    /// Executes the same level order as `backward` but without rayon parallelism.
    pub fn backward_sequential(&mut self, loss_node_id: usize) {
        // Reset gradient store for this backward pass.
        self.grad_store = GradStore::new();

        // Seed gradient at the loss node.
        let loss = self.nodes[loss_node_id]
            .output
            .as_ref()
            .expect("Loss node missing output");
        self.grad_store
            .set(loss_node_id, vec![1.0; loss.data.len()]);

        // Build topological levels starting from the loss and run backward sequentially.
        let levels = self.build_backward_levels(loss_node_id);
        for level in levels {
            for node_id in level {
                self.execute_backward_single(node_id);
            }
        }

        // Materialize gradients from GradStore into node.output.grad
        for (node_id, node) in self.nodes.iter_mut().enumerate() {
            let buffer = self.grad_store.take(node_id);
            if buffer.is_empty() {
                continue;
            }
            if let Some(output) = &mut node.output {
                assert_eq!(
                    buffer.len(),
                    output.data.len(),
                    "gradient length mismatch for node {} ({:?}) (sequential)",
                    node_id,
                    node.node_type
                );
                output.grad = Some(buffer);
            }
        }
    }

    fn execute_backward_single(&self, node_id: usize) {
        let op = match self.tape.get(node_id) {
            Some(op) => op,
            None => return,
        };

        if std::env::var("ATENIA_TRACE").unwrap_or_default() == "1" {
            eprintln!(
                "[APX TRACE] Backward op for node {} executed on thread {:?}",
                node_id,
                std::thread::current().id()
            );
        }

        // Take gradient for this node's output from the store.
        let grad_output = self.grad_store.take(op.output);
        if grad_output.is_empty() {
            return;
        }

        let output_template = self.nodes[op.output]
            .output
            .as_ref()
            .expect("Missing output tensor during backward");
        assert_eq!(
            grad_output.len(),
            output_template.data.len(),
            "gradient length mismatch for node {} ({:?})",
            op.output,
            self.nodes[op.output].node_type
        );

        let mut out_grad_tensor = output_template.clone();
        out_grad_tensor.data = grad_output;
        out_grad_tensor.grad = None;

        let input_refs: Vec<&Tensor> = op
            .inputs
            .iter()
            .map(|&id| {
                self.nodes[id]
                    .output
                    .as_ref()
                    .expect("Missing input tensor during backward")
            })
            .collect();

        (op.backward)(&self.grad_store, &input_refs, &out_grad_tensor);
    }

    fn build_backward_levels(&self, loss_id: usize) -> Vec<Vec<usize>> {
        let mut visited = vec![false; self.nodes.len()];
        let mut current = Vec::new();

        if self.tape.has_op(loss_id) {
            current.push(loss_id);
            visited[loss_id] = true;
        }

        let mut levels: Vec<Vec<usize>> = Vec::new();
        while !current.is_empty() {
            levels.push(current.clone());
            let mut next = Vec::new();
            for &node_id in levels.last().unwrap() {
                for &parent in &self.nodes[node_id].inputs {
                    if !visited[parent] && self.tape.has_op(parent) {
                        visited[parent] = true;
                        next.push(parent);
                    }
                }
            }
            current = next;
        }

        levels
    }

    fn collect_outputs(&self) -> Vec<Tensor> {
        self
            .nodes
            .iter()
            .filter_map(|node| match node.node_type {
                NodeType::Output => node.output.clone(),
                _ => None,
            })
            .collect()
    }

    fn execute_fused_add_mul(&mut self, add_node: usize, mul_node: usize) {
        execute_fused_add_mul_impl(&mut self.nodes, add_node, mul_node);
    }

    pub fn get_params_mut<'a>(&'a mut self, param_ids: &[usize]) -> Vec<&'a mut Tensor> {
        #[cfg(debug_assertions)]
        {
            let mut seen = HashSet::new();
            for &pid in param_ids {
                assert!(seen.insert(pid), "duplicate parameter id {pid}");
            }
        }

        let len = self.nodes.len();
        let base_ptr = self.nodes.as_mut_ptr();
        let mut tensors = Vec::with_capacity(param_ids.len());
        for &pid in param_ids {
            assert!(pid < len, "parameter id {pid} out of bounds");
            let node = unsafe { &mut *base_ptr.add(pid) };
            let tensor_ptr = node
                .output
                .as_mut()
                .expect("parameter node missing tensor output") as *mut Tensor;
            tensors.push(unsafe { &mut *tensor_ptr });
        }
        tensors
    }
}

fn execute_fused_add_mul_impl(nodes: &mut [Node], add_node: usize, mul_node: usize) {
    let add_inputs = nodes[add_node].inputs.clone();
    assert!(
        add_inputs.len() == 2,
        "Fused Add node must have exactly 2 inputs"
    );

    let a_id = add_inputs[0];
    let b_id = add_inputs[1];
    let c_id = nodes[mul_node].inputs[1];

    let a = nodes[a_id]
        .output
        .as_ref()
        .expect("Fused/Add missing A")
        .clone();
    let b = nodes[b_id]
        .output
        .as_ref()
        .expect("Fused/Add missing B")
        .clone();
    let c = nodes[c_id]
        .output
        .as_ref()
        .expect("Fused/Mul missing C")
        .clone();

    let tmp = a.add(&b);
    let out = tmp.mul(&c);

    nodes[add_node].set_output(tmp);
    nodes[mul_node].set_output(out);
}

fn add_to_grad_slice(store: &GradStore, node_id: usize, values: &[f32]) {
    store.add(node_id, values);
}

fn transpose_2d(t: &Tensor) -> Tensor {
    assert_eq!(t.shape.len(), 2, "transpose_2d expects a 2D tensor");
    let rows = t.shape[0];
    let cols = t.shape[1];
    let mut data = vec![0.0; t.data.len()];
    for r in 0..rows {
        for c in 0..cols {
            data[c * rows + r] = t.data[r * cols + c];
        }
    }
    let new_shape = vec![cols, rows];
    let strides = Tensor::compute_strides(&new_shape, &Layout::Contiguous);
    Tensor {
        shape: new_shape,
        data,
        device: t.device,
        dtype: t.dtype,
        layout: Layout::Contiguous,
        strides,
        grad: None,
        gpu: None,
        persistence: None,
        op: None,
    }
}

fn sum_rows(t: &Tensor) -> Vec<f32> {
    assert!(t.shape.len() >= 1, "sum_rows expects at least 1D tensor");
    let cols = *t.shape.last().unwrap();
    let rows = if t.shape.len() == 1 {
        1
    } else {
        t.shape[..t.shape.len() - 1].iter().product()
    };
    let mut result = vec![0.0; cols];
    for row in 0..rows {
        let start = row * cols;
        for i in 0..cols {
            result[i] += t.data[start + i];
        }
    }
    result
}

fn reshape_tensor(x: &Tensor, target: &Vec<isize>) -> Tensor {
    let mut new_shape = Vec::with_capacity(target.len());
    let mut inferred = None;
    let total: usize = x.shape.iter().product();
    let mut known = 1usize;
    for &dim in target {
        if dim == -1 {
            assert!(inferred.is_none(), "only one inferred dimension allowed");
            inferred = Some(new_shape.len());
            new_shape.push(1);
        } else {
            let d = dim as usize;
            known *= d.max(1);
            new_shape.push(d);
        }
    }
    if let Some(idx) = inferred {
        let inferred_dim = total / known.max(1);
        new_shape[idx] = inferred_dim;
    }
    assert_eq!(new_shape.iter().product::<usize>(), total, "reshape must preserve elements");
    Tensor {
        shape: new_shape.clone(),
        data: x.data.clone(),
        device: x.device,
        dtype: x.dtype,
        layout: Layout::Contiguous,
        strides: Tensor::compute_strides(&new_shape, &Layout::Contiguous),
        grad: None,
        gpu: None,
        persistence: None,
        op: None,
    }
}

fn reshape_back(x: &Tensor, original_shape: &Vec<usize>) -> Tensor {
    let mut t = x.clone();
    t.shape = original_shape.clone();
    t.strides = Tensor::compute_strides(original_shape, &Layout::Contiguous);
    t
}

fn transpose_last_two(x: &Tensor) -> Tensor {
    assert!(x.shape.len() >= 2, "TransposeLastTwo expects rank >= 2");
    let mut new_shape = x.shape.clone();
    let rank = new_shape.len();
    new_shape.swap(rank - 1, rank - 2);
    let mut out = Tensor::with_layout(new_shape.clone(), 0.0, x.device, Layout::Contiguous, x.dtype);
    let outer: usize = new_shape[..rank - 2].iter().product();
    let rows = new_shape[rank - 2];
    let cols = new_shape[rank - 1];
    for outer_idx in 0..outer {
        let offset_in = outer_idx * rows * cols;
        let offset_out = offset_in;
        for r in 0..rows {
            for c in 0..cols {
                let src = offset_in + c * rows + r;
                let dst = offset_out + r * cols + c;
                out.data[dst] = x.data[src];
            }
        }
    }
    out
}

fn batch_matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.shape.len() == 3 && b.shape.len() == 3, "BatchMatMul expects 3D tensors");
    let batch = a.shape[0];
    let m = a.shape[1];
    let k = a.shape[2];
    let n = b.shape[2];

    assert_eq!(b.shape[0], batch, "BatchMatMul batch mismatch");
    assert_eq!(b.shape[1], k, "BatchMatMul inner dim mismatch");

    let mut out = Tensor::with_layout(
        vec![batch, m, n],
        0.0,
        a.device,
        Layout::Contiguous,
        a.dtype,
    );

    crate::matmul_dispatcher::batch_matmul_dispatch(
        &a.data,
        &b.data,
        &mut out.data,
        batch,
        m,
        k,
        n,
    );

    out
}

fn batch_matmul_backward(a: &Tensor, b: &Tensor, out_grad: &Tensor) -> (Vec<f32>, Vec<f32>) {
    let batch = a.shape[0];
    let m = a.shape[1];
    let k = a.shape[2];
    let n = b.shape[2];
    let mut grad_a = vec![0.0; a.data.len()];
    let mut grad_b = vec![0.0; b.data.len()];
    for batch_idx in 0..batch {
        let a_offset = batch_idx * m * k;
        let b_offset = batch_idx * k * n;
        let out_offset = batch_idx * m * n;
        for i in 0..m {
            for kk in 0..k {
                let mut sum = 0.0;
                for j in 0..n {
                    let grad_idx = out_offset + i * n + j;
                    let b_idx = b_offset + kk * n + j;
                    sum += out_grad.data[grad_idx] * b.data[b_idx];
                }
                grad_a[a_offset + i * k + kk] = sum;
            }
        }
        for kk in 0..k {
            for j in 0..n {
                let mut sum = 0.0;
                for i in 0..m {
                    let grad_idx = out_offset + i * n + j;
                    let a_idx = a_offset + i * k + kk;
                    sum += out_grad.data[grad_idx] * a.data[a_idx];
                }
                grad_b[b_offset + kk * n + j] = sum;
            }
        }
    }
    (grad_a, grad_b)
}

fn broadcast_add(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.shape.len(), b.shape.len(), "BroadcastAdd ranks must match");
    let mut out = a.clone();
    add_broadcast_inplace(&mut out, b);
    out
}

fn add_broadcast_inplace(out: &mut Tensor, other: &Tensor) {
    let rank = out.shape.len();
    let mut index = vec![0usize; rank];
    loop {
        let out_offset = linear_index(&index, &out.shape);
        let mut other_index = vec![0usize; rank];
        for d in 0..rank {
            if other.shape[d] == 1 {
                other_index[d] = 0;
            } else {
                other_index[d] = index[d];
            }
        }
        let other_offset = linear_index(&other_index, &other.shape);
        out.data[out_offset] += other.data[other_offset];
        if !increment_multi_index(&mut index, &out.shape) {
            break;
        }
    }
}

fn reduce_broadcast_grad(out_grad: &Tensor, target_shape: &Vec<usize>) -> Vec<f32> {
    let mut grad = vec![0.0; target_shape.iter().product()];
    let rank = target_shape.len();
    let mut index = vec![0usize; rank];
    loop {
        let out_offset = linear_index(&index, &out_grad.shape);
        let mut target_index = vec![0usize; rank];
        for d in 0..rank {
            if target_shape[d] == 1 {
                target_index[d] = 0;
            } else {
                target_index[d] = index[d];
            }
        }
        let target_offset = linear_index(&target_index, target_shape);
        grad[target_offset] += out_grad.data[out_offset];
        if !increment_multi_index(&mut index, &out_grad.shape) {
            break;
        }
    }
    grad
}

fn linear_index(index: &[usize], shape: &[usize]) -> usize {
    let mut offset = 0usize;
    let mut stride = 1usize;
    for (&i, &dim) in index.iter().zip(shape.iter()).rev() {
        offset += i * stride;
        stride *= dim.max(1);
    }
    offset
}

fn increment_multi_index(index: &mut [usize], shape: &[usize]) -> bool {
    for axis in (0..index.len()).rev() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return true;
        }
        index[axis] = 0;
    }
    false
}

fn index_select_rows(table: &Tensor, indices: &Tensor) -> Tensor {
    assert_eq!(table.shape.len(), 2, "IndexSelect table must be 2D");
    let rows = table.shape[0];
    let cols = table.shape[1];

    let mut out_shape = indices.shape.clone();
    out_shape.push(cols);
    let mut out = Tensor::with_layout(
        out_shape,
        0.0,
        table.device,
        Layout::Contiguous,
        table.dtype,
    );

    for (slot, &raw_idx) in indices.data.iter().enumerate() {
        let idx = raw_idx.round() as isize;
        assert!(idx >= 0 && (idx as usize) < rows, "IndexSelect index out of bounds");
        let idx = idx as usize;
        let src_start = idx * cols;
        let dst_start = slot * cols;
        out.data[dst_start..dst_start + cols]
            .copy_from_slice(&table.data[src_start..src_start + cols]);
    }

    out
}

fn scatter_add_rows(grad_table: &mut [f32], indices: &Tensor, grad_out: &[f32], cols: usize) {
    assert_eq!(grad_out.len(), indices.data.len() * cols);
    for (slot, &raw_idx) in indices.data.iter().enumerate() {
        let idx = raw_idx.round() as isize;
        assert!(idx >= 0, "IndexSelect gradient index negative");
        let idx = idx as usize;
        let dst_start = idx * cols;
        let src_start = slot * cols;
        for i in 0..cols {
            grad_table[dst_start + i] += grad_out[src_start + i];
        }
    }
}
