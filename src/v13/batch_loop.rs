use super::async_executor::AsyncExecutor;
use super::graph_executor::GraphExecutor;
use super::hybrid_memory::HybridMemoryManager;
use super::memory_types::MemorySnapshot;
use super::offload_engine::SmartOffloadEngine;
use super::reconfigurable_graph::{GraphPlacementPlan, ReconfigurableGraph};

pub struct BatchLoopRunner {
    pub offload: SmartOffloadEngine,
    pub graph_exec: GraphExecutor,
}

impl BatchLoopRunner {
    pub fn new(offload: SmartOffloadEngine, graph_exec: GraphExecutor) -> Self {
        BatchLoopRunner { offload, graph_exec }
    }

    pub fn run_ticks(
        &mut self,
        graph: &ReconfigurableGraph,
        exec: &mut AsyncExecutor,
        mem: &mut HybridMemoryManager,
        snapshots: &[MemorySnapshot],
        gpu_available: bool,
    ) -> Vec<String> {
        for (i, snapshot) in snapshots.iter().enumerate() {
            let tick = i as u64;

            // Mark tick start.
            exec.timeline
                .push(format!("TICK_START tick={}", i));

            // Enqueue and run the graph for this snapshot.
            let _ = self.graph_exec.enqueue_graph(graph, exec, mem, snapshot, gpu_available);
            exec.run_to_completion();

            // Collect unique tensor ids from the graph.
            let mut ids: Vec<String> = Vec::new();
            for node in graph.nodes() {
                for id in &node.tensor_ids {
                    if !ids.contains(id) {
                        ids.push(id.clone());
                    }
                }
            }

            let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();

            // Plan offloads for this tick.
            let plan = self
                .offload
                .plan_with_tick(snapshot, &id_refs, mem, tick);

            let actions_count = plan.actions.len();
            exec.timeline.push(format!(
                "OFFLOAD_PLAN tick={} actions={} reason={}",
                i, actions_count, plan.reason
            ));

            if actions_count == 0 {
                exec.timeline
                    .push(format!("OFFLOAD_APPLY tick={} skipped", i));
                exec.timeline
                    .push(format!("TICK_END tick={}", i));
                continue;
            }

            // Apply offload plan; handle errors without panicking.
            match self.offload.apply(snapshot, &plan, mem) {
                Ok(()) => {
                    exec.timeline
                        .push(format!("OFFLOAD_APPLY tick={} ok", i));
                }
                Err(err) => {
                    exec.timeline.push(format!(
                        "OFFLOAD_APPLY tick={} error={:?}",
                        i, err
                    ));
                    exec.timeline
                        .push(format!("TICK_END tick={}", i));
                    break;
                }
            }

            exec.timeline
                .push(format!("TICK_END tick={}", i));
        }

        exec.timeline.clone()
    }

    pub fn run_ticks_with_plans(
        &mut self,
        graph: &ReconfigurableGraph,
        exec: &mut AsyncExecutor,
        mem: &mut HybridMemoryManager,
        snapshots: &[MemorySnapshot],
        gpu_available: bool,
    ) -> (Vec<String>, Vec<TickResult>) {
        let mut results: Vec<TickResult> = Vec::new();

        for (i, snapshot) in snapshots.iter().enumerate() {
            let tick = i as u64;

            // Capture placement plan *before* executing this tick, using
            // current memory tiers from HybridMemoryManager.
            let plan = graph.plan_for_snapshot_with_mem(mem, snapshot, gpu_available);
            results.push(TickResult { tick, plan });

            exec.timeline
                .push(format!("TICK_START tick={}", i));

            let _ = self.graph_exec.enqueue_graph(graph, exec, mem, snapshot, gpu_available);
            exec.run_to_completion();

            let mut ids: Vec<String> = Vec::new();
            for node in graph.nodes() {
                for id in &node.tensor_ids {
                    if !ids.contains(id) {
                        ids.push(id.clone());
                    }
                }
            }

            let id_refs: Vec<&str> = ids.iter().map(|s| s.as_str()).collect();

            let plan = self
                .offload
                .plan_with_tick(snapshot, &id_refs, mem, tick);

            let actions_count = plan.actions.len();
            exec.timeline.push(format!(
                "OFFLOAD_PLAN tick={} actions={} reason={}",
                i, actions_count, plan.reason
            ));

            if actions_count == 0 {
                exec.timeline
                    .push(format!("OFFLOAD_APPLY tick={} skipped", i));
                exec.timeline
                    .push(format!("TICK_END tick={}", i));
                continue;
            }

            match self.offload.apply(snapshot, &plan, mem) {
                Ok(()) => {
                    exec.timeline
                        .push(format!("OFFLOAD_APPLY tick={} ok", i));
                }
                Err(err) => {
                    exec.timeline.push(format!(
                        "OFFLOAD_APPLY tick={} error={:?}",
                        i, err
                    ));
                    exec.timeline
                        .push(format!("TICK_END tick={}", i));
                    break;
                }
            }

            exec.timeline
                .push(format!("TICK_END tick={}", i));
        }

        (exec.timeline.clone(), results)
    }
}

pub struct TickResult {
    pub tick: u64,
    pub plan: GraphPlacementPlan,
}
