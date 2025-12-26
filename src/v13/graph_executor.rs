use super::async_executor::AsyncExecutor;
use super::hybrid_memory::HybridMemoryManager;
use super::memory_types::MemorySnapshot;
use super::reconfigurable_graph::ReconfigurableGraph;
use super::stream_router::RoutedBundle;
use super::stream_router::StreamRouter;
use super::streams::StreamConfig;

pub struct GraphExecutor {
    pub cfg: StreamConfig,
}

impl GraphExecutor {
    pub fn new(cfg: StreamConfig) -> Self {
        GraphExecutor { cfg }
    }

    pub fn enqueue_graph(
        &self,
        graph: &ReconfigurableGraph,
        exec: &mut AsyncExecutor,
        mem: &mut HybridMemoryManager,
        snapshot: &MemorySnapshot,
        gpu_available: bool,
    ) -> Vec<RoutedBundle> {
        let mut bundles = Vec::new();

        for node in graph.nodes() {
            let tensor_id_refs: Vec<&str> = node.tensor_ids.iter().map(|s| s.as_str()).collect();

            let bundle = StreamRouter::route_kernel_with_memory(
                exec,
                mem,
                &node.kernel,
                &tensor_id_refs,
                snapshot,
                gpu_available,
            );

            bundles.push(bundle);
        }

        bundles
    }
}
