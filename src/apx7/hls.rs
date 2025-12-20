use crate::amg::graph::Graph;

#[derive(Clone, Debug)]
pub struct HLSCluster {
    pub nodes: Vec<usize>,
    pub score: f32,
}

pub struct HLSScheduler<'a> {
    pub graph: &'a Graph,
    pub children: Vec<Vec<usize>>,
    pub parents_left: Vec<usize>,
}

impl<'a> HLSScheduler<'a> {
    pub fn new(graph: &'a Graph) -> Self {
        let n = graph.nodes.len();
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut parents_left = vec![0usize; n];

        for (id, node) in graph.nodes.iter().enumerate() {
            parents_left[id] = node.inputs.len();
            for &inp in &node.inputs {
                if inp < n {
                    children[inp].push(id);
                }
            }
        }

        Self {
            graph,
            children,
            parents_left,
        }
    }

    fn topo_order(&self) -> Vec<usize> {
        let n = self.graph.nodes.len();
        let mut in_deg = self.parents_left.clone();
        let mut ready: Vec<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();
        let mut order = Vec::with_capacity(n);

        while let Some(u) = ready.pop() {
            order.push(u);
            for &v in &self.children[u] {
                if in_deg[v] > 0 {
                    in_deg[v] -= 1;
                    if in_deg[v] == 0 {
                        ready.push(v);
                    }
                }
            }
        }

        order
    }

    fn build_clusters(&self, topo: &[usize]) -> Vec<HLSCluster> {
        use std::collections::HashMap;

        let mut clusters: Vec<HLSCluster> = Vec::new();
        let mut map: HashMap<(usize, usize), usize> = HashMap::new();

        for &node_id in topo {
            let node = &self.graph.nodes[node_id];
            let parents_len = node.inputs.len();
            let children_len = self.children[node_id].len();
            let sig = (parents_len, children_len);

            let idx = map.entry(sig).or_insert_with(|| {
                let idx = clusters.len();
                clusters.push(HLSCluster { nodes: Vec::new(), score: 0.0 });
                idx
            });

            clusters[*idx].nodes.push(node_id);
            clusters[*idx].score += 1.0; // base cohesion by size
        }

        clusters
    }

    fn refine_by_locality(&self, clusters: &mut [HLSCluster]) {
        for cluster in clusters.iter_mut() {
            let nodes = &cluster.nodes;
            for i in 0..nodes.len() {
                for j in (i + 1)..nodes.len() {
                    let a = nodes[i];
                    let b = nodes[j];
                    let na = &self.graph.nodes[a];
                    let nb = &self.graph.nodes[b];

                    // Share immediate parent
                    let same_parent = na.inputs.get(0) == nb.inputs.get(0);

                    // Share any input
                    let shares_input = na
                        .inputs
                        .iter()
                        .any(|pa| nb.inputs.iter().any(|pb| pa == pb));

                    // Same op type also suggests some locality
                    let same_op = std::mem::discriminant(&na.node_type)
                        == std::mem::discriminant(&nb.node_type);

                    if same_parent || shares_input || same_op {
                        cluster.score += 1.0;
                    }
                }
            }
        }
    }

    fn order_clusters(&self, clusters: &mut Vec<HLSCluster>) {
        fn fanout_for(cluster: &HLSCluster, children: &Vec<Vec<usize>>) -> usize {
            cluster.nodes.iter().map(|&id| children[id].len()).sum()
        }

        clusters.sort_by(|a, b| {
            let fanout_a = fanout_for(a, &self.children);
            let fanout_b = fanout_for(b, &self.children);

            // score desc, size desc, fanout asc
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.nodes.len().cmp(&a.nodes.len()))
                .then_with(|| fanout_a.cmp(&fanout_b))
        });
    }

    pub fn run(&self) -> Vec<HLSCluster> {
        let topo = self.topo_order();
        if topo.is_empty() {
            return Vec::new();
        }

        let mut clusters = self.build_clusters(&topo);
        self.refine_by_locality(&mut clusters);
        self.order_clusters(&mut clusters);
        clusters
    }
}
