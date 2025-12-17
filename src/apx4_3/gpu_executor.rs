use crate::amg::graph::Graph;
use crate::amg::nodes::NodeType;
use crate::apx4_3::{gpu_enabled, log_gpu, gpu_plan::GpuSegment};
use crate::cuda::matmul::cuda_matmul;
use crate::cuda::linear::cuda_linear;
use crate::cuda::fused_linear_silu::cuda_fused_linear_silu;

impl Graph {
    pub fn exec_gpu_segment(&mut self, seg: &GpuSegment) {
        if !gpu_enabled() {
            return;
        }

        for id in seg.start..=seg.end {
            let node = &self.nodes[id];
            match &node.node_type {
                NodeType::MatMul => {
                    log_gpu(&format!("Executing MatMul on GPU (node {id})"));
                    self.exec_gpu_matmul(id);
                }
                NodeType::Add => {
                    log_gpu(&format!("Executing Add on GPU (node {id})"));
                    self.exec_gpu_add(id);
                }
                NodeType::Mul => {
                    log_gpu(&format!("Executing Mul on GPU (node {id})"));
                    self.exec_gpu_mul(id);
                }
                NodeType::Linear => {
                    log_gpu(&format!("Executing Linear on GPU (node {id})"));
                    self.exec_gpu_linear(id);
                }
                NodeType::FusedLinearActivation(crate::amg::nodes::ActType::SiLU) => {
                    log_gpu(&format!(
                        "Executing FusedLinearActivation(SiLU) on GPU (node {id})"
                    ));
                    self.exec_gpu_fused_linear_silu(id);
                }
                _ => {}
            }
        }
    }

    pub fn exec_gpu_matmul(&mut self, id: usize) {
        let a_id = self.nodes[id].inputs[0];
        let b_id = self.nodes[id].inputs[1];
        let a = self.nodes[a_id].output.as_ref().expect("MatMul missing A");
        let b = self.nodes[b_id].output.as_ref().expect("MatMul missing B");

        assert_eq!(a.shape.len(), 2);
        assert_eq!(b.shape.len(), 2);
        let m = a.shape[0];
        let k = a.shape[1];
        assert_eq!(b.shape[0], k);
        let n = b.shape[1];

        let gpu_out = cuda_matmul(a, b, m, k, n);

        self.nodes[id].output = Some(gpu_out);
    }

    pub fn exec_gpu_add(&mut self, id: usize) {
        let a_id = self.nodes[id].inputs[0];
        let b_id = self.nodes[id].inputs[1];
        let a = self.nodes[a_id].output.as_ref().expect("Add missing A");
        let b = self.nodes[b_id].output.as_ref().expect("Add missing B");

        // De momento, usar suma CPU pero ejecutada en el segmento GPU lógico.
        let gpu_out = a.add(b);

        self.nodes[id].output = Some(gpu_out);
    }

    pub fn exec_gpu_mul(&mut self, id: usize) {
        let inputs = self.nodes[id].inputs.clone();
        if inputs.len() < 2 {
            // Grafo inconsistente o nodo Mul parcial; no intentamos la ruta GPU.
            return;
        }

        let a_opt = self.nodes[inputs[0]].output.as_ref();
        let b_opt = self.nodes[inputs[1]].output.as_ref();

        if let (Some(a), Some(b)) = (a_opt, b_opt) {
            // Placeholder: multiplicación CPU dentro del segmento GPU hasta tener kernel dedicado.
            let gpu_out = a.mul(b);
            self.nodes[id].output = Some(gpu_out);
        }
    }

    pub fn exec_gpu_linear(&mut self, id: usize) {
        use crate::nn::linear::linear;

        let inputs = self.nodes[id].inputs.clone();
        // Necesitamos al menos [x, w]. Si no hay suficientes entradas, no
        // intentamos la ruta GPU.
        if inputs.len() < 2 {
            return;
        }

        let x_id = inputs[0];
        let w_id = inputs[1];

        let x_opt = self.nodes[x_id].output.as_ref();
        let w_opt = self.nodes[w_id].output.as_ref();

        // En algunos grafos (especialmente tras fusiones y trazas APX 4.9),
        // un segmento GPU puede empezar en un Linear cuyo input todavía no
        // ha sido materializado. En lugar de hacer panic, hacemos fallback
        // a la ruta CPU para ese nodo.
        if x_opt.is_none() || w_opt.is_none() {
            if !crate::apx_is_silent() {
                eprintln!(
                    "[APX 4.3 GPU] exec_gpu_linear fallback to CPU | node_id={} | has_x={} | has_w={}",
                    id,
                    x_opt.is_some(),
                    w_opt.is_some(),
                );
            }
            self.exec_cpu_linear_fallback(id);
            return;
        }

        let x = x_opt.expect("checked above as Some(x)");
        let w = w_opt.expect("checked above as Some(w)");

        // Ruta CUDA solo soporta, de momento, Linear con bias explícito
        // [x, w, b]. Si no hay bias, usamos siempre la ruta CPU.
        let maybe_b = if inputs.len() >= 3 {
            Some(
                self.nodes[inputs[2]]
                    .output
                    .as_ref()
                    .expect("Linear missing b"),
            )
        } else {
            None
        };

        let can_use_cuda = if let Some(b) = maybe_b {
            x.shape.len() == 2
                && w.shape.len() == 2
                && b.shape.len() == 1
                && w.shape[0] == x.shape[1]
                && b.shape[0] == w.shape[1]
        } else {
            false
        };

        if can_use_cuda {
            let b = maybe_b.expect("bias checked above");
            let m = x.shape[0];
            let k = x.shape[1];
            let n = w.shape[1];

            let mut out = crate::tensor::Tensor::zeros_new(&[m, n], x.device);

            cuda_linear(&x.data, &w.data, &b.data, &mut out.data, m, k, n);

            self.nodes[id].output = Some(out);
        } else {
            // Fallback CPU cuando no hay bias compatible con el kernel CUDA
            // o cuando las shapes no cumplen los requisitos.
            let out = linear(x, w, maybe_b);
            self.nodes[id].output = Some(out);
        }
    }

    pub fn exec_gpu_fused_linear_silu(&mut self, id: usize) {
        let inputs = self.nodes[id].inputs.clone();
        assert!(
            inputs.len() == 3,
            "exec_gpu_fused_linear_silu expects [x, w, b] inputs",
        );

        let x = self.nodes[inputs[0]].output.as_ref().expect("FusedLinear SiLU missing x");
        let w = self.nodes[inputs[1]].output.as_ref().expect("FusedLinear SiLU missing w");
        let b = self.nodes[inputs[2]].output.as_ref().expect("FusedLinear SiLU missing b");

        // Shapes: x [M,K], w [K,N], b [N]
        assert_eq!(x.shape.len(), 2);
        assert_eq!(w.shape.len(), 2);
        assert_eq!(b.shape.len(), 1);
        let m = x.shape[0];
        let k = x.shape[1];
        assert_eq!(w.shape[0], k);
        let n = w.shape[1];
        assert_eq!(b.shape[0], n);

        let mut out = crate::tensor::Tensor::zeros_new(&[m, n], x.device);

        cuda_fused_linear_silu(&x.data, &w.data, &b.data, &mut out.data, m, k, n);

        self.nodes[id].output = Some(out);
    }
}
