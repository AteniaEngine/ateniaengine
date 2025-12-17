// APX 5.3 — Prototipo genérico para adaptar al engine real

#[derive(Clone, Debug)]
pub struct ExecutionPlan5_3 {
    pub kernel_name: &'static str,
    pub layout: LayoutDecision,
    pub chunking: Option<ChunkInfo>,
    pub use_fp8: bool,
}

#[derive(Clone, Debug)]
pub enum LayoutDecision {
    Original,
    ForceContiguous,
    ForceChannelsFirst,
}

#[derive(Clone, Debug)]
pub struct ChunkInfo {
    pub chunks: usize,
}

#[derive(Clone, Debug, Default)]
pub struct NodeExecInfo {
    pub node_id: usize,
    pub op_name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub contiguous: bool,
    pub device_52: String,   // "CPU" o "GPU" según decisión de APX 5.2
    pub estimated_bytes: usize,
    pub estimated_flops: usize,
    pub vram_free: usize,
    pub kernel_time_avg: f32,
    pub preferred_kernel_size: Option<(usize, usize)>,
    pub tile_override: Option<(usize, usize, usize)>,
    pub scheduling_bias: Option<&'static str>,
    pub qkv_bias: Option<bool>,
    pub attention_bias: Option<bool>,
    pub exec_priority: Option<u8>,
    pub prefetch_hint: Option<&'static str>,
}

impl NodeExecInfo {
    pub fn apply_qkv_bias(&mut self) {
        self.preferred_kernel_size = Some((64, 64));
        self.scheduling_bias = Some("qkv_fastpath");
        self.qkv_bias = Some(true);
    }

    pub fn apply_attention_bias(&mut self) {
        self.preferred_kernel_size = Some((128, 128));
        self.tile_override = Some((64, 128, 32));
        self.scheduling_bias = Some("attn_large");
        self.attention_bias = Some(true);
    }

    // APX 6.12: hints de scheduling puro (no tocan la matemática de la op).
    pub fn bias_qkv_schedule(&mut self) {
        self.exec_priority = Some(250);
        self.prefetch_hint = Some("prefetch_qkv_weights");
    }

    pub fn bias_attention_schedule(&mut self) {
        self.exec_priority = Some(1);
        self.prefetch_hint = Some("prefetch_attn_weights");
    }
}

pub struct Planner5_3;

impl Planner5_3 {
    pub fn new() -> Self {
        Self
    }

    pub fn select_plan(&self, info: &NodeExecInfo) -> ExecutionPlan5_3 {
        // 1) Por defecto, no cambiar nada
        let mut plan = ExecutionPlan5_3 {
            kernel_name: "default",
            layout: LayoutDecision::Original,
            chunking: None,
            use_fp8: false,
        };

        // 2) Heurísticas específicas para MatMul. Estas decisiones sólo
        // afectan al plan (logs, diagnóstico); la ejecución real de MatMul
        // sigue controlada por APX 5.2 y el dispatcher actual.
        if info.op_name == "MatMul" {
            let num_elems: usize = info.shape.iter().product();

            // Umbrales simples iniciales.
            let small_threshold: usize = 4_096;
            let large_threshold: usize = 1_000_000;

            if num_elems <= small_threshold {
                // MatMul pequeño: mejor mantener CPU.
                plan.kernel_name = "matmul_cpu_small";
                plan.layout = LayoutDecision::Original;
            } else {
                // MatMul mediano/grande.
                if num_elems >= large_threshold
                    && info.dtype == "F32"
                    && info.contiguous
                {
                    // Buen candidato a GPU en un futuro: lo marcamos en el
                    // plan, aunque hoy la decisión real de CPU/GPU la toma
                    // APX 5.2.
                    plan.kernel_name = "matmul_gpu_candidate";
                } else {
                    plan.kernel_name = "matmul_cpu_medium";
                }

                // Si el tensor no es contiguo y es grande, sugerimos
                // contiguidad en el plan (sin aplicar aún realineado).
                if !info.contiguous && num_elems >= large_threshold {
                    plan.layout = LayoutDecision::ForceContiguous;
                }
            }
        } else if info.op_name == "BatchMatMul" {
            // 2b) Heurísticas análogas para BatchMatMul. Usamos el producto
            // total de la shape (batch * m * k * n) como proxy de tamaño.
            let num_elems: usize = info.shape.iter().product();

            // Umbrales específicos para BatchMatMul.
            let small_threshold: usize = 4_096;
            let medium_threshold: usize = 200_000;

            if num_elems <= small_threshold {
                plan.kernel_name = "batch_matmul_cpu_small";
                plan.layout = LayoutDecision::Original;
            } else if num_elems <= medium_threshold {
                plan.kernel_name = "batch_matmul_cpu_medium";
                plan.layout = LayoutDecision::Original;
            } else {
                plan.kernel_name = "batch_matmul_gpu_candidate";

                // Si el tensor no es contiguo y es grande, sugerimos
                // contigüidad en el plan.
                if !info.contiguous {
                    plan.layout = LayoutDecision::ForceContiguous;
                }
            }
        } else if info.op_name == "Linear" {
            // 2c) Heurísticas similares para Linear (FFN). Usamos el número
            // total de elementos del tensor de salida estimado como proxy.
            let num_elems: usize = info.shape.iter().product();

            let small_threshold: usize = 4_096;
            let large_threshold: usize = 1_000_000;

            if num_elems <= small_threshold {
                plan.kernel_name = "linear_cpu_small";
                plan.layout = LayoutDecision::Original;
            } else {
                if num_elems >= large_threshold && info.dtype == "F32" && info.contiguous {
                    plan.kernel_name = "linear_gpu_candidate";
                } else {
                    plan.kernel_name = "linear_cpu_medium";
                }

                if !info.contiguous && num_elems >= large_threshold {
                    plan.layout = LayoutDecision::ForceContiguous;
                }
            }
        }

        // 3) Fallback absoluto: si el plan no es seguro, restaurar plan original.
        if !self.plan_es_seguro(&plan) {
            return ExecutionPlan5_3 {
                kernel_name: "default",
                layout: LayoutDecision::Original,
                chunking: None,
                use_fp8: false,
            };
        }

        plan
    }

    fn plan_es_seguro(&self, _plan: &ExecutionPlan5_3) -> bool {
        // Aquí, cuando lo adaptes al engine real, puedes validar:
        // - que kernel_name corresponde a un kernel existente
        // - que layout es soportado por esa op
        // - que chunking/FP8 son realmente soportados por el nodo
        true
    }
}
