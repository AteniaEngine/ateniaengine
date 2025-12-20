// APX 5.3 — Generic prototype to adapt to the real engine

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
    pub device_52: String,   // "CPU" or "GPU" according to APX 5.2 decision
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

    // APX 6.12: pure scheduling hints (they do not touch the op's math).
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
        // 1) By default, do not change anything
        let mut plan = ExecutionPlan5_3 {
            kernel_name: "default",
            layout: LayoutDecision::Original,
            chunking: None,
            use_fp8: false,
        };

        // 2) MatMul-specific heuristics. These decisions only affect the plan
        // (logs, diagnostics); real MatMul execution remains controlled by
        // APX 5.2 and the current dispatcher.
        if info.op_name == "MatMul" {
            let num_elems: usize = info.shape.iter().product();

            // Simple initial thresholds.
            let small_threshold: usize = 4_096;
            let large_threshold: usize = 1_000_000;

            if num_elems <= small_threshold {
                // Small MatMul: better keep CPU.
                plan.kernel_name = "matmul_cpu_small";
                plan.layout = LayoutDecision::Original;
            } else {
                // Medium/large MatMul.
                if num_elems >= large_threshold
                    && info.dtype == "F32"
                    && info.contiguous
                {
                    // Good future GPU candidate: we mark it in the plan,
                    // although today the real CPU/GPU decision is made by
                    // APX 5.2.
                    plan.kernel_name = "matmul_gpu_candidate";
                } else {
                    plan.kernel_name = "matmul_cpu_medium";
                }

                // If the tensor is non-contiguous and large, suggest
                // contiguity in the plan (without yet applying realignment).
                if !info.contiguous && num_elems >= large_threshold {
                    plan.layout = LayoutDecision::ForceContiguous;
                }
            }
        } else if info.op_name == "BatchMatMul" {
            // 2b) Similar heuristics for BatchMatMul. Use the total shape
            // product (batch * m * k * n) as a size proxy.
            let num_elems: usize = info.shape.iter().product();

            // BatchMatMul-specific thresholds.
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

                // If the tensor is non-contiguous and large, suggest
                // contiguity in the plan.
                if !info.contiguous {
                    plan.layout = LayoutDecision::ForceContiguous;
                }
            }
        } else if info.op_name == "Linear" {
            // 2c) Similar heuristics for Linear (FFN). Use the total number
            // of elements of the estimated output tensor as proxy.
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

        // 3) Absolute fallback: if the plan is not safe, restore the original plan.
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
        // Here, when adapting this to the real engine, you can validate:
        // - that kernel_name corresponds to an existing kernel
        // - that layout is supported by that op
        // - that chunking/FP8 are actually supported by the node
        true
    }
}
