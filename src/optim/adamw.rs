use crate::tensor::Tensor;

pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub m: Vec<Vec<f32>>, // first moment
    pub v: Vec<Vec<f32>>, // second moment
    pub t: usize,
}

impl AdamW {
    pub fn new(param_count: usize, lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        let m = vec![Vec::new(); param_count];
        let v = vec![Vec::new(); param_count];
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            m,
            v,
            t: 0,
        }
    }

    fn ensure_slot(slot: &mut Vec<f32>, len: usize) {
        if slot.len() != len {
            slot.clear();
            slot.resize(len, 0.0);
        }
    }

    pub fn update(&mut self, params: &mut [&mut Tensor]) {
        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powf(self.t as f32);
        let bias_correction2 = 1.0 - self.beta2.powf(self.t as f32);

        for (idx, param) in params.iter_mut().enumerate() {
            let grad = match &param.grad {
                Some(g) => g,
                None => continue,
            };

            if idx >= self.m.len() {
                self.m.resize(idx + 1, Vec::new());
                self.v.resize(idx + 1, Vec::new());
            }

            Self::ensure_slot(&mut self.m[idx], grad.len());
            Self::ensure_slot(&mut self.v[idx], grad.len());

            // Split-borrow trick: `param.grad` (borrowed via `grad`) and
            // `param.storage` are disjoint fields, so we access `storage`
            // directly to avoid the `&mut self` method call that would
            // conflict with the `grad` borrow held across this loop.
            let param_data = match &mut param.storage {
                crate::tensor::TensorStorage::Cpu(v) => v,
                crate::tensor::TensorStorage::Cuda(_) => panic!(
                    "AdamW: parameter is GPU-resident; call ensure_cpu() \
                     before the optimizer step. Native GPU optimizer is \
                     scheduled for M3-d.4+."
                ),
                crate::tensor::TensorStorage::Disk(_) => panic!(
                    "AdamW: parameter is Disk-resident; call ensure_cpu() \
                     before the optimizer step to materialize data back in \
                     host memory. Disk-spilled parameters are M3-e.11 \
                     reactive-path territory, not a training-loop steady \
                     state."
                ),
                crate::tensor::TensorStorage::CpuShared(_)
                | crate::tensor::TensorStorage::CpuBf16Shared(_) => panic!(
                    "AdamW: parameter is Arc-shared (CpuShared / \
                     CpuBf16Shared); shared storage is read-only by \
                     construction (M5.c.2.a) and cannot be mutated by \
                     the optimizer. If a training scenario genuinely \
                     needs to update a shared parameter, call \
                     ensure_owned() first to clone-out into Cpu storage."
                ),
                crate::tensor::TensorStorage::CpuBf16(_) => panic!(
                    "AdamW: parameter is CpuBf16-resident; BF16 storage \
                     for trainable parameters is out of M4.7.2 scope \
                     (forward-only inference). A training pipeline \
                     against BF16 params requires F32 accumulators and \
                     a write-back path, which is M5+ territory."
                ),
            };
            for i in 0..grad.len() {
                let g = grad[i];
                self.m[idx][i] = self.beta1 * self.m[idx][i] + (1.0 - self.beta1) * g;
                self.v[idx][i] = self.beta2 * self.v[idx][i] + (1.0 - self.beta2) * g * g;

                let m_hat = self.m[idx][i] / bias_correction1;
                let v_hat = self.v[idx][i] / bias_correction2;

                let denom = v_hat.sqrt() + self.eps;
                let wd_term = self.weight_decay * param_data[i];
                param_data[i] -= self.lr * (m_hat / denom + wd_term);
            }

            param.clear_grad();
        }
    }
}
