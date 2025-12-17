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

            for i in 0..grad.len() {
                let g = grad[i];
                self.m[idx][i] = self.beta1 * self.m[idx][i] + (1.0 - self.beta1) * g;
                self.v[idx][i] = self.beta2 * self.v[idx][i] + (1.0 - self.beta2) * g * g;

                let m_hat = self.m[idx][i] / bias_correction1;
                let v_hat = self.v[idx][i] / bias_correction2;

                let denom = v_hat.sqrt() + self.eps;
                let wd_term = self.weight_decay * param.data[i];
                param.data[i] -= self.lr * (m_hat / denom + wd_term);
            }

            param.clear_grad();
        }
    }
}
