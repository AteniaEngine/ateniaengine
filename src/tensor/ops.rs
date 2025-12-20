//! Tensor operation kernels and composable primitives.

use super::tensor::{Device, Layout, Tensor};

pub mod matmul_cpu;
pub mod batch_matmul;

impl Tensor {
    /// Element-wise addition between two tensors, keeping the device from `self`.
    pub fn add(&self, other: &Tensor) -> Tensor {
        self.validate_binary_op(other, "add");

        if self.layout == Layout::Contiguous
            && other.layout == Layout::Contiguous
            && self.strides == other.strides
        {
            fast_add(self, other)
        } else {
            slow_add(self, other)
        }
    }

    /// Element-wise multiplication between two tensors, keeping the device from `self`.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        self.validate_binary_op(other, "mul");

        if self.layout == Layout::Contiguous
            && other.layout == Layout::Contiguous
            && self.strides == other.strides
        {
            fast_mul(self, other)
        } else {
            slow_mul(self, other)
        }
    }

    /// Element-wise subtraction between two tensors, keeping the device from `self`.
    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.validate_binary_op(other, "sub");

        if self.layout == Layout::Contiguous
            && other.layout == Layout::Contiguous
            && self.strides == other.strides
        {
            fast_sub(self, other)
        } else {
            slow_sub(self, other)
        }
    }

    fn validate_binary_op(&self, other: &Tensor, op: &str) {
        assert_eq!(self.shape.len(), other.shape.len(), "Tensor ranks must match for {}", op);
        assert_eq!(self.shape, other.shape, "Tensor shapes must match for {}", op);
        assert_eq!(self.strides.len(), other.strides.len(), "Stride ranks must match for {}", op);
        assert_eq!(self.dtype, other.dtype, "Tensor dtypes must match for {}", op);
        assert_eq!(
            self.num_elements(),
            other.num_elements(),
            "Tensor element counts must match for {}",
            op
        );
    }

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        self.validate_matmul_dims(other);
        if self.device == Device::CPU && other.device == Device::CPU {
            matmul_cpu::matmul_parallel(self, other)
        } else {
            matmul_serial(self, other)
        }
    }

    /// APX 7.0: public wrapper that enables PEX mode before delegating
    /// to the standard matmul path. The real parallel execution logic
    /// is controlled via runtime flags in the dispatcher.
    pub fn matmul_parallel(&self, rhs: &Tensor) -> Tensor {
        let mut flags = crate::config::get_runtime_flags();
        flags.enable_pex = true;
        drop(flags);
        self.matmul(rhs)
    }

    /// APX 7.1: variant that enables PEX + work-stealing before delegating
    /// to the standard matmul path.
    pub fn matmul_parallel_ws(&self, rhs: &Tensor) -> Tensor {
        let mut flags = crate::config::get_runtime_flags();
        flags.enable_pex = true;
        flags.enable_workstealing = true;
        drop(flags);
        self.matmul(rhs)
    }

    /// APX 7.3: optional wrapper that enables Adaptive PGL mode before
    /// delegating to the standard matmul path. The behavior remains
    /// mathematically identical; it only affects internal scheduling logic
    /// when APX_MODE >= 7.3.
    pub fn matmul_adaptive(&self, rhs: &Tensor) -> Tensor {
        let mut flags = crate::config::get_runtime_flags();
        flags.enable_adaptive_pgl = true;
        drop(flags);
        self.matmul(rhs)
    }

    fn validate_matmul_dims(&self, other: &Tensor) {
        assert_eq!(
            self.shape.len(), 2,
            "matmul expects lhs to be 2D, got rank {}",
            self.shape.len()
        );
        assert_eq!(
            other.shape.len(), 2,
            "matmul expects rhs to be 2D, got rank {}",
            other.shape.len()
        );
        assert_eq!(
            self.shape[1], other.shape[0],
            "matmul inner dims must match: {} vs {}",
            self.shape[1], other.shape[0]
        );
    }
}

fn fast_add(a: &Tensor, b: &Tensor) -> Tensor {
    record_fast_add_call();

    let data = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(lhs, rhs)| lhs + rhs)
        .collect();

    create_result_tensor(a, data)
}

fn fast_mul(a: &Tensor, b: &Tensor) -> Tensor {
    record_fast_mul_call();

    let data = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(lhs, rhs)| lhs * rhs)
        .collect();

    create_result_tensor(a, data)
}

fn fast_sub(a: &Tensor, b: &Tensor) -> Tensor {
    let data = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(lhs, rhs)| lhs - rhs)
        .collect();

    create_result_tensor(a, data)
}

fn slow_add(a: &Tensor, b: &Tensor) -> Tensor {
    record_slow_add_call();
    elementwise_with_strides(a, b, |lhs, rhs| lhs + rhs)
}

fn slow_mul(a: &Tensor, b: &Tensor) -> Tensor {
    record_slow_mul_call();
    elementwise_with_strides(a, b, |lhs, rhs| lhs * rhs)
}

fn slow_sub(a: &Tensor, b: &Tensor) -> Tensor {
    elementwise_with_strides(a, b, |lhs, rhs| lhs - rhs)
}

fn elementwise_with_strides<F>(a: &Tensor, b: &Tensor, op: F) -> Tensor
where
    F: Fn(f32, f32) -> f32,
{
    let total = a.num_elements();
    let mut data = vec![0.0; total];

    if total == 0 {
        return create_result_tensor(a, data);
    }

    if a.shape.is_empty() {
        data[0] = op(a.data[0], b.data[0]);
        return create_result_tensor(a, data);
    }

    let mut index = vec![0usize; a.shape.len()];
    loop {
        let offset_a = linear_offset(&index, &a.strides);
        let offset_b = linear_offset(&index, &b.strides);
        data[offset_a] = op(a.data[offset_a], b.data[offset_b]);

        if !increment_index(&mut index, &a.shape) {
            break;
        }
    }

    create_result_tensor(a, data)
}

fn linear_offset(index: &[usize], strides: &[usize]) -> usize {
    index
        .iter()
        .zip(strides)
        .map(|(i, stride)| i * stride)
        .sum()
}

fn increment_index(index: &mut [usize], shape: &[usize]) -> bool {
    for axis in (0..index.len()).rev() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return true;
        }
        index[axis] = 0;
    }
    false
}

fn create_result_tensor(proto: &Tensor, data: Vec<f32>) -> Tensor {
    Tensor {
        shape: proto.shape.clone(),
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
}

mod instrumentation {
    use std::cell::Cell;
    thread_local! {
        static FAST_ADD_CALLS: Cell<usize> = Cell::new(0);
        static SLOW_ADD_CALLS: Cell<usize> = Cell::new(0);
        static FAST_MUL_CALLS: Cell<usize> = Cell::new(0);
        static SLOW_MUL_CALLS: Cell<usize> = Cell::new(0);
    }

    pub fn reset() {
        FAST_ADD_CALLS.with(|c| c.set(0));
        SLOW_ADD_CALLS.with(|c| c.set(0));
        FAST_MUL_CALLS.with(|c| c.set(0));
        SLOW_MUL_CALLS.with(|c| c.set(0));
    }

    pub fn inc_fast_add() {
        FAST_ADD_CALLS.with(|c| c.set(c.get() + 1));
    }

    pub fn inc_slow_add() {
        SLOW_ADD_CALLS.with(|c| c.set(c.get() + 1));
    }

    pub fn inc_fast_mul() {
        FAST_MUL_CALLS.with(|c| c.set(c.get() + 1));
    }

    pub fn inc_slow_mul() {
        SLOW_MUL_CALLS.with(|c| c.set(c.get() + 1));
    }

    pub fn fast_add_calls() -> usize {
        FAST_ADD_CALLS.with(|c| c.get())
    }

    pub fn slow_add_calls() -> usize {
        SLOW_ADD_CALLS.with(|c| c.get())
    }

    pub fn fast_mul_calls() -> usize {
        FAST_MUL_CALLS.with(|c| c.get())
    }

    pub fn slow_mul_calls() -> usize {
        SLOW_MUL_CALLS.with(|c| c.get())
    }
}

pub mod testing {
    use super::instrumentation;

    pub fn reset_op_counters() {
        instrumentation::reset();
    }

    pub fn fast_add_calls() -> usize {
        instrumentation::fast_add_calls()
    }

    pub fn slow_add_calls() -> usize {
        instrumentation::slow_add_calls()
    }

    pub fn fast_mul_calls() -> usize {
        instrumentation::fast_mul_calls()
    }

    pub fn slow_mul_calls() -> usize {
        instrumentation::slow_mul_calls()
    }
}

fn record_fast_add_call() {
    instrumentation::inc_fast_add();
}

fn record_slow_add_call() {
    instrumentation::inc_slow_add();
}

fn record_fast_mul_call() {
    instrumentation::inc_fast_mul();
}

fn record_slow_mul_call() {
    instrumentation::inc_slow_mul();
}

fn matmul_serial(a: &Tensor, b: &Tensor) -> Tensor {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];
    let mut out = Tensor::with_layout(
        vec![m, n],
        0.0,
        a.device,
        Layout::Contiguous,
        a.dtype,
    );
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..k {
                let a_idx = i * k + kk;
                let b_idx = kk * n + j;
                sum += a.data[a_idx] * b.data[b_idx];
            }
            out.data[i * n + j] = sum;
        }
    }
    out
}
