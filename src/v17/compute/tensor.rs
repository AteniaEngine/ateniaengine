#![allow(dead_code)]

/// Minimal tensor representation used by the CPU backend.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(format!(
                "shape {:?} implies {} elements but data has {}",
                shape,
                expected,
                data.len()
            ));
        }
        Ok(Self { shape, data })
    }
}
