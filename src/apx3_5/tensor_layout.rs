#[derive(Debug, Clone)]
pub struct TensorLayoutInfo {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub num_elements: usize,
}

impl TensorLayoutInfo {
    pub fn from_shape_and_strides(shape: &[usize], strides: &[usize]) -> Self {
        let num_elements = shape.iter().product();
        Self {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            num_elements,
        }
    }
}
