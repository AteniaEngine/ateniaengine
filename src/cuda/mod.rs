use std::os::raw::c_int;

pub mod matmul;
pub mod linear;
pub mod batch_matmul;
pub mod fused_linear_silu;

#[link(name = "atenia_kernels", kind = "static")]
unsafe extern "C" {
    pub fn vec_add_cuda(
        a: *const f32,
        b: *const f32,
        out: *mut f32,
        n: c_int,
    );
}

pub fn cuda_available() -> bool {
    std::process::Command::new("nvidia-smi").output().is_ok()
}

pub fn vec_add_gpu(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());

    let n = a.len() as c_int;
    let mut out = vec![0.0f32; a.len()];

    unsafe {
        vec_add_cuda(
            a.as_ptr(),
            b.as_ptr(),
            out.as_mut_ptr(),
            n,
        );
    }

    out
}
