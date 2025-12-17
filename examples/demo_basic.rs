use atenia_engine::hal;
use atenia_engine::tensor::tensor::{Device, DType, Layout, Tensor};
use atenia_engine::training::trainer::{Trainer, TrainerConfig};

fn main() {
    println!("=== Atenia Engine Demo ===");

    // 1) HAL: CUDA detection
    let cuda_available = hal::detect_cuda();
    println!("CUDA available: {}", cuda_available);

    // 2) Define a sample tensor shape (1D vector of 1024 floats)
    let shape = vec![1024];
    let sample = Tensor::with_layout(
        shape.clone(),
        0.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    );

    // 3) Build a fake dataset of N samples
    let num_samples = 50;
    let mut dataset = Vec::new();
    for i in 0..num_samples {
        let mut t = Tensor::with_layout(
            shape.clone(),
            0.0,
            Device::CPU,
            Layout::Contiguous,
            DType::F32,
        );
        for v in t.data.iter_mut() {
            *v = i as f32;
        }
        dataset.push(t);
    }

    // 4) Configure Trainer with a conservative memory limit (256 MB)
    let config = TrainerConfig {
        memory_limit_bytes: 256 * 1024 * 1024,
        safety_margin_bytes: 32 * 1024 * 1024,
    };

    let trainer = Trainer::new(config);

    // 5) Estimate batch size
    let batch_size = trainer.batch_manager.estimate_max_batch_size(&sample);
    println!("Estimated batch size: {}", batch_size);

    if batch_size == 0 {
        println!("Batch size is zero. Memory limit is too small for this sample.");
        return;
    }

    // 6) Run training
    println!("Starting training over {} samples...", num_samples);
    let losses = trainer.train(&dataset, &sample);

    println!("Training finished. Collected {} loss values.", losses.len());
    if let Some(first) = losses.first() {
        println!("First loss: {}", first);
    }
    if let Some(last) = losses.last() {
        println!("Last loss: {}", last);
    }

    println!("=== Demo completed ===");
}
