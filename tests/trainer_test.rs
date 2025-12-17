use atenia_engine::tensor::tensor::{Device, DType, Layout, Tensor};
use atenia_engine::training::trainer::{Trainer, TrainerConfig};

fn sample_template(num_elements: usize) -> Tensor {
    Tensor::with_layout(
        vec![num_elements],
        1.0,
        Device::CPU,
        Layout::Contiguous,
        DType::F32,
    )
}

fn dataset_with_samples(count: usize, num_elements: usize) -> Vec<Tensor> {
    (0..count)
        .map(|i| {
            let mut tensor = sample_template(num_elements);
            tensor.data = vec![(i + 1) as f32; num_elements];
            tensor
        })
        .collect()
}

#[test]
fn trainer_initializes() {
    let config = TrainerConfig {
        memory_limit_bytes: 1024,
        safety_margin_bytes: 128,
    };
    let trainer = Trainer::new(config);
    assert_eq!(trainer.batch_manager.memory_limit_bytes, 1024);
}

#[test]
fn forward_sums_data() {
    let trainer = Trainer::new(TrainerConfig {
        memory_limit_bytes: 1024,
        safety_margin_bytes: 0,
    });
    let mut tensor = sample_template(4);
    tensor.data = vec![1.0, 2.0, 3.0, 4.0];
    assert_eq!(trainer.forward(&tensor), 10.0);
}

#[test]
fn train_returns_losses_per_batch() {
    let trainer = Trainer::new(TrainerConfig {
        memory_limit_bytes: 4096,
        safety_margin_bytes: 0,
    });
    let dataset = dataset_with_samples(6, 4);
    let template = sample_template(4);
    let losses = trainer.train(&dataset, &template);
    assert_eq!(losses.len(), 1); // All samples fit in one batch.
    assert!(losses[0] > 0.0);
}

#[test]
fn batch_size_respects_memory_limit() {
    let per_sample = sample_template(8);
    let config = TrainerConfig {
        memory_limit_bytes: per_sample.estimated_bytes() * 5,
        safety_margin_bytes: per_sample.estimated_bytes(),
    };
    let trainer = Trainer::new(config);
    let dataset = dataset_with_samples(20, 8);
    let losses = trainer.train(&dataset, &per_sample);

    assert!(losses.len() >= 4); // Expect multiple batches due to tight limit.
}

#[test]
fn tiny_memory_limit_yields_no_training() {
    let per_sample = sample_template(4);
    let trainer = Trainer::new(TrainerConfig {
        memory_limit_bytes: 0,
        safety_margin_bytes: 0,
    });
    let dataset = dataset_with_samples(3, 4);
    let losses = trainer.train(&dataset, &per_sample);
    assert!(losses.is_empty());
}
