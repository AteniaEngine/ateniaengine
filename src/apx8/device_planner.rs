// APX 8.18 — GPU Device Planner v0
// Simulated GPU device planner, without touching real hardware nor execution.

#[derive(Debug, Clone)]
pub struct SimulatedGPU {
    pub name: String,
    pub compute_units: u32,
    pub memory_gb: u32,
}

#[derive(Debug, Clone)]
pub struct DevicePlan {
    pub target_gpu: Option<SimulatedGPU>,
    pub split_hint: Option<u32>,
}

pub fn detect_simulated_gpus() -> Vec<SimulatedGPU> {
    vec![
        SimulatedGPU {
            name: "FakeCUDA_4090".into(),
            compute_units: 128,
            memory_gb: 24,
        },
        SimulatedGPU {
            name: "FakeAMD_7900".into(),
            compute_units: 96,
            memory_gb: 20,
        },
    ]
}

/// Given an IR name, suggest which GPU to send it to and how to split the work.
/// 100% simulated.
pub fn plan_for_ir(ir_name: &str) -> DevicePlan {
    let gpus = detect_simulated_gpus();

    if ir_name.contains("cuda") {
        DevicePlan {
            target_gpu: Some(gpus[0].clone()),
            split_hint: Some(2),
        }
    } else if ir_name.contains("hip") {
        DevicePlan {
            target_gpu: Some(gpus[1].clone()),
            split_hint: Some(4),
        }
    } else {
        DevicePlan {
            target_gpu: None,
            split_hint: None,
        }
    }
}
