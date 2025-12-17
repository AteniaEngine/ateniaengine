use std::process::Command;

pub fn gpu_memory_mb() -> (u32, u32) {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    if let Ok(out) = output {
        if let Ok(text) = String::from_utf8(out.stdout) {
            let parts: Vec<&str> = text.trim().split(',').collect();
            if parts.len() >= 2 {
                let used = parts[0].trim().parse().unwrap_or(0);
                let total = parts[1].trim().parse().unwrap_or(0);
                return (used, total);
            }
        }
    }

    (0, 0)
}

pub fn gpu_utilization() -> u32 {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();

    if let Ok(out) = output {
        if let Ok(text) = String::from_utf8(out.stdout) {
            return text.trim().parse().unwrap_or(0);
        }
    }

    0
}
