use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::v13::self_trainer::{
    BackendChoice, ChoiceStats, ContextBucket, SelfTrainer,
};

#[derive(Debug, Clone)]
pub enum PersistError {
    Io(String),
    Format(String),
}

const HEADER: &str = "ATENIA_SELFTRAINER_V1";

pub fn save_trainer_to_path(trainer: &SelfTrainer, path: &Path) -> Result<(), PersistError> {
    let parent_opt = path.parent();
    if let Some(parent) = parent_opt {
        if let Err(e) = fs::create_dir_all(parent) {
            return Err(PersistError::Io(e.to_string()));
        }
    }

    let tmp_path = path.with_extension("tmp");

    // Best-effort cleanup of any previous temp file.
    let _ = fs::remove_file(&tmp_path);

    let file = match fs::File::create(&tmp_path) {
        Ok(f) => f,
        Err(e) => return Err(PersistError::Io(e.to_string())),
    };

    let mut writer = std::io::BufWriter::new(file);

    if let Err(e) = writeln!(writer, "{}", HEADER) {
        return Err(PersistError::Io(e.to_string()));
    }

    let mut entries = trainer.all_stats();

    entries.sort_by(|(a_bucket, a_backend, _), (b_bucket, b_backend, _)| {
        let ag = if a_bucket.gpu_available { 1u8 } else { 0u8 };
        let bg = if b_bucket.gpu_available { 1u8 } else { 0u8 };
        ag.cmp(&bg)
            .then_with(|| a_bucket.vram_band.cmp(&b_bucket.vram_band))
            .then_with(|| a_bucket.ram_band.cmp(&b_bucket.ram_band))
            .then_with(|| a_backend.cmp(b_backend))
    });

    for (bucket, backend, stats) in entries {
        let backend_str = match backend {
            BackendChoice::Cpu => "cpu",
            BackendChoice::Gpu => "gpu",
        };

        if let Err(e) = writeln!(
            writer,
            "gpu_avail={};vram_band={};ram_band={};backend={};count={};success={};score_sum={};drift={}",
            if bucket.gpu_available { 1 } else { 0 },
            bucket.vram_band,
            bucket.ram_band,
            backend_str,
            stats.count,
            stats.success_count,
            stats.score_sum,
            stats.drift_count,
        ) {
            return Err(PersistError::Io(e.to_string()));
        }
    }

    if let Err(e) = writer.flush() {
        return Err(PersistError::Io(e.to_string()));
    }

    if let Err(e) = fs::rename(&tmp_path, path) {
        return Err(PersistError::Io(e.to_string()));
    }

    Ok(())
}

pub fn load_trainer_from_path(path: &Path) -> Result<SelfTrainer, PersistError> {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => {
            // If the file does not exist or cannot be opened, return an empty trainer.
            return Ok(SelfTrainer::new());
        }
    };

    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let header_line = match lines.next() {
        Some(Ok(line)) => line,
        Some(Err(_)) => return Ok(SelfTrainer::new()),
        None => return Ok(SelfTrainer::new()),
    };

    if header_line.trim() != HEADER {
        // Unknown format: return empty trainer, non-fatal.
        return Ok(SelfTrainer::new());
    }

    let mut trainer = SelfTrainer::new();

    for line_result in lines {
        let line = match line_result {
            Ok(l) => l,
            Err(_) => continue,
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut gpu_avail_opt: Option<bool> = None;
        let mut vram_band_opt: Option<u8> = None;
        let mut ram_band_opt: Option<u8> = None;
        let mut backend_opt: Option<BackendChoice> = None;
        let mut count_opt: Option<u32> = None;
        let mut success_opt: Option<u32> = None;
        let mut score_sum_opt: Option<i64> = None;
        let mut drift_opt: Option<u32> = None;

        for part in trimmed.split(';') {
            let mut it = part.splitn(2, '=');
            let key = match it.next() {
                Some(k) => k.trim(),
                None => continue,
            };
            let val = match it.next() {
                Some(v) => v.trim(),
                None => continue,
            };

            match key {
                "gpu_avail" => {
                    if val == "1" {
                        gpu_avail_opt = Some(true);
                    } else if val == "0" {
                        gpu_avail_opt = Some(false);
                    }
                }
                "vram_band" => {
                    if let Ok(b) = val.parse::<u8>() {
                        vram_band_opt = Some(b);
                    }
                }
                "ram_band" => {
                    if let Ok(b) = val.parse::<u8>() {
                        ram_band_opt = Some(b);
                    }
                }
                "backend" => {
                    if val.eq_ignore_ascii_case("cpu") {
                        backend_opt = Some(BackendChoice::Cpu);
                    } else if val.eq_ignore_ascii_case("gpu") {
                        backend_opt = Some(BackendChoice::Gpu);
                    }
                }
                "count" => {
                    if let Ok(v) = val.parse::<u32>() {
                        count_opt = Some(v);
                    }
                }
                "success" => {
                    if let Ok(v) = val.parse::<u32>() {
                        success_opt = Some(v);
                    }
                }
                "score_sum" => {
                    if let Ok(v) = val.parse::<i64>() {
                        score_sum_opt = Some(v);
                    }
                }
                "drift" => {
                    if let Ok(v) = val.parse::<u32>() {
                        drift_opt = Some(v);
                    }
                }
                _ => {
                    // Unknown key: ignore.
                }
            }
        }

        let gpu_available = match gpu_avail_opt {
            Some(v) => v,
            None => continue,
        };
        let vram_band = match vram_band_opt {
            Some(v) => v,
            None => continue,
        };
        let ram_band = match ram_band_opt {
            Some(v) => v,
            None => continue,
        };
        let backend = match backend_opt {
            Some(b) => b,
            None => continue,
        };
        let count = match count_opt {
            Some(c) => c,
            None => continue,
        };
        let success = match success_opt {
            Some(s) => s,
            None => continue,
        };
        let score_sum = match score_sum_opt {
            Some(s) => s,
            None => continue,
        };
        let drift = match drift_opt {
            Some(d) => d,
            None => continue,
        };

        let bucket = ContextBucket {
            gpu_available,
            vram_band,
            ram_band,
        };

        let stats = ChoiceStats {
            count,
            success_count: success,
            score_sum,
            drift_count: drift,
        };

        trainer.set_stats_entry(bucket, backend, stats);
    }

    Ok(trainer)
}
