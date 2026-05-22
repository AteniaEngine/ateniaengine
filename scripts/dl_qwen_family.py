"""Download the missing Qwen-family checkpoints for the mastery battery.

Safetensors models -> snapshot_download (config + tokenizer + shards).
GGUF single-file models -> hf_hub_download, plus tokenizer side-files
pulled from the source (non-GGUF) repo since bartowski GGUF repos do
not ship tokenizer.json / tokenizer_config.json.
"""
import os
import sys
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = r"F:\Proyectos\artenia_engine\atenia-engine\models"

# (repo_id, local_subdir, allow_patterns)
SAFETENSORS_JOBS = [
    (
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen2.5-3B-Instruct",
        ["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"],
    ),
    (
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen2.5-7B-Instruct",
        ["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"],
    ),
    (
        "Qwen/Qwen3-4B",
        "Qwen3-4B",
        ["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json"],
    ),
]

# (gguf_repo, gguf_filename, local_subdir, tokenizer_repo)
GGUF_JOBS = [
    (
        "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "Qwen2.5-7B-Instruct-Q4_K_M-GGUF",
        "Qwen/Qwen2.5-7B-Instruct",
    ),
    (
        "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "Qwen2.5-7B-Instruct-Q5_K_M.gguf",
        "Qwen2.5-7B-Instruct-Q5_K_M-GGUF",
        "Qwen/Qwen2.5-7B-Instruct",
    ),
    (
        "bartowski/Qwen2.5-7B-Instruct-GGUF",
        "Qwen2.5-7B-Instruct-Q6_K.gguf",
        "Qwen2.5-7B-Instruct-Q6_K-GGUF",
        "Qwen/Qwen2.5-7B-Instruct",
    ),
    (
        "bartowski/Qwen_Qwen3-8B-GGUF",
        "Qwen_Qwen3-8B-Q4_K_M.gguf",
        "Qwen3-8B-Q4_K_M-GGUF",
        "Qwen/Qwen3-8B",
    ),
    (
        "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
        "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
        "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M-GGUF",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    ),
]

TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]


def big(path):
    return os.path.exists(path) and os.path.getsize(path) > 100 * 1024 * 1024


def main():
    for repo, sub, patterns in SAFETENSORS_JOBS:
        dst = os.path.join(ROOT, sub)
        idx = os.path.join(dst, "model.safetensors.index.json")
        single = os.path.join(dst, "model.safetensors")
        if os.path.exists(idx) or big(single):
            print(f"[skip] {sub} already present")
            continue
        print(f"[snap] {repo} -> {sub}")
        try:
            snapshot_download(repo_id=repo, local_dir=dst, allow_patterns=patterns)
            print(f"       ok: {sub}")
        except Exception as e:
            print(f"       FAIL {sub}: {e}")

    for gguf_repo, fn, sub, tok_repo in GGUF_JOBS:
        dst = os.path.join(ROOT, sub)
        os.makedirs(dst, exist_ok=True)
        target = os.path.join(dst, fn)
        if big(target):
            print(f"[skip] {sub}/{fn} present ({os.path.getsize(target)/1e9:.2f} GB)")
        else:
            print(f"[gguf] {gguf_repo} :: {fn} -> {sub}")
            try:
                p = hf_hub_download(repo_id=gguf_repo, filename=fn, local_dir=dst)
                print(f"       ok: {p}")
            except Exception as e:
                print(f"       FAIL {sub}: {e}")
        for tf in TOKENIZER_FILES:
            tp = os.path.join(dst, tf)
            if os.path.exists(tp):
                continue
            try:
                hf_hub_download(repo_id=tok_repo, filename=tf, local_dir=dst)
                print(f"       +tok {tf} from {tok_repo}")
            except Exception as e:
                print(f"       tok-skip {tf}: {type(e).__name__}")


if __name__ == "__main__":
    main()
