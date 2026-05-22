"""Download the missing Llama-family GGUF quants + DeepSeek distill GGUF.

Run from anywhere; uses huggingface_hub directly so it works without the CLI.
"""
import os
import sys
from huggingface_hub import hf_hub_download

ROOT = r"F:\Proyectos\artenia_engine\atenia-engine\models"

JOBS = [
    # (repo_id, filename, local_dir, also_files)
    (
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "Llama-3.1-8B-Instruct-Q4_K_M-GGUF",
        ["tokenizer.json", "tokenizer_config.json"],
    ),
    (
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        "Llama-3.1-8B-Instruct-Q5_K_M-GGUF",
        [],
    ),
    (
        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
        "Llama-3.1-8B-Instruct-Q6_K-GGUF",
        [],
    ),
    (
        "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
        "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
        "DeepSeek-R1-Distill-Llama-8B-Q4_K_M-GGUF",
        ["tokenizer.json", "tokenizer_config.json"],
    ),
]


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for repo, fn, sub, extras in JOBS:
        if only and only not in sub:
            continue
        dst = os.path.join(ROOT, sub)
        os.makedirs(dst, exist_ok=True)
        target = os.path.join(dst, fn)
        if os.path.exists(target) and os.path.getsize(target) > 100 * 1024 * 1024:
            print(f"[skip] {sub}/{fn} already present ({os.path.getsize(target)/1e9:.2f} GB)")
        else:
            print(f"[get ] {repo} :: {fn} -> {sub}")
            try:
                p = hf_hub_download(repo_id=repo, filename=fn, local_dir=dst)
                print(f"       ok: {p}")
            except Exception as e:
                print(f"       FAIL: {e}")
        for extra in extras:
            ep = os.path.join(dst, extra)
            if os.path.exists(ep):
                continue
            try:
                p = hf_hub_download(repo_id=repo, filename=extra, local_dir=dst)
                print(f"       +extra: {extra}")
            except Exception as e:
                print(f"       extra-skip {extra}: {e}")


if __name__ == "__main__":
    main()
