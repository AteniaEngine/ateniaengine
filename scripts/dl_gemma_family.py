"""Download the missing Gemma-family checkpoints for the mastery battery.

Safetensors from the gated google/* repos (token already has Gemma
license). GGUF single-file from ungated mirrors, with tokenizer
side-files pulled from the matching google repo.
"""
import os
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = r"F:\Proyectos\artenia_engine\atenia-engine\models"

SAFETENSORS_JOBS = [
    ("google/gemma-2-9b-it", "gemma-2-9b-it",
     ["*.json", "*.safetensors", "*.model", "tokenizer*"]),
    ("google/gemma-3-1b-it", "gemma-3-1b-it",
     ["*.json", "*.safetensors", "*.model", "tokenizer*"]),
    ("google/gemma-3-4b-it", "gemma-3-4b-it",
     ["*.json", "*.safetensors", "*.model", "tokenizer*"]),
]

# (gguf_repo, gguf_filename, local_subdir, tokenizer_repo)
GGUF_JOBS = [
    ("bartowski/gemma-2-9b-it-GGUF", "gemma-2-9b-it-Q4_K_M.gguf",
     "gemma-2-9b-it-Q4_K_M-GGUF", "google/gemma-2-9b-it"),
    ("bartowski/gemma-2-9b-it-GGUF", "gemma-2-9b-it-Q5_K_M.gguf",
     "gemma-2-9b-it-Q5_K_M-GGUF", "google/gemma-2-9b-it"),
    ("bartowski/gemma-2-9b-it-GGUF", "gemma-2-9b-it-Q6_K.gguf",
     "gemma-2-9b-it-Q6_K-GGUF", "google/gemma-2-9b-it"),
    ("ggml-org/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf",
     "gemma-3-1b-it-Q4_K_M-GGUF", "google/gemma-3-1b-it"),
    ("ggml-org/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf",
     "gemma-3-4b-it-Q4_K_M-GGUF", "google/gemma-3-4b-it"),
]

TOK_FILES = ["tokenizer.json", "tokenizer_config.json"]


def big(p):
    return os.path.exists(p) and os.path.getsize(p) > 50 * 1024 * 1024


def main():
    for repo, sub, patterns in SAFETENSORS_JOBS:
        dst = os.path.join(ROOT, sub)
        idx = os.path.join(dst, "model.safetensors.index.json")
        single = os.path.join(dst, "model.safetensors")
        if os.path.exists(idx) or big(single):
            print(f"[skip] {sub} present")
            continue
        print(f"[snap] {repo} -> {sub}")
        try:
            snapshot_download(repo_id=repo, local_dir=dst, allow_patterns=patterns)
            print(f"       ok: {sub}")
        except Exception as e:
            print(f"       FAIL {sub}: {type(e).__name__}: {str(e)[:160]}")

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
                print(f"       FAIL {sub}: {type(e).__name__}: {str(e)[:160]}")
        for tf in TOK_FILES:
            tp = os.path.join(dst, tf)
            if os.path.exists(tp):
                continue
            try:
                hf_hub_download(repo_id=tok_repo, filename=tf, local_dir=dst)
                print(f"       +tok {tf}")
            except Exception as e:
                print(f"       tok-skip {tf}: {type(e).__name__}")


if __name__ == "__main__":
    main()
