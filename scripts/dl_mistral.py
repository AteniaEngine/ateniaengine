"""Download the missing Mistral-dense checkpoints for the mastery phase."""
import os
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = r"F:\Proyectos\artenia_engine\atenia-engine\models"

# (repo, subdir) safetensors snapshots — mistralai repos are gated.
SNAPSHOTS = [
    ("mistralai/Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.3"),
    ("mistralai/Mistral-7B-Instruct-v0.2", "Mistral-7B-Instruct-v0.2"),
]

# (repo, filename, subdir, tokenizer_repo) GGUF single files.
GGUF = [
    ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q5_K_M.gguf",
     "Mistral-7B-Instruct-v0.3-Q5_K_M-GGUF", "bartowski/Mistral-7B-Instruct-v0.3-GGUF"),
    ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q6_K.gguf",
     "Mistral-7B-Instruct-v0.3-Q6_K-GGUF", "bartowski/Mistral-7B-Instruct-v0.3-GGUF"),
    ("bartowski/Mistral-7B-Instruct-v0.2-GGUF", "Mistral-7B-Instruct-v0.2-Q4_K_M.gguf",
     "Mistral-7B-Instruct-v0.2-Q4_K_M-GGUF", "bartowski/Mistral-7B-Instruct-v0.2-GGUF"),
]

ST_PATTERNS = ["*.safetensors", "*.json", "tokenizer*", "*.model"]


def main():
    for repo, sub in SNAPSHOTS:
        dst = os.path.join(ROOT, sub)
        if os.path.isdir(dst) and any(f.endswith(".safetensors") for f in os.listdir(dst)):
            print(f"[skip] {sub} present")
            continue
        print(f"[snap] {repo} -> {sub}")
        try:
            snapshot_download(repo_id=repo, local_dir=dst, allow_patterns=ST_PATTERNS)
            print(f"       ok: {sub}")
        except Exception as e:
            print(f"       FAIL {sub}: {type(e).__name__}: {e}")

    for repo, fn, sub, tok_repo in GGUF:
        dst = os.path.join(ROOT, sub)
        os.makedirs(dst, exist_ok=True)
        target = os.path.join(dst, fn)
        if os.path.exists(target) and os.path.getsize(target) > 100 * 1024 * 1024:
            print(f"[skip] {sub}/{fn} present")
        else:
            print(f"[gguf] {repo} :: {fn} -> {sub}")
            try:
                hf_hub_download(repo_id=repo, filename=fn, local_dir=dst)
                print(f"       ok: {fn}")
            except Exception as e:
                print(f"       FAIL {fn}: {type(e).__name__}: {e}")
        for extra in ["tokenizer.json", "tokenizer_config.json"]:
            ep = os.path.join(dst, extra)
            if os.path.exists(ep):
                continue
            try:
                hf_hub_download(repo_id=tok_repo, filename=extra, local_dir=dst)
                print(f"       +tok {extra}")
            except Exception as e:
                print(f"       tok-skip {extra}: {type(e).__name__}")


if __name__ == "__main__":
    main()
