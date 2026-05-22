"""Download the missing Falcon / Falcon3 checkpoints for the mastery phase."""
import os
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = r"F:\Proyectos\artenia_engine\atenia-engine\models"

# (repo, subdir) safetensors snapshots.
SNAPSHOTS = [
    ("tiiuae/Falcon3-1B-Instruct", "Falcon3-1B-Instruct"),
    ("tiiuae/Falcon3-3B-Instruct", "Falcon3-3B-Instruct"),
    ("tiiuae/falcon-7b-instruct", "falcon-7b-instruct"),
]

# (repo, filename, subdir, tokenizer_repo) GGUF single files.
GGUF = [
    ("bartowski/Falcon3-7B-Instruct-GGUF", "Falcon3-7B-Instruct-Q4_K_M.gguf",
     "Falcon3-7B-Instruct-Q4_K_M-GGUF", "tiiuae/Falcon3-7B-Instruct"),
    ("bartowski/Falcon3-7B-Instruct-GGUF", "Falcon3-7B-Instruct-Q5_K_M.gguf",
     "Falcon3-7B-Instruct-Q5_K_M-GGUF", "tiiuae/Falcon3-7B-Instruct"),
    ("bartowski/Falcon3-7B-Instruct-GGUF", "Falcon3-7B-Instruct-Q6_K.gguf",
     "Falcon3-7B-Instruct-Q6_K-GGUF", "tiiuae/Falcon3-7B-Instruct"),
    ("maddes8cht/tiiuae-falcon-7b-instruct-gguf", "tiiuae-falcon-7b-instruct-Q4_K_M.gguf",
     "falcon-7b-instruct-Q4_K_M-GGUF", "tiiuae/falcon-7b-instruct"),
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
