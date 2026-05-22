"""Download the missing Phi-family checkpoints for the mastery phase."""
import os
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = r"F:\Proyectos\artenia_engine\atenia-engine\models"

# (repo, subdir, allow_patterns)  — safetensors snapshots
SNAPSHOTS = [
    ("microsoft/Phi-3-mini-4k-instruct", "phi-3-mini-4k-instruct"),
    ("microsoft/Phi-3-mini-128k-instruct", "phi-3-mini-128k-instruct"),
    ("microsoft/Phi-4-mini-instruct", "phi-4-mini-instruct"),
]

# (repo, filename, subdir, tokenizer_repo)  — GGUF single files
GGUF = [
    ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q5_K_M.gguf",
     "Phi-3.5-mini-instruct-Q5_K_M-GGUF", "microsoft/Phi-3.5-mini-instruct"),
    ("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q6_K.gguf",
     "Phi-3.5-mini-instruct-Q6_K-GGUF", "microsoft/Phi-3.5-mini-instruct"),
    ("bartowski/microsoft_Phi-4-mini-instruct-GGUF", "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf",
     "Phi-4-mini-instruct-Q4_K_M-GGUF", "microsoft/Phi-4-mini-instruct"),
]

ST_PATTERNS = ["*.safetensors", "*.json", "*.py", "tokenizer*", "*.model"]


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
