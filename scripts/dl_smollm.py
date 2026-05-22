"""Download the missing SmolLM / SmolLM2 checkpoints for the mastery phase."""
import os
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = r"F:\Proyectos\artenia_engine\atenia-engine\models"

SNAPSHOTS = [
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "SmolLM2-135M-Instruct"),
    ("HuggingFaceTB/SmolLM2-360M-Instruct", "SmolLM2-360M-Instruct"),
    ("HuggingFaceTB/SmolLM-135M-Instruct", "SmolLM-135M-Instruct"),
    ("HuggingFaceTB/SmolLM-360M-Instruct", "SmolLM-360M-Instruct"),
    ("HuggingFaceTB/SmolLM-1.7B-Instruct", "SmolLM-1.7B-Instruct"),
]

GGUF = [
    ("bartowski/SmolLM2-1.7B-Instruct-GGUF", "SmolLM2-1.7B-Instruct-Q5_K_M.gguf",
     "SmolLM2-1.7B-Instruct-Q5_K_M-GGUF", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("bartowski/SmolLM2-1.7B-Instruct-GGUF", "SmolLM2-1.7B-Instruct-Q6_K.gguf",
     "SmolLM2-1.7B-Instruct-Q6_K-GGUF", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
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
        if os.path.exists(target) and os.path.getsize(target) > 50 * 1024 * 1024:
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
