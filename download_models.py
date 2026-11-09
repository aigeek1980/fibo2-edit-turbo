"""
Download all models required by the Fibo 2.0 Edit Turbo workflow.

Skips any file that already exists (idempotent — safe to call on every startup).
If HF_TOKEN is set in the environment it is forwarded to huggingface_hub,
which is required if the repo is gated.

Usage:
    python download_models.py [--model-dir /path/to/ComfyUI/models]
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("[download] huggingface_hub not installed — running: pip install huggingface_hub")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "--quiet"])
    from huggingface_hub import hf_hub_download

# ──────────────────────────────────────────────
# Model manifest
# Each entry:
#   repo_id  - HuggingFace repo
#   filename - path inside the repo
#   dest     - path relative to <model_dir>
# ──────────────────────────────────────────────
MODELS = [
    {
        "repo_id":  "Comfy-Org/Fibo-2.0",
        "filename": "diffusion_models/fibo2_edit_turbo_bf16.safetensors",
        "dest":     "diffusion_models/fibo2_edit_turbo_bf16.safetensors",
        "size_hint": "~16 GB",
    },
    {
        "repo_id":  "Comfy-Org/Fibo-2.0",
        "filename": "text_encoders/qwen3vl_4b_bf16.safetensors",
        "dest":     "text_encoders/qwen3vl_4b_bf16.safetensors",
        "size_hint": "~8 GB",
    },
    {
        "repo_id":  "Comfy-Org/Fibo-2.0",
        "filename": "vae/flux2-vae.safetensors",
        "dest":     "vae/flux2-vae.safetensors",
        "size_hint": "~335 MB",
    },
]

# Optional: int8 diffusion model (smaller, slightly lower quality)
OPTIONAL_MODELS = [
    {
        "repo_id":  "Comfy-Org/Fibo-2.0",
        "filename": "diffusion_models/fibo2_edit_turbo_int8_convrot.safetensors",
        "dest":     "diffusion_models/fibo2_edit_turbo_int8_convrot.safetensors",
        "size_hint": "~8 GB",
    },
]


def download_models(model_dir: str, include_optional: bool = False) -> None:
    model_dir = Path(model_dir)
    token = os.environ.get("HF_TOKEN")
    manifest = MODELS + (OPTIONAL_MODELS if include_optional else [])

    total = len(manifest)
    for i, m in enumerate(manifest, 1):
        dest_path = model_dir / m["dest"]
        label = f"[{i}/{total}] {m['dest']} ({m['size_hint']})"

        if dest_path.exists():
            print(f"[skip]     {label} — already present")
            continue

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[download] {label}")
        print(f"           from {m['repo_id']}/{m['filename']}")

        try:
            downloaded = hf_hub_download(
                repo_id=m["repo_id"],
                filename=m["filename"],
                local_dir=str(model_dir),
                token=token,
            )
            print(f"[done]     → {downloaded}")
        except Exception as e:
            print(f"[ERROR]    Failed to download {m['dest']}: {e}", file=sys.stderr)
            sys.exit(1)

    print("\n[download] All models are present.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Fibo 2.0 models from HuggingFace")
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("COMFY_MODEL_DIR", "/ComfyUI/models"),
        help="Root ComfyUI models directory (default: /ComfyUI/models)",
    )
    parser.add_argument(
        "--optional",
        action="store_true",
        help="Also download optional int8 diffusion model variant",
    )
    args = parser.parse_args()

    print(f"[download] Model directory: {args.model_dir}")
    download_models(args.model_dir, include_optional=args.optional)
