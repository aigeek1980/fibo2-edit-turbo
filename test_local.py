"""
test_local.py — Test the RunPod handler locally (without RunPod infrastructure).

Usage:
    # 1. Start ComfyUI manually on port 8188
    # 2. Run this script:
    python test_local.py --image path/to/photo.jpg --prompt "Turn the coat into black leather"

The script calls handler() directly, bypassing runpod.serverless.start().
It saves the output image(s) to ./output_<n>.png.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

# ── Minimal stub so handler.py imports without the runpod package ──
import types, importlib
if "runpod" not in sys.modules:
    runpod_stub = types.ModuleType("runpod")
    runpod_stub.serverless = types.SimpleNamespace(start=lambda cfg: None)
    sys.modules["runpod"] = runpod_stub

# Import after stub so we can call handler() directly
sys.path.insert(0, "/")          # handler.py lives at /handler.py in Docker
# When running locally, adjust this path:
import importlib.util, os
_handler_path = os.path.join(os.path.dirname(__file__), "handler.py")
spec = importlib.util.spec_from_file_location("handler", _handler_path)
handler_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(handler_mod)
handler = handler_mod.handler


def image_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",   required=True,  help="Path to reference image (1st)")
    parser.add_argument("--image2",  default=None,   help="Path to 2nd reference image (optional)")
    parser.add_argument("--prompt",  required=True,  help="Edit prompt")
    parser.add_argument("--seed",    type=int, default=-1)
    parser.add_argument("--steps",   type=int, default=4)
    parser.add_argument("--cfg",     type=float, default=1.0)
    parser.add_argument("--denoise", type=float, default=1.0)
    args = parser.parse_args()

    images = [image_to_b64(args.image)]
    if args.image2:
        images.append(image_to_b64(args.image2))

    job = {
        "id": "local-test",
        "input": {
            "prompt":  args.prompt,
            "images":  images,
            "seed":    args.seed,
            "steps":   args.steps,
            "cfg":     args.cfg,
            "denoise": args.denoise,
        }
    }

    print(f"Sending job: prompt={args.prompt!r}, images={len(images)}")
    result = handler(job)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    print(f"Success — seed={result['seed']}, prompt_id={result['prompt_id']}")
    for i, img_b64 in enumerate(result["images"]):
        out_path = f"output_{i + 1}.png"
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(img_b64))
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
