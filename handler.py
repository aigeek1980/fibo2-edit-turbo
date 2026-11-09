"""
RunPod Serverless Handler for Fibo 2.0 Edit Turbo (ComfyUI)

Workflow: Fibo 2.0 image editing using BRIA's Fibo-2.0 model.
Models required (auto-downloaded at startup):
  - diffusion_models/fibo2_edit_turbo_bf16.safetensors  (~16 GB)
  - text_encoders/qwen3vl_4b_bf16.safetensors            (~8 GB)
  - vae/flux2-vae.safetensors                            (~335 MB)

API input schema:
  {
    "prompt":  str,          # Required. Describe the edit in final-state language.
    "images":  [str, ...],   # Required. 1 or 2 base64-encoded PNG/JPG images.
    "seed":    int,          # Optional. -1 = random. Default: -1
    "steps":   int,          # Optional. Default: 4  (turbo mode)
    "cfg":     float,        # Optional. Default: 1.0 (turbo mode)
    "denoise": float,        # Optional. Default: 1.0
    "width":   int,          # Optional. 0 = auto (recommended). Default: 0
    "height":  int           # Optional. 0 = auto (recommended). Default: 0
  }

API output schema:
  {
    "images":    [str, ...],  # Base64-encoded output PNG(s)
    "prompt_id": str,
    "seed":      int
  }
"""

import runpod
import requests
import json
import time
import base64
import uuid
import os
import io
import random
import subprocess
import threading

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_URL  = f"http://{COMFY_HOST}:{COMFY_PORT}"
WORKFLOW_PATH = "/workflow_api.json"

# ──────────────────────────────────────────────
# ComfyUI lifecycle
# ──────────────────────────────────────────────

def start_comfyui():
    """Launch ComfyUI as a background process."""
    model_dir = os.environ.get("COMFY_MODEL_DIR", "/ComfyUI/models")
    cmd = [
        "python", "/ComfyUI/main.py",
        "--listen", "0.0.0.0",
        "--port", str(COMFY_PORT),
        "--disable-auto-launch",
        "--models-dir", model_dir,
    ]
    print(f"[startup] Launching ComfyUI (models: {model_dir})")
    subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def wait_for_comfyui(timeout: int = 180):
    """Block until ComfyUI's /system_stats responds or timeout is reached."""
    print("[startup] Waiting for ComfyUI to be ready...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
            if r.status_code == 200:
                print("[startup] ComfyUI is ready.")
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"ComfyUI did not start within {timeout}s")


# ──────────────────────────────────────────────
# ComfyUI helpers
# ──────────────────────────────────────────────

def upload_image(image_b64: str, filename: str) -> str:
    """Upload a base64-encoded image to ComfyUI's input folder.

    Returns the filename ComfyUI assigned (may differ from requested name
    if a collision occurred, though overwrite=true prevents that).
    """
    image_bytes = base64.b64decode(image_b64)
    files = {
        "image":     (filename, io.BytesIO(image_bytes), "image/png"),
        "overwrite": (None, "true"),
    }
    r = requests.post(f"{COMFY_URL}/upload/image", files=files)
    r.raise_for_status()
    return r.json()["name"]


def build_prompt(
    prompt_text: str,
    image_filenames: list,
    seed: int,
    steps: int,
    cfg: float,
    denoise: float,
    width: int,
    height: int,
) -> dict:
    """Deep-copy the template workflow and patch in the job-specific values."""
    with open(WORKFLOW_PATH) as f:
        wf = json.load(f)

    # Text prompt → TextEncodeFibo2 node (id "4")
    wf["4"]["inputs"]["prompt"] = prompt_text

    # Dimensions → TextEncodeFibo2 node (0 = let model pick bucket)
    wf["4"]["inputs"]["width"]  = width
    wf["4"]["inputs"]["height"] = height

    # KSampler node (id "5")
    wf["5"]["inputs"]["seed"]    = seed
    wf["5"]["inputs"]["steps"]   = steps
    wf["5"]["inputs"]["cfg"]     = cfg
    wf["5"]["inputs"]["denoise"] = denoise

    # Reference images
    # Node 39 is used for image_1 and image_3; node 40 is used for image_2.
    img1 = image_filenames[0]
    img2 = image_filenames[1] if len(image_filenames) >= 2 else image_filenames[0]
    wf["39"]["inputs"]["image"] = img1
    wf["40"]["inputs"]["image"] = img2

    return wf


def queue_prompt(workflow: dict) -> tuple[str, str]:
    """POST the workflow to /prompt and return (prompt_id, client_id)."""
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}
    r = requests.post(f"{COMFY_URL}/prompt", json=payload)
    r.raise_for_status()
    data = r.json()

    if "error" in data:
        raise RuntimeError(f"ComfyUI rejected the prompt: {data['error']}\n{data.get('node_errors', '')}")

    return data["prompt_id"], client_id


def wait_for_job(prompt_id: str, timeout: int = 300) -> None:
    """Poll /history until the job appears (success or error)."""
    print(f"[job] Waiting for prompt_id={prompt_id}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10)
            r.raise_for_status()
            history = r.json()
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                if status.get("status_str") == "error" or status.get("completed") is False:
                    msgs = status.get("messages", [])
                    raise RuntimeError(f"ComfyUI execution failed: {msgs}")
                # Completed successfully
                return
        except RuntimeError:
            raise
        except Exception as e:
            print(f"[job] History poll error (retrying): {e}")
        time.sleep(2)
    raise TimeoutError(f"Job {prompt_id} timed out after {timeout}s")


def collect_output_images(prompt_id: str) -> list:
    """Fetch all output images from /history and return as base64 strings."""
    r = requests.get(f"{COMFY_URL}/history/{prompt_id}")
    r.raise_for_status()
    history = r.json()

    if prompt_id not in history:
        raise RuntimeError(f"No history entry for prompt_id={prompt_id}")

    outputs = history[prompt_id].get("outputs", {})
    images_b64 = []

    for _node_id, node_out in outputs.items():
        for img_info in node_out.get("images", []):
            img_r = requests.get(
                f"{COMFY_URL}/view",
                params={
                    "filename": img_info["filename"],
                    "subfolder": img_info.get("subfolder", ""),
                    "type":     img_info.get("type", "output"),
                },
                timeout=60,
            )
            img_r.raise_for_status()
            images_b64.append(base64.b64encode(img_r.content).decode("utf-8"))

    if not images_b64:
        raise RuntimeError("No output images found in ComfyUI history")

    return images_b64


# ──────────────────────────────────────────────
# RunPod handler
# ──────────────────────────────────────────────

def handler(job: dict) -> dict:
    job_input = job.get("input", {})
    job_id    = job.get("id", "unknown")

    # ── Validate inputs ──
    prompt_text = job_input.get("prompt", "").strip()
    if not prompt_text:
        return {"error": "'prompt' is required and must not be empty."}

    images_b64 = job_input.get("images", [])
    if not images_b64 or not isinstance(images_b64, list):
        return {"error": "'images' must be a non-empty list of base64-encoded image strings."}

    seed    = int(job_input.get("seed", -1))
    steps   = int(job_input.get("steps", 4))
    cfg     = float(job_input.get("cfg", 1.0))
    denoise = float(job_input.get("denoise", 1.0))
    width   = int(job_input.get("width", 0))
    height  = int(job_input.get("height", 0))

    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    print(f"[job:{job_id}] prompt={prompt_text!r}, images={len(images_b64)}, seed={seed}, steps={steps}")

    # ── Upload reference images ──
    image_filenames = []
    for idx, img_b64 in enumerate(images_b64[:2]):   # max 2 images
        fname = f"job_{job_id}_img{idx + 1}.png"
        try:
            uploaded = upload_image(img_b64, fname)
            image_filenames.append(uploaded)
            print(f"[job:{job_id}] Uploaded image {idx + 1} → {uploaded}")
        except Exception as e:
            return {"error": f"Failed to upload image {idx + 1}: {e}"}

    # ── Build & queue the workflow ──
    try:
        workflow  = build_prompt(prompt_text, image_filenames, seed, steps, cfg, denoise, width, height)
        prompt_id, client_id = queue_prompt(workflow)
        print(f"[job:{job_id}] Queued → prompt_id={prompt_id}")
    except Exception as e:
        return {"error": f"Failed to queue workflow: {e}"}

    # ── Wait for completion ──
    try:
        wait_for_job(prompt_id)
    except (TimeoutError, RuntimeError) as e:
        return {"error": str(e)}

    # ── Collect & return output images ──
    try:
        output_images = collect_output_images(prompt_id)
    except Exception as e:
        return {"error": f"Failed to retrieve output: {e}"}

    print(f"[job:{job_id}] Done — {len(output_images)} image(s) returned.")
    return {
        "images":    output_images,
        "prompt_id": prompt_id,
        "seed":      seed,
    }


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    start_comfyui()
    wait_for_comfyui()
    print("[startup] Starting RunPod serverless worker...")
    runpod.serverless.start({"handler": handler})
