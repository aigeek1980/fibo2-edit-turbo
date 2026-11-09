"""
RunPod Serverless Handler — Fibo 2.0 Edit Turbo (ComfyUI)

Workflow: Fibo 2.0 image editing / generation using BRIA's Fibo-2.0 model.

API input schema:
  {
    "prompt":    str,         # Required. Describe the edit in final-state language.
    "image":     str,         # Required. Base64-encoded image (PNG or JPG).
    "image2":    str,         # Optional. Second reference image (base64). Defaults to image.
    "image_url": str,         # Alternative to "image" — URL to fetch image from.
    "seed":      int,         # Optional. -1 = random. Default: -1
    "steps":     int,         # Optional. Default: 4  (turbo mode)
    "cfg":       float,       # Optional. Default: 1.0 (turbo mode — keep at 1.0)
    "denoise":   float        # Optional. Default: 1.0
  }

API output schema:
  { "image": str }            # Base64-encoded output PNG

Fibo 2.0 prompting tips:
  - Describe the FINAL state of the edited region only.
  - Do NOT write "keep X unchanged" — this causes model drift.
  - Stay near 1 MP output (width/height = 0 for auto-bucket).
  - Negative prompt left empty — that is Fibo's trained CFG null caption.

License: Weights © BRIA AI, CC BY-NC 4.0. Non-commercial only.
"""

import runpod
import os
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.parse
import urllib.error
import subprocess
import binascii
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SERVER_ADDRESS = os.getenv("SERVER_ADDRESS", "127.0.0.1")
CLIENT_ID = str(uuid.uuid4())

# ComfyUI input directory
INPUT_DIR = "/comfyui/input"
os.makedirs(INPUT_DIR, exist_ok=True)

# Load workflow template once at startup
WORKFLOW_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api-workflow.json")
with open(WORKFLOW_PATH) as f:
    WORKFLOW_TEMPLATE = json.load(f)

# Node IDs (match api-workflow.json)
NODE_LOAD_IMAGE_1 = "39"   # LoadImage — primary reference (image_1 + image_3)
NODE_LOAD_IMAGE_2 = "40"   # LoadImage — secondary reference (image_2)
NODE_ENCODE       = "4"    # TextEncodeFibo2 — prompt, dimensions
NODE_SAMPLER      = "5"    # KSampler — seed, steps, cfg, denoise


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────

def save_b64_image(b64_data: str, path: str) -> None:
    try:
        decoded = base64.b64decode(b64_data)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(decoded)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Base64 decode failed: {e}")


def download_url_to_file(url: str, path: str) -> None:
    result = subprocess.run(
        ["wget", "-q", "-O", path, url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"wget failed: {result.stderr}")


def queue_prompt(prompt: dict) -> str:
    url = f"http://{SERVER_ADDRESS}:8188/prompt"
    payload = json.dumps({"prompt": prompt, "client_id": CLIENT_ID}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req)
        return json.loads(resp.read())["prompt_id"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        logger.error(f"ComfyUI /prompt rejected: {body}")
        raise RuntimeError(f"ComfyUI 400: {body}")


def get_image_bytes(filename: str, subfolder: str, folder_type: str) -> bytes:
    params = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
    with urllib.request.urlopen(f"http://{SERVER_ADDRESS}:8188/view?{params}") as r:
        return r.read()


def get_history(prompt_id: str) -> dict:
    with urllib.request.urlopen(f"http://{SERVER_ADDRESS}:8188/history/{prompt_id}") as r:
        return json.loads(r.read())


# ──────────────────────────────────────────────
# WebSocket execution tracker
# ──────────────────────────────────────────────

def run_workflow_via_websocket(ws: websocket.WebSocket, prompt: dict) -> dict:
    """Queue prompt, wait for completion via websocket, return output images."""
    prompt_id = queue_prompt(prompt)
    logger.info(f"Queued prompt_id={prompt_id}")

    while True:
        raw = ws.recv()
        if not isinstance(raw, str):
            continue   # binary frame — skip
        msg = json.loads(raw)
        if msg["type"] == "executing":
            data = msg["data"]
            if data["node"] is None and data["prompt_id"] == prompt_id:
                logger.info("ComfyUI execution complete")
                break
        elif msg["type"] == "execution_error":
            raise RuntimeError(f"ComfyUI execution error: {msg['data']}")

    history = get_history(prompt_id).get(prompt_id, {})
    output_images = {}
    for node_id, node_out in history.get("outputs", {}).items():
        if "images" in node_out:
            imgs = []
            for img_info in node_out["images"]:
                raw_bytes = get_image_bytes(img_info["filename"], img_info.get("subfolder", ""), img_info.get("type", "output"))
                imgs.append(base64.b64encode(raw_bytes).decode("utf-8"))
            output_images[node_id] = imgs

    return output_images


# ──────────────────────────────────────────────
# RunPod handler
# ──────────────────────────────────────────────

def handler(job: dict) -> dict:
    job_input = job.get("input", {})
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    logger.info(f"[{task_id}] input keys: {list(job_input.keys())}")

    # ── Resolve primary image ──
    img1_path = f"{INPUT_DIR}/{task_id}_img1.png"

    if "image" in job_input:
        try:
            save_b64_image(job_input["image"], img1_path)
        except ValueError as e:
            return {"error": str(e)}
    elif "image_base64" in job_input:
        try:
            save_b64_image(job_input["image_base64"], img1_path)
        except ValueError as e:
            return {"error": str(e)}
    elif "image_url" in job_input:
        try:
            download_url_to_file(job_input["image_url"], img1_path)
        except RuntimeError as e:
            return {"error": str(e)}
    else:
        return {"error": "Provide 'image' (base64), 'image_base64', or 'image_url'."}

    # ── Resolve optional second image ──
    img2_path = img1_path   # default: same as first
    if "image2" in job_input:
        img2_path = f"{INPUT_DIR}/{task_id}_img2.png"
        try:
            save_b64_image(job_input["image2"], img2_path)
        except ValueError as e:
            return {"error": f"image2 decode failed: {e}"}

    # ── Parameters ──
    prompt_text = job_input.get("prompt", "").strip()
    if not prompt_text:
        return {"error": "'prompt' is required."}

    seed    = int(job_input.get("seed", -1))
    steps   = int(job_input.get("steps", 4))
    cfg     = float(job_input.get("cfg", 1.0))
    denoise = float(job_input.get("denoise", 1.0))

    if seed == -1:
        import random
        seed = random.randint(0, 2**32 - 1)

    logger.info(f"[{task_id}] prompt={prompt_text!r}, seed={seed}, steps={steps}, cfg={cfg}")

    # ── Patch workflow ──
    workflow = json.loads(json.dumps(WORKFLOW_TEMPLATE))   # deep copy
    workflow[NODE_LOAD_IMAGE_1]["inputs"]["image"] = img1_path
    workflow[NODE_LOAD_IMAGE_2]["inputs"]["image"] = img2_path
    workflow[NODE_ENCODE]["inputs"]["prompt"]      = prompt_text
    workflow[NODE_SAMPLER]["inputs"]["seed"]       = seed
    workflow[NODE_SAMPLER]["inputs"]["steps"]      = steps
    workflow[NODE_SAMPLER]["inputs"]["cfg"]        = cfg
    workflow[NODE_SAMPLER]["inputs"]["denoise"]    = denoise

    # ── Connect WebSocket (retry up to 3 min) ──
    ws_url = f"ws://{SERVER_ADDRESS}:8188/ws?clientId={CLIENT_ID}"
    ws = websocket.WebSocket()
    for attempt in range(36):
        try:
            ws.connect(ws_url)
            logger.info(f"WebSocket connected (attempt {attempt + 1})")
            break
        except Exception as e:
            logger.warning(f"WS connect attempt {attempt + 1}/36 failed: {e}")
            if attempt == 35:
                return {"error": "WebSocket connection timeout — ComfyUI may still be loading models"}
            time.sleep(5)

    # ── Run workflow ──
    try:
        images = run_workflow_via_websocket(ws, workflow)
    except RuntimeError as e:
        ws.close()
        return {"error": str(e)}
    finally:
        ws.close()

    if not images:
        return {"error": "No images returned from ComfyUI"}

    # Return first output image
    for node_id, node_images in images.items():
        if node_images:
            logger.info(f"[{task_id}] Done — returning image from node {node_id}")
            return {"image": node_images[0]}

    return {"error": "No output image found in ComfyUI history"}


runpod.serverless.start({"handler": handler})
