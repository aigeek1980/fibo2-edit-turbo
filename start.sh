#!/bin/bash
set -e

# ── Hot-patch: pull latest handler & workflow from GitHub on every boot ──────
# This means rp_handler.py and api-workflow.json changes never need a rebuild.
GITHUB_RAW="https://raw.githubusercontent.com/aigeek1980/fibo2-edit-turbo/main"

echo "Fetching latest handler from GitHub..."
curl -sf -o /rp_handler.py "${GITHUB_RAW}/rp_handler.py" \
    && echo "✅ rp_handler.py updated" \
    || echo "⚠️  Failed to fetch rp_handler.py — using baked-in version"

curl -sf -o /api-workflow.json "${GITHUB_RAW}/api-workflow.json" \
    && echo "✅ api-workflow.json updated" \
    || echo "⚠️  Failed to fetch api-workflow.json — using baked-in version"

# ── CUDA check ───────────────────────────────────────────────────────────────
echo "Checking CUDA..."
if ! python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'"; then
    echo "Error: CUDA not available. Exiting."
    exit 1
fi
echo "✅ CUDA available"

# ── Start ComfyUI in background ──────────────────────────────────────────────
echo "Starting ComfyUI..."
python /comfyui/main.py --listen 127.0.0.1 --port 8188 > /var/log/comfyui.log 2>&1 &

# ── Wait for ComfyUI to be ready ─────────────────────────────────────────────
echo "Waiting for ComfyUI to be ready..."
max_wait=180
waited=0
while [ $waited -lt $max_wait ]; do
    if curl -s http://127.0.0.1:8188/ > /dev/null 2>&1; then
        echo "✅ ComfyUI is ready! (${waited}s)"
        break
    fi
    sleep 2
    waited=$((waited + 2))
done

if [ $waited -ge $max_wait ]; then
    echo "Error: ComfyUI failed to start within ${max_wait}s — check /var/log/comfyui.log"
    exit 1
fi

# ── Start RunPod handler ──────────────────────────────────────────────────────
echo "Starting handler..."
exec python -u /rp_handler.py
