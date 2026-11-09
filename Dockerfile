FROM runpod/worker-comfyui:5.8.4-base

ARG HF_TOKEN=""

# Install curl (needed by start.sh hot-patch) and websocket-client
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir websocket-client

# ── Models ────────────────────────────────────────────────────
# Total ~24 GB — downloaded once during docker build, baked into the image.
# NOTE: RunPod GitHub builds have a 30-min Docker build timeout.
# If the build times out, build locally and push to Docker Hub instead:
#   docker build --platform linux/amd64 -t yourname/fibo2-serverless:v1 .
#   docker push yourname/fibo2-serverless:v1

# 1. Fibo 2.0 Edit Turbo diffusion model (bf16, ~16 GB)
RUN BACKOFFS="10 20 30 60 90" && for i in 1 2 3 4 5; do \
    comfy model download \
      --url 'https://huggingface.co/Comfy-Org/Fibo-2.0/resolve/main/diffusion_models/fibo2_edit_turbo_bf16.safetensors' \
      --relative-path models/diffusion_models \
      --filename 'fibo2_edit_turbo_bf16.safetensors' && break; \
    if [ $i -eq 5 ]; then echo "model-download failed" >&2; exit 1; fi; \
    SLEEP=$(echo $BACKOFFS | cut -d ' ' -f $i) && sleep $SLEEP; done

# 2. Qwen3-VL 4B text encoder (bf16, ~8 GB)
RUN BACKOFFS="10 20 30 60 90" && for i in 1 2 3 4 5; do \
    comfy model download \
      --url 'https://huggingface.co/Comfy-Org/Fibo-2.0/resolve/main/text_encoders/qwen3vl_4b_bf16.safetensors' \
      --relative-path models/text_encoders \
      --filename 'qwen3vl_4b_bf16.safetensors' && break; \
    if [ $i -eq 5 ]; then echo "model-download failed" >&2; exit 1; fi; \
    SLEEP=$(echo $BACKOFFS | cut -d ' ' -f $i) && sleep $SLEEP; done

# 3. Flux2 VAE (~335 MB)
RUN BACKOFFS="10 20 30 60 90" && for i in 1 2 3 4 5; do \
    comfy model download \
      --url 'https://huggingface.co/Comfy-Org/Fibo-2.0/resolve/main/vae/flux2-vae.safetensors' \
      --relative-path models/vae \
      --filename 'flux2-vae.safetensors' && break; \
    if [ $i -eq 5 ]; then echo "model-download failed" >&2; exit 1; fi; \
    SLEEP=$(echo $BACKOFFS | cut -d ' ' -f $i) && sleep $SLEEP; done

# ── Handler & workflow ────────────────────────────────────────
# start.sh hot-patches these from GitHub on every boot,
# so handler/workflow changes never require a full rebuild.
COPY rp_handler.py     /rp_handler.py
COPY api-workflow.json /api-workflow.json
COPY start.sh          /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
