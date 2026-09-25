#!/bin/bash
# GCE VM startup script: installs vLLM and serves the configured model.
# Attached to a VM via --metadata-from-file startup-script=, runs
# automatically on first boot -- see deploy-vllm-gpu-vm.sh.

set -euo pipefail

VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"

apt-get update -y
apt-get install -y python3-pip

pip3 install vllm

# Serve on 0.0.0.0:8000 as an OpenAI-compatible /v1/chat/completions endpoint --
# matches what VLLMProvider/OpenAICompatibleProvider expect via VLLM_SERVER_URL.
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "${VLLM_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port 8000 \
    > /var/log/vllm.log 2>&1 &
