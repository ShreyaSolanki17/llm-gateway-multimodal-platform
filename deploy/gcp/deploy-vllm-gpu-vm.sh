#!/bin/bash
# Provisions a single GCE VM with an NVIDIA T4 GPU running vLLM, serving the
# model configured via VLLM_MODEL_NAME (defaults to the small Qwen2.5-0.5B
# model this project uses for GPU-less local simulation).
#
# COST NOTE: a T4 + n1-standard-4 runs roughly $0.35-$0.55/hr in us-central1
# (varies by region and whether you use Spot pricing). Delete the instance
# when you're done -- see the teardown command printed at the end.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (`gcloud auth login`)
#   - A GCP project with billing enabled and the Compute Engine API enabled
#   - GPU quota for the target region/zone -- new projects default to 0 GPU
#     quota, request an increase via the GCP Console (Quotas page) first

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID first}"
ZONE="${GCP_ZONE:-us-central1-a}"
INSTANCE_NAME="${VLLM_INSTANCE_NAME:-llm-gateway-vllm}"

gcloud config set project "${PROJECT_ID}"

gcloud compute instances create "${INSTANCE_NAME}" \
    --zone="${ZONE}" \
    --machine-type=n1-standard-4 \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --maintenance-policy=TERMINATE \
    --image-family=common-cu121-debian-11 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --metadata-from-file=startup-script="$(dirname "$0")/vllm-startup-script.sh" \
    --tags=vllm-server

gcloud compute firewall-rules create allow-vllm-8000 \
    --allow=tcp:8000 \
    --target-tags=vllm-server \
    --description="Allow inbound to vLLM OpenAI-compatible server" \
    2>/dev/null || echo "Firewall rule already exists, skipping"

echo ""
echo "VM created. vLLM takes a few minutes to install and start serving on first boot."
echo ""
echo "Get its external IP:"
echo "  gcloud compute instances describe ${INSTANCE_NAME} --zone=${ZONE} --format='get(networkInterfaces[0].accessConfigs[0].natIP)'"
echo ""
echo "Then point this gateway at it in .env:"
echo "  VLLM_SERVER_URL=http://<EXTERNAL_IP>:8000/v1"
echo "  VLLM_SIMULATE_LOCAL=false"
echo ""
echo "Teardown when done (stops billing):"
echo "  gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE}"
echo "  gcloud compute firewall-rules delete allow-vllm-8000"
