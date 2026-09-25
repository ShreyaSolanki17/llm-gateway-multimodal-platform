#!/bin/bash
# Builds the gateway image and deploys it to Cloud Run -- a serverless,
# CPU-only, pay-per-request platform. Well suited to this gateway since it
# doesn't need a GPU itself (only the vLLM server does -- see
# deploy-vllm-gpu-vm.sh for that piece).
#
# COST NOTE: Cloud Run's free tier covers light usage; beyond that it's
# billed per request/CPU-second, with no idle cost when nothing's calling it
# (unlike the GPU VM, which bills for every hour it's running).
#
# Prerequisites:
#   - gcloud CLI installed and authenticated, Docker installed locally
#   - A GCP project with billing enabled, Cloud Run + Artifact Registry APIs
#     enabled
#   - Real secrets (OPENAI_API_KEY, GATEWAY_API_KEY) should go through Secret
#     Manager for an actual deployment, not plain --set-env-vars -- see the
#     commented example at the bottom

set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID first}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${CLOUD_RUN_SERVICE_NAME:-llm-gateway}"
REPO_NAME="${ARTIFACT_REPO_NAME:-llm-gateway-repo}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/gateway:latest"

gcloud config set project "${PROJECT_ID}"

gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    2>/dev/null || echo "Artifact Registry repo already exists, skipping"

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

docker build -t "${IMAGE}" .
docker push "${IMAGE}"

# NOTE: GATEWAY_API_KEY is intentionally not set here -- this service is
# reachable from the public internet once deployed (--allow-unauthenticated
# is a network-level setting; it does not disable this gateway's own
# app-level auth from Milestone 13). Set GATEWAY_API_KEY via Secret Manager
# before treating this as anything beyond a demo.
gcloud run deploy "${SERVICE_NAME}" \
    --image="${IMAGE}" \
    --region="${REGION}" \
    --platform=managed \
    --allow-unauthenticated \
    --port=8000 \
    --set-env-vars="ENVIRONMENT=production,VLLM_SIMULATE_LOCAL=true" \
    --memory=512Mi

echo ""
echo "Deployed. Get the service URL:"
echo "  gcloud run services describe ${SERVICE_NAME} --region=${REGION} --format='value(status.url)'"
echo ""
echo "To point this gateway at a real GPU-backed vLLM server instead of"
echo "simulated responses (see deploy-vllm-gpu-vm.sh):"
echo "  gcloud run services update ${SERVICE_NAME} --region=${REGION} \\"
echo "    --set-env-vars=VLLM_SERVER_URL=http://<VLLM_VM_IP>:8000/v1,VLLM_SIMULATE_LOCAL=false"
echo ""
echo "Set real secrets via Secret Manager (do this before real use):"
echo "  echo -n 'sk-...' | gcloud secrets create openai-api-key --data-file=-"
echo "  gcloud run services update ${SERVICE_NAME} --region=${REGION} \\"
echo "    --set-secrets=OPENAI_API_KEY=openai-api-key:latest,GATEWAY_API_KEY=gateway-api-key:latest"
echo ""
echo "Teardown when done (stops billing):"
echo "  gcloud run services delete ${SERVICE_NAME} --region=${REGION}"
