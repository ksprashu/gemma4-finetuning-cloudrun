#!/bin/bash
set -e

# Colors for professional output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
INFO='\033[0;34m'
YELLOW='\033[1;33m'

echo -e "${INFO}======================================================================${NC}"
echo -e "${INFO}🚀 STEP 5: BUILD & DEPLOY INFERENCE SERVICE TO CLOUD RUN${NC}"
echo -e "${INFO}======================================================================${NC}"

# 1. Resolve PROJECT_ID
if [ -z "$PROJECT_ID" ]; then
    DEFAULT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
    if [ -t 0 ]; then
        read -p "Enter GCP Project ID [Current: $DEFAULT_PROJECT]: " input_project
        PROJECT_ID=${input_project:-$DEFAULT_PROJECT}
    else
        PROJECT_ID=$DEFAULT_PROJECT
    fi
fi

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: GCP Project ID is required.${NC}"
    exit 1
fi

# Set project
gcloud config set project "$PROJECT_ID"

# 2. Resolve BUCKET_NAME
if [ -z "$BUCKET_NAME" ]; then
    SUGGESTED_BUCKET="${PROJECT_ID}-gemma4-finetuning"
    if [ -t 0 ]; then
        read -p "Enter GCS Bucket Name where trained adapters are stored [Default: $SUGGESTED_BUCKET]: " input_bucket
        BUCKET_NAME=${input_bucket:-$SUGGESTED_BUCKET}
    else
        BUCKET_NAME=$SUGGESTED_BUCKET
    fi
fi

# 3. Resolve REGION
if [ -z "$REGION" ]; then
    DEFAULT_REGION="us-central1"
    if [ -t 0 ]; then
        read -p "Enter GCP Region [Default: us-central1]: " input_region
        REGION=${input_region:-$DEFAULT_REGION}
    else
        REGION=$DEFAULT_REGION
    fi
fi

# 4. Resolve configurable options
REPO_NAME=${REPO_NAME:-"gemma-4-repo"}
SERVICE_NAME=${SERVICE_NAME:-"gemma-4-serve"}
MODEL_ID=${MODEL_ID:-"google/gemma-4-E4B-it"}
GCS_PREFIX=${GCS_PREFIX:-"gemma-4-adapters"}

echo -e "\n${INFO}Serving Deployment Configuration Summary:${NC}"
echo -e " - PROJECT_ID:   ${YELLOW}${PROJECT_ID}${NC}"
echo -e " - BUCKET_NAME:  ${YELLOW}${BUCKET_NAME}${NC}"
echo -e " - REGION:       ${YELLOW}${REGION}${NC}"
echo -e " - REPO_NAME:    ${YELLOW}${REPO_NAME}${NC}"
echo -e " - SERVICE_NAME: ${YELLOW}${SERVICE_NAME}${NC}"
echo -e " - MODEL_ID:     ${YELLOW}${MODEL_ID}${NC}"
echo -e " - GCS_PREFIX:   ${YELLOW}${GCS_PREFIX}${NC}"
echo -e "----------------------------------------------------------------------"

# 5. Build Serving Container using Google Cloud Build (Serverless)
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}:latest"
echo -e "\n${INFO}Building and pushing serving image using Cloud Build...${NC}"
echo -e "${INFO}Image destination: ${IMAGE_TAG}${NC}"
gcloud builds submit --tag "$IMAGE_TAG" \
    --config=<(echo '
steps:
- name: "gcr.io/cloud-builders/docker"
  args: ["build", "-t", "'"$IMAGE_TAG"'", "-f", "Dockerfile.serve", "."]
images:
- "'"$IMAGE_TAG"'"
') \
    --project="$PROJECT_ID" \
    --timeout="2h"

# 6. Deploy Cloud Run Service using templated YAML or directly via gcloud run deploy
echo -e "\n${INFO}Generating service deployment YAML from template...${NC}"

# Generate customized service config
cat <<EOF > deploy_service.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: ${SERVICE_NAME}
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/gpu-zonal-redundancy-disabled: 'true'
        run.googleapis.com/cpu-throttling: 'false'
        autoscaling.knative.dev/minScale: '1'
        autoscaling.knative.dev/maxScale: '5'
    spec:
      containerConcurrency: 10
      containers:
      - image: ${IMAGE_TAG}
        resources:
          limits:
            cpu: '8'
            memory: '32Gi'
            nvidia.com/gpu: '1'
        env:
        - name: PROJECT_ID
          value: "${PROJECT_ID}"
        - name: GCP_PROJECT
          value: "${PROJECT_ID}"
        - name: MODEL_ID
          value: "${MODEL_ID}"
        - name: LORA_ADAPTER_PATH
          value: "gs://${BUCKET_NAME}/${GCS_PREFIX}"
        - name: HF_TOKEN_SECRET_NAME
          value: "HF_TOKEN"
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: HF_TOKEN
              key: "latest"
        ports:
        - containerPort: 8080
        startupProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 120
          periodSeconds: 10
          failureThreshold: 15
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          periodSeconds: 20
      nodeSelector:
        run.googleapis.com/accelerator: nvidia-l4
EOF

echo -e "${INFO}Deploying Cloud Run Service '${SERVICE_NAME}'...${NC}"
gcloud run services replace deploy_service.yaml --region="$REGION" --project="$PROJECT_ID"

# Clean up temp deploy service config
rm deploy_service.yaml

# Fetch deployed URL
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --format="value(status.url)" --region="$REGION" --project="$PROJECT_ID" 2>/dev/null || echo "")

echo -e "\n${GREEN}✅ Step 5 complete! Cloud Run Inference Service is deployed and active.${NC}"
echo -e "${GREEN}   Service URL: ${SERVICE_URL}${NC}"
echo -e "${GREEN}   You can now test inference using Step 6!${NC}"
