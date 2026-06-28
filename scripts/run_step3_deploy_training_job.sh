#!/bin/bash
set -e

# Colors for professional output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
INFO='\033[0;34m'
YELLOW='\033[1;33m'

echo -e "${INFO}======================================================================${NC}"
echo -e "${INFO}🚀 STEP 3: BUILD & DEPLOY CLOUD RUN TRAINING JOB (GPU)${NC}"
echo -e "${INFO}======================================================================${NC}"

# Determine repo root directory relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Always execute relative to the repository root directory
cd "${REPO_ROOT}"

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
        read -p "Enter your GCS Bucket Name [Default: $SUGGESTED_BUCKET]: " input_bucket
        BUCKET_NAME=${input_bucket:-$SUGGESTED_BUCKET}
    else
        BUCKET_NAME=$SUGGESTED_BUCKET
    fi
fi

# 3. Resolve HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    if [ -t 0 ]; then
        read -p "Enter Hugging Face Token (will be secured in GCP Secret Manager): " HF_TOKEN
    else
        echo -e "${YELLOW}⚠️ HF_TOKEN is not set in environment. Checking Secret Manager fallback...${NC}"
    fi
fi

# 4. Resolve REGION
if [ -z "$REGION" ]; then
    DEFAULT_REGION="us-central1"
    if [ -t 0 ]; then
        read -p "Enter GCP Region [Default: $DEFAULT_REGION]: " input_region
        REGION=${input_region:-$DEFAULT_REGION}
    else
        REGION=$DEFAULT_REGION
    fi
fi

# 5. Resolve fully generic configurable pipeline variables
REPO_NAME=${REPO_NAME:-"gemma-4-repo"}
JOB_NAME=${JOB_NAME:-"gemma-4-finetune"}
MODEL_ID=${MODEL_ID:-"google/gemma-4-E4B-it"}
DATASET_FILE_NAME=$(basename "${OUTPUT_FILE:-sentiment_dataset.jsonl}")
DATASET_PATH=${DATASET_PATH:-"gs://${BUCKET_NAME}/data/${DATASET_FILE_NAME}"}
GCS_PREFIX=${GCS_PREFIX:-"gemma-4-adapters"}

echo -e "\n${INFO}Deployment Configuration Summary:${NC}"
echo -e " - PROJECT_ID:   ${YELLOW}${PROJECT_ID}${NC}"
echo -e " - BUCKET_NAME:  ${YELLOW}${BUCKET_NAME}${NC}"
echo -e " - REGION:       ${YELLOW}${REGION}${NC}"
echo -e " - REPO_NAME:    ${YELLOW}${REPO_NAME}${NC}"
echo -e " - JOB_NAME:     ${YELLOW}${JOB_NAME}${NC}"
echo -e " - MODEL_ID:     ${YELLOW}${MODEL_ID}${NC}"
echo -e " - DATASET_PATH: ${YELLOW}${DATASET_PATH}${NC}"
echo -e " - GCS_PREFIX:   ${YELLOW}${GCS_PREFIX}${NC}"
echo -e "----------------------------------------------------------------------"

# 6. Enable Required Services
echo -e "\n${INFO}Enabling required Google Cloud APIs...${NC}"
gcloud services enable \
    artifactregistry.googleapis.com \
    run.googleapis.com \
    secretmanager.googleapis.com \
    cloudbuild.googleapis.com

# 7. Setup Secret Manager for HF_TOKEN
if [ -n "$HF_TOKEN" ]; then
    echo -e "\n${INFO}Storing HF_TOKEN securely in Google Secret Manager...${NC}"
    # Create secret if it doesn't exist
    if ! gcloud secrets describe HF_TOKEN &>/dev/null; then
        gcloud secrets create HF_TOKEN --replication-policy="automatic" --project="$PROJECT_ID"
    fi
    # Add new secret version
    echo -n "$HF_TOKEN" | gcloud secrets versions add HF_TOKEN --data-file=- --project="$PROJECT_ID"
    echo -e "${GREEN}HF_TOKEN secret created/updated successfully!${NC}"
fi

# 8. Create Artifact Registry Repository
echo -e "\n${INFO}Creating Artifact Registry repository: ${REPO_NAME} in ${REGION}...${NC}"
if ! gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" &>/dev/null; then
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --description="Docker repository for Gemma 4 training and serving images" \
        --project="$PROJECT_ID"
else
    echo -e "${GREEN}Repository ${REPO_NAME} already exists.${NC}"
fi

# 9. Resolve Service Account and IAM bindings
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo -e "\n${INFO}Assigning storage and secret permissions to the service account: ${SERVICE_ACCOUNT}...${NC}"
# Grant GCS permissions
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/storage.objectAdmin" \
    --condition=none &>/dev/null

# Grant Secret Manager permissions
gcloud secrets add-iam-policy-binding HF_TOKEN \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor" \
    --project="$PROJECT_ID" \
    --condition=none &>/dev/null

echo -e "${GREEN}IAM permissions set up successfully!${NC}"

# 10. Build Training Container using Google Cloud Build (Serverless)
IMAGE_TAG="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/gemma-4-train:latest"
echo -e "\n${INFO}Building and pushing training image using Cloud Build...${NC}"
echo -e "${INFO}Image destination: ${IMAGE_TAG}${NC}"
gcloud builds submit --tag "$IMAGE_TAG" \
    --config=<(echo '
steps:
- name: "gcr.io/cloud-builders/docker"
  args: ["build", "-t", "'"$IMAGE_TAG"'", "-f", "deploy/Dockerfile.train", "."]
images:
- "'"$IMAGE_TAG"'"
') \
    --project="$PROJECT_ID" \
    --timeout="2h"

# 11. Deploy Cloud Run Job
echo -e "\n${INFO}Generating deployment job YAML from template...${NC}"

# Generate customized job config
cat <<EOF > deploy_job.yaml
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  labels:
    cloud.googleapis.com/location: ${REGION}
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/gpu-zonal-redundancy-disabled: 'true'
    spec:
      template:
        spec:
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
            - name: HF_TOKEN_SECRET_NAME
              value: "HF_TOKEN"
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: HF_TOKEN
                  key: "latest"
            args:
            - "--model_id"
            - "${MODEL_ID}"
            - "--dataset_name_or_path"
            - "${DATASET_PATH}"
            - "--gcs_bucket"
            - "${BUCKET_NAME}"
            - "--gcs_prefix"
            - "${GCS_PREFIX}"
            nodeSelector:
              run.googleapis.com/accelerator: nvidia-l4
          timeoutSeconds: 86400
EOF

echo -e "${INFO}Deploying Cloud Run Job '${JOB_NAME}'...${NC}"
gcloud run jobs replace deploy_job.yaml --region="$REGION" --project="$PROJECT_ID"

# Clean up temp deploy job
rm deploy_job.yaml

echo -e "\n${GREEN}✅ Step 3 complete! Cloud Run Job has been registered successfully.${NC}"
echo -e "${GREEN}   To execute the training job, run:${NC}"
echo -e "${YELLOW}   gcloud run jobs execute ${JOB_NAME} --region=${REGION} --project=${PROJECT_ID}${NC}"
