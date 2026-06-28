#!/bin/bash
set -e

# Colors for professional output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
INFO='\033[0;34m'
YELLOW='\033[1;33m'

echo -e "${INFO}======================================================================${NC}"
echo -e "${INFO}🚀 STEP 2: CREATE GCS BUCKET & UPLOAD DATASET${NC}"
echo -e "${INFO}======================================================================${NC}"

# Resolve dataset file path
OUTPUT_FILE=${OUTPUT_FILE:-"sentiment_dataset.jsonl"}

# Check for local dataset file
if [ ! -f "$OUTPUT_FILE" ]; then
    echo -e "${YELLOW}⚠️ '$OUTPUT_FILE' not found in workspace root. Running Step 1 dataset generator...${NC}"
    export OUTPUT_FILE="$OUTPUT_FILE"
    python3 generate_dataset.py
fi

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
    echo -e "${RED}❌ Error: GCP Project ID is required. Please set PROJECT_ID env var or select a gcloud project.${NC}"
    exit 1
fi

# 2. Resolve BUCKET_NAME
if [ -z "$BUCKET_NAME" ]; then
    SUGGESTED_BUCKET="${PROJECT_ID}-gemma4-finetuning"
    if [ -t 0 ]; then
        read -p "Enter GCS Bucket Name [Default: $SUGGESTED_BUCKET]: " input_bucket
        BUCKET_NAME=${input_bucket:-$SUGGESTED_BUCKET}
    else
        BUCKET_NAME=$SUGGESTED_BUCKET
    fi
fi

# 3. Resolve REGION
if [ -z "$REGION" ]; then
    DEFAULT_REGION="us-central1"
    if [ -t 0 ]; then
        read -p "Enter GCP Region [Default: $DEFAULT_REGION]: " input_region
        REGION=${input_region:-$DEFAULT_REGION}
    else
        REGION=$DEFAULT_REGION
    fi
fi

echo -e "\n${INFO}Configuring gcloud project to: ${PROJECT_ID}...${NC}"
gcloud config set project "$PROJECT_ID"

# Check if bucket exists
if gcloud storage buckets describe "gs://${BUCKET_NAME}" &>/dev/null; then
    echo -e "${GREEN}Bucket gs://${BUCKET_NAME} already exists.${NC}"
else
    echo -e "${INFO}Creating bucket gs://${BUCKET_NAME} in region ${REGION}...${NC}"
    gcloud storage buckets create "gs://${BUCKET_NAME}" \
        --project="$PROJECT_ID" \
        --location="$REGION" \
        --uniform-bucket-level-access
fi

# Upload the dataset
DATASET_BASENAME=$(basename "$OUTPUT_FILE")
echo -e "\n${INFO}Uploading '$OUTPUT_FILE' to gs://${BUCKET_NAME}/data/${DATASET_BASENAME}...${NC}"
gcloud storage cp "$OUTPUT_FILE" "gs://${BUCKET_NAME}/data/${DATASET_BASENAME}"

echo -e "\n${GREEN}✅ Step 2 complete! Dataset successfully uploaded to GCS.${NC}"
echo -e "${GREEN}   Path: gs://${BUCKET_NAME}/data/${DATASET_BASENAME}${NC}"
