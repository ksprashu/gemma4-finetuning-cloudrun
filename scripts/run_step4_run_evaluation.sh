#!/bin/bash
set -e

# Colors for professional output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
INFO='\033[0;34m'
YELLOW='\033[1;33m'

echo -e "${INFO}======================================================================${NC}"
echo -e "${INFO}🚀 STEP 4: RUN AN EVALUATION CHECK ON MODEL PRECISION${NC}"
echo -e "${INFO}======================================================================${NC}"

# Determine repo root directory relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo -e "This script evaluates the sentiment classification precision of your model."
echo -e "You can evaluate a remote Cloud Run endpoint (highly recommended & fast) or run a local PyTorch model check."
echo -e "----------------------------------------------------------------------"

# Resolve PROJECT_ID (used for defaults in gcloud run describe)
if [ -z "$PROJECT_ID" ]; then
    DEFAULT_PROJECT=$(gcloud config get-value project 2>/dev/null || echo "")
    if [ -t 0 ]; then
        read -p "Enter GCP Project ID [Current: $DEFAULT_PROJECT]: " input_project
        PROJECT_ID=${input_project:-$DEFAULT_PROJECT}
    else
        PROJECT_ID=$DEFAULT_PROJECT
    fi
fi

# Resolve REGION (used for defaults in gcloud run describe)
if [ -z "$REGION" ]; then
    DEFAULT_REGION="us-central1"
    if [ -t 0 ]; then
        read -p "Enter GCP Region [Default: $DEFAULT_REGION]: " input_region
        REGION=${input_region:-$DEFAULT_REGION}
    else
        REGION=$DEFAULT_REGION
    fi
fi

# Resolve SERVICE_NAME
SERVICE_NAME=${SERVICE_NAME:-"gemma-4-serve"}

# Resolve EVAL_MODE
if [ -z "$EVAL_MODE" ]; then
    if [ -t 0 ]; then
        echo -e "Choose your evaluation mode:"
        echo -e "1) Remote: Evaluate against a running Cloud Run Service (Default)"
        echo -e "2) Local: Evaluate locally using PyTorch (Requires GPU/MPS and model weights download)"
        read -p "Enter your choice [1 or 2, default: 1]: " input_mode
        EVAL_MODE=${input_mode:-1}
    else
        EVAL_MODE=1
    fi
fi

if [ "$EVAL_MODE" = "1" ]; then
    # Remote mode
    if [ -z "$SERVICE_URL" ]; then
        # Try to fetch default Cloud Run service URL
        DEFAULT_URL=""
        if [ -n "$PROJECT_ID" ]; then
            DEFAULT_URL=$(gcloud run services describe "$SERVICE_NAME" --format="value(status.url)" --region="$REGION" --project="$PROJECT_ID" 2>/dev/null || echo "")
        fi
        
        if [ -t 0 ]; then
            read -p "Enter Cloud Run Service URL [Current: $DEFAULT_URL]: " input_url
            SERVICE_URL=${input_url:-$DEFAULT_URL}
        else
            SERVICE_URL=$DEFAULT_URL
        fi
    fi
    
    if [ -z "$SERVICE_URL" ]; then
        echo -e "${RED}❌ Error: Cloud Run Service URL is required for remote evaluation.${NC}"
        echo -e "${YELLOW}Hint: Deploy your serving container first using Step 5 or set SERVICE_URL env var!${NC}"
        exit 1
    fi
    
    echo -e "${INFO}Evaluating remote service URL: ${SERVICE_URL}...${NC}"
    python3 "${REPO_ROOT}/src/evaluation_check.py" --url "$SERVICE_URL"
else
    # Local mode
    if [ -z "$MODEL_ID" ]; then
        DEFAULT_MODEL="google/gemma-4-E4B-it"
        if [ -t 0 ]; then
            read -p "Enter Base Model ID [Default: $DEFAULT_MODEL]: " input_model
            MODEL_ID=${input_model:-$DEFAULT_MODEL}
        else
            MODEL_ID=$DEFAULT_MODEL
        fi
    fi
    
    if [ -z "$ADAPTER_PATH" ] && [ -t 0 ]; then
        read -p "Enter Local/GCS Adapter Path (leave blank to test un-fine-tuned base model): " ADAPTER_PATH
    fi
    
    echo -e "${INFO}Evaluating local model: ${MODEL_ID}...${NC}"
    if [ -n "$ADAPTER_PATH" ]; then
        echo -e "${INFO}With adapter path: ${ADAPTER_PATH}...${NC}"
        python3 "${REPO_ROOT}/src/evaluation_check.py" --local --model_id "$MODEL_ID" --adapter_path "$ADAPTER_PATH"
    else
        python3 "${REPO_ROOT}/src/evaluation_check.py" --local --model_id "$MODEL_ID"
    fi
fi

echo -e "\n${GREEN}✅ Step 4 evaluation run finished successfully!${NC}"
