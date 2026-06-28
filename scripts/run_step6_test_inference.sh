#!/bin/bash
set -e

# Colors for professional output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color
INFO='\033[0;34m'
YELLOW='\033[1;33m'

echo -e "${INFO}======================================================================${NC}"
echo -e "${INFO}🚀 STEP 6: TEST INFERENCE WITH CURL${NC}"
echo -e "${INFO}======================================================================${NC}"

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

# Try to fetch default Cloud Run service URL
if [ -z "$SERVICE_URL" ]; then
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
    echo -e "${RED}❌ Error: Cloud Run Service URL is required to test inference.${NC}"
    exit 1
fi

CLEAN_URL="${SERVICE_URL%/}"

echo -e "\n${INFO}1. Checking Service Health Status (/health)...${NC}"
curl -s -w "\nHTTP Status: %{http_code}\n\n" "${CLEAN_URL}/health"

echo -e "${INFO}2. Testing /generate (Raw Prompt)...${NC}"
GENERATE_PAYLOAD='{
  "prompt": "Classify the sentiment: '\''The delivery was extremely fast, but the gadget itself was dead on arrival.'\''",
  "max_new_tokens": 10,
  "temperature": 0.0
}'

echo -e "${YELLOW}Sending Payload:${NC} $GENERATE_PAYLOAD"
curl -s -X POST "${CLEAN_URL}/generate" \
     -H "Content-Type: application/json" \
     -d "$GENERATE_PAYLOAD" | json_pp || curl -s -X POST "${CLEAN_URL}/generate" -H "Content-Type: application/json" -d "$GENERATE_PAYLOAD"
echo -e "\n"

echo -e "${INFO}3. Testing OpenAI-Compatible Chat Completion (/v1/chat/completions)...${NC}"
CHAT_PAYLOAD='{
  "messages": [
    {"role": "user", "content": "Classify the sentiment: '\''Oh great, another update that breaks the entire app. Exactly what I needed.'\''"}
  ],
  "max_new_tokens": 10,
  "temperature": 0.0
}'

echo -e "${YELLOW}Sending Payload:${NC} $CHAT_PAYLOAD"
curl -s -X POST "${CLEAN_URL}/v1/chat/completions" \
     -H "Content-Type: application/json" \
     -d "$CHAT_PAYLOAD" | json_pp || curl -s -X POST "${CLEAN_URL}/v1/chat/completions" -H "Content-Type: application/json" -d "$CHAT_PAYLOAD"
echo -e "\n"

echo -e "${GREEN}✅ Step 6 complete! Inference tested successfully with curl.${NC}"
