#!/bin/bash
set -e

# Colors for professional output
GREEN='\033[0;32m'
NC='\033[0m' # No Color
INFO='\033[0;34m'
YELLOW='\033[1;33m'

echo -e "${INFO}======================================================================${NC}"
echo -e "${INFO}🚀 STEP 1: GENERATE SYNTHETIC SENTIMENT DATASET${NC}"
echo -e "${INFO}======================================================================${NC}"

# Configure defaults and inform the user
DATASET_SIZE=${DATASET_SIZE:-2000}
OUTPUT_FILE=${OUTPUT_FILE:-"sentiment_dataset.jsonl"}

echo -e "${INFO}Configuration:${NC}"
echo -e " - DATASET_SIZE: ${YELLOW}${DATASET_SIZE}${NC} (Set via DATASET_SIZE env var)"
echo -e " - OUTPUT_FILE:  ${YELLOW}${OUTPUT_FILE}${NC} (Set via OUTPUT_FILE env var)"
echo -e "----------------------------------------------------------------------"

# Run the python dataset generator with the resolved environment variables
export DATASET_SIZE="$DATASET_SIZE"
export OUTPUT_FILE="$OUTPUT_FILE"

python3 generate_dataset.py

echo -e "\n${GREEN}✅ Step 1 complete! '${OUTPUT_FILE}' has been successfully generated.${NC}"
