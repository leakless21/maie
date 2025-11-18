#!/bin/bash
# Offline Operation Validation Script
# Verifies that MAIE is configured for fully offline operation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MAIE Offline Operation Validation ===${NC}\n"

# Track results
FAILED=0
PASSED=0

# Helper functions
check_file() {
    local file="$1"
    local pattern="$2"
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Found '$pattern' in $file"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} Missing '$pattern' in $file"
        ((FAILED++))
        return 1
    fi
}

check_model_exists() {
    local model_path="$1"
    if [ -d "$model_path" ]; then
        local file_count=$(find "$model_path" -type f | wc -l)
        echo -e "${GREEN}✓${NC} Model found: $model_path ($file_count files)"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} Model NOT found: $model_path"
        ((FAILED++))
        return 1
    fi
}

check_env_var() {
    local var_name="$1"
    local file="$2"
    local pattern="$var_name"
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} Environment variable $var_name configured"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗${NC} Environment variable $var_name NOT configured"
        ((FAILED++))
        return 1
    fi
}

# Check 1: Environment variables disabled in config/__init__.py
echo -e "${BLUE}1. Checking telemetry environment variables...${NC}"
check_env_var "PYANNOTE_DISABLE_TELEMETRY" "src/config/__init__.py" || true
check_env_var "PYANNOTE_NO_ANALYTICS" "src/config/__init__.py" || true
check_env_var "HUGGINGFACE_HUB_OFFLINE" "src/config/__init__.py" || true
check_env_var "TRANSFORMERS_OFFLINE" "src/config/__init__.py" || true
echo

# Check 2: Diarizer uses local model path
echo -e "${BLUE}2. Checking diarizer model loading...${NC}"
check_file "src/processors/audio/diarizer.py" "FULLY OFFLINE" || true
check_file "src/processors/audio/diarizer.py" "Pipeline.from_pretrained(self.model_path)" || true
check_file "src/processors/audio/diarizer.py" "LOCAL PATH" || true
echo

# Check 3: Default model path is local
echo -e "${BLUE}3. Checking model configuration...${NC}"
check_file "src/config/model.py" 'default="data/models/pyannote-speaker-diarization-community-1"' || true
check_file "src/config/model.py" "FULLY OFFLINE operation" || true
echo

# Check 4: Local model exists
echo -e "${BLUE}4. Checking local models...${NC}"
check_model_exists "data/models/pyannote-speaker-diarization-community-1" || true
if [ -d "data/models/pyannote-speaker-diarization-community-1" ]; then
    # Check for key files
    echo -e "   Checking model files:"
    for file in config.yaml embedding/pytorch_model.bin segmentation/pytorch_model.bin plda/plda.npz; do
        if [ -f "data/models/pyannote-speaker-diarization-community-1/$file" ]; then
            echo -e "   ${GREEN}✓${NC} $file"
            ((PASSED++))
        else
            echo -e "   ${RED}✗${NC} $file (MISSING)"
            ((FAILED++))
        fi
    done
fi
echo

# Check 5: No HuggingFace model IDs in code
echo -e "${BLUE}5. Checking for hardcoded HuggingFace model IDs...${NC}"
if grep -r "pyannote/speaker-diarization" src/ --include="*.py" 2>/dev/null | grep -v "# " | grep -v '"""' | grep -v "'" > /tmp/hf_refs.txt 2>&1; then
    if [ -s /tmp/hf_refs.txt ]; then
        echo -e "${YELLOW}⚠${NC} Found HuggingFace model ID references (might be in comments/docs):"
        while IFS= read -r line; do
            echo "   $line"
        done < /tmp/hf_refs.txt
    else
        echo -e "${GREEN}✓${NC} No HuggingFace model ID references in active code"
        ((PASSED++))
    fi
else
    echo -e "${GREEN}✓${NC} No HuggingFace model ID references in active code"
    ((PASSED++))
fi
echo

# Check 6: Documentation
echo -e "${BLUE}6. Checking documentation...${NC}"
if [ -f "docs/OFFLINE_OPERATION.md" ]; then
    echo -e "${GREEN}✓${NC} Offline operation guide exists: docs/OFFLINE_OPERATION.md"
    ((PASSED++))
else
    echo -e "${RED}✗${NC} Offline operation guide missing"
    ((FAILED++))
fi
echo

# Check 7: Test updates
echo -e "${BLUE}7. Checking test configurations...${NC}"
check_file "tests/integration/test_enable_diarization_e2e.py" 'data/models/pyannote-speaker-diarization-community-1' || true
echo

# Summary
echo -e "${BLUE}=== Validation Summary ===${NC}"
echo -e "Passed: ${GREEN}${PASSED}${NC}"
echo -e "Failed: ${RED}${FAILED}${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "\n${GREEN}✓ All checks passed! System is configured for fully offline operation.${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Some checks failed. Please review the issues above.${NC}"
    exit 1
fi
