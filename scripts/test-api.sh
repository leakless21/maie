#!/usr/bin/env bash
# Simple API testing script for MAIE
# Tests the API endpoints with curl commands

set -euo pipefail

# Configuration
API_HOST="${API_HOST:-localhost}"
API_PORT="${API_PORT:-8000}"
API_KEY="${SECRET_API_KEY:-your-32-char-or-longer-api-key-here}"
BASE_URL="http://${API_HOST}:${API_PORT}"  # NOTE: http:// not https://

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${YELLOW}=== $1 ===${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Test 1: Health check (no auth required)
print_header "Test 1: Health Check"
echo "GET ${BASE_URL}/health"
response=$(curl -s -w "\n%{http_code}" "${BASE_URL}/health" || true)
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [[ "$http_code" == "200" ]]; then
    print_success "Health check passed"
    echo "$body" | jq . || echo "$body"
else
    print_error "Health check failed (HTTP $http_code)"
    echo "$body"
    exit 1
fi

# Test 2: List models (no auth required)
print_header "Test 2: List Available Models"
echo "GET ${BASE_URL}/v1/models"
response=$(curl -s -w "\n%{http_code}" "${BASE_URL}/v1/models" || true)
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [[ "$http_code" == "200" ]]; then
    print_success "Models list retrieved"
    echo "$body" | jq . || echo "$body"
else
    print_error "Failed to list models (HTTP $http_code)"
    echo "$body"
fi

# Test 3: List templates (no auth required)
print_header "Test 3: List Available Templates"
echo "GET ${BASE_URL}/v1/templates"
response=$(curl -s -w "\n%{http_code}" "${BASE_URL}/v1/templates" || true)
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [[ "$http_code" == "200" ]]; then
    print_success "Templates list retrieved"
    echo "$body" | jq . || echo "$body"
else
    print_error "Failed to list templates (HTTP $http_code)"
    echo "$body"
fi

# Test 4: Process text (requires API key)
print_header "Test 4: Process Text (with API key)"
echo "POST ${BASE_URL}/v1/process_text"
echo "API Key: ${API_KEY:0:10}..."

request_json='{
  "text": "This is a test transcript. Speaker A discusses the project requirements. Speaker B agrees with the timeline.",
  "template": "meeting_notes_v2",
  "language": "en",
  "options": {
    "enable_diarization": false
  }
}'

response=$(curl -s -w "\n%{http_code}" \
    -X POST "${BASE_URL}/v1/process_text" \
    -H "X-API-Key: ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "$request_json" || true)
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n-1)

if [[ "$http_code" == "202" ]]; then
    print_success "Text processing task accepted"
    task_id=$(echo "$body" | jq -r '.task_id')
    echo "Task ID: $task_id"
    
    # Test 5: Check status
    print_header "Test 5: Check Task Status"
    echo "GET ${BASE_URL}/v1/status/${task_id}"
    
    # Poll for completion (max 30 seconds)
    max_attempts=30
    for i in $(seq 1 $max_attempts); do
        sleep 1
        response=$(curl -s -w "\n%{http_code}" \
            -H "X-API-Key: ${API_KEY}" \
            "${BASE_URL}/v1/status/${task_id}" || true)
        http_code=$(echo "$response" | tail -n1)
        body=$(echo "$response" | head -n-1)
        
        status=$(echo "$body" | jq -r '.status')
        echo "[$i/$max_attempts] Status: $status"
        
        if [[ "$status" == "COMPLETED" ]]; then
            print_success "Task completed successfully"
            echo "$body" | jq .
            break
        elif [[ "$status" == "FAILED" ]]; then
            print_error "Task failed"
            echo "$body" | jq .
            break
        fi
        
        if [[ $i -eq $max_attempts ]]; then
            print_error "Timeout waiting for task completion"
        fi
    done
else
    print_error "Failed to submit text processing task (HTTP $http_code)"
    echo "$body"
    
    if [[ "$http_code" == "401" ]]; then
        echo -e "\n${RED}Authentication failed! Please set SECRET_API_KEY in your .env file${NC}"
    fi
fi

print_header "Test Summary"
echo "To process an audio file, use:"
echo ""
echo "curl -X POST '${BASE_URL}/v1/process' \\"
echo "  -H 'X-API-Key: YOUR_API_KEY' \\"
echo "  -F 'file=@path/to/audio.mp3' \\"
echo "  -F 'template=meeting_notes_v2'"
echo ""
echo -e "${YELLOW}IMPORTANT: Use http:// not https://${NC}"
echo "If you see SSL errors, you're likely using https:// instead of http://"
