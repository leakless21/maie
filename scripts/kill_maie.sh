#!/usr/bin/env bash
# Quick script to kill MAIE processes

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 MAIE Process Killer${NC}"
echo "================================"

# Check if process_manager.py exists
if [[ ! -f "scripts/process_manager.py" ]]; then
    echo -e "${RED}❌ Error: scripts/process_manager.py not found${NC}"
    echo "Please run this script from the MAIE project root directory"
    exit 1
fi

# Show current processes
echo -e "${YELLOW}📊 Current MAIE processes:${NC}"
pixi run python scripts/process_manager.py

echo ""
echo -e "${YELLOW}🤔 What would you like to do?${NC}"
echo "1) Kill all processes (SIGTERM)"
echo "2) Force kill all processes (SIGKILL)"
echo "3) Interactive kill (ask for each process)"
echo "4) Just show processes (no kill)"
echo "5) Exit"

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo -e "${GREEN}💀 Killing all processes with SIGTERM...${NC}"
        pixi run python scripts/process_manager.py --kill
        ;;
    2)
        echo -e "${RED}💀 Force killing all processes with SIGKILL...${NC}"
        pixi run python scripts/process_manager.py --kill --force
        ;;
    3)
        echo -e "${YELLOW}🤔 Starting interactive kill mode...${NC}"
        pixi run python scripts/process_manager.py --interactive
        ;;
    4)
        echo -e "${BLUE}📊 Showing processes only...${NC}"
        pixi run python scripts/process_manager.py
        ;;
    5)
        echo -e "${GREEN}👋 Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✅ Done!${NC}"
