#!/usr/bin/env bash
# =============================================================================
# sync-from-main.sh - Sync jetson branch with main while preserving jetson-specific files
# =============================================================================
#
# Usage:
#   ./scripts/sync-from-main.sh [--dry-run]
#
# This script:
#   1. Fetches the latest main branch from origin
#   2. Starts a merge with main (no auto-commit)
#   3. Preserves jetson-specific files (pyproject.toml, edge_main.py, etc.)
#   4. Shows you the status and next steps
#
# See docs/JETSON_NANO_BRANCH_PLAN.md for full documentation on branch strategy.
# =============================================================================

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Files that are specific to the jetson branch and should be preserved during merge
JETSON_SPECIFIC_FILES=(
    "pyproject.toml"
    "pixi.lock"
    "Dockerfile.jetson"
    "src/api/edge_main.py"
    "src/config/profiles.py"
    "docs/JETSON_NANO_BRANCH_PLAN.md"
    ".env.jetson"
    ".env.jetson.example"
)

# Parse arguments
DRY_RUN=false
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run]"
            echo ""
            echo "Sync the jetson branch with main while preserving jetson-specific files."
            echo ""
            echo "Options:"
            echo "  --dry-run    Show what would be done without making changes"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
    esac
done

echo -e "${BLUE}=== MAIE Jetson Branch Sync ===${NC}"
echo ""

# Ensure we're in a git repository
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo -e "${RED}Error: Not inside a git repository${NC}"
    exit 1
fi

# Get the current branch
current_branch=$(git branch --show-current)
echo -e "Current branch: ${GREEN}${current_branch}${NC}"

# Ensure we're on the jetson branch
if [[ "$current_branch" != "jetson" ]]; then
    echo -e "${RED}Error: Must be on 'jetson' branch. Currently on '${current_branch}'${NC}"
    echo ""
    echo "To switch to the jetson branch:"
    echo "  git checkout jetson"
    exit 1
fi

# Check for uncommitted changes
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo -e "${YELLOW}Warning: You have uncommitted changes${NC}"
    echo ""
    git status --short
    echo ""
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}[DRY RUN MODE - No changes will be made]${NC}"
    echo ""
fi

# Fetch latest main
echo -e "${BLUE}Fetching latest main branch...${NC}"
if [[ "$DRY_RUN" != "true" ]]; then
    git fetch origin main
fi

# Show what's new in main
echo ""
echo -e "${BLUE}Commits in main not in jetson:${NC}"
git log --oneline jetson..origin/main | head -20
echo ""

# Count commits
commit_count=$(git rev-list --count jetson..origin/main)
if [[ "$commit_count" -eq 0 ]]; then
    echo -e "${GREEN}Already up to date with main!${NC}"
    exit 0
fi
echo -e "Found ${YELLOW}${commit_count}${NC} new commit(s) in main"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}[DRY RUN] Would merge origin/main into jetson${NC}"
    echo ""
    echo "Files that would be preserved:"
    for file in "${JETSON_SPECIFIC_FILES[@]}"; do
        if [[ -f "$file" ]]; then
            echo -e "  ${GREEN}✓${NC} $file"
        else
            echo -e "  ${YELLOW}○${NC} $file (not present)"
        fi
    done
    echo ""
    echo "To perform the actual sync, run without --dry-run:"
    echo "  $0"
    exit 0
fi

# Start merge without committing
echo -e "${BLUE}Starting merge...${NC}"
if ! git merge origin/main --no-commit --no-ff; then
    echo ""
    echo -e "${YELLOW}Merge has conflicts. This is expected if both branches modified the same files.${NC}"
fi

# Preserve jetson-specific files
echo ""
echo -e "${BLUE}Preserving jetson-specific files...${NC}"
for file in "${JETSON_SPECIFIC_FILES[@]}"; do
    if git ls-files --error-unmatch "$file" &>/dev/null 2>&1; then
        echo -e "  Keeping: ${GREEN}$file${NC}"
        git checkout --ours "$file" 2>/dev/null || true
        git add "$file" 2>/dev/null || true
    elif [[ -f "$file" ]]; then
        echo -e "  Keeping (untracked): ${YELLOW}$file${NC}"
    fi
done

# Show status
echo ""
echo -e "${BLUE}=== Merge Status ===${NC}"
git status --short

# Check for remaining conflicts
conflict_count=$(git diff --name-only --diff-filter=U | wc -l)
if [[ "$conflict_count" -gt 0 ]]; then
    echo ""
    echo -e "${YELLOW}⚠️  There are ${conflict_count} file(s) with conflicts:${NC}"
    git diff --name-only --diff-filter=U
fi

echo ""
echo -e "${BLUE}=== Next Steps ===${NC}"
echo ""
echo "1. Review the changes:"
echo -e "   ${GREEN}git diff --cached${NC}"
echo ""
if [[ "$conflict_count" -gt 0 ]]; then
    echo "2. Resolve the ${conflict_count} remaining conflict(s):"
    echo -e "   ${GREEN}# Edit the conflicting files, then:${NC}"
    echo -e "   ${GREEN}git add <resolved-files>${NC}"
    echo ""
    echo "3. Commit the merge:"
else
    echo "2. Commit the merge:"
fi
echo -e "   ${GREEN}git commit -m 'chore: sync with main branch'${NC}"
echo ""
echo "To abort the merge:"
echo -e "   ${RED}git merge --abort${NC}"
echo ""
echo -e "${GREEN}Done! Review the changes and commit when ready.${NC}"
