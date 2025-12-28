#!/bin/bash

################################################################################
# setup.sh
#
# Complete setup script for QuantumBench LLM project.
# This script:
# 1. Creates folder structure
# 2. Sets up Python virtual environment
# 3. Installs dependencies
# 4. Initializes Git repository
# 5. Sets up remote origin
# 6. Makes initial commit and push
#
# Usage:
#   bash setup.sh
#   or
#   chmod +x setup.sh && ./setup.sh
################################################################################

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root directory (where this script is located)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}=== QuantumBench LLM Setup ===${NC}"
echo ""

# Step 1: Create folder structure
echo -e "${YELLOW}Step 1: Creating folder structure...${NC}"
mkdir -p data scripts notebooks models
echo -e "${GREEN}✓ Folder structure created${NC}"
echo ""

# Step 2: Create Python virtual environment
echo -e "${YELLOW}Step 2: Setting up Python virtual environment...${NC}"
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping creation..."
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Step 3: Upgrade pip
echo -e "${YELLOW}Step 3: Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ Pip upgraded${NC}"
echo ""

# Step 4: Install dependencies
echo -e "${YELLOW}Step 4: Installing dependencies from requirements.txt...${NC}"
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found!${NC}"
    exit 1
fi

pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Step 5: Make scripts executable
echo -e "${YELLOW}Step 5: Making scripts executable...${NC}"
chmod +x scripts/*.py
echo -e "${GREEN}✓ Scripts made executable${NC}"
echo ""

# Step 6: Initialize Git repository
echo -e "${YELLOW}Step 6: Initializing Git repository...${NC}"
if [ -d ".git" ]; then
    echo "Git repository already initialized, skipping..."
else
    git init
    echo -e "${GREEN}✓ Git repository initialized${NC}"
fi

# Step 7: Check if remote origin exists, add if not
echo -e "${YELLOW}Step 7: Setting up Git remote...${NC}"
REMOTE_URL="https://github.com/crurp/quantumbench-llm.git"
if git remote get-url origin &>/dev/null; then
    echo "Remote 'origin' already exists: $(git remote get-url origin)"
    echo "Updating to: $REMOTE_URL"
    git remote set-url origin "$REMOTE_URL"
else
    git remote add origin "$REMOTE_URL"
    echo -e "${GREEN}✓ Remote origin added${NC}"
fi

# Step 8: Add all files and make initial commit
echo -e "${YELLOW}Step 8: Staging files for commit...${NC}"
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "No changes to commit (repository might already be up to date)"
else
    echo -e "${YELLOW}Step 9: Creating initial commit...${NC}"
    git commit -m "Initial commit: QuantumBench LLM project setup

- Created folder structure (data/, scripts/, notebooks/, models/)
- Added Python scripts for data preparation, training, distillation, and evaluation
- Set up .gitignore for Python, data files, and models
- Added requirements.txt with all dependencies
- Configured for QLoRA/PEFT fine-tuning with 16GB RAM optimizations"
    echo -e "${GREEN}✓ Initial commit created${NC}"
    echo ""
    
    # Step 10: Push to remote (if main branch exists or create it)
    echo -e "${YELLOW}Step 10: Pushing to remote repository...${NC}"
    CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
    
    # Create main branch if it doesn't exist
    if ! git show-ref --verify --quiet refs/heads/main; then
        if [ "$CURRENT_BRANCH" != "main" ]; then
            git branch -M main
            CURRENT_BRANCH="main"
        fi
    fi
    
    echo "Pushing to origin/$CURRENT_BRANCH..."
    # Note: This will fail if remote doesn't exist yet, which is expected
    # User can push manually after creating the GitHub repo
    if git push -u origin "$CURRENT_BRANCH" 2>/dev/null; then
        echo -e "${GREEN}✓ Pushed to remote repository${NC}"
    else
        echo -e "${YELLOW}⚠ Could not push to remote (this is normal if the GitHub repo doesn't exist yet)${NC}"
        echo "   You can push manually after creating the repository on GitHub:"
        echo "   git push -u origin $CURRENT_BRANCH"
    fi
fi

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Prepare your QuantumBench data:"
echo "   python scripts/prepare_quantumbench.py --input data/your_data.csv --output data/quantumbench.jsonl"
echo "3. Train the teacher model:"
echo "   python scripts/train_quantumbench.py --model-name meta-llama/Llama-2-7b-hf --data-path data/quantumbench.jsonl"
echo "4. Distill to student model:"
echo "   python scripts/distill_quantumbench.py --teacher-model models/teacher-model"
echo "5. Evaluate models:"
echo "   python scripts/evaluate_quantumbench.py --teacher-model models/teacher-model --student-model models/student-model"
echo ""
echo "Note: Make sure to activate the virtual environment before running any scripts!"
echo "      source venv/bin/activate"

