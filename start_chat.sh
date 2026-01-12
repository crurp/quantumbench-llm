#!/bin/bash
# start_chat.sh
#
# Quick launcher for the distilled model interactive interface

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="models/student-model-fixed"

echo "======================================================================"
echo "🚀 QUANTUM COMPUTING ASSISTANT - DISTILLED MODEL"
echo "======================================================================"
echo ""

# Check if virtual environment exists
if [ -f "venv/bin/activate" ]; then
    echo "✓ Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠ Warning: Virtual environment not found. Using system Python."
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo ""
    echo "❌ Error: Distilled model not found at $MODEL_PATH"
    echo "   Please ensure training has completed."
    echo ""
    exit 1
fi

echo "✓ Model found at $MODEL_PATH"
echo ""
echo "Loading model (this may take 30-60 seconds on CPU)..."
echo ""

# Run the interactive script with the fixed model
python3 scripts/interactive_distilled_model.py --model-path "$MODEL_PATH" "$@"

