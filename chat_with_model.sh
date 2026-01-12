#!/bin/bash
# chat_with_model.sh
#
# Simple launcher script for the interactive distilled model interface.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "🚀 QUANTUM COMPUTING ASSISTANT - LAUNCHING"
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
if [ ! -d "models/student-model" ]; then
    echo ""
    echo "❌ Error: Student model not found at models/student-model"
    echo "   Please ensure training has completed."
    echo ""
    exit 1
fi

echo "✓ Model found at models/student-model"
echo ""
echo "Loading model (this may take 30-60 seconds)..."
echo ""

# Run the interactive script
python3 scripts/interactive_distilled_model.py --model-path models/student-model "$@"

