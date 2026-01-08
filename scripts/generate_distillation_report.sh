#!/bin/bash
# generate_distillation_report.sh
#
# Monitors student training and automatically generates comprehensive
# distillation performance metrics once training completes.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Configuration
TEACHER_MODEL="models/teacher-model"
STUDENT_MODEL="models/student-model"
DATA_PATH="data/quantumbench.jsonl"
OUTPUT_METRICS="metrics/distillation_metrics.json"
TRAINING_LOG="training_student.log"
CHECK_INTERVAL=300  # Check every 5 minutes

echo "======================================================================"
echo "DISTILLATION PERFORMANCE METRICS GENERATOR"
echo "======================================================================"
echo ""
echo "This script will:"
echo "  1. Monitor student training completion"
echo "  2. Generate comprehensive performance metrics"
echo "  3. Create detailed comparison report"
echo ""

# Check if training is running
check_training_status() {
    ps aux | grep -E "[p]ython.*distill" > /dev/null 2>&1
    return $?
}

# Wait for training to complete
echo "Monitoring training status..."
while check_training_status; do
    CURRENT_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    LATEST_STEP=$(grep -E "[0-9]+/[0-9]+" "$TRAINING_LOG" 2>/dev/null | tail -1 | grep -oE "[0-9]+/[0-9]+" | head -1 || echo "N/A")
    echo "[$CURRENT_TIME] Training in progress... Latest: $LATEST_STEP (checking again in $CHECK_INTERVAL seconds)"
    sleep $CHECK_INTERVAL
done

echo ""
echo "✓ Training appears to have completed!"
echo "  Waiting 30 seconds for final log writes..."
sleep 30

# Verify training completed successfully
if grep -q "Training completed\|Saving model\|Training finished" "$TRAINING_LOG" 2>/dev/null; then
    echo "✓ Training log confirms completion"
else
    echo "⚠ Warning: Training log doesn't show explicit completion message"
    echo "  Proceeding with metrics generation anyway..."
fi

echo ""
echo "======================================================================"
echo "GENERATING DISTILLATION PERFORMANCE METRICS"
echo "======================================================================"
echo ""

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Warning: Virtual environment not found at venv/bin/activate"
fi

# Generate metrics
echo "Running comprehensive metrics generation..."
python scripts/distillation_performance_metrics.py \
    --teacher-model "$TEACHER_MODEL" \
    --student-model "$STUDENT_MODEL" \
    --data-path "$DATA_PATH" \
    --output "$OUTPUT_METRICS" \
    --benchmark-samples 10

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "✓ METRICS GENERATION COMPLETED SUCCESSFULLY"
    echo "======================================================================"
    echo ""
    echo "Results saved to: $OUTPUT_METRICS"
    echo ""
    echo "To view the metrics:"
    echo "  cat $OUTPUT_METRICS | python -m json.tool"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "✗ METRICS GENERATION FAILED"
    echo "======================================================================"
    echo ""
    echo "Please check the error messages above and ensure:"
    echo "  1. Both teacher and student models exist"
    echo "  2. Training has completed successfully"
    echo "  3. All dependencies are installed"
    exit 1
fi

# Also run the full comparison script
echo "Running additional comparison analysis..."
python scripts/compare_teacher_student.py \
    --teacher-model "$TEACHER_MODEL" \
    --student-model "$STUDENT_MODEL" \
    --data-path "$DATA_PATH" \
    --output "metrics/comparison_results.json"

echo ""
echo "======================================================================"
echo "DISTILLATION ANALYSIS COMPLETE"
echo "======================================================================"
echo ""
echo "Generated files:"
echo "  - $OUTPUT_METRICS (comprehensive metrics)"
echo "  - metrics/comparison_results.json (detailed comparison)"
echo ""

