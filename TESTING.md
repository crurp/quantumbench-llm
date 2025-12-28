# QuantumBench LLM Testing and Visualization Guide

Complete guide to testing and visualizing the QuantumBench LLM workflow.

## Overview

The testing suite includes scripts for:
- Dataset verification
- Model evaluation (teacher and student)
- Teacher-student comparison and metrics
- Comprehensive visualizations
- System checks and memory testing
- Full workflow orchestration

## Scripts

### 1. `test_dataset.py` - Dataset Verification

Verifies and displays information about the prepared JSONL dataset.

**Usage:**
```bash
python scripts/test_dataset.py --data-path data/quantumbench.jsonl --num-samples 5
```

**Features:**
- Loads and validates JSONL format
- Displays first N examples
- Shows dataset statistics (size, average lengths, etc.)
- Verifies required fields ('instruction', 'response')

### 2. `test_models.py` - Model Evaluation

Evaluates teacher or student models on QuantumBench questions.

**Usage:**
```bash
# Evaluate teacher model
python scripts/test_models.py \
    --model-path models/teacher-model \
    --model-type teacher \
    --data-path data/quantumbench.jsonl \
    --num-samples 5

# Evaluate student model
python scripts/test_models.py \
    --model-path models/student-model \
    --model-type student \
    --data-path data/quantumbench.jsonl \
    --num-samples 5
```

**Features:**
- Runs inference on sample questions
- Computes accuracy
- Shows predicted vs correct answers
- Supports full dataset evaluation (omit `--num-samples`)

### 3. `compare_teacher_student.py` - Comparison and Metrics

Compares teacher and student models and computes interaction metrics.

**Usage:**
```bash
python scripts/compare_teacher_student.py \
    --teacher-model models/teacher-model \
    --student-model models/student-model \
    --data-path data/quantumbench.jsonl \
    --output comparison_results.json
```

**Metrics Computed:**
- Teacher accuracy
- Student accuracy
- Delta accuracy (teacher - student)
- Exact match rate (predictions match exactly)
- Consistency (proportion of matching predictions)
- Student agreement when teacher is correct
- Student correct when teacher is wrong
- F1 scores

**Output:**
- Detailed summary table
- Per-example comparisons
- JSON file with all metrics (optional)

### 4. `visualize_results.py` - Visualizations

Generates comprehensive visualizations from comparison results.

**Usage:**
```bash
python scripts/visualize_results.py \
    --comparison-results plots/comparison_results.json \
    --output-dir plots/
```

**Visualizations Generated:**
1. **accuracy_comparison.png** - Bar chart comparing teacher vs student accuracy
2. **delta_accuracy.png** - Bar chart showing accuracy delta
3. **comprehensive_metrics.png** - Multi-panel chart with all key metrics
4. **confusion_matrices.png** - Confusion matrices for both models
5. **prediction_heatmap.png** - Heatmap showing prediction agreement across examples
6. **delta_across_examples.png** - Line plot showing cumulative accuracy and delta

### 5. `system_check.py` - System Checks

Verifies system setup, dependencies, and tests memory usage.

**Usage:**
```bash
# Basic checks
python scripts/system_check.py

# Include training step test (takes longer)
python scripts/system_check.py --test-training
```

**Checks Performed:**
- Required folder structure (data/, scripts/, models/, plots/, notebooks/)
- Model existence (teacher and student)
- System memory (RAM, CPU, GPU)
- Python dependencies
- Optional: Single training step test (memory and pipeline)

### 6. `test_full_workflow.py` - Complete Workflow Test

Orchestrates all tests and generates complete analysis.

**Usage:**
```bash
# Full workflow test
python scripts/test_full_workflow.py \
    --data-path data/quantumbench.jsonl

# Test on subset of data (faster)
python scripts/test_full_workflow.py \
    --data-path data/quantumbench.jsonl \
    --num-samples 50

# Save full results
python scripts/test_full_workflow.py \
    --data-path data/quantumbench.jsonl \
    --save-results results/full_test_results.json
```

**Workflow Steps:**
1. System checks (folders, dependencies, memory)
2. Dataset verification
3. Teacher model evaluation
4. Student model evaluation
5. Comparison metrics computation
6. Visualization generation

**Output:**
- All visualizations in `plots/` directory
- Comparison results JSON file
- Full results JSON (if `--save-results` specified)
- Summary printed to console

## Quick Start

### 1. Verify Setup
```bash
python scripts/system_check.py
```

### 2. Verify Dataset
```bash
python scripts/test_dataset.py --data-path data/quantumbench.jsonl
```

### 3. Run Complete Test Suite
```bash
python scripts/test_full_workflow.py --data-path data/quantumbench.jsonl
```

This will:
- Verify system setup
- Check dataset
- Evaluate both models
- Compute all metrics
- Generate all visualizations
- Save results

### 4. View Results

All visualizations are saved to `plots/`:
- `accuracy_comparison.png` - Quick accuracy comparison
- `comprehensive_metrics.png` - All metrics at a glance
- `confusion_matrices.png` - Model performance details
- `prediction_heatmap.png` - Agreement analysis
- `delta_across_examples.png` - Performance over dataset

## Metrics Explained

### Accuracy
- **Teacher Accuracy**: Percentage of correct predictions by teacher model
- **Student Accuracy**: Percentage of correct predictions by student model

### Delta
- **Delta Accuracy**: Difference between teacher and student accuracy
  - Positive: Teacher performs better
  - Negative: Student performs better (rare)

### Match Rate
- **Exact Match Rate**: Proportion of examples where teacher and student predictions match exactly
- **Consistency**: Same as exact match rate, measures prediction agreement

### Agreement Metrics
- **Student Agreement (Teacher Correct)**: When teacher is correct, how often does student agree?
- **Student Correct (Teacher Wrong)**: When teacher is wrong, how often is student correct?

These metrics help understand:
- How well knowledge was transferred (low delta is good)
- Prediction consistency (high match rate is good)
- Student's ability to correct teacher mistakes (student correct when teacher wrong)

## Example Output

```
======================================================================
QUANTUMBENCH LLM FULL WORKFLOW TEST
======================================================================

[1/6] Running system checks...
✓ System checks complete

[2/6] Verifying dataset...
✓ Dataset verification complete

[3/6] Evaluating teacher model...
✓ Teacher evaluation complete

[4/6] Evaluating student model...
✓ Student evaluation complete

[5/6] Computing comparison metrics...
✓ Comparison metrics saved to plots/comparison_results.json

[6/6] Generating visualizations...
✓ All visualizations generated successfully!

======================================================================
WORKFLOW TEST SUMMARY
======================================================================
Teacher Model Accuracy: 0.8500 (85.00%)
Student Model Accuracy: 0.8200 (82.00%)
Accuracy Delta: +0.0300 (+3.00%)
Exact Match Rate: 0.7500 (75.00%)

Visualizations saved to: plots
======================================================================
```

## Troubleshooting

### Models Not Found
If models haven't been trained yet:
```bash
# Train teacher model first
python scripts/train_quantumbench.py --data-path data/quantumbench.jsonl

# Then train student model
python scripts/distill_quantumbench.py --teacher-model models/teacher-model
```

### Memory Issues
If you run out of memory:
- Use `--num-samples` to test on a subset
- Close other applications
- Use smaller batch sizes in training scripts

### Missing Dependencies
Install all dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Visualization Errors
Ensure matplotlib and seaborn are installed:
```bash
pip install matplotlib seaborn
```

## Advanced Usage

### Custom Evaluation
```bash
# Evaluate specific number of examples
python scripts/test_models.py \
    --model-path models/teacher-model \
    --num-samples 100 \
    --max-new-tokens 512
```

### Compare on Different Datasets
```bash
python scripts/compare_teacher_student.py \
    --teacher-model models/teacher-model \
    --student-model models/student-model \
    --data-path data/test_set.jsonl
```

### Generate Only Specific Visualizations
Edit `visualize_results.py` to comment out unwanted plots, or create a custom script.

## Output Files

All outputs are saved to:
- `plots/` - All visualization images
- `plots/comparison_results.json` - Full comparison metrics
- `results/` - Optional full test results (if specified)

## Integration with Training

The test scripts work seamlessly with the training workflow:

1. Prepare data: `prepare_quantumbench.py`
2. Train teacher: `train_quantumbench.py`
3. Distill student: `distill_quantumbench.py`
4. **Test everything**: `test_full_workflow.py`

This provides a complete pipeline from data preparation to evaluation and visualization.

