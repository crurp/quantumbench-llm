# Knowledge Distillation Performance Metrics

This document describes the comprehensive performance metrics that will be computed once training completes.

## Overview

The `distillation_performance_metrics.py` script provides a complete analysis of the knowledge distillation process, comparing the teacher and student models across multiple dimensions:

1. **Model Size Metrics**
2. **Accuracy Metrics**
3. **Inference Performance Metrics**
4. **Memory Usage Metrics**
5. **Knowledge Retention Metrics**
6. **Efficiency Scores**

## Metrics Provided

### 1. Model Size Comparison

- **Disk Size**: Physical storage requirements (MB, GB)
- **Parameter Count**: Total and trainable parameters
- **Size Reduction**: Absolute and percentage reduction
- **Compression Ratio**: Ratio of student to teacher size

### 2. Accuracy Metrics

- **Teacher Accuracy**: Accuracy on QuantumBench test set
- **Student Accuracy**: Accuracy on QuantumBench test set
- **Accuracy Delta**: Difference (Teacher - Student)
- **Accuracy Retention**: Percentage of teacher accuracy retained
- **Accuracy Loss**: Negative delta (how much accuracy was lost)

### 3. Inference Performance

- **Inference Time**: Average time per sample
- **Load Time**: Model loading time
- **Throughput**: Samples per second
- **Token Generation Speed**: Tokens per second
- **Speed Improvement**: Time saved by using student model

### 4. Memory Usage

- **Model Memory**: RAM used by loaded model
- **Peak Memory**: Maximum memory during inference
- **Memory Reduction**: Memory saved by using student model

### 5. Knowledge Retention Metrics

- **Knowledge Retention %**: `(Student Accuracy / Teacher Accuracy) * 100`
- **Performance Efficiency**: Accuracy per parameter ratio
- **Distillation Efficacy**: Overall success of knowledge transfer

### 6. Efficiency Scores

- **Efficiency Score**: Accuracy normalized by parameter ratio
- **Performance per MB**: Accuracy divided by model size
- **Performance per Million Params**: Accuracy normalized by parameter count

## Usage

### Automatic Generation (Recommended)

Once training completes, run:

```bash
bash scripts/generate_distillation_report.sh
```

This script will:
1. Monitor training completion
2. Automatically generate all metrics
3. Create comparison analysis
4. Save results to JSON files

### Manual Generation

```bash
source venv/bin/activate

python scripts/distillation_performance_metrics.py \
    --teacher-model models/teacher-model \
    --student-model models/student-model \
    --data-path data/quantumbench.jsonl \
    --benchmark-samples 10 \
    --output metrics/distillation_metrics.json
```

### Viewing Results

The script generates a comprehensive console report and saves detailed JSON metrics:

```bash
# View JSON metrics
cat metrics/distillation_metrics.json | python -m json.tool

# View console report (shown during execution)
```

## Output Files

### 1. `metrics/distillation_metrics.json`

Comprehensive metrics in JSON format:
- `teacher`: Teacher model metrics (size, accuracy, benchmark)
- `student`: Student model metrics (size, accuracy, benchmark)
- `distillation`: Comparison and efficiency metrics

### 2. `metrics/comparison_results.json`

Detailed comparison from `compare_teacher_student.py`:
- Prediction-by-prediction comparison
- Exact match rates
- Consistency metrics
- F1 scores

## Expected Metrics (Example)

Based on typical distillation results:

```
Model Size:
  Teacher: ~500-1000 MB
  Student: ~500 MB
  Reduction: 50-80%

Accuracy:
  Teacher: 70-90%
  Student: 60-85%
  Retention: 85-95%

Speed:
  Teacher: 2-5 seconds/sample
  Student: 1-3 seconds/sample
  Improvement: 30-50%

Efficiency:
  Student achieves 85-95% of teacher accuracy
  with 50-80% fewer parameters
  and 30-50% faster inference
```

## Interpreting Results

### Good Distillation:
- **High Retention (>85%)**: Student maintains most teacher knowledge
- **Significant Size Reduction (>50%)**: Student is substantially smaller
- **Speed Improvement (>30%)**: Student is noticeably faster
- **High Efficiency Score**: Good accuracy-to-size ratio

### Poor Distillation:
- **Low Retention (<70%)**: Student lost significant knowledge
- **Minimal Size Reduction (<30%)**: Not much compression benefit
- **No Speed Improvement**: Student not faster than teacher
- **Low Efficiency Score**: Poor trade-off

## Troubleshooting

### Model Loading Errors

If you see PEFT loading errors:
- The script automatically detects PEFT/LoRA models
- It loads the base model first, then applies adapters
- Ensure the base model is accessible (may need to download)

### Memory Issues

If running out of memory:
- Reduce `--benchmark-samples` (default: 10)
- Use CPU mode (slower but uses less memory)
- Close other applications

### Slow Execution

The metrics generation:
- Evaluates on full dataset (can take 30-60 minutes)
- Benchmarks inference speed (10 samples)
- Can take 1-2 hours total on CPU

## Next Steps

After generating metrics:

1. Review the console report
2. Check `metrics/distillation_metrics.json` for detailed numbers
3. Generate visualizations: `python scripts/visualize_results.py`
4. Compare with other distillation experiments
5. Document findings in project report

## References

- Knowledge Distillation: Hinton et al. (2015)
- QLoRA: Dettmers et al. (2023)
- PEFT: Hu et al. (2021)

