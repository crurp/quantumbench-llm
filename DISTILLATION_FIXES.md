# Knowledge Distillation Training Script - Fixes Applied

## Summary

Fixed numerical stability issues in `distill_quantumbench.py` that were causing NaN gradients and zero loss during training.

## Fixes Applied

### 1. ✅ Gradient Clipping
- **Added**: `max_grad_norm=1.0` in `TrainingArguments`
- **Location**: Line ~520
- **Impact**: Prevents gradient explosion that causes NaN

### 2. ✅ NaN Gradient Detection & Fixes
- **Added**: Comprehensive NaN/Inf checks in `compute_loss()`:
  - Check student logits for NaN/Inf before loss computation
  - Check teacher logits for NaN/Inf before loss computation
  - Check logit shapes match between teacher and student
  - Check distillation loss for NaN/Inf
  - Check cross-entropy loss for NaN/Inf
  - Check final combined loss for NaN/Inf
- **Fixed**: KL divergence implementation:
  - Explicitly using `log_target=False` for clarity
  - Ensured both log_softmax and softmax are applied correctly
  - Temperature^2 scaling already present (maintained)

### 3. ✅ KD Loss Scaling
- **Status**: Already correct - temperature^2 scaling maintained
- **Location**: Line ~230
- **Formula**: `loss_distill * (temperature ** 2)`

### 4. ✅ Learning Rate Adjustments
- **Changed**: Default learning rate from 5e-5 to 2e-5
- **Added**: Warmup set to 10% of total steps (was fixed 50 steps)
- **Location**: Line ~490-495
- **Calculation**: `warmup_steps = max(1, int(total_steps * 0.1))`

### 5. ✅ CPU Optimization
- **Fixed**: `fp16=False` by default for CPU
- **Added**: Automatic detection - FP16 only used if:
  - `--fp16` flag is explicitly set AND
  - Running on CUDA GPU (not CPU)
- **Changed**: Model loading uses `torch.float32` for CPU, `torch.float16` only when explicitly requested for GPU
- **Location**: Lines ~365-370, ~395-405, ~450-455

### 6. ✅ Balanced Alpha Weighting
- **Changed**: Default alpha from 0.7 to 0.5 (50/50 split)
- **Location**: Line ~400
- **Impact**: Balanced weighting between distillation loss and cross-entropy loss

### 7. ✅ Sanity Check Features

#### 7a. Dry Run Mode
- **Added**: `--dry-run` flag
- **Behavior**: 
  - Sets `max_steps=5`
  - Sets `eval_steps=5`
  - Sets `save_steps=5`
  - Overrides epochs to 1
- **Location**: Lines ~415-425, ~500-510

#### 7b. NaN Detector Callback
- **Added**: `NaNDetectorCallback` class
- **Functionality**: 
  - Checks loss at every step
  - Raises `ValueError` immediately if NaN detected
  - Provides descriptive error message with troubleshooting tips
- **Location**: Lines ~103-120
- **Integration**: Automatically added to trainer (Line ~540)

#### 7c. Logit Shape Check
- **Added**: Automatic shape checking on first forward pass
- **Output**: Prints to console:
  - Student logits shape
  - Teacher logits shape
  - Temperature value
  - Alpha (distillation weight)
- **Location**: Lines ~135-150
- **Validation**: Ensures shapes match before loss computation

#### 7d. Loss Component Printing
- **Added**: Prints individual loss components for first 5 steps
- **Output**: For each of first 5 steps:
  - Distillation loss value
  - Cross-entropy loss value
  - Combined loss value
  - Temperature and Temperature^2
- **Location**: Lines ~260-270

## Usage

### Standard Training
```bash
python scripts/distill_quantumbench.py \
    --teacher-model models/teacher-model \
    --student-model-name gpt2 \
    --data-path data/quantumbench.jsonl \
    --epochs 3 \
    --batch-size 2 \
    --learning-rate 2e-5
```

### Dry Run (Recommended First Step)
```bash
python scripts/distill_quantumbench.py \
    --teacher-model models/teacher-model \
    --student-model-name gpt2 \
    --data-path data/quantumbench.jsonl \
    --dry-run
```

### With FP16 (GPU Only)
```bash
python scripts/distill_quantumbench.py \
    --teacher-model models/teacher-model \
    --student-model-name gpt2 \
    --data-path data/quantumbench.jsonl \
    --fp16
```

## Expected Output

### First Forward Pass
```
======================================================================
LOGIT SHAPE CHECK (First Forward Pass)
======================================================================
Student logits shape: torch.Size([batch_size, seq_len, vocab_size])
Teacher logits shape: torch.Size([batch_size, seq_len, vocab_size])
Temperature: 3.0
Alpha (distillation weight): 0.5, CE weight: 0.5
======================================================================
```

### First 5 Steps
```
[Step 0] Loss Components:
  Distillation Loss (α=0.5): 2.345678
  Cross-Entropy Loss (1-α=0.5): 4.567890
  Combined Loss: 3.456784
  Temperature: 3.0, Temperature^2: 9.0

[Step 1] Loss Components:
  ...
```

### Training Configuration
```
Training Configuration:
  Learning rate: 2e-05
  Warmup steps: 42 (10% of 420 total steps)
  Temperature: 3.0
  Alpha (distillation weight): 0.5, CE weight: 0.5
  Max grad norm: 1.0 (gradient clipping)
  FP16: False
  Device: cpu
```

## Error Handling

If NaN is detected, training will stop immediately with a clear error message:

```
======================================================================
❌ TRAINING STOPPED DUE TO NUMERICAL INSTABILITY
======================================================================
NaN loss detected at step 123! Loss value: nan. Training stopped...
Suggested fixes:
  1. Lower learning rate (try 1e-5 or lower)
  2. Reduce batch size
  3. Check input data for invalid values
  4. Ensure teacher model is valid
  5. Try --dry-run first to diagnose
======================================================================
```

## Testing Recommendations

1. **Always run dry-run first**: `--dry-run` to verify setup
2. **Monitor first 5 steps**: Check loss components are reasonable
3. **Verify logit shapes**: Ensure shapes match in first forward pass
4. **Watch for NaN warnings**: NaN detector will catch issues immediately
5. **Check gradient norms**: Should be < 1.0 due to clipping

## Technical Details

### Loss Computation
```python
# Student log probabilities (log_softmax)
student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

# Teacher probabilities (softmax)
teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

# KL divergence with temperature^2 scaling
loss_distill = F.kl_div(
    student_log_probs,
    teacher_probs,
    reduction='batchmean',
    log_target=False
) * (temperature ** 2)

# Cross-entropy loss
loss_ce = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)

# Combined (balanced 50/50)
loss = 0.5 * loss_distill + 0.5 * loss_ce
```

### Numerical Stability Measures
1. **NaN checks**: Every tensor checked before and after operations
2. **Shape validation**: Ensures compatibility before computation
3. **Gradient clipping**: Prevents explosion
4. **FP32 for CPU**: Avoids precision issues on CPU
5. **Temperature scaling**: Maintains gradient magnitudes

## Files Modified

- `scripts/distill_quantumbench.py` - Main training script with all fixes

## Next Steps

1. Test with `--dry-run` first
2. Monitor loss components in first 5 steps
3. If successful, proceed with full training
4. Watch for NaN detector alerts
5. Adjust learning rate or batch size if needed

