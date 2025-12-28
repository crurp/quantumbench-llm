# Training Setup Instructions

## Prerequisites for LLaMA-2-7B Training

### 1. Hugging Face Access

LLaMA-2-7B is a gated model on Hugging Face. You need to:

1. **Request Access:**
   - Go to: https://huggingface.co/meta-llama/Llama-2-7b-hf
   - Click "Request access" and fill out the form
   - Wait for approval (usually within a few hours)

2. **Login to Hugging Face:**
   ```bash
   pip install huggingface_hub[cli]
   huggingface-cli login
   ```
   Enter your Hugging Face access token (create at https://huggingface.co/settings/tokens)

3. **Alternative: Set Token in Environment:**
   ```bash
   export HUGGING_FACE_HUB_TOKEN=your_token_here
   ```

### 2. System Requirements

- **GPU Recommended:** Training LLaMA-2-7B on CPU is extremely slow (days/weeks)
- **RAM:** At least 16GB (32GB+ recommended)
- **Disk Space:** ~15GB for model weights

### 3. Training Command

Once authenticated, run:

```bash
source venv/bin/activate
python scripts/train_quantumbench.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --data-path data/quantumbench.jsonl \
    --output-dir models/teacher-model \
    --epochs 3 \
    --batch-size 1 \
    --gradient-accumulation-steps 8
```

### 4. Expected Training Time

- **With GPU (16GB+ VRAM):** 2-4 hours
- **With GPU (8GB VRAM):** 4-8 hours  
- **CPU only:** Days to weeks (not recommended)

### 5. Monitor Training

```bash
# View training progress
tail -f training_teacher.log

# Check GPU usage (if available)
nvidia-smi
```

## Current Status

✅ **Dataset:** 104 examples ready in `data/quantumbench.jsonl`  
✅ **Scripts:** All training scripts ready  
⏳ **Training:** Pending Hugging Face authentication  
⏳ **Models:** Teacher and student models to be trained  

## Next Steps After Training

1. **Train Teacher Model** (see above)
2. **Train Student Model:**
   ```bash
   python scripts/distill_quantumbench.py \
       --teacher-model models/teacher-model \
       --student-model-name gpt2 \
       --data-path data/quantumbench.jsonl
   ```
3. **Run Full Evaluation:**
   ```bash
   python scripts/test_full_workflow.py --data-path data/quantumbench.jsonl
   ```

