# QuantumBench LLM

A complete workflow for fine-tuning language models on QuantumBench using QLoRA (Quantized Low-Rank Adaptation) and knowledge distillation.

## Project Structure

```
quantumbench-llm/
├── data/              # QuantumBench datasets (CSV/JSON/JSONL)
├── scripts/           # Python scripts for data preparation, training, and evaluation
├── notebooks/         # Jupyter notebooks for experimentation
├── models/            # Saved model checkpoints (teacher and student)
├── venv/              # Python virtual environment (created by setup.sh)
├── requirements.txt   # Python dependencies
├── setup.sh          # Automated setup script
└── README.md         # This file
```

## Quick Start

### 1. Run Setup Script

```bash
bash setup.sh
```

This will:
- Create the folder structure
- Set up a Python virtual environment
- Install all dependencies
- Initialize Git repository
- Set up remote origin
- Make initial commit

### 2. Activate Virtual Environment

```bash
source venv/bin/activate
```

### 3. Prepare Data

Convert your QuantumBench CSV or JSON files to JSONL format:

```bash
python scripts/prepare_quantumbench.py \
    --input data/quantumbench.csv \
    --output data/quantumbench.jsonl
```

### 4. Train Teacher Model

Fine-tune a large model (e.g., LLaMA-2-7B or Qwen-7B) using QLoRA:

```bash
python scripts/train_quantumbench.py \
    --model-name meta-llama/Llama-2-7b-hf \
    --data-path data/quantumbench.jsonl \
    --output-dir models/teacher-model \
    --epochs 3
```

**Note:** The script automatically:
- Checks if the model is already downloaded and skips if present
- Uses 4-bit quantization for memory efficiency
- Applies gradient checkpointing for 16GB RAM systems
- Saves the model to avoid re-downloading

### 5. Distill to Student Model

Perform knowledge distillation to create a smaller, faster student model:

```bash
python scripts/distill_quantumbench.py \
    --teacher-model models/teacher-model \
    --student-model-name gpt2 \
    --data-path data/quantumbench.jsonl \
    --output-dir models/student-model \
    --epochs 5
```

### 6. Evaluate Models

Evaluate both teacher and student models:

```bash
python scripts/evaluate_quantumbench.py \
    --teacher-model models/teacher-model \
    --student-model models/student-model \
    --data-path data/quantumbench.jsonl
```

## Scripts Overview

### `prepare_quantumbench.py`

Converts QuantumBench CSV/JSON files into JSONL format suitable for instruction-following fine-tuning.

**Features:**
- Supports CSV and JSON input formats
- Automatically detects question/answer columns
- Formats data as instruction-response pairs
- Handles various column naming conventions

### `train_quantumbench.py`

Fine-tunes a language model using QLoRA and PEFT.

**Features:**
- 4-bit quantization for memory efficiency
- LoRA adapters for parameter-efficient fine-tuning
- Gradient checkpointing for 16GB RAM systems
- Automatic model caching (won't re-download if already present)
- Configurable hyperparameters

### `distill_quantumbench.py`

Performs knowledge distillation from teacher to student model.

**Features:**
- Custom distillation loss combining soft and hard labels
- Configurable temperature and alpha parameters
- Supports various student model architectures
- Automatic model caching

### `evaluate_quantumbench.py`

Evaluates models on QuantumBench questions.

**Features:**
- Computes accuracy by comparing predictions to ground truth
- Supports evaluating teacher and/or student models
- Normalizes text for fair comparison
- Optional prediction saving for analysis

## Requirements

See `requirements.txt` for full list of dependencies. Key packages:

- `torch>=2.0.0` - PyTorch
- `transformers>=4.35.0` - Hugging Face Transformers
- `peft>=0.6.0` - Parameter-Efficient Fine-Tuning
- `bitsandbytes>=0.41.0` - Quantization support
- `accelerate>=0.24.0` - Training acceleration
- `datasets>=2.14.0` - Dataset handling

## Hardware Requirements

- **Minimum:** 16GB RAM, GPU with 8GB+ VRAM (for QLoRA training)
- **Recommended:** 32GB+ RAM, GPU with 16GB+ VRAM

The scripts are optimized for 16GB RAM systems with:
- Gradient checkpointing
- 4-bit quantization
- Small batch sizes with gradient accumulation

## Model Support

### Teacher Models
- `meta-llama/Llama-2-7b-hf` - LLaMA-2 7B (requires Hugging Face access)
- `Qwen/Qwen-7B-Chat` - Qwen 7B Chat
- Any compatible Hugging Face causal LM model

### Student Models
- `gpt2` - GPT-2 1.5B
- `distilgpt2` - DistilGPT-2
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` - TinyLlama 1.1B
- Any smaller Hugging Face causal LM model

## Git Workflow

The setup script initializes a Git repository and sets up the remote:

```bash
# Remote repository
https://github.com/crurp/quantumbench-llm.git
```

After running `setup.sh`, you can:

```bash
# Check status
git status

# Make changes and commit
git add .
git commit -m "Your commit message"

# Push to remote
git push origin main
```

## Troubleshooting

### Model Download Issues

If model downloads fail:
1. Ensure you have Hugging Face access tokens set up (for gated models)
2. Check internet connection
3. Models are cached locally after first download

### Memory Issues

If you encounter OOM errors:
- Reduce `--batch-size` in training scripts
- Increase `--gradient-accumulation-steps` to maintain effective batch size
- Reduce `--max-length` for shorter sequences

### CUDA/GPU Issues

If CUDA is not available:
- Scripts will automatically fall back to CPU (much slower)
- For training, GPU is strongly recommended

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]

