#!/usr/bin/env python3
"""
train_quantumbench.py

Fine-tunes a moderate-sized language model (LLaMA-2-7B or Qwen-7B) on QuantumBench
using QLoRA (Quantized Low-Rank Adaptation) and PEFT (Parameter-Efficient Fine-Tuning).

This script is optimized for 16GB RAM systems with:
- Gradient checkpointing
- 4-bit quantization
- LoRA adapters
- Low-memory optimizations

Usage:
    python scripts/train_quantumbench.py --model-name meta-llama/Llama-2-7b-hf --data-path data/quantumbench.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from accelerate import Accelerator


def load_jsonl_data(data_path: Path) -> List[Dict[str, str]]:
    """
    Load JSONL dataset from file.
    
    Args:
        data_path: Path to JSONL file
        
    Returns:
        List of dictionaries with 'instruction' and 'response' keys
    """
    print(f"Loading dataset from {data_path}...")
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"Loaded {len(data)} examples")
    return data


def format_prompt(instruction: str, response: str) -> str:
    """
    Format instruction and response into a single prompt for training.
    
    Args:
        instruction: The instruction/question
        response: The response/answer
        
    Returns:
        Formatted prompt string
    """
    # Format: Instruction: ... Response: ...
    # Adjust format based on model's training format
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}\n\n### End\n"


def tokenize_function(examples, tokenizer, max_length: int = 512):
    """
    Tokenize the dataset examples.
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    # Combine instruction and response
    texts = [
        format_prompt(inst, resp)
        for inst, resp in zip(examples['instruction'], examples['response'])
    ]
    
    # Tokenize with padding and truncation
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized['labels'] = tokenized['input_ids'].clone()
    
    return tokenized


def setup_model_and_tokenizer(model_name: str, model_dir: Path, device_map: str = "auto"):
    """
    Load or download model and tokenizer.
    Checks if model is already downloaded to avoid re-downloading.
    
    Args:
        model_name: Hugging Face model identifier
        model_dir: Directory to save/load model from
        device_map: Device mapping strategy ("auto", "cpu", etc.)
        
    Returns:
        Tuple of (model, tokenizer)
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model is already downloaded
    model_path = model_dir / model_name.split('/')[-1]
    
    if model_path.exists() and any(model_path.iterdir()):
        print(f"Model found at {model_path}, loading from local directory...")
        model_load_path = str(model_path)
    else:
        print(f"Model not found locally, will download from Hugging Face: {model_name}")
        model_load_path = model_name
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    
    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure quantization based on device availability
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # Use 4-bit quantization for GPU
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        print("Loading model with 4-bit quantization (GPU)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_load_path,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        # For CPU, load without quantization (use float32 for stability)
        print("Loading model for CPU (no quantization)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_load_path,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    
    # If model was downloaded, save it locally for future use
    if model_load_path == model_name:
        print(f"Saving model to {model_path} for future use...")
        model.save_pretrained(str(model_path))
        tokenizer.save_pretrained(str(model_path))
    
    # Prepare model for training
    if use_cuda:
        # Prepare for k-bit training with quantization
        model = prepare_model_for_kbit_training(model)
    else:
        # For CPU, enable gradient computation on all parameters
        model.gradient_checkpointing_enable()
        for param in model.parameters():
            param.requires_grad = False
    
    # Configure LoRA - adjust target modules based on model architecture
    # For GPT-2, use 'c_attn' instead of 'q_proj', 'k_proj', etc.
    if 'gpt2' in model_name.lower():
        target_modules = ["c_attn"]  # GPT-2 uses different module names
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]  # LLaMA-style models
    
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA alpha (scaling parameter)
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model on QuantumBench using QLoRA"
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='meta-llama/Llama-2-7b-hf',
        help='Hugging Face model identifier (e.g., meta-llama/Llama-2-7b-hf or Qwen/Qwen-7B-Chat)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/quantumbench.jsonl',
        help='Path to JSONL training data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/teacher-model',
        help='Directory to save fine-tuned model'
    )
    parser.add_argument(
        '--model-cache-dir',
        type=str,
        default='models',
        help='Directory to cache downloaded models'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size (adjust based on GPU memory)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=4,
        help='Gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    model_cache_dir = Path(args.model_cache_dir)
    
    # Validate data file exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading and preparing dataset...")
    raw_data = load_jsonl_data(data_path)
    
    # Convert to Hugging Face dataset
    dataset_dict = {
        'instruction': [item['instruction'] for item in raw_data],
        'response': [item['response'] for item in raw_data]
    }
    from datasets import Dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split into train/val (80/20)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    
    print(f"Train examples: {len(train_dataset)}, Validation examples: {len(val_dataset)}")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(args.model_name, model_cache_dir)
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_tokenized = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),  # Use mixed precision only if CUDA is available
        logging_steps=10,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",  # Use 8-bit optimizer only on GPU
        report_to=None,  # Disable wandb/tensorboard if not needed
        warmup_steps=50,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving model to {output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(str(output_dir))
    
    print("Training complete!")


if __name__ == "__main__":
    main()

