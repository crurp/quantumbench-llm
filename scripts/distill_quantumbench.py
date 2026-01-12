#!/usr/bin/env python3
"""
distill_quantumbench.py

Performs knowledge distillation from a teacher model to a smaller student model.
The teacher model (fine-tuned on QuantumBench) is used to generate soft labels
that guide the training of a smaller, more efficient student model.

Usage:
    python scripts/distill_quantumbench.py --teacher-model models/teacher-model --student-model-name gpt2
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import Dataset
from tqdm import tqdm


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


def format_prompt(instruction: str, response: str = None) -> str:
    """
    Format instruction (and optionally response) into a prompt.
    
    Args:
        instruction: The instruction/question
        response: Optional response/answer
        
    Returns:
        Formatted prompt string
    """
    if response:
        return f"### Instruction:\n{instruction}\n\n### Response:\n{response}\n\n### End\n"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_teacher_logits(teacher_model, teacher_tokenizer, instruction: str, max_length: int = 512) -> torch.Tensor:
    """
    Generate logits from teacher model for a given instruction.
    
    Args:
        teacher_model: Fine-tuned teacher model
        teacher_tokenizer: Teacher model tokenizer
        instruction: Input instruction/question
        max_length: Maximum sequence length
        
    Returns:
        Teacher model logits
    """
    teacher_model.eval()
    
    # Format prompt
    prompt = format_prompt(instruction)
    
    # Tokenize
    inputs = teacher_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True
    ).to(teacher_model.device)
    
    # Generate logits (no gradient tracking)
    with torch.no_grad():
        outputs = teacher_model(**inputs)
        logits = outputs.logits
    
    return logits


class NaNDetectorCallback(TrainerCallback):
    """Callback to detect NaN losses and stop training immediately."""
    
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        """Check for NaN loss after each step."""
        if logs is not None and 'loss' in logs:
            loss = logs['loss']
            if torch.isnan(torch.tensor(loss)) if not isinstance(loss, torch.Tensor) else torch.isnan(loss):
                raise ValueError(
                    f"NaN loss detected at step {state.global_step}! "
                    f"Loss value: {loss}. Training stopped to prevent further corruption. "
                    f"Check: 1) Learning rate (may be too high), 2) Gradient clipping, "
                    f"3) Model initialization, 4) Input data quality."
                )
        return control


class DistillationTrainer(Trainer):
    """
    Custom Trainer class that implements knowledge distillation loss.
    Combines hard labels (ground truth) with soft labels (teacher predictions).
    """
    
    def __init__(self, teacher_model, temperature=3.0, alpha=0.5, step_count=None, **kwargs):
        """
        Initialize distillation trainer.
        
        Args:
            teacher_model: Teacher model for generating soft labels
            temperature: Temperature for softmax (higher = softer probabilities)
            alpha: Weight for distillation loss (1-alpha for hard labels)
            step_count: Reference to step counter for logging (optional)
        """
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        self.step_count = step_count
        self._first_forward_done = False
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute distillation loss combining hard and soft labels.
        
        Args:
            model: Student model
            inputs: Input batch
            return_outputs: Whether to return model outputs
            **kwargs: Additional arguments (e.g., num_items_in_batch) for compatibility
            
        Returns:
            Loss value (and optionally outputs)
        """
        # Student model forward pass
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        # Get labels (hard labels)
        labels = inputs.get("labels")
        
        # Logit shape check (first forward pass only)
        if not self._first_forward_done:
            print("\n" + "="*70)
            print("LOGIT SHAPE CHECK (First Forward Pass)")
            print("="*70)
            print(f"Student logits shape: {student_logits.shape}")
            self._first_forward_done = True
        
        # Teacher model forward pass (for soft labels)
        self.teacher_model.eval()
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**{k: v for k, v in inputs.items() if k != "labels"})
            teacher_logits = teacher_outputs.logits
        
        # Complete logit shape check
        if not hasattr(self, '_logit_shapes_printed'):
            print(f"Teacher logits shape: {teacher_logits.shape}")
            print(f"Temperature: {self.temperature}")
            print(f"Alpha (distillation weight): {self.alpha}, CE weight: {1 - self.alpha}")
            print("="*70 + "\n")
            self._logit_shapes_printed = True
        
        # Check for NaN/Inf in logits BEFORE loss computation
        if not torch.all(torch.isfinite(student_logits)):
            nan_count = (~torch.isfinite(student_logits)).sum().item()
            raise ValueError(
                f"NaN/Inf detected in student logits! Count: {nan_count}/{student_logits.numel()}. "
                f"This indicates numerical instability. Check model initialization and input data."
            )
        
        if not torch.all(torch.isfinite(teacher_logits)):
            nan_count = (~torch.isfinite(teacher_logits)).sum().item()
            raise ValueError(
                f"NaN/Inf detected in teacher logits! Count: {nan_count}/{teacher_logits.numel()}. "
                f"Teacher model may be corrupted or input data contains invalid values."
            )
        
        # Ensure logits shapes match for distillation
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"Logit shape mismatch! Student: {student_logits.shape}, Teacher: {teacher_logits.shape}. "
                f"Models must produce logits of the same shape for distillation."
            )
        
        # Compute distillation loss (KL divergence between teacher and student)
        # Use log_target=False since we're applying softmax to teacher_logits
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Check for NaN/Inf after softmax operations
        if not torch.all(torch.isfinite(student_log_probs)):
            raise ValueError("NaN/Inf in student log probabilities after log_softmax!")
        if not torch.all(torch.isfinite(teacher_probs)):
            raise ValueError("NaN/Inf in teacher probabilities after softmax!")
        
        # KL divergence: KL(student || teacher) = sum(teacher_probs * log(teacher_probs / student_probs))
        # Using F.kl_div with log_target=False means: input=log(student), target=teacher_probs
        loss_distill = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False
        ) * (self.temperature ** 2)  # Scale by T^2 to keep gradients consistent
        
        # Check distillation loss for NaN/Inf
        if not torch.isfinite(loss_distill):
            raise ValueError(
                f"NaN/Inf in distillation loss: {loss_distill}. "
                f"Temperature may be too high, or logits contain extreme values."
            )
        
        # Compute standard cross-entropy loss (hard labels)
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        # Check CE loss for NaN/Inf
        if not torch.isfinite(loss_ce):
            raise ValueError(
                f"NaN/Inf in cross-entropy loss: {loss_ce}. "
                f"Check labels and student logits."
            )
        
        # Combine losses with balanced weighting (alpha=0.5 means 50/50 split)
        loss = self.alpha * loss_distill + (1 - self.alpha) * loss_ce
        
        # Print loss components for first 5 steps
        current_step = self.state.global_step if hasattr(self, 'state') and hasattr(self.state, 'global_step') else None
        if current_step is not None and current_step < 5:
            print(f"\n[Step {current_step}] Loss Components:")
            print(f"  Distillation Loss (α={self.alpha}): {loss_distill.item():.6f}")
            print(f"  Cross-Entropy Loss (1-α={1-self.alpha}): {loss_ce.item():.6f}")
            print(f"  Combined Loss: {loss.item():.6f}")
            print(f"  Temperature: {self.temperature}, Temperature^2: {self.temperature**2}")
        
        # Final NaN check on combined loss
        if not torch.isfinite(loss):
            raise ValueError(
                f"NaN/Inf in final combined loss: {loss}. "
                f"Distill loss: {loss_distill.item()}, CE loss: {loss_ce.item()}"
            )
        
        return (loss, outputs) if return_outputs else loss


def setup_teacher_model(teacher_model_path: Path, device: str = "cuda", use_fp16: bool = False):
    """
    Load teacher model and tokenizer.
    
    Args:
        teacher_model_path: Path to teacher model directory
        device: Device to load model on
        use_fp16: Whether to use float16 (only for GPU with tensor cores)
        
    Returns:
        Tuple of (teacher_model, teacher_tokenizer)
    """
    print(f"Loading teacher model from {teacher_model_path}...")
    
    teacher_tokenizer = AutoTokenizer.from_pretrained(str(teacher_model_path))
    
    # Use float32 for CPU, float16 only for GPU if requested
    dtype = torch.float16 if (use_fp16 and device == "cuda" and torch.cuda.is_available()) else torch.float32
    print(f"Using dtype: {dtype} (device: {device})")
    
    teacher_model = AutoModelForCausalLM.from_pretrained(
        str(teacher_model_path),
        device_map=device if device == "cuda" else "cpu",
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    teacher_model.eval()
    print("Teacher model loaded successfully")
    
    return teacher_model, teacher_tokenizer


def setup_student_model(student_model_name: str, model_cache_dir: Path, device: str = "cuda", use_fp16: bool = False):
    """
    Load or download student model (smaller model like GPT-2 1.5B or similar).
    
    Args:
        student_model_name: Hugging Face model identifier for student
        model_cache_dir: Directory to cache models
        device: Device to load model on
        use_fp16: Whether to use float16 (only for GPU with tensor cores)
        
    Returns:
        Tuple of (student_model, student_tokenizer)
    """
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model is already downloaded
    model_path = model_cache_dir / student_model_name.split('/')[-1]
    
    if model_path.exists() and any(model_path.iterdir()):
        print(f"Student model found at {model_path}, loading from local directory...")
        model_load_path = str(model_path)
    else:
        print(f"Student model not found locally, will download: {student_model_name}")
        model_load_path = student_model_name
    
    print(f"Loading student model: {student_model_name}...")
    student_tokenizer = AutoTokenizer.from_pretrained(model_load_path)
    
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id
    
    # Use float32 for CPU, float16 only for GPU if requested
    dtype = torch.float16 if (use_fp16 and device == "cuda" and torch.cuda.is_available()) else torch.float32
    print(f"Using dtype: {dtype} (device: {device})")
    
    student_model = AutoModelForCausalLM.from_pretrained(
        model_load_path,
        device_map=device if device == "cuda" else "cpu",
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Save locally if downloaded
    if model_load_path == student_model_name:
        print(f"Saving student model to {model_path} for future use...")
        student_model.save_pretrained(str(model_path))
        student_tokenizer.save_pretrained(str(model_path))
    
    print("Student model loaded successfully")
    
    return student_model, student_tokenizer


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
    texts = [
        format_prompt(inst, resp)
        for inst, resp in zip(examples['instruction'], examples['response'])
    ]
    
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    tokenized['labels'] = tokenized['input_ids'].clone()
    
    return tokenized


def main():
    """Main distillation function."""
    parser = argparse.ArgumentParser(
        description="Perform knowledge distillation from teacher to student model"
    )
    parser.add_argument(
        '--teacher-model',
        type=str,
        default='models/teacher-model',
        help='Path to fine-tuned teacher model'
    )
    parser.add_argument(
        '--student-model-name',
        type=str,
        default='gpt2',
        help='Hugging Face model identifier for student (e.g., gpt2, distilgpt2, or TinyLlama/TinyLlama-1.1B-Chat-v1.0)'
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
        default='models/student-model',
        help='Directory to save distilled student model'
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
        default=5,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Training batch size'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=3.0,
        help='Temperature for distillation (higher = softer probabilities)'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Weight for distillation loss (1-alpha for hard labels). Default: 0.5 (balanced)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode: train for only 5 steps with eval every 5 steps for sanity checking'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Use mixed precision (fp16) training. Only recommended for GPU with tensor cores. Disabled by default for CPU stability.'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    teacher_model_path = Path(args.teacher_model)
    data_path = Path(args.data_path)
    output_dir = Path(args.output_dir)
    model_cache_dir = Path(args.model_cache_dir)
    
    # Validate paths
    if not teacher_model_path.exists():
        raise FileNotFoundError(f"Teacher model not found: {teacher_model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Determine if we should use FP16 (only for GPU with tensor cores, and if explicitly requested)
    use_fp16 = args.fp16 and device == "cuda" and torch.cuda.is_available()
    if device == "cpu" and args.fp16:
        print("⚠ Warning: FP16 requested but running on CPU. Using FP32 for numerical stability.")
        use_fp16 = False
    
    # Load teacher model
    teacher_model, teacher_tokenizer = setup_teacher_model(teacher_model_path, device, use_fp16=use_fp16)
    
    # Load student model
    student_model, student_tokenizer = setup_student_model(args.student_model_name, model_cache_dir, device, use_fp16=use_fp16)
    
    # Load dataset
    print("Loading and preparing dataset...")
    raw_data = load_jsonl_data(data_path)
    
    # Convert to Hugging Face dataset
    dataset_dict = {
        'instruction': [item['instruction'] for item in raw_data],
        'response': [item['response'] for item in raw_data]
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split into train/val
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset['train']
    val_dataset = dataset['test']
    
    print(f"Train examples: {len(train_dataset)}, Validation examples: {len(val_dataset)}")
    
    # Tokenize datasets (using student tokenizer)
    print("Tokenizing datasets...")
    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, student_tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_tokenized = val_dataset.map(
        lambda x: tokenize_function(x, student_tokenizer, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Calculate total steps and warmup steps
    total_steps = len(train_tokenized) // args.batch_size * args.epochs
    warmup_steps = max(1, int(total_steps * 0.1))  # 10% warmup
    
    # Adjust for dry-run mode
    if args.dry_run:
        print("\n" + "="*70)
        print("DRY RUN MODE ENABLED")
        print("="*70)
        print("Training limited to 5 steps for sanity checking.")
        print("="*70 + "\n")
        max_steps = 5
        eval_steps = 5
        save_steps = 5
        num_train_epochs = 1  # Override epochs for dry run
    else:
        max_steps = -1  # Use epochs instead
        eval_steps = 100
        save_steps = 100
        num_train_epochs = args.epochs
    
    # Set learning rate (default to 2e-5 as recommended, or use provided value)
    learning_rate = 2e-5 if args.learning_rate == 5e-5 else args.learning_rate
    
    print(f"\nTraining Configuration:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Warmup steps: {warmup_steps} (10% of {total_steps} total steps)")
    print(f"  Temperature: {args.temperature}")
    print(f"  Alpha (distillation weight): {args.alpha}, CE weight: {1 - args.alpha}")
    print(f"  Max grad norm: 1.0 (gradient clipping)")
    print(f"  FP16: {use_fp16}")
    print(f"  Device: {device}")
    if args.dry_run:
        print(f"  DRY RUN: max_steps={max_steps}, eval_steps={eval_steps}")
    print()
    
    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=learning_rate,
        fp16=use_fp16,  # Only use FP16 if explicitly requested and on GPU
        logging_steps=10,
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=not args.dry_run,
        gradient_checkpointing=True,
        warmup_steps=warmup_steps,
        max_grad_norm=1.0,  # Gradient clipping for numerical stability
        report_to=None,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=student_tokenizer,
        mlm=False,
    )
    
    # Initialize custom distillation trainer
    trainer = DistillationTrainer(
        teacher_model=teacher_model,
        temperature=args.temperature,
        alpha=args.alpha,
        model=student_model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
    )
    
    # Add NaN detector callback
    trainer.add_callback(NaNDetectorCallback())
    
    # Train student model
    print("Starting knowledge distillation...")
    print("NaN detection enabled - training will stop immediately if NaN loss is detected.\n")
    
    try:
        trainer.train()
    except ValueError as e:
        if "NaN" in str(e) or "Inf" in str(e):
            print("\n" + "="*70)
            print("❌ TRAINING STOPPED DUE TO NUMERICAL INSTABILITY")
            print("="*70)
            print(str(e))
            print("\nSuggested fixes:")
            print("  1. Lower learning rate (try 1e-5 or lower)")
            print("  2. Reduce batch size")
            print("  3. Check input data for invalid values")
            print("  4. Ensure teacher model is valid")
            print("  5. Try --dry-run first to diagnose")
            print("="*70)
            raise
        else:
            raise
    
    if args.dry_run:
        print("\n" + "="*70)
        print("✓ DRY RUN COMPLETED SUCCESSFULLY")
        print("="*70)
        print("No NaN/Inf detected in first 5 steps. Safe to proceed with full training.")
        print("="*70 + "\n")
    
    # Save final model
    print(f"Saving student model to {output_dir}...")
    trainer.save_model()
    student_tokenizer.save_pretrained(str(output_dir))
    
    print("Knowledge distillation complete!")


if __name__ == "__main__":
    main()


