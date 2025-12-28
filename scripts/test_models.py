#!/usr/bin/env python3
"""
test_models.py

Model evaluation script for teacher and student models.
Runs inference on sample questions and computes accuracy.

Usage:
    python scripts/test_models.py --model-path models/teacher-model --data-path data/quantumbench.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_jsonl(data_path: Path) -> List[Dict]:
    """Load JSONL dataset."""
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def format_prompt(instruction: str) -> str:
    """Format instruction into prompt for generation."""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def generate_response(model, tokenizer, instruction: str, max_new_tokens: int = 256) -> str:
    """
    Generate response from model.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        instruction: Input instruction/question
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated response text
    """
    model.eval()
    
    prompt = format_prompt(instruction)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response part
    if "### Response:\n" in generated_text:
        response = generated_text.split("### Response:\n")[-1].strip()
        response = response.split("### End")[0].strip()
    else:
        response = generated_text[len(prompt):].strip()
    
    return response


def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Compute accuracy by comparing predictions to ground truth."""
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths")
    
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_norm = normalize_text(pred)
        gt_norm = normalize_text(gt)
        
        if pred_norm == gt_norm or pred_norm in gt_norm or gt_norm in pred_norm:
            correct += 1
    
    return correct / len(predictions)


def evaluate_model(
    model_path: Path,
    data: List[Dict],
    model_type: str = "unknown",
    num_samples: int = None,
    max_new_tokens: int = 256
) -> Dict:
    """
    Evaluate a model on the dataset.
    
    Args:
        model_path: Path to model directory
        data: List of evaluation examples
        model_type: Type of model ("teacher" or "student")
        num_samples: Number of samples to evaluate (None = all)
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {model_type.upper()} MODEL")
    print(f"{'='*70}")
    print(f"Model path: {model_path}")
    
    if not model_path.exists():
        print(f"⚠ Warning: Model not found at {model_path}")
        return {
            'model_path': str(model_path),
            'model_type': model_type,
            'exists': False,
            'predictions': [],
            'accuracy': 0.0
        }
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return {
            'model_path': str(model_path),
            'model_type': model_type,
            'exists': False,
            'error': str(e),
            'predictions': [],
            'accuracy': 0.0
        }
    
    # Select samples
    eval_data = data[:num_samples] if num_samples else data
    print(f"Evaluating on {len(eval_data)} examples...")
    
    # Generate predictions
    predictions = []
    ground_truths = []
    instructions = []
    
    for item in tqdm(eval_data, desc="Generating predictions"):
        instruction = item['instruction']
        # Clean instruction
        if "Answer the following quantum computing question:\n\n" in instruction:
            instruction_clean = instruction.split("Answer the following quantum computing question:\n\n")[-1]
        else:
            instruction_clean = instruction
        
        ground_truth = item['response']
        
        # Generate prediction
        prediction = generate_response(model, tokenizer, instruction_clean, max_new_tokens)
        
        predictions.append(prediction)
        ground_truths.append(ground_truth)
        instructions.append(instruction_clean)
    
    # Compute accuracy
    accuracy = compute_accuracy(predictions, ground_truths)
    
    print(f"\n{'='*70}")
    print(f"{model_type.upper()} Model Results:")
    print(f"{'='*70}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Evaluated examples: {len(eval_data)}")
    print()
    
    # Display sample predictions
    if num_samples and num_samples <= 5:
        print(f"{'='*70}")
        print("Sample Predictions:")
        print(f"{'='*70}\n")
        
        for i, (inst, pred, gt) in enumerate(zip(instructions, predictions, ground_truths), 1):
            print(f"Example {i}:")
            print(f"  Question: {inst[:100]}{'...' if len(inst) > 100 else ''}")
            print(f"  Predicted: {pred[:150]}{'...' if len(pred) > 150 else ''}")
            print(f"  Correct: {gt[:150]}{'...' if len(gt) > 150 else ''}")
            
            # Check if correct
            pred_norm = normalize_text(pred)
            gt_norm = normalize_text(gt)
            is_correct = pred_norm == gt_norm or pred_norm in gt_norm or gt_norm in pred_norm
            print(f"  ✓ Correct" if is_correct else f"  ✗ Incorrect")
            print()
    
    return {
        'model_path': str(model_path),
        'model_type': model_type,
        'exists': True,
        'predictions': predictions,
        'ground_truths': ground_truths,
        'instructions': instructions,
        'accuracy': accuracy,
        'num_evaluated': len(eval_data)
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate teacher or student model on QuantumBench"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to model directory'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['teacher', 'student'],
        default='teacher',
        help='Type of model being evaluated'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/quantumbench.jsonl',
        help='Path to JSONL evaluation data'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to evaluate (default: 5, use None for all)'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=256,
        help='Maximum tokens to generate'
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    
    # Load data
    print("Loading dataset...")
    data = load_jsonl(data_path)
    print(f"Loaded {len(data)} examples")
    
    # Evaluate model
    results = evaluate_model(
        model_path,
        data,
        args.model_type,
        args.num_samples,
        args.max_new_tokens
    )
    
    return results


if __name__ == "__main__":
    main()

