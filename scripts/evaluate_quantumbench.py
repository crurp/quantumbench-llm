#!/usr/bin/env python3
"""
evaluate_quantumbench.py

Evaluates teacher and/or student models on QuantumBench questions.
Computes accuracy by comparing model predictions to ground truth answers.

Usage:
    python scripts/evaluate_quantumbench.py --model-path models/teacher-model --data-path data/quantumbench.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_jsonl_data(data_path: Path) -> List[Dict[str, str]]:
    """
    Load JSONL evaluation dataset.
    
    Args:
        data_path: Path to JSONL file
        
    Returns:
        List of dictionaries with 'instruction' and 'response' keys
    """
    print(f"Loading evaluation dataset from {data_path}...")
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"Loaded {len(data)} examples for evaluation")
    return data


def format_prompt(instruction: str) -> str:
    """
    Format instruction into a prompt for generation.
    
    Args:
        instruction: The instruction/question
        
    Returns:
        Formatted prompt string
    """
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_response(model, tokenizer, instruction: str, max_new_tokens: int = 256) -> str:
    """
    Generate response from model for a given instruction.
    
    Args:
        model: Language model to generate from
        tokenizer: Model tokenizer
        instruction: Input instruction/question
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated response text
    """
    model.eval()
    
    # Format prompt
    prompt = format_prompt(instruction)
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding for consistency
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the response part (after "### Response:\n")
    if "### Response:\n" in generated_text:
        response = generated_text.split("### Response:\n")[-1].strip()
        # Remove "### End" if present
        response = response.split("### End")[0].strip()
    else:
        # Fallback: use everything after the prompt
        response = generated_text[len(prompt):].strip()
    
    return response


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison (lowercase, strip whitespace, remove punctuation).
    
    Args:
        text: Text to normalize
        
    Returns:
        Normalized text
    """
    import re
    text = text.lower().strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation (optional, can be adjusted)
    # text = re.sub(r'[^\w\s]', '', text)
    return text


def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute accuracy by comparing predictions to ground truth.
    Uses normalized text comparison.
    
    Args:
        predictions: List of predicted responses
        ground_truths: List of ground truth responses
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths")
    
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_norm = normalize_text(pred)
        gt_norm = normalize_text(gt)
        
        # Check for exact match or substring match
        if pred_norm == gt_norm or pred_norm in gt_norm or gt_norm in pred_norm:
            correct += 1
    
    accuracy = correct / len(predictions)
    return accuracy


def evaluate_model(model_path: Path, data: List[Dict[str, str]], max_new_tokens: int = 256) -> Tuple[List[str], float]:
    """
    Evaluate a model on the dataset.
    
    Args:
        model_path: Path to model directory
        data: List of evaluation examples
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Tuple of (predictions, accuracy)
    """
    print(f"Loading model from {model_path}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    print(f"Model loaded. Evaluating on {len(data)} examples...")
    
    # Generate predictions
    predictions = []
    ground_truths = []
    
    for item in tqdm(data, desc="Generating predictions"):
        instruction = item['instruction']
        # Remove the instruction template prefix if present
        if "Answer the following quantum computing question:\n\n" in instruction:
            instruction = instruction.split("Answer the following quantum computing question:\n\n")[-1]
        
        ground_truth = item['response']
        
        # Generate prediction
        prediction = generate_response(model, tokenizer, instruction, max_new_tokens)
        
        predictions.append(prediction)
        ground_truths.append(ground_truth)
    
    # Compute accuracy
    accuracy = compute_accuracy(predictions, ground_truths)
    
    return predictions, accuracy


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate teacher and/or student models on QuantumBench"
    )
    parser.add_argument(
        '--teacher-model',
        type=str,
        default=None,
        help='Path to teacher model directory (optional)'
    )
    parser.add_argument(
        '--student-model',
        type=str,
        default=None,
        help='Path to student model directory (optional)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to single model to evaluate (alternative to --teacher-model/--student-model)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/quantumbench.jsonl',
        help='Path to JSONL evaluation data'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=256,
        help='Maximum number of tokens to generate per response'
    )
    parser.add_argument(
        '--output-predictions',
        type=str,
        default=None,
        help='Optional path to save predictions JSON file'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.model_path is None and args.teacher_model is None and args.student_model is None:
        raise ValueError("Must provide either --model-path, --teacher-model, or --student-model")
    
    # Convert to Path objects
    data_path = Path(args.data_path)
    
    # Validate data file exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load evaluation data
    eval_data = load_jsonl_data(data_path)
    
    # Evaluate models
    results = {}
    
    # Evaluate teacher model if provided
    if args.teacher_model:
        teacher_path = Path(args.teacher_model)
        if not teacher_path.exists():
            print(f"Warning: Teacher model not found at {teacher_path}, skipping...")
        else:
            print("\n" + "="*60)
            print("Evaluating TEACHER model...")
            print("="*60)
            predictions, accuracy = evaluate_model(teacher_path, eval_data, args.max_new_tokens)
            results['teacher'] = {
                'accuracy': accuracy,
                'predictions': predictions
            }
            print(f"\nTeacher Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Evaluate student model if provided
    if args.student_model:
        student_path = Path(args.student_model)
        if not student_path.exists():
            print(f"Warning: Student model not found at {student_path}, skipping...")
        else:
            print("\n" + "="*60)
            print("Evaluating STUDENT model...")
            print("="*60)
            predictions, accuracy = evaluate_model(student_path, eval_data, args.max_new_tokens)
            results['student'] = {
                'accuracy': accuracy,
                'predictions': predictions
            }
            print(f"\nStudent Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Evaluate single model if provided
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Warning: Model not found at {model_path}, skipping...")
        else:
            print("\n" + "="*60)
            print(f"Evaluating model at {model_path}...")
            print("="*60)
            predictions, accuracy = evaluate_model(model_path, eval_data, args.max_new_tokens)
            results['model'] = {
                'accuracy': accuracy,
                'predictions': predictions
            }
            print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for model_name, result in results.items():
        print(f"{model_name.capitalize()} Model: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
    print("="*60)
    
    # Save predictions if requested
    if args.output_predictions:
        output_path = Path(args.output_predictions)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()

