#!/usr/bin/env python3
"""
distillation_performance_metrics.py

Comprehensive performance metrics for knowledge distillation.
Computes model size, accuracy, inference speed, memory usage, and
knowledge retention metrics comparing teacher and student models.

Usage:
    python scripts/distillation_performance_metrics.py \
        --teacher-model models/teacher-model \
        --student-model models/student-model \
        --data-path data/quantumbench.jsonl \
        --output metrics/distillation_metrics.json
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm


def get_model_size(model_path: Path) -> Dict[str, float]:
    """
    Calculate model size in MB and number of parameters.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Dictionary with size metrics
    """
    size_bytes = 0
    num_files = 0
    
    # Calculate total disk size
    if model_path.exists():
        for dirpath, dirnames, filenames in os.walk(model_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    size_bytes += os.path.getsize(fp)
                    num_files += 1
    
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_mb / 1024
    
    # Try to get parameter count from model
    try:
        model, _ = load_model_with_peft(model_path)
        num_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            'disk_size_mb': size_mb,
            'disk_size_gb': size_gb,
            'num_files': num_files,
            'num_parameters': num_params,
            'trainable_parameters': trainable_params,
            'num_parameters_millions': num_params / 1e6,
            'num_parameters_billions': num_params / 1e9,
        }
    except Exception as e:
        print(f"Warning: Could not load model to count parameters: {e}")
        return {
            'disk_size_mb': size_mb,
            'disk_size_gb': size_gb,
            'num_files': num_files,
            'num_parameters': None,
            'trainable_parameters': None,
        }


def format_prompt(instruction: str) -> str:
    """Format instruction into a prompt."""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_response(
    model, tokenizer, instruction: str, 
    max_new_tokens: int = 256,
    measure_time: bool = False
) -> Tuple[str, float]:
    """
    Generate response and optionally measure inference time.
    
    Returns:
        Tuple of (response_text, inference_time_seconds)
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
    
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    inference_time = time.time() - start_time
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:\n" in generated_text:
        response = generated_text.split("### Response:\n")[-1].strip()
        response = response.split("### End")[0].strip()
    else:
        response = generated_text[len(prompt):].strip()
    
    return response, inference_time


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    import re
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def compute_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """Compute accuracy."""
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Mismatch: {len(predictions)} predictions vs {len(ground_truths)} ground truths")
    
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_norm = normalize_text(pred)
        gt_norm = normalize_text(gt)
        
        if pred_norm == gt_norm or pred_norm in gt_norm or gt_norm in pred_norm:
            correct += 1
    
    return correct / len(predictions)


def load_model_with_peft(model_path: Path):
    """
    Load model, handling both regular models and PEFT/LoRA adapters.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    # Check if this is a PEFT model (has adapter_config.json)
    adapter_config = model_path / "adapter_config.json"
    
    if adapter_config.exists():
        # This is a PEFT/LoRA model - try to load directly first
        print("  Detected PEFT/LoRA model, attempting direct load...")
        try:
            # Try loading directly - sometimes PEFT models can be loaded this way
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            print("  ✓ Loaded PEFT model directly")
            model.eval()
            return model, tokenizer
        except Exception as e1:
            print(f"  Direct load failed: {e1}")
            print("  Attempting to load with PEFT library...")
            
            # Load base model and adapter separately
            import json
            with open(adapter_config, 'r') as f:
                adapter_info = json.load(f)
                base_model_path = adapter_info.get('base_model_name_or_path', 'gpt2-large')
            
            # Try local path first
            base_model_path_obj = Path(base_model_path)
            if base_model_path_obj.exists() and (base_model_path_obj / "config.json").exists():
                base_model_name = str(base_model_path_obj)
            else:
                # Use HuggingFace model name
                base_model_name = base_model_path.split('/')[-1] if '/' in base_model_path else base_model_path
                # If it's a relative path, try HuggingFace directly
                if not Path(base_model_path).exists():
                    base_model_name = "gpt2-large"  # Default fallback
            
            print(f"  Loading base model: {base_model_name}")
            try:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            except Exception as e2:
                # Try HuggingFace directly
                print(f"  Trying HuggingFace model directly: {base_model_path}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path if '/' not in base_model_path or base_model_path.startswith('gpt2') else "gpt2-large",
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            
            print(f"  Loading PEFT adapter...")
            model = PeftModel.from_pretrained(base_model, str(model_path))
            model = model.merge_and_unload()  # Merge for inference
            model.eval()
    else:
        # Regular model
        print("  Loading standard model...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.eval()
    
    return model, tokenizer


def benchmark_inference(
    model_path: Path,
    data: List[Dict[str, str]],
    num_samples: int = 10,
    max_new_tokens: int = 256
) -> Dict[str, Any]:
    """
    Benchmark inference speed and memory usage.
    
    Args:
        model_path: Path to model
        data: Evaluation dataset
        num_samples: Number of samples to benchmark
        max_new_tokens: Max tokens to generate
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"Benchmarking inference for {num_samples} samples...")
    
    # Get initial memory
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Load model
    start_load = time.time()
    model, tokenizer = load_model_with_peft(model_path)
    load_time = time.time() - start_load
    
    # Memory after loading
    mem_after_load = process.memory_info().rss / (1024 * 1024)  # MB
    model_memory = mem_after_load - mem_before
    
    # Benchmark inference
    inference_times = []
    predictions = []
    
    eval_data = data[:num_samples] if num_samples else data[:10]
    
    for item in tqdm(eval_data, desc="Benchmarking"):
        instruction = item['instruction']
        if "Answer the following quantum computing question:\n\n" in instruction:
            instruction = instruction.split("Answer the following quantum computing question:\n\n")[-1]
        
        response, inf_time = generate_response(model, tokenizer, instruction, max_new_tokens, measure_time=True)
        inference_times.append(inf_time)
        predictions.append(response)
    
    # Memory after inference
    mem_after_inf = process.memory_info().rss / (1024 * 1024)  # MB
    
    avg_inference_time = sum(inference_times) / len(inference_times)
    total_inference_time = sum(inference_times)
    tokens_per_second = max_new_tokens / avg_inference_time if avg_inference_time > 0 else 0
    samples_per_second = 1.0 / avg_inference_time if avg_inference_time > 0 else 0
    
    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        'load_time_seconds': load_time,
        'model_memory_mb': model_memory,
        'peak_memory_mb': mem_after_inf,
        'inference_times': inference_times,
        'avg_inference_time_seconds': avg_inference_time,
        'total_inference_time_seconds': total_inference_time,
        'tokens_per_second': tokens_per_second,
        'samples_per_second': samples_per_second,
        'min_inference_time': min(inference_times) if inference_times else 0,
        'max_inference_time': max(inference_times) if inference_times else 0,
    }


def evaluate_model_full(
    model_path: Path,
    data: List[Dict[str, str]],
    max_new_tokens: int = 256
) -> Tuple[List[str], float]:
    """
    Full evaluation with predictions and accuracy.
    
    Returns:
        Tuple of (predictions, accuracy)
    """
    print(f"Evaluating model on {len(data)} examples...")
    
    model, tokenizer = load_model_with_peft(model_path)
    
    predictions = []
    ground_truths = []
    
    for item in tqdm(data, desc="Evaluating"):
        instruction = item['instruction']
        if "Answer the following quantum computing question:\n\n" in instruction:
            instruction = instruction.split("Answer the following quantum computing question:\n\n")[-1]
        
        ground_truth = item['response']
        prediction, _ = generate_response(model, tokenizer, instruction, max_new_tokens)
        
        predictions.append(prediction)
        ground_truths.append(ground_truth)
    
    accuracy = compute_accuracy(predictions, ground_truths)
    
    del model
    del tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return predictions, accuracy


def compute_distillation_metrics(
    teacher_metrics: Dict,
    student_metrics: Dict
) -> Dict[str, Any]:
    """
    Compute knowledge distillation efficacy metrics.
    
    Args:
        teacher_metrics: Teacher model metrics
        student_metrics: Student model metrics
        
    Returns:
        Dictionary with distillation metrics
    """
    # Size reduction
    teacher_size = teacher_metrics.get('size', {}).get('disk_size_mb', 0)
    student_size = student_metrics.get('size', {}).get('disk_size_mb', 0)
    size_reduction = teacher_size - student_size if teacher_size > 0 else 0
    size_reduction_percent = (size_reduction / teacher_size * 100) if teacher_size > 0 else 0
    size_ratio = student_size / teacher_size if teacher_size > 0 else 0
    
    # Parameter reduction
    teacher_params = teacher_metrics.get('size', {}).get('num_parameters', 0) or 0
    student_params = student_metrics.get('size', {}).get('num_parameters', 0) or 0
    param_reduction = teacher_params - student_params
    param_reduction_percent = (param_reduction / teacher_params * 100) if teacher_params > 0 else 0
    param_ratio = student_params / teacher_params if teacher_params > 0 else 0
    
    # Accuracy metrics
    teacher_acc = teacher_metrics.get('accuracy', 0)
    student_acc = student_metrics.get('accuracy', 0)
    accuracy_delta = teacher_acc - student_acc
    accuracy_retention = (student_acc / teacher_acc * 100) if teacher_acc > 0 else 0
    
    # Speed metrics
    teacher_speed = teacher_metrics.get('benchmark', {}).get('avg_inference_time_seconds', 0)
    student_speed = student_metrics.get('benchmark', {}).get('avg_inference_time_seconds', 0)
    speed_improvement = teacher_speed - student_speed if teacher_speed > 0 else 0
    speed_improvement_percent = (speed_improvement / teacher_speed * 100) if teacher_speed > 0 else 0
    speed_ratio = student_speed / teacher_speed if teacher_speed > 0 else 1.0
    
    # Memory metrics
    teacher_mem = teacher_metrics.get('benchmark', {}).get('model_memory_mb', 0)
    student_mem = student_metrics.get('benchmark', {}).get('model_memory_mb', 0)
    memory_reduction = teacher_mem - student_mem
    memory_reduction_percent = (memory_reduction / teacher_mem * 100) if teacher_mem > 0 else 0
    memory_ratio = student_mem / teacher_mem if teacher_mem > 0 else 0
    
    # Efficiency metrics
    efficiency_score = student_acc / param_ratio if param_ratio > 0 else 0  # Accuracy per parameter ratio
    throughput_improvement = (student_metrics.get('benchmark', {}).get('samples_per_second', 0) /
                            teacher_metrics.get('benchmark', {}).get('samples_per_second', 1)) if teacher_metrics.get('benchmark', {}).get('samples_per_second', 0) > 0 else 0
    
    return {
        # Size metrics
        'size_reduction_mb': size_reduction,
        'size_reduction_percent': size_reduction_percent,
        'size_ratio': size_ratio,
        'compression_ratio': 1.0 / size_ratio if size_ratio > 0 else 0,
        
        # Parameter metrics
        'parameter_reduction': param_reduction,
        'parameter_reduction_percent': param_reduction_percent,
        'parameter_ratio': param_ratio,
        'compression_ratio_parameters': 1.0 / param_ratio if param_ratio > 0 else 0,
        
        # Accuracy metrics
        'accuracy_delta': accuracy_delta,
        'accuracy_retention_percent': accuracy_retention,
        'accuracy_loss': -accuracy_delta,
        'accuracy_loss_percent': -accuracy_delta / teacher_acc * 100 if teacher_acc > 0 else 0,
        
        # Speed metrics
        'speed_improvement_seconds': speed_improvement,
        'speed_improvement_percent': speed_improvement_percent,
        'speed_ratio': speed_ratio,
        'throughput_improvement': throughput_improvement,
        
        # Memory metrics
        'memory_reduction_mb': memory_reduction,
        'memory_reduction_percent': memory_reduction_percent,
        'memory_ratio': memory_ratio,
        
        # Efficiency scores
        'efficiency_score': efficiency_score,
        'performance_per_mb': student_acc / student_size if student_size > 0 else 0,
        'performance_per_million_params': student_acc / (student_params / 1e6) if student_params > 0 else 0,
    }


def print_metrics_report(
    teacher_metrics: Dict,
    student_metrics: Dict,
    distillation_metrics: Dict
):
    """Print a comprehensive metrics report."""
    print("\n" + "="*80)
    print("KNOWLEDGE DISTILLATION PERFORMANCE METRICS")
    print("="*80)
    
    print("\n" + "-"*80)
    print("MODEL SIZE COMPARISON")
    print("-"*80)
    teacher_size = teacher_metrics.get('size', {})
    student_size = student_metrics.get('size', {})
    
    print(f"\nTeacher Model:")
    print(f"  Disk Size: {teacher_size.get('disk_size_mb', 0):.2f} MB ({teacher_size.get('disk_size_gb', 0):.4f} GB)")
    print(f"  Parameters: {teacher_size.get('num_parameters_millions', 0):.2f}M parameters" if teacher_size.get('num_parameters') else "  Parameters: N/A")
    print(f"  Trainable Parameters: {teacher_size.get('trainable_parameters', 0):,}" if teacher_size.get('trainable_parameters') else "")
    
    print(f"\nStudent Model:")
    print(f"  Disk Size: {student_size.get('disk_size_mb', 0):.2f} MB ({student_size.get('disk_size_gb', 0):.4f} GB)")
    print(f"  Parameters: {student_size.get('num_parameters_millions', 0):.2f}M parameters" if student_size.get('num_parameters') else "  Parameters: N/A")
    print(f"  Trainable Parameters: {student_size.get('trainable_parameters', 0):,}" if student_size.get('trainable_parameters') else "")
    
    print(f"\nSize Reduction:")
    print(f"  Reduction: {distillation_metrics.get('size_reduction_mb', 0):.2f} MB ({distillation_metrics.get('size_reduction_percent', 0):.1f}%)")
    print(f"  Compression Ratio: {distillation_metrics.get('compression_ratio', 0):.2f}x")
    
    print("\n" + "-"*80)
    print("ACCURACY METRICS")
    print("-"*80)
    teacher_acc = teacher_metrics.get('accuracy', 0)
    student_acc = student_metrics.get('accuracy', 0)
    
    print(f"\nTeacher Accuracy: {teacher_acc:.4f} ({teacher_acc*100:.2f}%)")
    print(f"Student Accuracy: {student_acc:.4f} ({student_acc*100:.2f}%)")
    print(f"\nKnowledge Retention: {distillation_metrics.get('accuracy_retention_percent', 0):.2f}%")
    print(f"Accuracy Loss: {distillation_metrics.get('accuracy_loss', 0):.4f} ({distillation_metrics.get('accuracy_loss_percent', 0):.2f}%)")
    
    print("\n" + "-"*80)
    print("INFERENCE PERFORMANCE")
    print("-"*80)
    teacher_bench = teacher_metrics.get('benchmark', {})
    student_bench = student_metrics.get('benchmark', {})
    
    print(f"\nTeacher Model:")
    print(f"  Avg Inference Time: {teacher_bench.get('avg_inference_time_seconds', 0):.4f} seconds")
    print(f"  Tokens/Second: {teacher_bench.get('tokens_per_second', 0):.2f}")
    print(f"  Samples/Second: {teacher_bench.get('samples_per_second', 0):.4f}")
    print(f"  Model Memory: {teacher_bench.get('model_memory_mb', 0):.2f} MB")
    
    print(f"\nStudent Model:")
    print(f"  Avg Inference Time: {student_bench.get('avg_inference_time_seconds', 0):.4f} seconds")
    print(f"  Tokens/Second: {student_bench.get('tokens_per_second', 0):.2f}")
    print(f"  Samples/Second: {student_bench.get('samples_per_second', 0):.4f}")
    print(f"  Model Memory: {student_bench.get('model_memory_mb', 0):.2f} MB")
    
    print(f"\nSpeed Improvement:")
    print(f"  Time Saved: {distillation_metrics.get('speed_improvement_seconds', 0):.4f} seconds ({distillation_metrics.get('speed_improvement_percent', 0):.1f}%)")
    print(f"  Throughput Improvement: {distillation_metrics.get('throughput_improvement', 0):.2f}x")
    
    print("\n" + "-"*80)
    print("DISTILLATION EFFICACY SUMMARY")
    print("-"*80)
    print(f"\n{'Metric':<50} {'Value':>28}")
    print("-"*80)
    print(f"{'Size Reduction':<50} {distillation_metrics.get('size_reduction_percent', 0):>27.2f}%")
    print(f"{'Parameter Reduction':<50} {distillation_metrics.get('parameter_reduction_percent', 0):>27.2f}%")
    print(f"{'Accuracy Retention':<50} {distillation_metrics.get('accuracy_retention_percent', 0):>27.2f}%")
    print(f"{'Speed Improvement':<50} {distillation_metrics.get('speed_improvement_percent', 0):>27.2f}%")
    print(f"{'Memory Reduction':<50} {distillation_metrics.get('memory_reduction_percent', 0):>27.2f}%")
    print(f"{'Efficiency Score':<50} {distillation_metrics.get('efficiency_score', 0):>27.4f}")
    print(f"{'Performance per MB':<50} {distillation_metrics.get('performance_per_mb', 0):>27.6f}")
    print(f"{'Performance per Million Params':<50} {distillation_metrics.get('performance_per_million_params', 0):>27.6f}")
    print("-"*80)
    
    print("\n" + "="*80)
    print("END OF METRICS REPORT")
    print("="*80 + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compute comprehensive distillation performance metrics"
    )
    parser.add_argument(
        '--teacher-model',
        type=str,
        required=True,
        help='Path to teacher model directory'
    )
    parser.add_argument(
        '--student-model',
        type=str,
        required=True,
        help='Path to student model directory'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/quantumbench.jsonl',
        help='Path to JSONL evaluation data'
    )
    parser.add_argument(
        '--benchmark-samples',
        type=int,
        default=10,
        help='Number of samples for inference benchmarking'
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=256,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='metrics/distillation_metrics.json',
        help='Path to save metrics JSON file'
    )
    
    args = parser.parse_args()
    
    teacher_path = Path(args.teacher_model)
    student_path = Path(args.student_model)
    data_path = Path(args.data_path)
    
    # Validate paths
    if not teacher_path.exists():
        raise FileNotFoundError(f"Teacher model not found: {teacher_path}")
    if not student_path.exists():
        raise FileNotFoundError(f"Student model not found: {student_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load data
    print("Loading evaluation dataset...")
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(f"Loaded {len(data)} examples")
    
    results = {}
    
    # Evaluate Teacher Model
    print("\n" + "="*80)
    print("EVALUATING TEACHER MODEL")
    print("="*80)
    
    teacher_metrics = {}
    
    print("\n1. Computing model size...")
    teacher_metrics['size'] = get_model_size(teacher_path)
    
    print("\n2. Benchmarking inference...")
    teacher_metrics['benchmark'] = benchmark_inference(
        teacher_path, data, args.benchmark_samples, args.max_new_tokens
    )
    
    print("\n3. Computing accuracy...")
    teacher_predictions, teacher_accuracy = evaluate_model_full(
        teacher_path, data, args.max_new_tokens
    )
    teacher_metrics['accuracy'] = teacher_accuracy
    teacher_metrics['num_examples'] = len(data)
    
    results['teacher'] = teacher_metrics
    
    # Evaluate Student Model
    print("\n" + "="*80)
    print("EVALUATING STUDENT MODEL")
    print("="*80)
    
    student_metrics = {}
    
    print("\n1. Computing model size...")
    student_metrics['size'] = get_model_size(student_path)
    
    print("\n2. Benchmarking inference...")
    student_metrics['benchmark'] = benchmark_inference(
        student_path, data, args.benchmark_samples, args.max_new_tokens
    )
    
    print("\n3. Computing accuracy...")
    student_predictions, student_accuracy = evaluate_model_full(
        student_path, data, args.max_new_tokens
    )
    student_metrics['accuracy'] = student_accuracy
    student_metrics['num_examples'] = len(data)
    
    results['student'] = student_metrics
    
    # Compute Distillation Metrics
    print("\n" + "="*80)
    print("COMPUTING DISTILLATION METRICS")
    print("="*80)
    
    distillation_metrics = compute_distillation_metrics(teacher_metrics, student_metrics)
    results['distillation'] = distillation_metrics
    
    # Print report
    print_metrics_report(teacher_metrics, student_metrics, distillation_metrics)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare JSON-safe results (exclude non-serializable data)
    json_results = {
        'teacher': {
            k: v for k, v in teacher_metrics.items()
            if k != 'benchmark' or isinstance(v, dict)
        },
        'student': {
            k: v for k, v in student_metrics.items()
            if k != 'benchmark' or isinstance(v, dict)
        },
        'distillation': distillation_metrics
    }
    
    # Include benchmark summary (not full inference times list)
    json_results['teacher']['benchmark_summary'] = {
        k: v for k, v in teacher_metrics.get('benchmark', {}).items()
        if k != 'inference_times'
    }
    json_results['student']['benchmark_summary'] = {
        k: v for k, v in student_metrics.get('benchmark', {}).items()
        if k != 'inference_times'
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Metrics saved to {output_path}")
    
    return results


if __name__ == "__main__":
    main()

