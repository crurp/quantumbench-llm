#!/usr/bin/env python3
"""
compare_teacher_student.py

Compare teacher and student models, compute interaction metrics,
and analyze distillation efficacy.

Usage:
    python scripts/compare_teacher_student.py \
        --teacher-model models/teacher-model \
        --student-model models/student-model \
        --data-path data/quantumbench.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

from test_models import load_jsonl, evaluate_model, normalize_text


def compare_predictions(
    teacher_results: Dict,
    student_results: Dict,
    data: List[Dict]
) -> Dict:
    """
    Compare teacher and student predictions and compute metrics.
    
    Args:
        teacher_results: Results from teacher model evaluation
        student_results: Results from student model evaluation
        data: Original dataset
        
    Returns:
        Dictionary with comparison metrics
    """
    print("\n" + "="*70)
    print("TEACHER-STUDENT COMPARISON METRICS")
    print("="*70)
    
    if not teacher_results.get('exists') or not student_results.get('exists'):
        print("⚠ Warning: One or both models not found. Skipping comparison.")
        return {}
    
    teacher_preds = teacher_results['predictions']
    student_preds = student_results['predictions']
    ground_truths = teacher_results['ground_truths']
    
    if len(teacher_preds) != len(student_preds):
        raise ValueError("Mismatch in number of predictions")
    
    # Compute metrics
    n = len(teacher_preds)
    
    # 1. Exact match rate between teacher and student
    exact_matches = 0
    for t_pred, s_pred in zip(teacher_preds, student_preds):
        if normalize_text(t_pred) == normalize_text(s_pred):
            exact_matches += 1
    exact_match_rate = exact_matches / n
    
    # 2. Teacher accuracy
    teacher_correct = 0
    for t_pred, gt in zip(teacher_preds, ground_truths):
        t_norm = normalize_text(t_pred)
        gt_norm = normalize_text(gt)
        if t_norm == gt_norm or t_norm in gt_norm or gt_norm in t_norm:
            teacher_correct += 1
    teacher_accuracy = teacher_correct / n
    
    # 3. Student accuracy
    student_correct = 0
    for s_pred, gt in zip(student_preds, ground_truths):
        s_norm = normalize_text(s_pred)
        gt_norm = normalize_text(gt)
        if s_norm == gt_norm or s_norm in gt_norm or gt_norm in s_norm:
            student_correct += 1
    student_accuracy = student_correct / n
    
    # 4. Delta (teacher accuracy - student accuracy)
    delta_accuracy = teacher_accuracy - student_accuracy
    
    # 5. Consistency: proportion of student predictions matching teacher predictions
    # This includes both correct and incorrect matches
    consistency = exact_match_rate
    
    # 6. Agreement when teacher is correct
    teacher_correct_indices = []
    for i, (t_pred, gt) in enumerate(zip(teacher_preds, ground_truths)):
        t_norm = normalize_text(t_pred)
        gt_norm = normalize_text(gt)
        if t_norm == gt_norm or t_norm in gt_norm or gt_norm in t_norm:
            teacher_correct_indices.append(i)
    
    if teacher_correct_indices:
        student_agreement_when_teacher_correct = sum(
            1 for i in teacher_correct_indices
            if normalize_text(teacher_preds[i]) == normalize_text(student_preds[i])
        ) / len(teacher_correct_indices)
    else:
        student_agreement_when_teacher_correct = 0.0
    
    # 7. Student correct when teacher is incorrect
    teacher_incorrect_indices = [i for i in range(n) if i not in teacher_correct_indices]
    if teacher_incorrect_indices:
        student_correct_when_teacher_wrong = sum(
            1 for i in teacher_incorrect_indices
            if normalize_text(student_preds[i]) == normalize_text(ground_truths[i])
        ) / len(teacher_incorrect_indices)
    else:
        student_correct_when_teacher_wrong = 0.0
    
    # Compile metrics
    metrics = {
        'num_examples': n,
        'teacher_accuracy': teacher_accuracy,
        'student_accuracy': student_accuracy,
        'delta_accuracy': delta_accuracy,
        'exact_match_rate': exact_match_rate,
        'consistency': consistency,
        'student_agreement_when_teacher_correct': student_agreement_when_teacher_correct,
        'student_correct_when_teacher_wrong': student_correct_when_teacher_wrong,
        'teacher_correct_count': teacher_correct,
        'student_correct_count': student_correct,
        'exact_match_count': exact_matches
    }
    
    # Print summary table
    print("\n" + "-"*70)
    print("SUMMARY TABLE")
    print("-"*70)
    print(f"{'Metric':<50} {'Value':>18}")
    print("-"*70)
    print(f"{'Number of examples evaluated':<50} {n:>18}")
    print(f"{'Teacher accuracy':<50} {teacher_accuracy:>17.4f} ({teacher_accuracy*100:>5.2f}%)")
    print(f"{'Student accuracy':<50} {student_accuracy:>17.4f} ({student_accuracy*100:>5.2f}%)")
    print(f"{'Delta (Teacher - Student)':<50} {delta_accuracy:>17.4f} ({delta_accuracy*100:>5.2f}%)")
    print(f"{'Exact match rate (Teacher = Student)':<50} {exact_match_rate:>17.4f} ({exact_match_rate*100:>5.2f}%)")
    print(f"{'Consistency (predictions match)':<50} {consistency:>17.4f} ({consistency*100:>5.2f}%)")
    print(f"{'Student agreement when teacher correct':<50} {student_agreement_when_teacher_correct:>17.4f} ({student_agreement_when_teacher_correct*100:>5.2f}%)")
    print(f"{'Student correct when teacher wrong':<50} {student_correct_when_teacher_wrong:>17.4f} ({student_correct_when_teacher_wrong*100:>5.2f}%)")
    print("-"*70)
    print()
    
    # Print detailed comparison for first few examples
    print("\n" + "="*70)
    print("DETAILED COMPARISON (First 5 examples)")
    print("="*70 + "\n")
    
    for i in range(min(5, n)):
        t_pred = teacher_preds[i]
        s_pred = student_preds[i]
        gt = ground_truths[i]
        
        t_norm = normalize_text(t_pred)
        s_norm = normalize_text(s_pred)
        gt_norm = normalize_text(gt)
        
        t_correct = t_norm == gt_norm or t_norm in gt_norm or gt_norm in t_norm
        s_correct = s_norm == gt_norm or s_norm in gt_norm or gt_norm in s_norm
        match = t_norm == s_norm
        
        print(f"Example {i+1}:")
        print(f"  Ground Truth: {gt[:100]}{'...' if len(gt) > 100 else ''}")
        print(f"  Teacher: {t_pred[:100]}{'...' if len(t_pred) > 100 else ''} {'✓' if t_correct else '✗'}")
        print(f"  Student: {s_pred[:100]}{'...' if len(s_pred) > 100 else ''} {'✓' if s_correct else '✗'}")
        print(f"  Match: {'✓' if match else '✗'}")
        print()
    
    return metrics


def compute_f1_score(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute F1 score using word-level matching.
    
    Args:
        predictions: List of predicted responses
        ground_truths: List of ground truth responses
        
    Returns:
        F1 score
    """
    from sklearn.metrics import precision_recall_fscore_support
    
    # Convert to binary labels (correct/incorrect)
    labels = []
    for pred, gt in zip(predictions, ground_truths):
        pred_norm = normalize_text(pred)
        gt_norm = normalize_text(gt)
        is_correct = pred_norm == gt_norm or pred_norm in gt_norm or gt_norm in pred_norm
        labels.append(1 if is_correct else 0)
    
    # F1 score for correct predictions
    true_labels = [1] * len(labels)  # All should be correct
    _, _, f1, _ = precision_recall_fscore_support(true_labels, labels, average='binary', zero_division=0)
    
    return f1


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Compare teacher and student models"
    )
    parser.add_argument(
        '--teacher-model',
        type=str,
        default='models/teacher-model',
        help='Path to teacher model directory'
    )
    parser.add_argument(
        '--student-model',
        type=str,
        default='models/student-model',
        help='Path to student model directory'
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
        default=None,
        help='Number of samples to evaluate (None = all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save comparison results JSON (optional)'
    )
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data_path)
    print("Loading dataset...")
    data = load_jsonl(data_path)
    print(f"Loaded {len(data)} examples")
    
    # Evaluate teacher
    from test_models import evaluate_model as eval_model
    teacher_results = eval_model(
        Path(args.teacher_model),
        data,
        'teacher',
        args.num_samples
    )
    
    # Evaluate student
    student_results = eval_model(
        Path(args.student_model),
        data,
        'student',
        args.num_samples
    )
    
    # Compare
    metrics = compare_predictions(teacher_results, student_results, data)
    
    # Compute F1 scores
    if teacher_results.get('exists') and student_results.get('exists'):
        teacher_f1 = compute_f1_score(
            teacher_results['predictions'],
            teacher_results['ground_truths']
        )
        student_f1 = compute_f1_score(
            student_results['predictions'],
            student_results['ground_truths']
        )
        
        metrics['teacher_f1'] = teacher_f1
        metrics['student_f1'] = student_f1
        
        print(f"\nF1 Scores:")
        print(f"  Teacher F1: {teacher_f1:.4f}")
        print(f"  Student F1: {student_f1:.4f}")
        print()
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'teacher_results': {k: v for k, v in teacher_results.items() if k != 'predictions'},  # Exclude full predictions
                'student_results': {k: v for k, v in student_results.items() if k != 'predictions'}
            }, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return metrics


if __name__ == "__main__":
    main()



