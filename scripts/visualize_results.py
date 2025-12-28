#!/usr/bin/env python3
"""
visualize_results.py

Generate visualizations for QuantumBench LLM workflow results.
Creates bar charts, line plots, confusion matrices, and heatmaps.

Usage:
    python scripts/visualize_results.py \
        --comparison-results comparison_results.json \
        --output-dir plots/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_comparison_results(results_path: Path) -> Dict:
    """Load comparison results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_accuracy_comparison(metrics: Dict, output_dir: Path):
    """
    Generate bar chart comparing teacher vs student accuracy.
    
    Args:
        metrics: Dictionary with comparison metrics
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['Teacher', 'Student']
    accuracies = [metrics['teacher_accuracy'], metrics['student_accuracy']]
    colors = ['#2ecc71', '#3498db']
    
    bars = ax.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Teacher vs Student Model Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(accuracies) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'accuracy_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_delta_accuracy(metrics: Dict, output_dir: Path):
    """
    Generate bar chart showing delta accuracy.
    
    Args:
        metrics: Dictionary with comparison metrics
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    delta = metrics['delta_accuracy']
    color = 'red' if delta > 0 else 'green'
    
    ax.barh(['Delta'], [delta], color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Accuracy Difference (Teacher - Student)', fontsize=12, fontweight='bold')
    ax.set_title('Distillation Efficacy: Accuracy Delta', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value label
    ax.text(delta, 0, f'{delta:+.2%}',
            ha='left' if delta > 0 else 'right', va='center',
            fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'delta_accuracy.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_comprehensive_metrics(metrics: Dict, output_dir: Path):
    """
    Generate comprehensive bar chart with multiple metrics.
    
    Args:
        metrics: Dictionary with comparison metrics
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comprehensive Teacher-Student Comparison Metrics', fontsize=16, fontweight='bold')
    
    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    models = ['Teacher', 'Student']
    accuracies = [metrics['teacher_accuracy'], metrics['student_accuracy']]
    bars = ax1.bar(models, accuracies, color=['#2ecc71', '#3498db'], alpha=0.7)
    ax1.set_ylabel('Accuracy', fontweight='bold')
    ax1.set_title('Accuracy Comparison', fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Exact match rate
    ax2 = axes[0, 1]
    match_rate = metrics['exact_match_rate']
    ax2.barh(['Match Rate'], [match_rate], color='#e74c3c', alpha=0.7)
    ax2.set_xlabel('Rate', fontweight='bold')
    ax2.set_title('Exact Match Rate', fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3)
    ax2.text(match_rate, 0, f'{match_rate:.2%}',
            ha='left', va='center', fontweight='bold')
    
    # 3. Agreement metrics
    ax3 = axes[1, 0]
    agreement_metrics = [
        metrics['student_agreement_when_teacher_correct'],
        metrics['student_correct_when_teacher_wrong']
    ]
    agreement_labels = ['Agreement\n(Teacher Correct)', 'Student Correct\n(Teacher Wrong)']
    bars = ax3.barh(agreement_labels, agreement_metrics, color=['#9b59b6', '#f39c12'], alpha=0.7)
    ax3.set_xlabel('Rate', fontweight='bold')
    ax3.set_title('Agreement Metrics', fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.grid(axis='x', alpha=0.3)
    for bar, val in zip(bars, agreement_metrics):
        ax3.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                f'{val:.2%}', ha='left', va='center', fontweight='bold')
    
    # 4. Delta accuracy
    ax4 = axes[1, 1]
    delta = metrics['delta_accuracy']
    color = 'red' if delta > 0 else 'green'
    ax4.barh(['Delta'], [abs(delta)], color=color, alpha=0.7)
    ax4.set_xlabel('Absolute Accuracy Difference', fontweight='bold')
    ax4.set_title(f'Accuracy Delta: {delta:+.2%}', fontweight='bold')
    ax4.set_xlim(0, max(0.5, abs(delta) * 1.5))
    ax4.grid(axis='x', alpha=0.3)
    ax4.text(abs(delta), 0, f'{delta:+.2%}',
            ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'comprehensive_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_confusion_matrices(teacher_results: Dict, student_results: Dict, output_dir: Path):
    """
    Generate confusion matrices for teacher and student predictions.
    
    Args:
        teacher_results: Teacher model evaluation results
        student_results: Student model evaluation results
        output_dir: Directory to save plots
    """
    if not teacher_results.get('exists') or not student_results.get('exists'):
        print("⚠ Skipping confusion matrices: models not found")
        return
    
    def create_confusion_matrix_data(predictions, ground_truths):
        """Create binary confusion matrix (correct/incorrect)."""
        def normalize_text(text):
            import re
            return re.sub(r'\s+', ' ', text.lower().strip())
        
        y_true = []
        y_pred = []
        
        for pred, gt in zip(predictions, ground_truths):
            pred_norm = normalize_text(pred)
            gt_norm = normalize_text(gt)
            is_correct = pred_norm == gt_norm or pred_norm in gt_norm or gt_norm in pred_norm
            
            y_true.append(1)  # Always should be correct
            y_pred.append(1 if is_correct else 0)
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return cm
    
    teacher_cm = create_confusion_matrix_data(
        teacher_results['predictions'],
        teacher_results['ground_truths']
    )
    
    student_cm = create_confusion_matrix_data(
        student_results['predictions'],
        student_results['ground_truths']
    )
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Confusion Matrices: Teacher vs Student', fontsize=16, fontweight='bold')
    
    for idx, (cm, model_name) in enumerate([(teacher_cm, 'Teacher'), (student_cm, 'Student')]):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Incorrect', 'Correct'],
                   yticklabels=['Incorrect', 'Correct'])
        ax.set_title(f'{model_name} Model', fontweight='bold')
        ax.set_ylabel('Actual', fontweight='bold')
        ax.set_xlabel('Predicted', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'confusion_matrices.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_prediction_heatmap(teacher_results: Dict, student_results: Dict, output_dir: Path, max_examples: int = 50):
    """
    Generate heatmap showing prediction agreement across examples.
    
    Args:
        teacher_results: Teacher model evaluation results
        student_results: Student model evaluation results
        output_dir: Directory to save plots
        max_examples: Maximum number of examples to show
    """
    if not teacher_results.get('exists') or not student_results.get('exists'):
        print("⚠ Skipping prediction heatmap: models not found")
        return
    
    def normalize_text(text):
        import re
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    teacher_preds = teacher_results['predictions'][:max_examples]
    student_preds = student_results['predictions'][:max_examples]
    ground_truths = teacher_results['ground_truths'][:max_examples]
    
    # Create agreement matrix
    agreement_matrix = []
    for t_pred, s_pred, gt in zip(teacher_preds, student_preds, ground_truths):
        t_norm = normalize_text(t_pred)
        s_norm = normalize_text(s_pred)
        gt_norm = normalize_text(gt)
        
        t_correct = t_norm == gt_norm or t_norm in gt_norm or gt_norm in t_norm
        s_correct = s_norm == gt_norm or s_norm in gt_norm or gt_norm in s_norm
        match = t_norm == s_norm
        
        # 0: Both wrong, 1: Teacher correct only, 2: Student correct only, 3: Both correct
        # 4: Match but both wrong, 5: Match and both correct
        if match and t_correct:
            value = 5  # Perfect match and correct
        elif match and not t_correct:
            value = 4  # Match but wrong
        elif t_correct and s_correct:
            value = 3  # Both correct but different
        elif t_correct:
            value = 2  # Teacher correct only
        elif s_correct:
            value = 1  # Student correct only
        else:
            value = 0  # Both wrong
        
        agreement_matrix.append([value])
    
    agreement_array = np.array(agreement_matrix).T
    
    fig, ax = plt.subplots(figsize=(max(12, len(agreement_matrix) // 5), 4))
    
    cmap = plt.cm.get_cmap('RdYlGn', 6)
    sns.heatmap(agreement_array, cmap=cmap, cbar_kws={'label': 'Agreement Level'},
                yticklabels=['Examples'], xticklabels=False, ax=ax)
    
    ax.set_title('Prediction Agreement Heatmap (Teacher vs Student)', fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Example Index (showing {len(agreement_matrix)} examples)', fontweight='bold')
    
    # Add colorbar labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    cbar.set_ticklabels(['Both Wrong', 'Student Only', 'Teacher Only', 'Both Correct (Diff)', 'Match Wrong', 'Match Correct'])
    
    plt.tight_layout()
    output_path = output_dir / 'prediction_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_delta_across_examples(teacher_results: Dict, student_results: Dict, output_dir: Path):
    """
    Generate line plot showing accuracy delta across examples.
    
    Args:
        teacher_results: Teacher model evaluation results
        student_results: Student model evaluation results
        output_dir: Directory to save plots
    """
    if not teacher_results.get('exists') or not student_results.get('exists'):
        print("⚠ Skipping delta plot: models not found")
        return
    
    def normalize_text(text):
        import re
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    teacher_preds = teacher_results['predictions']
    student_preds = student_results['predictions']
    ground_truths = teacher_results['ground_truths']
    
    # Compute cumulative accuracy
    teacher_cumulative = []
    student_cumulative = []
    
    teacher_correct = 0
    student_correct = 0
    
    for i, (t_pred, s_pred, gt) in enumerate(zip(teacher_preds, student_preds, ground_truths)):
        t_norm = normalize_text(t_pred)
        s_norm = normalize_text(s_pred)
        gt_norm = normalize_text(gt)
        
        if t_norm == gt_norm or t_norm in gt_norm or gt_norm in t_norm:
            teacher_correct += 1
        if s_norm == gt_norm or s_norm in gt_norm or gt_norm in s_norm:
            student_correct += 1
        
        teacher_cumulative.append(teacher_correct / (i + 1))
        student_cumulative.append(student_correct / (i + 1))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(1, len(teacher_cumulative) + 1)
    ax.plot(x, teacher_cumulative, label='Teacher', linewidth=2, color='#2ecc71')
    ax.plot(x, student_cumulative, label='Student', linewidth=2, color='#3498db')
    
    # Compute delta
    delta = [t - s for t, s in zip(teacher_cumulative, student_cumulative)]
    ax.plot(x, delta, label='Delta (Teacher - Student)', linewidth=2, linestyle='--', color='#e74c3c')
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1)
    
    ax.set_xlabel('Example Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Accuracy and Delta Across Examples', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'delta_across_examples.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for QuantumBench LLM results"
    )
    parser.add_argument(
        '--comparison-results',
        type=str,
        required=True,
        help='Path to comparison results JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots',
        help='Directory to save plots'
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.comparison_results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    print(f"Loading results from: {results_path}")
    
    results = load_comparison_results(results_path)
    metrics = results.get('metrics', {})
    
    # Check if we have valid metrics
    if not metrics or 'teacher_accuracy' not in metrics:
        print("⚠ No valid comparison metrics found. Models may not be trained yet.")
        print("   Skipping visualization generation.")
        return
    
    print("\nGenerating plots...")
    
    # Generate all plots
    plot_accuracy_comparison(metrics, output_dir)
    plot_delta_accuracy(metrics, output_dir)
    plot_comprehensive_metrics(metrics, output_dir)
    plot_confusion_matrices(results.get('teacher_results', {}), results.get('student_results', {}), output_dir)
    plot_prediction_heatmap(results.get('teacher_results', {}), results.get('student_results', {}), output_dir)
    plot_delta_across_examples(results.get('teacher_results', {}), results.get('student_results', {}), output_dir)
    
    print("\n" + "="*70)
    print("✓ All visualizations generated successfully!")
    print(f"   Saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

