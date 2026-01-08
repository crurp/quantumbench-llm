#!/usr/bin/env python3
"""
visualize_training.py

Generate comprehensive visualizations of the training process from training log.
Creates charts showing loss progression, learning rate schedule, training speed, etc.

Usage:
    python scripts/visualize_training.py --log-file training_teacher.log --output-dir plots/
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def parse_training_log(log_path: Path) -> Dict:
    """
    Parse training log file to extract metrics.
    
    Args:
        log_path: Path to training log file
        
    Returns:
        Dictionary with extracted training metrics
    """
    print(f"Parsing training log: {log_path}")
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    # Extract metrics from log entries
    steps = []
    losses = []
    epochs = []
    learning_rates = []
    grad_norms = []
    times = []
    
    # Parse line by line to find metric dictionaries
    for i, line in enumerate(lines):
        # Look for dictionary entries with metrics
        if "'loss'" in line and "'epoch'" in line:
            # Extract step number from previous progress bar line
            step_num = None
            for j in range(max(0, i-5), i):
                step_match = re.search(r'\| (\d+)/(\d+) \[', lines[j])
                if step_match:
                    step_num = int(step_match.group(1))
                    total_steps = int(step_match.group(2))
                    break
            
            if step_num is not None:
                # Extract metrics from dict
                loss_match = re.search(r"'loss': ([0-9.]+)", line)
                epoch_match = re.search(r"'epoch': ([0-9.]+)", line)
                lr_match = re.search(r"'learning_rate': ([0-9.e-]+)", line)
                grad_match = re.search(r"'grad_norm': ([0-9.]+)", line)
                
                if loss_match and epoch_match:
                    steps.append(step_num)
                    losses.append(float(loss_match.group(1)))
                    epochs.append(float(epoch_match.group(1)))
                    if lr_match:
                        learning_rates.append(float(lr_match.group(1)))
                    if grad_match:
                        grad_norms.append(float(grad_match.group(1)))
                    
                    # Extract time from progress bar
                    time_match = re.search(r'\[([\d:]+)<', lines[max(0, i-1)])
                    if time_match:
                        time_str = time_match.group(1)
                        parts = time_str.split(':')
                        if len(parts) == 3:
                            hours, mins, secs = map(int, parts)
                            total_seconds = hours * 3600 + mins * 60 + secs
                            times.append(total_seconds)
    
    # Extract total steps
    total_steps = 0
    step_pattern = r'\| (\d+)/(\d+) \['
    for line in lines:
        match = re.search(step_pattern, line)
        if match:
            total_steps = max(total_steps, int(match.group(2)))
    
    # Extract runtime stats
    runtime_match = re.search(r"'train_runtime': ([0-9.]+)", ''.join(lines))
    train_loss_match = re.search(r"'train_loss': ([0-9.]+)", ''.join(lines))
    
    # Fill missing values with last known value
    if len(learning_rates) < len(steps):
        learning_rates = learning_rates + [learning_rates[-1]] * (len(steps) - len(learning_rates)) if learning_rates else [0] * len(steps)
    if len(grad_norms) < len(steps):
        grad_norms = grad_norms + [grad_norms[-1]] * (len(steps) - len(grad_norms)) if grad_norms else [0] * len(steps)
    
    return {
        'steps': steps,
        'total_steps': total_steps if total_steps else len(steps),
        'losses': losses,
        'epochs': epochs,
        'learning_rates': learning_rates[:len(steps)] if learning_rates else [],
        'grad_norms': grad_norms[:len(steps)] if grad_norms else [],
        'times': times[:len(steps)] if times else [],
        'total_runtime': float(runtime_match.group(1)) if runtime_match else None,
        'final_train_loss': float(train_loss_match.group(1)) if train_loss_match else None,
    }


def plot_loss_progression(metrics: Dict, output_dir: Path):
    """Plot training loss over steps."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = metrics['steps']
    losses = metrics['losses']
    
    ax.plot(steps, losses, marker='o', linewidth=2, markersize=6, color='#2ecc71')
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Teacher Model Training: Loss Progression', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    if len(losses) > 0:
        initial_loss = losses[0]
        final_loss = losses[-1]
        improvement = ((initial_loss - final_loss) / initial_loss) * 100
        
        ax.annotate(f'Initial: {initial_loss:.4f}', 
                   xy=(steps[0], initial_loss), 
                   xytext=(10, 20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.annotate(f'Final: {final_loss:.4f}\n({improvement:.1f}% improvement)', 
                   xy=(steps[-1], final_loss), 
                   xytext=(10, -30), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    output_path = output_dir / 'teacher_training_loss.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_loss_by_epoch(metrics: Dict, output_dir: Path):
    """Plot loss progression by epoch."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    epochs = metrics['epochs']
    losses = metrics['losses']
    
    ax.plot(epochs, losses, marker='o', linewidth=2, markersize=6, color='#3498db')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title('Teacher Model Training: Loss by Epoch', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(0, max(epochs) + 0.5, 0.5))
    
    plt.tight_layout()
    output_path = output_dir / 'teacher_training_loss_by_epoch.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_learning_rate_schedule(metrics: Dict, output_dir: Path):
    """Plot learning rate schedule over training."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = metrics['steps']
    learning_rates = metrics['learning_rates']
    
    ax.plot(steps, learning_rates, marker='s', linewidth=2, markersize=6, color='#e74c3c')
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    ax.set_title('Teacher Model Training: Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale for learning rate
    
    plt.tight_layout()
    output_path = output_dir / 'teacher_learning_rate.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_gradient_norms(metrics: Dict, output_dir: Path):
    """Plot gradient norms over training."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = metrics['steps']
    grad_norms = metrics['grad_norms']
    
    ax.plot(steps, grad_norms, marker='^', linewidth=2, markersize=6, color='#9b59b6')
    ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
    ax.set_ylabel('Gradient Norm', fontsize=12, fontweight='bold')
    ax.set_title('Teacher Model Training: Gradient Norms', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at typical good range
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Target norm (~1.0)')
    ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / 'teacher_gradient_norms.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_training_speed(metrics: Dict, output_dir: Path):
    """Plot training speed (time per step) over training."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = metrics['steps']
    times = metrics['times']
    
    if len(times) > 1 and len(steps) > 1:
        # Calculate time per step
        step_times = []
        for i in range(1, len(times)):
            step_time = times[i] - times[i-1]  # seconds
            step_times.append(step_time / 60)  # convert to minutes
        
        if step_times and len(steps[1:]) == len(step_times):
            ax.plot(steps[1:], step_times, marker='d', linewidth=2, markersize=6, color='#f39c12')
            ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
            ax.set_ylabel('Time per Step (minutes)', fontsize=12, fontweight='bold')
            ax.set_title('Teacher Model Training: Training Speed', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            avg_time = np.mean(step_times)
            ax.axhline(y=avg_time, color='r', linestyle='--', alpha=0.5, 
                      label=f'Average: {avg_time:.1f} min/step')
            ax.legend()
    
    plt.tight_layout()
    output_path = output_dir / 'teacher_training_speed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_comprehensive_dashboard(metrics: Dict, output_dir: Path):
    """Create comprehensive dashboard with all metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Teacher Model Training: Comprehensive Dashboard', fontsize=16, fontweight='bold')
    
    steps = metrics['steps']
    losses = metrics['losses']
    epochs = metrics['epochs']
    learning_rates = metrics['learning_rates']
    grad_norms = metrics['grad_norms']
    
    # 1. Loss progression
    ax1 = axes[0, 0]
    ax1.plot(steps, losses, marker='o', linewidth=2, color='#2ecc71')
    ax1.set_xlabel('Step', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Loss Progression', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss by epoch
    ax2 = axes[0, 1]
    ax2.plot(epochs, losses, marker='o', linewidth=2, color='#3498db')
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Loss', fontweight='bold')
    ax2.set_title('Loss by Epoch', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(np.arange(0, max(epochs) + 0.5, 0.5))
    
    # 3. Learning rate
    ax3 = axes[1, 0]
    ax3.plot(steps, learning_rates, marker='s', linewidth=2, color='#e74c3c')
    ax3.set_xlabel('Step', fontweight='bold')
    ax3.set_ylabel('Learning Rate', fontweight='bold')
    ax3.set_title('Learning Rate Schedule', fontweight='bold')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # 4. Gradient norms
    ax4 = axes[1, 1]
    ax4.plot(steps, grad_norms, marker='^', linewidth=2, color='#9b59b6')
    ax4.set_xlabel('Step', fontweight='bold')
    ax4.set_ylabel('Gradient Norm', fontweight='bold')
    ax4.set_title('Gradient Norms', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    output_path = output_dir / 'teacher_training_dashboard.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def plot_training_summary_stats(metrics: Dict, output_dir: Path):
    """Create summary statistics visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Teacher Model Training: Summary Statistics', fontsize=16, fontweight='bold')
    
    # 1. Loss statistics
    ax1 = axes[0]
    losses = metrics['losses']
    if losses:
        loss_data = {
            'Initial': losses[0],
            'Final': losses[-1],
            'Minimum': min(losses),
            'Average': np.mean(losses),
        }
        bars = ax1.bar(loss_data.keys(), loss_data.values(), color=['#e74c3c', '#2ecc71', '#3498db', '#f39c12'])
        ax1.set_ylabel('Loss', fontweight='bold')
        ax1.set_title('Loss Statistics', fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, loss_data.values()):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Training metrics
    ax2 = axes[1]
    if metrics['total_runtime']:
        runtime_hours = metrics['total_runtime'] / 3600
        total_steps = metrics['total_steps']
        avg_time_per_step = metrics['total_runtime'] / total_steps / 60  # minutes
        
        metrics_data = {
            'Runtime\n(hours)': runtime_hours,
            'Total\nSteps': total_steps,
            'Avg Time/Step\n(minutes)': avg_time_per_step,
        }
        bars = ax2.bar(metrics_data.keys(), metrics_data.values(), color=['#3498db', '#9b59b6', '#f39c12'])
        ax2.set_ylabel('Value', fontweight='bold')
        ax2.set_title('Training Metrics', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, metrics_data.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.1f}' if val < 100 else f'{int(val)}',
                    ha='center', va='bottom', fontweight='bold')
    
    # 3. Improvement metrics
    ax3 = axes[2]
    if len(losses) > 1:
        initial_loss = losses[0]
        final_loss = losses[-1]
        improvement_pct = ((initial_loss - final_loss) / initial_loss) * 100
        reduction = initial_loss - final_loss
        
        improvement_data = {
            'Loss\nReduction': reduction,
            'Improvement\n(%)': improvement_pct,
        }
        bars = ax3.bar(improvement_data.keys(), improvement_data.values(), color=['#2ecc71', '#27ae60'])
        ax3.set_ylabel('Value', fontweight='bold')
        ax3.set_title('Training Improvement', fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, improvement_data.values()):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}' if val < 10 else f'{val:.1f}%',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / 'teacher_training_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations from training log"
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default='training_teacher.log',
        help='Path to training log file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots',
        help='Directory to save plots'
    )
    
    args = parser.parse_args()
    
    log_path = Path(args.log_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        return
    
    print("="*70)
    print("GENERATING TEACHER TRAINING VISUALIZATIONS")
    print("="*70)
    
    # Parse log
    metrics = parse_training_log(log_path)
    
    if not metrics['steps']:
        print("Warning: No training metrics found in log file")
        return
    
    print(f"Extracted {len(metrics['steps'])} training steps")
    
    # Generate all plots
    print("\nGenerating visualizations...")
    plot_loss_progression(metrics, output_dir)
    plot_loss_by_epoch(metrics, output_dir)
    plot_learning_rate_schedule(metrics, output_dir)
    plot_gradient_norms(metrics, output_dir)
    plot_training_speed(metrics, output_dir)
    plot_comprehensive_dashboard(metrics, output_dir)
    plot_training_summary_stats(metrics, output_dir)
    
    print("\n" + "="*70)
    print("✓ All teacher training visualizations generated!")
    print(f"   Saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

