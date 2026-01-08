#!/usr/bin/env python3
"""
generate_simple_metrics.py

Simplified metrics generation that uses existing evaluation scripts.
Generates distillation performance metrics without complex PEFT loading.

Usage:
    python scripts/generate_simple_metrics.py
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

def get_model_size_mb(model_path: Path) -> float:
    """Get model size in MB."""
    if not model_path.exists():
        return 0.0
    
    total_size = 0
    for file_path in model_path.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size / (1024 * 1024)


def run_evaluation(model_type: str, model_path: str) -> dict:
    """Run evaluation script and parse results."""
    print(f"\n{'='*70}")
    print(f"Evaluating {model_type.upper()} model...")
    print(f"{'='*70}")
    
    try:
        result = subprocess.run(
            [
                sys.executable,
                'scripts/evaluate_quantumbench.py',
                f'--{model_type}-model',
                model_path,
                '--data-path',
                'data/quantumbench.jsonl',
            ],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        output = result.stdout
        
        # Parse accuracy from output
        accuracy = None
        for line in output.split('\n'):
            if 'Accuracy:' in line:
                # Look for pattern like "Accuracy: 0.8500 (85.00%)"
                parts = line.split('Accuracy:')
                if len(parts) > 1:
                    acc_str = parts[1].strip().split()[0]
                    try:
                        accuracy = float(acc_str)
                        break
                    except:
                        pass
        
        return {
            'success': result.returncode == 0,
            'accuracy': accuracy,
            'output': output[:1000]  # First 1000 chars
        }
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def main():
    """Generate simplified metrics."""
    print("="*70)
    print("DISTILLATION PERFORMANCE METRICS - SIMPLIFIED")
    print("="*70)
    
    teacher_path = Path('models/teacher-model')
    student_path = Path('models/student-model')
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'teacher': {},
        'student': {},
        'distillation': {}
    }
    
    # Get model sizes
    print("\n1. Computing model sizes...")
    teacher_size = get_model_size_mb(teacher_path)
    student_size = get_model_size_mb(student_path)
    
    results['teacher']['disk_size_mb'] = teacher_size
    results['student']['disk_size_mb'] = student_size
    results['teacher']['disk_size_gb'] = teacher_size / 1024
    results['student']['disk_size_gb'] = student_size / 1024
    
    print(f"  Teacher: {teacher_size:.2f} MB ({teacher_size/1024:.4f} GB)")
    print(f"  Student: {student_size:.2f} MB ({student_size/1024:.4f} GB)")
    
    # Evaluate teacher
    teacher_eval = run_evaluation('teacher', str(teacher_path))
    if teacher_eval['success'] and teacher_eval.get('accuracy') is not None:
        results['teacher']['accuracy'] = teacher_eval['accuracy']
    
    # Evaluate student
    student_eval = run_evaluation('student', str(student_path))
    if student_eval['success'] and student_eval.get('accuracy') is not None:
        results['student']['accuracy'] = student_eval['accuracy']
    
    # Compute distillation metrics
    if results['teacher'].get('accuracy') and results['student'].get('accuracy'):
        teacher_acc = results['teacher']['accuracy']
        student_acc = results['student']['accuracy']
        
        results['distillation'] = {
            'accuracy_delta': teacher_acc - student_acc,
            'accuracy_retention_percent': (student_acc / teacher_acc * 100) if teacher_acc > 0 else 0,
            'size_reduction_mb': teacher_size - student_size,
            'size_reduction_percent': ((teacher_size - student_size) / teacher_size * 100) if teacher_size > 0 else 0,
            'compression_ratio': teacher_size / student_size if student_size > 0 else 0,
        }
    
    # Print summary
    print("\n" + "="*70)
    print("METRICS SUMMARY")
    print("="*70)
    print(f"\nTeacher Model:")
    print(f"  Size: {teacher_size:.2f} MB")
    print(f"  Accuracy: {results['teacher'].get('accuracy', 'N/A')}")
    
    print(f"\nStudent Model:")
    print(f"  Size: {student_size:.2f} MB")
    print(f"  Accuracy: {results['student'].get('accuracy', 'N/A')}")
    
    if results.get('distillation'):
        print(f"\nDistillation Metrics:")
        print(f"  Accuracy Retention: {results['distillation'].get('accuracy_retention_percent', 0):.2f}%")
        print(f"  Size Reduction: {results['distillation'].get('size_reduction_percent', 0):.2f}%")
        print(f"  Compression Ratio: {results['distillation'].get('compression_ratio', 0):.2f}x")
    
    # Save results
    output_path = Path('metrics/distillation_metrics_simple.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Metrics saved to {output_path}")
    print("="*70)


if __name__ == "__main__":
    main()

