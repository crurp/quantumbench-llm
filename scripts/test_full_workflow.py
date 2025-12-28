#!/usr/bin/env python3
"""
test_full_workflow.py

Main orchestration script for testing and visualizing the QuantumBench LLM workflow.
Runs all tests, evaluations, comparisons, and generates visualizations.

Usage:
    python scripts/test_full_workflow.py --data-path data/quantumbench.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

# Import test modules
import test_dataset
import test_models
import compare_teacher_student
import visualize_results
import system_check


def run_full_workflow(
    data_path: Path,
    teacher_model_path: Path = None,
    student_model_path: Path = None,
    num_samples: int = None,
    output_dir: Path = None,
    skip_system_check: bool = False,
    skip_training_test: bool = True
) -> Dict:
    """
    Run complete workflow testing and visualization.
    
    Args:
        data_path: Path to JSONL dataset
        teacher_model_path: Path to teacher model (default: models/teacher-model)
        student_model_path: Path to student model (default: models/student-model)
        num_samples: Number of samples to evaluate (None = all)
        output_dir: Directory for output files (default: plots/)
        skip_system_check: Skip system checks
        skip_training_test: Skip training step test (faster)
        
    Returns:
        Dictionary with all results
    """
    print("\n" + "="*70)
    print("QUANTUMBENCH LLM FULL WORKFLOW TEST")
    print("="*70 + "\n")
    
    # Set defaults
    if teacher_model_path is None:
        teacher_model_path = Path('models/teacher-model')
    if student_model_path is None:
        student_model_path = Path('models/student-model')
    if output_dir is None:
        output_dir = Path('plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'dataset_verification': None,
        'system_check': None,
        'teacher_evaluation': None,
        'student_evaluation': None,
        'comparison_metrics': None,
        'visualizations': None
    }
    
    # 1. System checks
    if not skip_system_check:
        print("\n[1/6] Running system checks...")
        try:
            folder_results = system_check.check_folders()
            model_results = system_check.check_models()
            memory_info = system_check.get_memory_info()
            dependency_results = system_check.check_dependencies()
            
            results['system_check'] = {
                'folders': folder_results,
                'models': model_results,
                'memory': memory_info,
                'dependencies': dependency_results
            }
            print("✓ System checks complete")
        except Exception as e:
            print(f"✗ System checks failed: {e}")
            results['system_check'] = {'error': str(e)}
    
    # 2. Dataset verification
    print("\n[2/6] Verifying dataset...")
    try:
        dataset_results = test_dataset.verify_dataset(data_path)
        results['dataset_verification'] = dataset_results['stats']
        print("✓ Dataset verification complete")
    except Exception as e:
        print(f"✗ Dataset verification failed: {e}")
        results['dataset_verification'] = {'error': str(e)}
        return results
    
    # Load dataset for model evaluation
    data = dataset_results['data']
    
    # 3. Teacher model evaluation
    print("\n[3/6] Evaluating teacher model...")
    try:
        teacher_results = test_models.evaluate_model(
            teacher_model_path,
            data,
            'teacher',
            num_samples,
            256
        )
        results['teacher_evaluation'] = {
            'accuracy': teacher_results.get('accuracy', 0.0),
            'num_evaluated': teacher_results.get('num_evaluated', 0),
            'exists': teacher_results.get('exists', False)
        }
        print("✓ Teacher evaluation complete")
    except Exception as e:
        print(f"✗ Teacher evaluation failed: {e}")
        results['teacher_evaluation'] = {'error': str(e)}
        teacher_results = {'exists': False}
    
    # 4. Student model evaluation
    print("\n[4/6] Evaluating student model...")
    try:
        student_results = test_models.evaluate_model(
            student_model_path,
            data,
            'student',
            num_samples,
            256
        )
        results['student_evaluation'] = {
            'accuracy': student_results.get('accuracy', 0.0),
            'num_evaluated': student_results.get('num_evaluated', 0),
            'exists': student_results.get('exists', False)
        }
        print("✓ Student evaluation complete")
    except Exception as e:
        print(f"✗ Student evaluation failed: {e}")
        results['student_evaluation'] = {'error': str(e)}
        student_results = {'exists': False}
    
    # 5. Comparison and metrics
    print("\n[5/6] Computing comparison metrics...")
    comparison_output_path = output_dir / 'comparison_results.json'
    try:
        comparison_metrics = compare_teacher_student.compare_predictions(
            teacher_results,
            student_results,
            data
        )
        results['comparison_metrics'] = comparison_metrics
        
        # Save comparison results for visualization
        comparison_data = {
            'metrics': comparison_metrics,
            'teacher_results': {
                k: v for k, v in teacher_results.items()
                if k not in ['predictions', 'ground_truths', 'instructions']
            },
            'student_results': {
                k: v for k, v in student_results.items()
                if k not in ['predictions', 'ground_truths', 'instructions']
            }
        }
        
        # Include limited predictions for visualization
        if teacher_results.get('exists') and student_results.get('exists'):
            comparison_data['teacher_results']['predictions'] = teacher_results.get('predictions', [])
            comparison_data['teacher_results']['ground_truths'] = teacher_results.get('ground_truths', [])
            comparison_data['student_results']['predictions'] = student_results.get('predictions', [])
            comparison_data['student_results']['ground_truths'] = student_results.get('ground_truths', [])
        
        with open(comparison_output_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"✓ Comparison metrics saved to {comparison_output_path}")
        
    except Exception as e:
        print(f"✗ Comparison failed: {e}")
        results['comparison_metrics'] = {'error': str(e)}
    
    # 6. Generate visualizations
    print("\n[6/6] Generating visualizations...")
    try:
        comparison_output_path = output_dir / 'comparison_results.json'
        if comparison_output_path.exists():
            # Import and run visualization
            import sys
            old_argv = sys.argv
            sys.argv = ['visualize_results.py',
                       '--comparison-results', str(comparison_output_path),
                       '--output-dir', str(output_dir)]
            try:
                visualize_results.main()
            finally:
                sys.argv = old_argv
            
            results['visualizations'] = {'success': True, 'output_dir': str(output_dir)}
            print("✓ Visualizations generated")
        else:
            print("⚠ Skipping visualizations: comparison results not found")
            results['visualizations'] = {'skipped': True}
    except Exception as e:
        print(f"✗ Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()
        results['visualizations'] = {'error': str(e)}
    
    # Print final summary
    print("\n" + "="*70)
    print("WORKFLOW TEST SUMMARY")
    print("="*70)
    
    if results['teacher_evaluation'] and results['teacher_evaluation'].get('exists'):
        print(f"Teacher Model Accuracy: {results['teacher_evaluation']['accuracy']:.4f} ({results['teacher_evaluation']['accuracy']*100:.2f}%)")
    
    if results['student_evaluation'] and results['student_evaluation'].get('exists'):
        print(f"Student Model Accuracy: {results['student_evaluation']['accuracy']:.4f} ({results['student_evaluation']['accuracy']*100:.2f}%)")
    
    if results['comparison_metrics'] and 'delta_accuracy' in results['comparison_metrics']:
        delta = results['comparison_metrics']['delta_accuracy']
        print(f"Accuracy Delta: {delta:+.4f} ({delta*100:+.2f}%)")
        print(f"Exact Match Rate: {results['comparison_metrics']['exact_match_rate']:.4f} ({results['comparison_metrics']['exact_match_rate']*100:.2f}%)")
    
    print(f"\nVisualizations saved to: {output_dir}")
    print("="*70 + "\n")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Test and visualize complete QuantumBench LLM workflow"
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/quantumbench.jsonl',
        help='Path to JSONL dataset'
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
        '--num-samples',
        type=int,
        default=None,
        help='Number of samples to evaluate (None = all, default: None for full evaluation)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='plots',
        help='Directory for output files'
    )
    parser.add_argument(
        '--skip-system-check',
        action='store_true',
        help='Skip system checks'
    )
    parser.add_argument(
        '--save-results',
        type=str,
        default=None,
        help='Path to save full results JSON (optional)'
    )
    
    args = parser.parse_args()
    
    # Run full workflow
    results = run_full_workflow(
        Path(args.data_path),
        Path(args.teacher_model),
        Path(args.student_model),
        args.num_samples,
        Path(args.output_dir),
        args.skip_system_check
    )
    
    # Save results if requested
    if args.save_results:
        output_path = Path(args.save_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Full results saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

