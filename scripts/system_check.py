#!/usr/bin/env python3
"""
system_check.py

System checks and memory testing for QuantumBench LLM workflow.
Verifies required folders, checks models, and tests pipeline functionality.

Usage:
    python scripts/system_check.py
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import psutil
import gc


def check_folders() -> Dict[str, bool]:
    """
    Verify required folders exist.
    
    Returns:
        Dictionary with folder existence status
    """
    print("="*70)
    print("FOLDER STRUCTURE CHECK")
    print("="*70)
    
    required_folders = {
        'data': Path('data'),
        'scripts': Path('scripts'),
        'models': Path('models'),
        'plots': Path('plots'),
        'notebooks': Path('notebooks'),
    }
    
    results = {}
    for name, folder_path in required_folders.items():
        exists = folder_path.exists() and folder_path.is_dir()
        results[name] = exists
        status = "✓" if exists else "✗"
        print(f"  {status} {name}/: {'EXISTS' if exists else 'MISSING'}")
    
    print()
    return results


def check_models() -> Dict[str, bool]:
    """
    Check if teacher and student models exist.
    
    Returns:
        Dictionary with model existence status
    """
    print("="*70)
    print("MODEL CHECK")
    print("="*70)
    
    teacher_path = Path('models/teacher-model')
    student_path = Path('models/student-model')
    
    teacher_exists = teacher_path.exists() and any(teacher_path.iterdir())
    student_exists = student_path.exists() and any(student_path.iterdir())
    
    results = {
        'teacher': teacher_exists,
        'student': student_exists
    }
    
    print(f"  {'✓' if teacher_exists else '✗'} Teacher model: {'EXISTS' if teacher_exists else 'NOT FOUND'}")
    if teacher_exists:
        # Check for key files
        config_file = teacher_path / 'config.json'
        print(f"    Config file: {'✓' if config_file.exists() else '✗'}")
    
    print(f"  {'✓' if student_exists else '✗'} Student model: {'EXISTS' if student_exists else 'NOT FOUND'}")
    if student_exists:
        config_file = student_path / 'config.json'
        print(f"    Config file: {'✓' if config_file.exists() else '✗'}")
    
    print()
    return results


def get_memory_info() -> Dict:
    """
    Get system memory information.
    
    Returns:
        Dictionary with memory statistics
    """
    print("="*70)
    print("MEMORY INFORMATION")
    print("="*70)
    
    # System memory
    mem = psutil.virtual_memory()
    print(f"  Total RAM: {mem.total / (1024**3):.2f} GB")
    print(f"  Available RAM: {mem.available / (1024**3):.2f} GB")
    print(f"  Used RAM: {mem.used / (1024**3):.2f} GB ({mem.percent}%)")
    
    # CPU info
    print(f"\n  CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"  CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    # GPU info (if available)
    gpu_info = {}
    if torch.cuda.is_available():
        print(f"\n  GPU Available: ✓")
        print(f"  GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            gpu_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            gpu_cached = torch.cuda.memory_reserved(i) / (1024**3)
            
            print(f"    GPU {i}: {gpu_name}")
            print(f"      Total Memory: {gpu_mem:.2f} GB")
            print(f"      Allocated: {gpu_allocated:.2f} GB")
            print(f"      Cached: {gpu_cached:.2f} GB")
            
            gpu_info[f'gpu_{i}'] = {
                'name': gpu_name,
                'total_memory': gpu_mem,
                'allocated': gpu_allocated,
                'cached': gpu_cached
            }
    else:
        print(f"\n  GPU Available: ✗")
    
    print()
    
    return {
        'ram_total': mem.total / (1024**3),
        'ram_available': mem.available / (1024**3),
        'ram_used': mem.used / (1024**3),
        'cpu_cores': psutil.cpu_count(logical=False),
        'cpu_logical': psutil.cpu_count(logical=True),
        'gpu_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_info': gpu_info
    }


def test_training_step():
    """
    Run a single training step to test memory usage and pipeline functionality.
    
    Returns:
        Dictionary with test results
    """
    print("="*70)
    print("TRAINING STEP TEST")
    print("="*70)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig
    except ImportError as e:
        print(f"  ✗ Missing dependencies: {e}")
        return {'success': False, 'error': str(e)}
    
    # Check if we have a small model available for testing
    test_model_name = "gpt2"  # Small model for testing
    
    print(f"  Testing with model: {test_model_name}")
    print("  This will test memory usage and basic pipeline functionality...")
    
    try:
        # Get initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_mem = torch.cuda.memory_allocated() / (1024**2)  # MB
        
        # Load tokenizer
        print("    Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(test_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create a tiny dataset
        print("    Creating test dataset...")
        test_data = {
            'instruction': ['Test question 1', 'Test question 2'],
            'response': ['Test answer 1', 'Test answer 2']
        }
        dataset = Dataset.from_dict(test_data)
        
        # Tokenize
        def tokenize(examples):
            texts = [f"Question: {inst}\nAnswer: {resp}" for inst, resp in zip(examples['instruction'], examples['response'])]
            return tokenizer(texts, truncation=True, padding='max_length', max_length=128)
        
        tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
        
        # Try to load model (without quantization for speed)
        print("    Loading model (this may take a moment)...")
        model = AutoModelForCausalLM.from_pretrained(
            test_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Get memory after model load
        if torch.cuda.is_available():
            model_mem = torch.cuda.memory_allocated() / (1024**2) - initial_mem
            print(f"    Model memory usage: {model_mem:.2f} MB")
        
        # Create a minimal training setup
        print("    Setting up training step...")
        training_args = TrainingArguments(
            output_dir='./test_training',
            num_train_epochs=1,
            per_device_train_batch_size=1,
            max_steps=1,  # Only 1 step
            logging_steps=1,
            report_to=None,
        )
        
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=data_collator,
        )
        
        # Run one training step
        print("    Running single training step...")
        trainer.train()
        
        # Get memory after training
        if torch.cuda.is_available():
            final_mem = torch.cuda.memory_allocated() / (1024**2) - initial_mem
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2) - initial_mem
            print(f"    Peak memory usage: {peak_mem:.2f} MB")
        
        # Cleanup
        del model, trainer, tokenizer, tokenized, dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("  ✓ Training step test passed!")
        print()
        
        return {
            'success': True,
            'model_memory_mb': model_mem if torch.cuda.is_available() else None,
            'peak_memory_mb': peak_mem if torch.cuda.is_available() else None
        }
        
    except Exception as e:
        print(f"  ✗ Training step test failed: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def check_dependencies() -> Dict:
    """
    Check if required Python packages are installed.
    
    Returns:
        Dictionary with dependency check results
    """
    print("="*70)
    print("DEPENDENCY CHECK")
    print("="*70)
    
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'peft',
        'bitsandbytes',
        'accelerate',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tqdm',
    ]
    
    results = {}
    for package in required_packages:
        try:
            __import__(package)
            results[package] = True
            print(f"  ✓ {package}")
        except ImportError:
            results[package] = False
            print(f"  ✗ {package} (MISSING)")
    
    print()
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="System checks for QuantumBench LLM workflow"
    )
    parser.add_argument(
        '--test-training',
        action='store_true',
        help='Run training step test (takes longer)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("QUANTUMBENCH LLM SYSTEM CHECK")
    print("="*70 + "\n")
    
    # Run all checks
    folder_results = check_folders()
    model_results = check_models()
    memory_info = get_memory_info()
    dependency_results = check_dependencies()
    
    training_results = None
    if args.test_training:
        training_results = test_training_step()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    all_folders_ok = all(folder_results.values())
    all_deps_ok = all(dependency_results.values())
    
    print(f"  Folders: {'✓ All present' if all_folders_ok else '✗ Some missing'}")
    print(f"  Dependencies: {'✓ All installed' if all_deps_ok else '✗ Some missing'}")
    print(f"  Teacher Model: {'✓ Present' if model_results['teacher'] else '✗ Not found'}")
    print(f"  Student Model: {'✓ Present' if model_results['student'] else '✗ Not found'}")
    print(f"  GPU: {'✓ Available' if memory_info['gpu_available'] else '✗ Not available'}")
    
    if training_results:
        print(f"  Training Test: {'✓ Passed' if training_results['success'] else '✗ Failed'}")
    
    print("\n" + "="*70)
    
    # Return status code
    if all_folders_ok and all_deps_ok:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())



