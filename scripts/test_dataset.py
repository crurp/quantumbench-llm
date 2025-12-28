#!/usr/bin/env python3
"""
test_dataset.py

Dataset verification script for QuantumBench LLM project.
Loads and verifies the JSONL dataset prepared by prepare_quantumbench.py.

Usage:
    python scripts/test_dataset.py --data-path data/quantumbench.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict


def load_jsonl(data_path: Path) -> List[Dict]:
    """
    Load JSONL dataset from file.
    
    Args:
        data_path: Path to JSONL file
        
    Returns:
        List of dictionaries with 'instruction' and 'response' keys
    """
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def verify_dataset(data_path: Path, num_samples: int = 5) -> Dict:
    """
    Verify and display dataset information.
    
    Args:
        data_path: Path to JSONL dataset file
        num_samples: Number of samples to display
        
    Returns:
        Dictionary with verification results
    """
    print("="*70)
    print("DATASET VERIFICATION")
    print("="*70)
    
    # Load dataset
    print(f"\nLoading dataset from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    data = load_jsonl(data_path)
    total_examples = len(data)
    
    print(f"✓ Dataset loaded successfully")
    print(f"  Total examples: {total_examples}")
    
    # Verify structure
    if total_examples == 0:
        raise ValueError("Dataset is empty!")
    
    # Check required keys
    required_keys = ['instruction', 'response']
    missing_keys = []
    for i, item in enumerate(data[:10]):  # Check first 10
        for key in required_keys:
            if key not in item:
                missing_keys.append((i, key))
    
    if missing_keys:
        raise ValueError(f"Missing required keys in dataset: {missing_keys}")
    
    print(f"✓ Dataset structure verified (contains 'instruction' and 'response')")
    
    # Display first N samples
    print(f"\n{'='*70}")
    print(f"First {num_samples} examples:")
    print(f"{'='*70}\n")
    
    for i, item in enumerate(data[:num_samples], 1):
        instruction = item.get('instruction', '')
        response = item.get('response', '')
        
        # Clean instruction (remove template prefix if present)
        if "Answer the following quantum computing question:\n\n" in instruction:
            instruction = instruction.split("Answer the following quantum computing question:\n\n")[-1]
        
        print(f"Example {i}:")
        print(f"  Instruction: {instruction[:150]}{'...' if len(instruction) > 150 else ''}")
        print(f"  Response: {response[:150]}{'...' if len(response) > 150 else ''}")
        print()
    
    # Dataset statistics
    print(f"{'='*70}")
    print("Dataset Statistics:")
    print(f"{'='*70}")
    
    instruction_lengths = [len(item['instruction']) for item in data]
    response_lengths = [len(item['response']) for item in data]
    
    stats = {
        'total_examples': total_examples,
        'avg_instruction_length': sum(instruction_lengths) / len(instruction_lengths),
        'avg_response_length': sum(response_lengths) / len(response_lengths),
        'min_instruction_length': min(instruction_lengths),
        'max_instruction_length': max(instruction_lengths),
        'min_response_length': min(response_lengths),
        'max_response_length': max(response_lengths),
    }
    
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  Average instruction length: {stats['avg_instruction_length']:.1f} characters")
    print(f"  Average response length: {stats['avg_response_length']:.1f} characters")
    print(f"  Instruction length range: {stats['min_instruction_length']} - {stats['max_instruction_length']} chars")
    print(f"  Response length range: {stats['min_response_length']} - {stats['max_response_length']} chars")
    print()
    
    return {
        'data': data,
        'stats': stats,
        'verified': True
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Verify QuantumBench JSONL dataset"
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/quantumbench.jsonl',
        help='Path to JSONL dataset file'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of samples to display'
    )
    
    args = parser.parse_args()
    
    data_path = Path(args.data_path)
    results = verify_dataset(data_path, args.num_samples)
    
    print(f"{'='*70}")
    print("✓ Dataset verification complete!")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    main()

