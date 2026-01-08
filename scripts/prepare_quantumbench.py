#!/usr/bin/env python3
"""
prepare_quantumbench.py

Converts QuantumBench CSV or JSON files into JSONL format suitable for
instruction-following fine-tuning. Each line in the output JSONL file contains
an instruction-response pair formatted for training language models.

Usage:
    python scripts/prepare_quantumbench.py --input data/quantumbench.csv --output data/quantumbench.jsonl
"""

import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any


def load_quantumbench_data(input_path: Path) -> List[Dict[str, Any]]:
    """
    Load QuantumBench data from CSV or JSON file.
    
    Args:
        input_path: Path to input file (CSV or JSON)
        
    Returns:
        List of dictionaries containing question-answer pairs
    """
    print(f"Loading data from {input_path}...")
    
    if input_path.suffix == '.csv':
        # Load CSV file
        df = pd.read_csv(input_path)
        print(f"Loaded {len(df)} rows from CSV")
        
        # Expected columns: 'question', 'answer' (or similar)
        # Adjust column names based on actual QuantumBench format
        if 'question' in df.columns and 'answer' in df.columns:
            data = df[['question', 'answer']].to_dict('records')
        elif 'Question' in df.columns and 'Answer' in df.columns:
            data = df[['Question', 'Answer']].rename(
                columns={'Question': 'question', 'Answer': 'answer'}
            ).to_dict('records')
        else:
            # Try to infer columns (assume first is question, second is answer)
            cols = df.columns.tolist()
            if len(cols) >= 2:
                data = df[[cols[0], cols[1]]].rename(
                    columns={cols[0]: 'question', cols[1]: 'answer'}
                ).to_dict('records')
            else:
                raise ValueError(f"Could not identify question/answer columns in CSV. Found columns: {cols}")
    
    elif input_path.suffix == '.json':
        # Load JSON file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from JSON")
        
        # Handle both list and dict formats
        if isinstance(data, dict):
            # If it's a dict, try to extract a list
            if 'data' in data:
                data = data['data']
            elif 'questions' in data:
                data = data['questions']
            else:
                data = [data]
    
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .csv or .json")
    
    return data


def convert_to_jsonl_format(data: List[Dict[str, Any]], instruction_template: str = None) -> List[Dict[str, str]]:
    """
    Convert raw QuantumBench data into JSONL instruction-response format.
    
    Args:
        data: List of dictionaries with 'question' and 'answer' keys
        instruction_template: Optional template for instructions. If None, uses default.
        
    Returns:
        List of dictionaries with 'instruction' and 'response' keys
    """
    if instruction_template is None:
        instruction_template = "Answer the following quantum computing question:"
    
    jsonl_data = []
    
    for item in data:
        # Extract question and answer
        question = item.get('question', item.get('Question', ''))
        answer = item.get('answer', item.get('Answer', item.get('solution', '')))
        
        if not question or not answer:
            print(f"Warning: Skipping item with missing question or answer: {item}")
            continue
        
        # Format as instruction-response pair
        formatted_item = {
            "instruction": f"{instruction_template}\n\n{question}",
            "response": str(answer)
        }
        
        jsonl_data.append(formatted_item)
    
    print(f"Converted {len(jsonl_data)} items to JSONL format")
    return jsonl_data


def save_jsonl(data: List[Dict[str, str]], output_path: Path) -> None:
    """
    Save data to JSONL file (one JSON object per line).
    
    Args:
        data: List of dictionaries to save
        output_path: Path to output JSONL file
    """
    print(f"Saving JSONL to {output_path}...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    print(f"Successfully saved {len(data)} examples to {output_path}")


def main():
    """Main function to orchestrate the conversion process."""
    parser = argparse.ArgumentParser(
        description="Convert QuantumBench CSV/JSON to JSONL instruction-response format"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV or JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output JSONL file'
    )
    parser.add_argument(
        '--instruction-template',
        type=str,
        default=None,
        help='Custom instruction template (optional)'
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Validate input file exists
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load data
    raw_data = load_quantumbench_data(input_path)
    
    # Convert to JSONL format
    jsonl_data = convert_to_jsonl_format(raw_data, args.instruction_template)
    
    # Save to JSONL file
    save_jsonl(jsonl_data, output_path)
    
    print("Conversion complete!")


if __name__ == "__main__":
    main()



