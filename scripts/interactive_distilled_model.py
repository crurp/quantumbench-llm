#!/usr/bin/env python3
"""
interactive_distilled_model.py

Interactive interface for the distilled (student) model.
Allows users to ask quantum computing questions and get responses
from the fine-tuned student model.

Usage:
    python scripts/interactive_distilled_model.py --model-path models/student-model
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_prompt(instruction: str) -> str:
    """Format instruction into a prompt for generation."""
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_response(
    model, 
    tokenizer, 
    instruction: str, 
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True
) -> str:
    """
    Generate response from model for a given instruction.
    
    Args:
        model: Language model to generate from
        tokenizer: Model tokenizer
        instruction: Input instruction/question
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        do_sample: Whether to use sampling (True) or greedy decoding (False)
        
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
            do_sample=do_sample,
            temperature=temperature,
            top_p=0.9 if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
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


def load_model(model_path: Path):
    """
    Load model and tokenizer.
    
    Args:
        model_path: Path to model directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    # Check if pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    model.eval()
    print("✓ Model loaded successfully!\n")
    
    return model, tokenizer


def print_welcome():
    """Print welcome message."""
    print("="*70)
    print("QUANTUM COMPUTING ASSISTANT")
    print("="*70)
    print("Ask questions about quantum computing and get answers from")
    print("the fine-tuned distilled model.")
    print()
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit', 'exit', or 'q' to exit")
    print("  - Type 'settings' to adjust generation parameters")
    print("  - Type 'examples' to see example questions")
    print("="*70)
    print()


def print_examples():
    """Print example questions."""
    examples = [
        "What is a qubit?",
        "What is quantum entanglement?",
        "How does quantum superposition work?",
        "What is a quantum gate?",
        "Explain quantum teleportation.",
        "What is the difference between classical and quantum computing?",
        "How does quantum error correction work?",
        "What is a quantum algorithm?",
    ]
    print("\nExample Questions:")
    print("-" * 70)
    for i, ex in enumerate(examples, 1):
        print(f"{i}. {ex}")
    print("-" * 70)
    print()


def interactive_loop(model, tokenizer, default_max_tokens: int = 256):
    """
    Main interactive loop.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        default_max_tokens: Default max tokens for generation
    """
    # Settings
    max_tokens = default_max_tokens
    temperature = 0.7
    do_sample = True
    
    print_welcome()
    print_examples()
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 Your question: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            elif user_input.lower() == 'examples':
                print_examples()
                continue
            
            elif user_input.lower() == 'settings':
                print("\n⚙️  Current Settings:")
                print(f"  Max tokens: {max_tokens}")
                print(f"  Temperature: {temperature}")
                print(f"  Sampling: {'Enabled' if do_sample else 'Disabled (greedy)'}")
                print("\nTo change settings, edit the script or restart with different defaults.")
                continue
            
            # Generate response
            print("\n🤔 Thinking...")
            
            # If user input doesn't look like a question, prepend "Answer the following quantum computing question:"
            if not user_input.lower().endswith('?'):
                instruction = f"Answer the following quantum computing question:\n\n{user_input}"
            else:
                instruction = user_input
            
            response = generate_response(
                model, 
                tokenizer, 
                instruction, 
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=do_sample
            )
            
            # Print response
            print("\n" + "="*70)
            print("📝 Answer:")
            print("="*70)
            print(response)
            print("="*70)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Interactive interface for the distilled student model"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/student-model',
        help='Path to student model directory'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=256,
        help='Maximum tokens to generate per response'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='Sampling temperature (0.0-2.0, higher = more creative)'
    )
    parser.add_argument(
        '--greedy',
        action='store_true',
        help='Use greedy decoding instead of sampling'
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    
    # Validate model path
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("Please ensure the student model has been trained and saved.")
        sys.exit(1)
    
    # Load model
    try:
        model, tokenizer = load_model(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Start interactive loop
    try:
        interactive_loop(
            model, 
            tokenizer, 
            default_max_tokens=args.max_tokens
        )
    finally:
        # Cleanup
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

