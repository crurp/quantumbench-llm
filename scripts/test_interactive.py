#!/usr/bin/env python3
"""
test_interactive.py

Quick test to verify the interactive model can be loaded and generate responses.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from interactive_distilled_model import load_model, generate_response

def test_model_loading():
    """Test that the model can be loaded."""
    print("Testing model loading...")
    model_path = Path("models/student-model")
    
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        return False
    
    try:
        model, tokenizer = load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Test generation
        print("\nTesting generation with sample question...")
        test_question = "What is a qubit?"
        response = generate_response(
            model, 
            tokenizer, 
            test_question,
            max_new_tokens=100,
            do_sample=False  # Greedy for testing
        )
        
        print(f"\nQuestion: {test_question}")
        print(f"Response: {response[:200]}...")
        
        if response and len(response) > 10:
            print("✅ Generation test passed!")
            return True
        else:
            print("⚠️  Warning: Response seems too short")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)

