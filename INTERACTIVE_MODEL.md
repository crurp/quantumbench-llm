# Interactive Distilled Model Interface

This guide explains how to interact with the distilled (student) model that was trained through knowledge distillation.

## Quick Start

### Start Interactive Session

```bash
cd quantumbench-llm
source venv/bin/activate
python scripts/interactive_distilled_model.py
```

Or specify a custom model path:

```bash
python scripts/interactive_distilled_model.py --model-path models/student-model
```

## Usage

### Basic Interaction

1. **Ask Questions**: Type any quantum computing question and press Enter
2. **View Examples**: Type `examples` to see example questions
3. **Check Settings**: Type `settings` to view current generation parameters
4. **Exit**: Type `quit`, `exit`, or `q` to exit

### Example Questions

- "What is a qubit?"
- "What is quantum entanglement?"
- "How does quantum superposition work?"
- "What is a quantum gate?"
- "Explain quantum teleportation."
- "What is the difference between classical and quantum computing?"
- "How does quantum error correction work?"

### Command Line Options

```bash
python scripts/interactive_distilled_model.py \
    --model-path models/student-model \
    --max-tokens 256 \
    --temperature 0.7 \
    --greedy
```

**Options:**
- `--model-path`: Path to the student model directory (default: `models/student-model`)
- `--max-tokens`: Maximum tokens to generate per response (default: 256)
- `--temperature`: Sampling temperature 0.0-2.0, higher = more creative (default: 0.7)
- `--greedy`: Use greedy decoding instead of sampling (default: False)

## Features

### Intelligent Prompt Formatting

The script automatically formats your questions to match the training format:
- Adds "Answer the following quantum computing question:" prefix if needed
- Handles questions with or without question marks

### Generation Settings

- **Temperature**: Controls randomness (0.0 = deterministic, 2.0 = very creative)
- **Sampling**: By default uses sampling with temperature for varied responses
- **Greedy Mode**: Use `--greedy` flag for deterministic, consistent responses

### Response Quality

The distilled model has been trained on QuantumBench dataset, so it:
- Understands quantum computing terminology
- Provides accurate explanations
- Generates coherent, educational responses
- Maintains context within each response

## Tips for Best Results

1. **Be Specific**: More specific questions get better answers
   - Good: "What is quantum entanglement and how does it work?"
   - Less effective: "quantum"

2. **Ask Follow-ups**: The model can handle follow-up questions in the same session

3. **Adjust Temperature**: 
   - Lower (0.3-0.5): More focused, deterministic answers
   - Higher (0.8-1.2): More creative, varied explanations

4. **Use Examples**: Type `examples` to see question formats that work well

## Troubleshooting

### Model Not Found

If you see "Model not found", ensure:
1. Student model training has completed
2. Model is saved at `models/student-model/`
3. You're running from the project root directory

### Slow Responses

- The model runs on CPU by default (slower but works everywhere)
- If you have GPU, it will automatically use it
- Reduce `--max-tokens` for faster responses

### Poor Quality Responses

- Try lowering temperature: `--temperature 0.5`
- Use greedy mode: `--greedy`
- Check that training completed successfully
- Ensure the model path is correct

## Example Session

```
$ python scripts/interactive_distilled_model.py

======================================================================
QUANTUM COMPUTING ASSISTANT
======================================================================
Ask questions about quantum computing and get answers from
the fine-tuned distilled model.

Commands:
  - Type your question and press Enter
  - Type 'quit', 'exit', or 'q' to exit
  - Type 'settings' to adjust generation parameters
  - Type 'examples' to see example questions
======================================================================

Loading model from models/student-model...
Loading model weights...
✓ Model loaded successfully!

💬 Your question: What is a qubit?

🤔 Thinking...

======================================================================
📝 Answer:
======================================================================
A qubit is the basic unit of quantum information, analogous to a 
classical bit but capable of existing in superposition states. Unlike 
a classical bit which can only be in one of two states (0 or 1), a 
qubit can be in a superposition of both states simultaneously, 
represented by a quantum state vector.
======================================================================

💬 Your question: quit

👋 Goodbye!
```

## Integration

You can also use the model programmatically:

```python
from scripts.interactive_distilled_model import load_model, generate_response

model, tokenizer = load_model(Path("models/student-model"))
response = generate_response(
    model, 
    tokenizer, 
    "What is quantum entanglement?",
    max_new_tokens=256
)
print(response)
```

## Next Steps

After using the interactive interface, you can:
1. Review the performance metrics: `metrics/distillation_metrics.json`
2. Compare with teacher model: `python scripts/compare_teacher_student.py`
3. Generate visualizations: `python scripts/visualize_results.py`
4. Run full evaluation: `python scripts/test_full_workflow.py`

