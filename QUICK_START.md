# Quick Start Guide - Interactive Distilled Model

## 🚀 Start Chatting with the Model

### Option 1: Use the Launcher Script (Easiest)

```bash
cd quantumbench-llm
./chat_with_model.sh
```

### Option 2: Manual Start

```bash
cd quantumbench-llm
source venv/bin/activate
python3 scripts/interactive_distilled_model.py
```

## 📝 Usage

Once the interface loads:

1. **Type your question** and press Enter
   - Example: `What is a qubit?`
   
2. **Available commands:**
   - `examples` - Show example questions
   - `settings` - View current generation settings
   - `quit` or `exit` - Exit the interface

## 💡 Example Questions

- "What is a qubit?"
- "Explain quantum entanglement"
- "How does quantum superposition work?"
- "What is a quantum gate?"
- "What is the difference between classical and quantum computing?"

## ⚙️ Customization

You can customize the generation:

```bash
python3 scripts/interactive_distilled_model.py \
    --max-tokens 512 \
    --temperature 0.8 \
    --greedy
```

**Options:**
- `--max-tokens`: Maximum response length (default: 256)
- `--temperature`: Creativity level 0.0-2.0 (default: 0.7)
- `--greedy`: Use deterministic decoding instead of sampling

## 📊 First Time Setup

If this is your first time running:

1. Ensure training completed: Check `models/student-model/` exists
2. Activate virtual environment: `source venv/bin/activate`
3. Verify dependencies: All packages should be installed from `requirements.txt`

## 🐛 Troubleshooting

**Model loading takes long time?**
- This is normal on CPU (30-60 seconds)
- The model loads once, then responses are fast

**Model not found error?**
- Ensure training completed successfully
- Check `models/student-model/` directory exists

**Poor quality responses?**
- Try lowering temperature: `--temperature 0.5`
- Use greedy mode: `--greedy`
- Increase max tokens: `--max-tokens 512`

## 📚 More Information

- See `INTERACTIVE_MODEL.md` for detailed documentation
- See `DISTILLATION_METRICS.md` for performance metrics

