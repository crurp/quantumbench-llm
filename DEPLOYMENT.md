# Distilled Model Deployment Guide

Quick guide for deploying and using the distilled student model for interactive queries.

## 🚀 Quick Start

### Option 1: Use the Launcher Script (Recommended)

```bash
./start_chat.sh
```

### Option 2: Manual Start

```bash
source venv/bin/activate
python3 scripts/interactive_distilled_model.py --model-path models/student-model-fixed
```

## 📋 Model Information

- **Model Location**: `models/student-model-fixed/`
- **Model Type**: GPT-2 (distilled from GPT-2 Large teacher)
- **Training**: Knowledge distillation with numerical stability fixes
- **Training Status**: ✅ Completed (3 epochs, 249 steps)
- **Performance**: 76% loss reduction, stable training

## 💬 Usage

Once the interface loads:

1. **Ask Questions**: Type any quantum computing question and press Enter
   - Example: `What is a qubit?`
   - Example: `Explain quantum entanglement`

2. **Commands**:
   - `examples` - Show example questions
   - `settings` - View generation settings
   - `quit` or `exit` - Exit the interface

## 📝 Example Questions

- "What is a qubit?"
- "What is quantum entanglement?"
- "How does quantum superposition work?"
- "What is a quantum gate?"
- "Explain quantum teleportation."
- "What is the difference between classical and quantum computing?"
- "How does quantum error correction work?"
- "What is a quantum algorithm?"

## ⚙️ Customization

You can customize generation parameters:

```bash
python3 scripts/interactive_distilled_model.py \
    --model-path models/student-model-fixed \
    --max-tokens 512 \
    --temperature 0.8 \
    --greedy
```

**Options:**
- `--max-tokens`: Maximum response length (default: 256)
- `--temperature`: Creativity level 0.0-2.0 (default: 0.7)
  - Lower (0.3-0.5): More focused, deterministic
  - Higher (0.8-1.2): More creative, varied
- `--greedy`: Use deterministic decoding (no sampling)

## 🔧 Technical Details

### Model Specifications
- **Base Model**: GPT-2 (124M parameters)
- **Precision**: FP32 (CPU-optimized)
- **Device**: CPU (GPU automatic if available)
- **Tokenizer**: GPT-2 tokenizer

### Performance Expectations
- **Load Time**: 30-60 seconds (first time)
- **Response Time**: 5-15 seconds per question (CPU)
- **Memory Usage**: ~500-800 MB

## 📊 Model Training Summary

- **Training Method**: Knowledge distillation from GPT-2 Large
- **Training Duration**: 6.7 hours
- **Loss Reduction**: 76% (131.27 → 31.40)
- **Evaluation Improvement**: 15% (29.38 → 25.04)
- **Numerical Stability**: ✅ No NaN/Inf issues
- **Gradient Clipping**: ✅ Active (max_norm=1.0)

## 🐛 Troubleshooting

### Model Loading Takes Long Time?
- Normal on CPU (30-60 seconds)
- Model loads once, then responses are faster
- GPU will be faster if available

### Model Not Found Error?
```bash
# Check if model exists
ls -la models/student-model-fixed/
```
- Ensure training completed successfully
- Check that model directory contains:
  - `config.json`
  - `tokenizer.json` or `tokenizer_config.json`
  - `model.safetensors` or `pytorch_model.bin`

### Poor Quality Responses?
- Try lower temperature: `--temperature 0.5`
- Use greedy mode: `--greedy`
- Increase max tokens: `--max-tokens 512`
- Model is trained on quantum computing domain

### Memory Issues?
- Close other applications
- Use smaller max-tokens
- Model uses ~500-800 MB RAM

## 📚 Related Scripts

- **Evaluation**: `scripts/evaluate_quantumbench.py`
- **Comparison**: `scripts/compare_teacher_student.py`
- **Metrics**: `scripts/generate_simple_metrics.py`

## 🎯 Example Session

```
$ ./start_chat.sh

======================================================================
QUANTUM COMPUTING ASSISTANT
======================================================================

Loading model from models/student-model-fixed...
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
qubit can be in a superposition of both states simultaneously...
======================================================================

💬 Your question: quit

👋 Goodbye!
```

## ✅ Deployment Checklist

- [x] Model trained successfully
- [x] Model saved to `models/student-model-fixed/`
- [x] Interactive script created (`interactive_distilled_model.py`)
- [x] Launcher script created (`start_chat.sh`)
- [x] Documentation created (`DEPLOYMENT.md`)
- [ ] Model tested (run `./start_chat.sh` to test)
- [ ] Performance validated
- [ ] Ready for use

## 🚀 Next Steps After Deployment

1. **Test the Interface**: Run `./start_chat.sh` and ask a few questions
2. **Evaluate Performance**: Run evaluation scripts to measure accuracy
3. **Compare with Teacher**: See how well knowledge was distilled
4. **Generate Metrics**: Create performance reports

## 📞 Support

For issues or questions:
- Check training log: `training_student_fixed.log`
- Review training summary in log file
- Check model files: `ls -lh models/student-model-fixed/`

