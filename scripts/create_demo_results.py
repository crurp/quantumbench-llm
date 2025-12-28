#!/usr/bin/env python3
"""
Create demo results for visualization testing when models aren't trained yet.
Generates sample comparison results to test the visualization suite.
"""
import json
from pathlib import Path

# Load the actual dataset
with open('data/quantumbench.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# Generate demo predictions (for testing visualization suite)
teacher_preds = []
student_preds = []
ground_truths = []

for item in data:
    gt = item['response']
    # Simulate teacher predictions (85% accuracy)
    import random
    if random.random() < 0.85:
        teacher_preds.append(gt)  # Correct
    else:
        teacher_preds.append(gt[:len(gt)//2] + " (partial answer)")  # Partial
    
    # Simulate student predictions (80% accuracy, 75% match with teacher)
    if random.random() < 0.75:
        student_preds.append(teacher_preds[-1])  # Match teacher
    elif random.random() < 0.80:
        student_preds.append(gt)  # Correct but different wording
    else:
        student_preds.append(gt[:len(gt)//2])  # Partial
    
    ground_truths.append(gt)

# Compute metrics
def normalize_text(text):
    import re
    return re.sub(r'\s+', ' ', text.lower().strip())

n = len(data)
teacher_correct = sum(1 for p, gt in zip(teacher_preds, ground_truths) 
                     if normalize_text(p) == normalize_text(gt) or 
                        normalize_text(p) in normalize_text(gt) or
                        normalize_text(gt) in normalize_text(p))
student_correct = sum(1 for p, gt in zip(student_preds, ground_truths)
                     if normalize_text(p) == normalize_text(gt) or
                        normalize_text(p) in normalize_text(gt) or
                        normalize_text(gt) in normalize_text(p))
exact_matches = sum(1 for t, s in zip(teacher_preds, student_preds)
                   if normalize_text(t) == normalize_text(s))

metrics = {
    'num_examples': n,
    'teacher_accuracy': teacher_correct / n,
    'student_accuracy': student_correct / n,
    'delta_accuracy': (teacher_correct - student_correct) / n,
    'exact_match_rate': exact_matches / n,
    'consistency': exact_matches / n,
    'student_agreement_when_teacher_correct': 0.80,  # Simulated
    'student_correct_when_teacher_wrong': 0.30,  # Simulated
    'teacher_correct_count': teacher_correct,
    'student_correct_count': student_correct,
    'exact_match_count': exact_matches
}

# Save demo results
output = {
    'metrics': metrics,
    'teacher_results': {
        'model_path': 'models/teacher-model',
        'model_type': 'teacher',
        'exists': True,
        'accuracy': metrics['teacher_accuracy'],
        'predictions': teacher_preds,
        'ground_truths': ground_truths,
        'instructions': [item['instruction'] for item in data]
    },
    'student_results': {
        'model_path': 'models/student-model',
        'model_type': 'student',
        'exists': True,
        'accuracy': metrics['student_accuracy'],
        'predictions': student_preds,
        'ground_truths': ground_truths,
        'instructions': [item['instruction'] for item in data]
    }
}

Path('plots').mkdir(exist_ok=True)
with open('plots/comparison_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Created demo results with {n} examples")
print(f"Teacher accuracy: {metrics['teacher_accuracy']:.2%}")
print(f"Student accuracy: {metrics['student_accuracy']:.2%}")
