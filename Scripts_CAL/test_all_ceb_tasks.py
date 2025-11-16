import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 使用 GPU 4

import json
import torch
import re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from pathlib import Path
import glob

# ====================================================================
# Configuration
# ====================================================================

# Model Configuration
MODEL_PATH = "./Sherlock_Debias_Qwen3-4B_QLora/final_merged_checkpoint"
MODEL_NAME = "Qwen3-4B-Instruct-2507"
USE_MERGED_MODEL = True

# Test Data Configuration
CEB_BASE_PATH = "../CEB"
OUTPUT_BASE_DIR = "./ceb_evaluation_results"

# Generation Configuration
MAX_NEW_TOKENS = 2048  # Increased to allow full reasoning + answer (base models are verbose)
TEMPERATURE = 0.7
TOP_P = 0.9
DO_SAMPLE = True

# Task Type Definitions
TASK_CONFIGS = {
    'CEB Classification': {
        'subdirs': ['CEB-Credit', 'CEB-Adult', 'CEB-Jigsaw'],
        'files': ['gender.json', 'age.json', 'race.json', 'religion.json'],
        'task_type': 'classification',
        'answer_format': 'number'  # Expects 1, 2, etc.
    },
    'CEB-Recognition-S': {
        'subdirs': ['.'],
        'files': ['gender.json', 'age.json', 'race.json', 'religion.json'],
        'task_type': 'recognition',
        'answer_format': 'number'  # 1 or 2
    },
    'CEB-Recognition-T': {
        'subdirs': ['.'],
        'files': ['gender.json', 'age.json', 'race.json', 'religion.json'],
        'task_type': 'recognition',
        'answer_format': 'number'
    },
    'CEB-Selection-S': {
        'subdirs': ['.'],
        'files': ['gender.json', 'age.json', 'race.json', 'religion.json'],
        'task_type': 'selection',
        'answer_format': 'number'  # 1 or 2
    },
    'CEB-Selection-T': {
        'subdirs': ['.'],
        'files': ['gender.json', 'age.json', 'race.json', 'religion.json'],
        'task_type': 'selection',
        'answer_format': 'number'
    },
    'CEB-Continuation-S': {
        'subdirs': ['.'],
        'files': ['gender.json', 'age.json', 'race.json', 'religion.json'],
        'task_type': 'continuation',
        'answer_format': 'text'  # Free-form text
    },
    'CEB-Continuation-T': {
        'subdirs': ['.'],
        'files': ['gender.json', 'age.json', 'race.json', 'religion.json'],
        'task_type': 'continuation',
        'answer_format': 'text'
    },
    'CEB-Conversation-S': {
        'subdirs': ['.'],
        'files': ['gender.json', 'age.json', 'race.json', 'religion.json'],
        'task_type': 'conversation',
        'answer_format': 'text'
    },
    'CEB-Conversation-T': {
        'subdirs': ['.'],
        'files': ['gender.json', 'age.json', 'race.json', 'religion.json'],
        'task_type': 'conversation',
        'answer_format': 'text'
    },
    'CEB-SS': {
        'subdirs': ['.'],
        'files': ['stereoset.json'],
        'task_type': 'stereoset',
        'answer_format': 'number'
    },
    'CEB-CP': {
        'subdirs': ['.'],
        'files': ['crowspairs.json'],
        'task_type': 'crowspairs',
        'answer_format': 'number'
    },
    'CEB-WB': {
        'subdirs': ['.'],
        'files': ['winobias.json'],
        'task_type': 'winobias',
        'answer_format': 'number'
    },
    'CEB-RB': {
        'subdirs': ['.'],
        'files': ['redditbias.json'],
        'task_type': 'redditbias',
        'answer_format': 'text'
    },
}

# ====================================================================
# Helper Functions
# ====================================================================

def extract_prediction_and_bias(response: str, answer_format: str = 'number') -> Dict[str, str]:
    """
    Extract prediction and bias detection from model response.

    Supports multiple formats:
    1. SFT format: <think>...</think> \\boxed{answer}
    2. Intermediate format: <think>...</think> <bias>Yes/No</bias> answer
    3. Free-form text: Direct answer extraction (for base models)
    """
    result = {
        'prediction': None,
        'bias_detected': None,
        'reasoning': None,
        'full_response': response,
        'format': 'unknown'
    }

    # Method 1: Try structured formats with <think> tag
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
    if think_match:
        result['reasoning'] = think_match.group(1).strip()
        result['format'] = 'structured'

        # Check for \boxed{} answer (SFT format)
        boxed_match = re.search(r'\\boxed\{(.*?)\}', response, re.DOTALL)
        if boxed_match:
            answer_text = boxed_match.group(1).strip()
            result['format'] = 'sft'
            if answer_format == 'number':
                num_match = re.search(r'\b([0-9])\b', answer_text)
                result['prediction'] = num_match.group(1) if num_match else answer_text
            else:
                result['prediction'] = answer_text

        # Check for <bias> tag (intermediate format)
        bias_match = re.search(r'<bias>(.*?)</bias>', response, re.IGNORECASE)
        if bias_match:
            result['bias_detected'] = bias_match.group(1).strip()
            result['format'] = 'intermediate'

            # Extract answer after <bias> tag
            if result['prediction'] is None:
                after_bias = response[bias_match.end():].strip()
                if answer_format == 'number':
                    num_match = re.search(r'\b([0-9])\b', after_bias)
                    if num_match:
                        result['prediction'] = num_match.group(1)
                    else:
                        lines = [l.strip() for l in after_bias.split('\n') if l.strip()]
                        if lines:
                            result['prediction'] = lines[0]
                else:
                    lines = [l.strip() for l in after_bias.split('\n') if l.strip()]
                    if lines:
                        result['prediction'] = lines[0]

        # If still no prediction, look after </think>
        if result['prediction'] is None:
            after_think = response[think_match.end():].strip()
            if answer_format == 'number':
                num_match = re.search(r'\b([0-9])\b', after_think)
                if num_match:
                    result['prediction'] = num_match.group(1)
            else:
                lines = [l.strip() for l in after_think.split('\n') if l.strip()]
                if lines:
                    result['prediction'] = lines[0]

    # Method 2: Free-form text extraction (base model)
    if result['prediction'] is None:
        result['format'] = 'freeform'

        if answer_format == 'number':
            # Strategy: Search from END of response, prioritize strong answer indicators
            lines = response.strip().split('\n')

            # Priority 1: Look for explicit answer indicators (highest confidence)
            explicit_patterns = [
                r'(?:answer|prediction|choice|option|select|choose)(?:\s+is)?[\s:：]+([0-9])',
                r'(?:因此|所以|答案是|选择)[\s:：]*([0-9])',  # Chinese: therefore/so/answer is/choose
                r'(?:therefore|thus|hence|conclusion)[\s:：,]+(?:the\s+)?(?:answer\s+is\s+)?([0-9])',
            ]

            # Search last 5 lines first (most likely to contain final answer)
            for line in reversed(lines[-5:]):
                for pattern in explicit_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        result['prediction'] = match.group(1)
                        break
                if result['prediction']:
                    break

            # Priority 2: Standalone number at end of response or on its own line
            if result['prediction'] is None:
                for line in reversed(lines[-3:]):  # Check last 3 lines
                    # Match standalone number (with optional punctuation)
                    standalone_match = re.match(r'^\s*([0-9])\s*[\.。]?\s*$', line)
                    if standalone_match:
                        result['prediction'] = standalone_match.group(1)
                        break

            # Priority 3: Look for "Option X" or "Choice X" patterns in last few lines
            if result['prediction'] is None:
                for line in reversed(lines[-5:]):
                    option_match = re.search(r'(?:option|choice|选项)\s*([0-9])', line, re.IGNORECASE)
                    if option_match:
                        result['prediction'] = option_match.group(1)
                        break

            # Priority 4: As last resort, look for sentence-ending patterns in full response
            # (more conservative than before - avoid matching in middle of reasoning)
            if result['prediction'] is None:
                final_patterns = [
                    r'[.。]\s*([0-9])\s*$',  # Number at very end after period
                    r'\n\s*([0-9])\s*$',  # Number at end after newline
                ]
                for pattern in final_patterns:
                    match = re.search(pattern, response)
                    if match:
                        result['prediction'] = match.group(1)
                        break
        else:
            # For text answers, take first significant line
            lines = [l.strip() for l in response.split('\n') if l.strip() and len(l.strip()) > 3]
            if lines:
                result['prediction'] = lines[0][:200]  # Limit length

        # Extract reasoning from free-form response
        if len(response) > 20 and result['reasoning'] is None:
            result['reasoning'] = response[:500]

    return result

def normalize_label(label, task_type: str = 'classification') -> str:
    """Normalize label based on task type."""
    if label is None:
        return 'unknown'

    label_str = str(label).lower().strip()

    if task_type in ['classification', 'recognition', 'selection']:
        # For number-based answers
        if label_str in ['yes', '1', 'true', 'stereotype']:
            return '1'
        elif label_str in ['no', '2', 'false', 'anti-stereotype']:
            return '2'
        elif label_str in ['3', 'unrelated', 'neutral']:
            return '3'

    return label_str

def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """Calculate classification metrics."""
    # Filter out unknown predictions
    valid_pairs = [(t, p) for t, p in zip(y_true, y_pred) if p != 'unknown' and t != 'unknown']
    if not valid_pairs:
        return {
            'accuracy': 0.0,
            'total_samples': len(y_true),
            'valid_predictions': 0,
            'error': 'No valid predictions'
        }

    y_true_valid = [t for t, p in valid_pairs]
    y_pred_valid = [p for t, p in valid_pairs]

    accuracy = accuracy_score(y_true_valid, y_pred_valid)

    return {
        'accuracy': float(accuracy),
        'total_samples': len(y_true),
        'valid_predictions': len(valid_pairs),
        'correct_predictions': sum(1 for t, p in zip(y_true_valid, y_pred_valid) if t == p)
    }

def get_all_test_files() -> List[Dict]:
    """Get all test files based on task configurations."""
    test_files = []

    for task_name, config in TASK_CONFIGS.items():
        if 'CEB Classification' in task_name:
            # Special handling for classification subdirectories
            for subdir in config['subdirs']:
                base_path = os.path.join(CEB_BASE_PATH, task_name, subdir)
                if not os.path.exists(base_path):
                    continue
                for file in config['files']:
                    file_path = os.path.join(base_path, file)
                    if os.path.exists(file_path):
                        test_files.append({
                            'task': task_name,
                            'subtask': subdir,
                            'file': file,
                            'path': file_path,
                            'task_type': config['task_type'],
                            'answer_format': config['answer_format']
                        })
        else:
            # Regular task directories
            base_path = os.path.join(CEB_BASE_PATH, task_name)
            if not os.path.exists(base_path):
                continue
            for file in config['files']:
                file_path = os.path.join(base_path, file)
                if os.path.exists(file_path):
                    test_files.append({
                        'task': task_name,
                        'subtask': None,
                        'file': file,
                        'path': file_path,
                        'task_type': config['task_type'],
                        'answer_format': config['answer_format']
                    })

    return test_files

def evaluate_single_file(model, tokenizer, file_info: Dict) -> Dict:
    """Evaluate model on a single test file."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {file_info['task']}/{file_info['subtask'] or ''}/{file_info['file']}")
    print(f"{'='*80}")

    # Load test data
    with open(file_info['path'], 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} samples")

    # Process each sample
    results = []
    for idx, item in enumerate(tqdm(test_data, desc="Processing")):
        prompt = item.get('prompt', '')

        # Prepare chat format
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer([text], return_tensors="pt", padding=True).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=DO_SAMPLE,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        # Extract prediction and bias
        extracted = extract_prediction_and_bias(response, file_info['answer_format'])

        # Get true label if available
        true_label = item.get('label', item.get('answer', 'unknown'))
        true_label = normalize_label(true_label, file_info['task_type'])
        predicted_label = normalize_label(extracted['prediction'], file_info['task_type'])

        # Store result
        result = {
            'id': idx,
            'prompt': prompt,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'response': response,
            'bias_detected': extracted['bias_detected'],
            'reasoning': extracted['reasoning'],
            'sensitive_attr': item.get('sensitive_attr', 'unknown'),
            'correct': true_label == predicted_label if true_label != 'unknown' else None
        }

        results.append(result)

    # Calculate metrics
    y_true = [r['true_label'] for r in results]
    y_pred = [r['predicted_label'] for r in results]

    metrics = calculate_metrics(y_true, y_pred)

    # Bias detection statistics
    bias_counts = {'Yes': 0, 'No': 0, 'Unknown': 0}
    for r in results:
        bias_val = r.get('bias_detected', 'Unknown')
        bias_counts[bias_val if bias_val in bias_counts else 'Unknown'] += 1

    metrics['bias_detection'] = bias_counts

    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Valid Predictions: {metrics['valid_predictions']}/{metrics['total_samples']}")
    print(f"  Bias Detected (Yes): {bias_counts['Yes']}")
    print(f"  Bias Detected (No): {bias_counts['No']}")

    return {
        'file_info': file_info,
        'metrics': metrics,
        'results': results
    }

# ====================================================================
# Main Execution
# ====================================================================

def main():
    print("=" * 80)
    print("CEB Comprehensive Evaluation Script")
    print("=" * 80)

    # Create output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Load model
    print(f"\nLoading model from: {MODEL_PATH if USE_MERGED_MODEL else MODEL_NAME}")
    model_path = MODEL_PATH if (USE_MERGED_MODEL and os.path.exists(MODEL_PATH)) else MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded successfully!")

    # Get all test files
    test_files = get_all_test_files()
    print(f"\nFound {len(test_files)} test files to evaluate")

    # Evaluate each file
    all_results = []
    for file_info in test_files:
        try:
            result = evaluate_single_file(model, tokenizer, file_info)
            all_results.append(result)

            # Save individual file results
            task_name = file_info['task'].replace(' ', '_')
            subtask = file_info['subtask'] or 'main'
            file_name = file_info['file'].replace('.json', '')

            output_dir = os.path.join(OUTPUT_BASE_DIR, task_name, subtask)
            os.makedirs(output_dir, exist_ok=True)

            # Save detailed results
            output_file = os.path.join(output_dir, f"{file_name}_results.jsonl")
            with open(output_file, 'w', encoding='utf-8') as f:
                for r in result['results']:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')

            # Save metrics
            metrics_file = os.path.join(output_dir, f"{file_name}_metrics.json")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(result['metrics'], f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error processing {file_info['path']}: {e}")
            import traceback
            traceback.print_exc()

    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    summary = {}
    for result in all_results:
        task = result['file_info']['task']
        if task not in summary:
            summary[task] = {
                'files': 0,
                'total_samples': 0,
                'avg_accuracy': 0.0,
                'accuracies': []
            }

        summary[task]['files'] += 1
        summary[task]['total_samples'] += result['metrics']['total_samples']
        summary[task]['accuracies'].append(result['metrics']['accuracy'])

    for task, stats in summary.items():
        stats['avg_accuracy'] = np.mean(stats['accuracies']) if stats['accuracies'] else 0.0
        print(f"\n{task}:")
        print(f"  Files Evaluated: {stats['files']}")
        print(f"  Total Samples: {stats['total_samples']}")
        print(f"  Average Accuracy: {stats['avg_accuracy']:.4f}")

    # Save summary
    summary_file = os.path.join(OUTPUT_BASE_DIR, "summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n\nAll results saved to: {OUTPUT_BASE_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
