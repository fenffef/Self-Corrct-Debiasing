# CAL Dataset Generation and Processing

This directory contains scripts for generating and processing the CAL (Contextual Answer Labeling) inference dataset.

## Overview

The workflow consists of three main steps:
1. **API Data Generation**: Generate inference data using API calls
2. **Dataset Splitting**: Randomly split the generated dataset into two parts
3. **SFT Training**: Train models on the split datasets using supervised fine-tuning

## Files

- `CAL_inference.jsonl` - Main dataset file containing 10,908 examples
- `split_dataset.py` - Python script to randomly split the dataset
- `sft.py` - Supervised fine-tuning script for model training
- `DA/train.jsonl` - First half of split dataset (5,454 examples)
- `DB/train.jsonl` - Second half of split dataset (5,454 examples)
- `README.md` - This file

## Dataset Structure

Each line in the JSONL file contains a JSON object with the following fields:
- `prompt`: The question/scenario presented to the model
- `response`: The model's reasoning, reflection, and answer

Example:
```json
{
  "prompt": "Sam Zhang and Alex Chan go to school together...",
  "response": "Reasoning: The question provides information...\n\nReflection: NO\n\nAnswer: C: Unknown"
}
```

## Step 1: API Data Generation

Generate the dataset using your API service:

```bash
# Your API generation command here
# This produces CAL_inference.jsonl with 10,908 examples
```

**Output**: `CAL_inference.jsonl` (10,908 lines)

## Step 2: Dataset Splitting

Split the dataset randomly into two parts using the provided script.

### Usage

#### Recommended: Split into DA and DB folders
```bash
# Create directories
mkdir -p DA DB

# Split dataset into two train.jsonl files
python split_dataset.py CAL_inference.jsonl --output1 DA/train.jsonl --output2 DB/train.jsonl
```

This will create:
- `DA/train.jsonl` (50% of data, ~5,454 examples)
- `DB/train.jsonl` (50% of data, ~5,454 examples)

#### Basic usage (50/50 split with default names):
```bash
python split_dataset.py CAL_inference.jsonl
```

This will create:
- `CAL_inference_part1.jsonl` (50% of data, ~5,454 examples)
- `CAL_inference_part2.jsonl` (50% of data, ~5,454 examples)

#### Custom split ratio:
```bash
# 70/30 split
python split_dataset.py CAL_inference.jsonl --ratio 0.7
```

#### Custom output filenames:
```bash
python split_dataset.py CAL_inference.jsonl \
  --output1 CAL_train.jsonl \
  --output2 CAL_test.jsonl
```

#### Custom random seed:
```bash
# Use different seed for different random splits
python split_dataset.py CAL_inference.jsonl --seed 123
```

### Script Parameters

- `input_file`: Path to the input JSONL file (required)
- `--output1`: Path to first output file (default: `{input}_part1.jsonl`)
- `--output2`: Path to second output file (default: `{input}_part2.jsonl`)
- `--ratio`: Split ratio for first file, 0.0-1.0 (default: 0.5)
- `--seed`: Random seed for reproducibility (default: 42)

### Examples

```bash
# Equal split (default)
python split_dataset.py CAL_inference.jsonl

# 80% train, 20% test
python split_dataset.py CAL_inference.jsonl \
  --ratio 0.8 \
  --output1 train.jsonl \
  --output2 test.jsonl

# Different random split
python split_dataset.py CAL_inference.jsonl --seed 2024
```

## Features

- **Random shuffling**: Data is randomly shuffled before splitting
- **Reproducible**: Uses random seed for consistent splits
- **Flexible ratios**: Support any split ratio (not just 50/50)
- **Encoding safe**: Handles UTF-8 encoding properly
- **Progress tracking**: Shows split statistics after completion

## Output Example

```
Reading data from CAL_inference.jsonl...
Total lines: 10908
Shuffling data...
Writing 5454 lines to DA/train.jsonl...
Writing 5454 lines to DB/train.jsonl...

Split complete!
  Part 1: 5454 lines (50.0%)
  Part 2: 5454 lines (50.0%)
```

## Requirements

- Python 3.6+
- No external dependencies (uses only standard library)

## Step 3: SFT Training

Train models on the split datasets using supervised fine-tuning (SFT).

### Configuration

Edit the following parameters in `sft.py` before training:

```python
# Model configuration
MODEL_NAME = "Qwen3-4B-Instruct-2507"  # Base model to fine-tune
DATASET_PATH = ""  # Path to dataset directory (containing DA/DB folders)
DATASET_SUBSET = "DA"  # Choose "DA" or "DB" for different splits
SEQ_LENGTH = 1024  # Maximum sequence length

# Training mode
USE_QLORA = True  # True: QLoRA (4-bit), False: Full fine-tuning

# GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # Set GPU ID
```

### Usage

#### Train on DA dataset:
```bash
# Edit sft.py: set DATASET_SUBSET = "DA"
python sft.py
```

#### Train on DB dataset:
```bash
# Edit sft.py: set DATASET_SUBSET = "DB"
python sft.py
```

### Training Features

- **Two Training Modes**:
  - **QLoRA (default)**: 4-bit quantization + LoRA for memory efficiency
  - **Full Fine-tuning**: Full model fine-tuning in bfloat16

- **QLoRA Configuration**:
  - 4-bit NF4 quantization
  - LoRA rank: 8, alpha: 16
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Batch size: 4 per device

- **Training Parameters**:
  - Epochs: 3
  - Learning rate: 1e-4 with cosine scheduling
  - Gradient accumulation: 8 steps
  - Optimizer: paged_adamw_32bit
  - Precision: bfloat16

- **Logging & Checkpointing**:
  - Wandb integration for experiment tracking
  - Save strategy: every epoch
  - Output: `./V0_Qwen3-4B-Instruct-2507/`

### Output Structure

After training, the following directories will be created:

```
V0_Qwen3-4B-Instruct-2507/
├── final_adapter/              # LoRA adapter weights (QLoRA mode)
├── final_merged_checkpoint/    # Merged full model (QLoRA mode)
└── final_checkpoint/           # Full model checkpoint (Full FT mode)
```

### Training Output Example

```
Loading dataset from: , subset: DA
Dataset loaded successfully ('prompt' and 'response' columns).
Size of the train set: 5454. (No validation set)
Loading model with 4-bit BNB quantization (QLoRA)...
Model loaded.
Configuring QLoRA (PEFT)...
trainable params: 6815744 || all params: 3628849152 || trainable%: 0.18780...
Starting training (QLoRA)...
...
Training finished.
Saving LoRA adapter...
LoRA adapter saved to: ./V0_Qwen3-4B-Instruct-2507/final_adapter
Merging LoRA adapter into the base model...
Full merged model saved to: ./V0_Qwen3-4B-Instruct-2507/final_merged_checkpoint
```

### Requirements

- Python 3.8+
- PyTorch with CUDA support
- transformers, datasets, peft, trl, bitsandbytes
- wandb (for experiment tracking)

Install dependencies:
```bash
pip install torch transformers datasets peft trl bitsandbytes wandb
```

## Notes

- The original file `CAL_inference.jsonl` is not modified
- The split is random but reproducible with the same seed
- All files maintain JSONL format (one JSON object per line)
- UTF-8 encoding is preserved
- QLoRA training requires less GPU memory (~19GB) compared to full fine-tuning
- Training supports both single-GPU and multi-GPU setups (DDP)
