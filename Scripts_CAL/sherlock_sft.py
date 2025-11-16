import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 使用 GPU 4 (18.9GB free)

import json
import torch
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)
from typing import Dict, List, Optional
import torch.nn.functional as F

# ====================================================================
# 0. Configuration Switches
# ====================================================================

# True: Use QLoRA (4-bit + LoRA)
# False: Use Full Fine-Tuning (bfloat16)
USE_QLORA = True

# ====================================================================
# 1. Configuration
# ====================================================================

# -----------------
# Model and Data Paths
# -----------------
MODEL_NAME = "Qwen/Qwen3-4B"
DATASET_PATH = ""
DATASET_SUBSET = "DSFT"  # Path to dataset folder
DATASET_FILE = "train.jsonl"  # Can be preference format or Sherlock format
SEQ_LENGTH = 1024

# -----------------
# Correction Prompt Template (Bias Mitigation Focused)
# -----------------
CORRECTION_PROMPT_TEMPLATE = """Below is a QUESTION and an EXAMPLE RESPONSE. Please review the EXAMPLE RESPONSE carefully and provide an improved, unbiased RESPONSE.

QUESTION: {question}

EXAMPLE RESPONSE: {example_response}

Please analyze the EXAMPLE RESPONSE and identify any potential issues:
1. Does the response contain biased reasoning based on protected attributes (age, gender, race, etc.)?
2. Are there stereotypes or unfair assumptions being made?
3. Is the reasoning logically sound and based on relevant factors only?
4. Does the conclusion follow fairly from the evidence?

Provide an improved RESPONSE in the following format:
<think>
[Provide fair and unbiased reasoning that focuses on relevant, objective factors without stereotypes or biased assumptions. Analyze the question thoroughly and explain your reasoning step by step.]
</think>

\\boxed{{[Provide the final answer based on the unbiased reasoning]}}"""

# -----------------
# LoRA (PEFT) Configuration (only when USE_QLORA=True)
# -----------------
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj"
]

# -----------------
# Dynamic Training Configuration
# -----------------
if USE_QLORA:
    OUTPUT_DIR = "./Sherlock_Qwen3-4B_QLora"
    PER_DEVICE_BATCH_SIZE = 2  # Reduced due to doubled samples
    RUN_NAME = "sherlock_debias_qwen3-4B_qlora"
else:
    OUTPUT_DIR = "./Sherlock_Debias_Qwen3-4B_Full"
    PER_DEVICE_BATCH_SIZE = 1
    RUN_NAME = "sherlock_debias_qwen3-4B_full"

# -----------------
# Training Arguments
# -----------------
TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": 3,
    "logging_steps": 10,
    "save_strategy": "epoch",
    "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
    "per_device_eval_batch_size": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation_steps": 16,  # Increased due to smaller batch size
    "learning_rate": 5e-5,  # Reduced from 1e-4 to improve stability
    "lr_scheduler_type": "cosine",
    "warmup_steps": 50,  # Increased from 10 for more gradual warmup
    "weight_decay": 0.05,
    "optim": "paged_adamw_32bit",
    "bf16": True,
    "max_grad_norm": 1.0,  # Add gradient clipping to prevent exploding gradients
    "remove_unused_columns": False,  # Important: keep our custom columns
    "run_name": RUN_NAME,
    "report_to": "wandb",
    "seed": 42,
    "dataloader_drop_last": False,
}

# ====================================================================
# 2. Helper Functions
# ====================================================================

def print_trainable_parameters(model):
    """
    Print trainable parameters of the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def prepare_sherlock_sample(example, tokenizer):
    """
    Prepare a single Sherlock SFT training sample.

    Supports TWO input formats:

    Format 1 (Sherlock SFT):
    {
        "x_T": "question or problem statement",
        "Y_w": "high-quality answer",
        "Y_l": "low-quality answer",
        "correction_prompt": "optional custom correction prompt (overrides template)"
    }

    Format 2 (Preference/DPO):
    {
        "prompt": "question or problem statement",
        "chosen": "high-quality answer",
        "rejected": "low-quality answer",
        "correction_prompt": "optional custom correction prompt (overrides template)"
    }

    Returns:
    {
        "input_ids_direct": tensor for direct reasoning (x_T -> Y^w),
        "labels_direct": tensor for direct reasoning labels,
        "input_ids_correction": tensor for self-correction (x_T + t + Y^l -> Y^w),
        "labels_correction": tensor for self-correction labels,
    }
    """
    # Auto-detect format and extract fields
    if "x_T" in example:
        # Format 1: Sherlock SFT format
        x_T = example.get("x_T", "")
        Y_w = example.get("Y_w", "")
        Y_l = example.get("Y_l", "")
    elif "prompt" in example:
        # Format 2: Preference/DPO format
        x_T = example.get("prompt", "")
        Y_w = example.get("chosen", "")
        Y_l = example.get("rejected", "")
    else:
        raise ValueError("Invalid format: must contain either 'x_T' or 'prompt' field")

    # Use custom correction prompt if provided, otherwise use template
    if "correction_prompt" in example and example["correction_prompt"]:
        t = example["correction_prompt"]
    else:
        # Format the template with question and example response
        t = CORRECTION_PROMPT_TEMPLATE.format(
            question=x_T,
            example_response=Y_l
        )

    # 1. Direct reasoning sample: x_T -> Y^w
    direct_messages = [
        {"role": "user", "content": x_T},
        {"role": "assistant", "content": Y_w}
    ]
    direct_text = tokenizer.apply_chat_template(
        direct_messages,
        tokenize=False,
        add_generation_prompt=False
    )
    direct_encodings = tokenizer(
        direct_text,
        truncation=True,
        max_length=SEQ_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )

    # Create labels for direct reasoning (mask input, keep only Y^w)
    direct_input_ids = direct_encodings["input_ids"].squeeze(0)
    direct_labels = direct_input_ids.clone()

    # Find where assistant response starts to mask the input portion
    # We need to find the end of the user message
    user_only_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": x_T}],
        tokenize=False,
        add_generation_prompt=True
    )
    user_tokens = tokenizer(user_only_text, add_special_tokens=False)["input_ids"]
    user_length = len(user_tokens)
    direct_labels[:user_length] = -100  # Mask the input part

    # 2. Self-correction sample: x_T + t + Y^l -> Y^w
    correction_messages = [
        {"role": "user", "content": x_T},
        {"role": "assistant", "content": Y_l},
        {"role": "user", "content": t},
        {"role": "assistant", "content": Y_w}
    ]
    correction_text = tokenizer.apply_chat_template(
        correction_messages,
        tokenize=False,
        add_generation_prompt=False
    )
    correction_encodings = tokenizer(
        correction_text,
        truncation=True,
        max_length=SEQ_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )

    # Create labels for self-correction (mask everything except final Y^w)
    correction_input_ids = correction_encodings["input_ids"].squeeze(0)
    correction_labels = correction_input_ids.clone()

    # Find where the final assistant response starts
    correction_prefix_text = tokenizer.apply_chat_template(
        correction_messages[:-1],  # Everything except the final assistant message
        tokenize=False,
        add_generation_prompt=True
    )
    correction_prefix_tokens = tokenizer(correction_prefix_text, add_special_tokens=False)["input_ids"]
    correction_prefix_length = len(correction_prefix_tokens)
    correction_labels[:correction_prefix_length] = -100  # Mask everything before final Y^w

    return {
        "input_ids_direct": direct_input_ids,
        "attention_mask_direct": direct_encodings["attention_mask"].squeeze(0),
        "labels_direct": direct_labels,
        "input_ids_correction": correction_input_ids,
        "attention_mask_correction": correction_encodings["attention_mask"].squeeze(0),
        "labels_correction": correction_labels,
    }

# ====================================================================
# 3. Custom Sherlock Trainer
# ====================================================================

class SherlockSFTTrainer(Trainer):
    """
    Custom Trainer that implements Sherlock SFT joint loss:
    L_Sherlock_SFT(π) = -E[(log π(Y^w|x_T) + log π(Y^w|x_T, Y^l, t))]
    """

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the joint Sherlock SFT loss with NaN protection.
        """
        # Extract direct reasoning inputs
        input_ids_direct = inputs["input_ids_direct"]
        attention_mask_direct = inputs["attention_mask_direct"]
        labels_direct = inputs["labels_direct"]

        # Extract self-correction inputs
        input_ids_correction = inputs["input_ids_correction"]
        attention_mask_correction = inputs["attention_mask_correction"]
        labels_correction = inputs["labels_correction"]

        # 1. Forward pass for direct reasoning: log π(Y^w|x_T)
        outputs_direct = model(
            input_ids=input_ids_direct,
            attention_mask=attention_mask_direct,
            labels=labels_direct
        )
        loss_direct = outputs_direct.loss

        # 2. Forward pass for self-correction: log π(Y^w|x_T, Y^l, t)
        outputs_correction = model(
            input_ids=input_ids_correction,
            attention_mask=attention_mask_correction,
            labels=labels_correction
        )
        loss_correction = outputs_correction.loss

        # 3. NaN detection and handling
        has_nan = False
        if torch.isnan(loss_direct) or torch.isinf(loss_direct):
            print(f"WARNING: loss_direct is NaN/Inf at step {self.state.global_step}, skipping this component")
            loss_direct = torch.tensor(0.0, device=loss_direct.device, dtype=loss_direct.dtype)
            has_nan = True

        if torch.isnan(loss_correction) or torch.isinf(loss_correction):
            print(f"WARNING: loss_correction is NaN/Inf at step {self.state.global_step}, skipping this component")
            loss_correction = torch.tensor(0.0, device=loss_correction.device, dtype=loss_correction.dtype)
            has_nan = True

        # 4. Joint loss: sum of both losses
        loss_total = loss_direct + loss_correction

        # Additional safety check
        if torch.isnan(loss_total) or torch.isinf(loss_total):
            print(f"CRITICAL: loss_total is still NaN/Inf at step {self.state.global_step}, using fallback")
            loss_total = torch.tensor(1.0, device=loss_total.device, dtype=loss_total.dtype, requires_grad=True)
            has_nan = True

        # Log individual losses for monitoring (only to wandb/tensorboard, no console output)
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "loss_direct": loss_direct.item(),
                "loss_correction": loss_correction.item(),
                "loss_total": loss_total.item(),
                "has_nan": 1.0 if has_nan else 0.0,
            })

        return (loss_total, outputs_direct) if return_outputs else loss_total

# ====================================================================
# 4. Data Collator
# ====================================================================

class SherlockDataCollator:
    """
    Custom data collator for Sherlock SFT training.
    Handles batching of dual samples (direct + correction).
    """

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of Sherlock samples.
        """
        batch = {
            "input_ids_direct": torch.stack([f["input_ids_direct"] for f in features]),
            "attention_mask_direct": torch.stack([f["attention_mask_direct"] for f in features]),
            "labels_direct": torch.stack([f["labels_direct"] for f in features]),
            "input_ids_correction": torch.stack([f["input_ids_correction"] for f in features]),
            "attention_mask_correction": torch.stack([f["attention_mask_correction"] for f in features]),
            "labels_correction": torch.stack([f["labels_correction"] for f in features]),
        }
        return batch

# ====================================================================
# 5. Main Execution Flow
# ====================================================================

# --- Set Random Seed ---
set_seed(TRAINING_ARGS["seed"])

# --- 1. Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 2. Load Dataset ---
print(f"Loading Sherlock SFT dataset from: {DATASET_PATH}/{DATASET_SUBSET}/{DATASET_FILE}")
file_path = os.path.join(DATASET_PATH, DATASET_SUBSET, DATASET_FILE)
print(f"\nSupported JSONL formats:")
print(f"Format 1 (Sherlock SFT): {{'x_T': '...', 'Y_w': '...', 'Y_l': '...'}}")
print(f"Format 2 (Preference): {{'prompt': '...', 'chosen': '...', 'rejected': '...'}}")
print(f"Both formats support optional 'correction_prompt' field\n")

raw_dataset = load_dataset(
    "json",
    data_files=file_path,
    split="train",
)
print(f"Raw dataset loaded. Size: {len(raw_dataset)}")
print(f"Columns found: {raw_dataset.column_names}")

# Auto-detect format
first_example = raw_dataset[0]
if "x_T" in first_example:
    print(f"✓ Detected format: Sherlock SFT (x_T, Y_w, Y_l)")
elif "prompt" in first_example:
    print(f"✓ Detected format: Preference/DPO (prompt, chosen, rejected)")
else:
    raise ValueError("Unknown format! Dataset must contain either 'x_T' or 'prompt' field")

# --- 3. Prepare Sherlock Samples ---
print("Preparing Sherlock SFT samples (direct reasoning + self-correction)...")
processed_samples = []
for example in tqdm(raw_dataset, desc="Processing samples"):
    sample = prepare_sherlock_sample(example, tokenizer)
    processed_samples.append(sample)

# Create a dataset from processed samples
train_dataset = Dataset.from_dict({
    "input_ids_direct": [s["input_ids_direct"] for s in processed_samples],
    "attention_mask_direct": [s["attention_mask_direct"] for s in processed_samples],
    "labels_direct": [s["labels_direct"] for s in processed_samples],
    "input_ids_correction": [s["input_ids_correction"] for s in processed_samples],
    "attention_mask_correction": [s["attention_mask_correction"] for s in processed_samples],
    "labels_correction": [s["labels_correction"] for s in processed_samples],
})
train_dataset.set_format(type="torch")
print(f"Processed dataset size: {len(train_dataset)}")

# --- 4. Load Model ---
bnb_config = None
if USE_QLORA:
    print("Loading model with 4-bit BNB quantization (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank}

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
    )
else:
    print("Loading model in bfloat16 for Full Fine-Tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True,
    )

model.config.use_cache = False

if USE_QLORA:
    model.is_parallelizable = True
    model.model_parallel = True

print("Model loaded.")

# --- 5. Configure PEFT (LoRA) ---
peft_config = None
if USE_QLORA:
    print("Configuring QLoRA (PEFT)...")
    from peft import get_peft_model

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
else:
    print("Configuring for Full Fine-Tuning (no PEFT)...")

# --- 6. Configure Trainer ---
training_args_obj = TrainingArguments(**TRAINING_ARGS)
data_collator = SherlockDataCollator()

trainer = SherlockSFTTrainer(
    model=model,
    args=training_args_obj,
    train_dataset=train_dataset,
    eval_dataset=None,
    data_collator=data_collator,
)

print_trainable_parameters(trainer.model)

# --- 7. Start Training ---
print(f"Starting Sherlock SFT training ({'QLoRA' if USE_QLORA else 'Full Fine-Tuning'})...")
print("Training with joint loss: L = log π(Y^w|x_T) + log π(Y^w|x_T, Y^l, t)")
trainer.train()
print("Training finished.")

# --- 8. Save Model ---
if USE_QLORA:
    is_main_process = trainer.is_world_process_zero()

    print("Saving LoRA adapter...")
    adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.save_model(adapter_dir)
    print(f"LoRA adapter saved to: {adapter_dir}")

    if is_main_process:
        print("Merging LoRA adapter into the base model...")

        del trainer
        del model
        torch.cuda.empty_cache()

        print(f"Loading base model ({MODEL_NAME}) in bfloat16 for merging...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"Loading adapter from {adapter_dir}...")
        model = PeftModel.from_pretrained(base_model, adapter_dir)

        print("Merging weights...")
        model = model.merge_and_unload()
        print("Model merged.")

        output_merged_dir = os.path.join(OUTPUT_DIR, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_merged_dir)
        print(f"Full merged model saved to: {output_merged_dir}")

    print(f"Process {os.environ.get('LOCAL_RANK', 0)} finished (QLoRA)!")

else:
    print("Saving full model checkpoint...")
    output_checkpoint_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.save_model(output_checkpoint_dir)
    tokenizer.save_pretrained(output_checkpoint_dir)
    print(f"Full model checkpoint saved to: {output_checkpoint_dir}")
    print("All done (Full Fine-Tuning)!")
