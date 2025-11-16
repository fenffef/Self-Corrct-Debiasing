import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"  # 使用 GPU 4

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from typing import Dict, List, Tuple, Optional
import random
import numpy as np
from torch.utils.data import DataLoader
import wandb
from datetime import datetime

# ====================================================================
# 0. Configuration
# ====================================================================

# Model Paths
SFT_MODEL_PATH = "./Sherlock_Qwen3-4B_QLora"  # Stage I 训练好的模型
BASE_MODEL_NAME = "Qwen/Qwen3-4B"
DATASET_PATH = "./DA/train_preference.jsonl"  # 偏好数据集
OUTPUT_DIR = "./Sherlock_Offline_Qwen3-4B"

# Training Hyperparameters (Table 6)
LEARNING_RATE = 5e-6
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 32
NUM_EPOCHS = 1
MAX_LENGTH = 512

# Sherlock Loss Weights
ALPHA = 0.25  # Weight for DPO loss
BETA_BASE = 0.25  # Base beta for DPO
USE_DYNAMIC_BETA = True  # Whether to use dynamic beta

# LoRA Configuration
USE_QLORA = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# Random seed
SEED = 42

# Wandb Configuration
WANDB_PROJECT = "sherlock-offline"
WANDB_RUN_NAME = f"sherlock_offline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
USE_WANDB = True

# ====================================================================
# 1. Data Processing
# ====================================================================

def load_preference_data(file_path: str) -> List[Dict]:
    """加载偏好数据集"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                # 只保留成功注入偏见的样本
                if item.get('has_bias', False):
                    data.append(item)
    print(f"加载 {len(data)} 个有效偏好对")
    return data


def find_divergence_point(chosen_tokens: List[int], rejected_tokens: List[int]) -> int:
    """
    找到 chosen 和 rejected 响应的分歧点
    返回第一个不同 token 的索引
    """
    min_len = min(len(chosen_tokens), len(rejected_tokens))
    for i in range(min_len):
        if chosen_tokens[i] != rejected_tokens[i]:
            return i
    return min_len


def compute_dynamic_beta(divergence_ratio: float, base_beta: float = 0.25) -> float:
    """
    计算动态 beta
    divergence_ratio: 分歧点位置占总长度的比例 (0-1)
    分歧点越靠前 (ratio 越小)，质量差距越大，beta 越大
    """
    # beta = base_beta * (1 + (1 - divergence_ratio))
    # 范围: base_beta ~ 2*base_beta
    return base_beta * (2 - divergence_ratio)


def prepare_dpo_batch(
    batch: List[Dict],
    tokenizer,
    max_length: int = 512
) -> Dict[str, torch.Tensor]:
    """
    准备 DPO 训练的 batch
    """
    prompts = []
    chosen_responses = []
    rejected_responses = []
    divergence_indices = []

    for item in batch:
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']

        prompts.append(prompt)
        chosen_responses.append(chosen)
        rejected_responses.append(rejected)

        # 计算分歧点
        chosen_tokens = tokenizer.encode(chosen, add_special_tokens=False)
        rejected_tokens = tokenizer.encode(rejected, add_special_tokens=False)
        div_idx = find_divergence_point(chosen_tokens, rejected_tokens)
        divergence_indices.append(div_idx)

    # Tokenize chosen sequences
    chosen_inputs = []
    chosen_labels = []
    for prompt, chosen in zip(prompts, chosen_responses):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encodings = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")

        # Create labels (mask prompt, only compute loss on response)
        input_ids = encodings["input_ids"].squeeze(0)
        labels = input_ids.clone()

        # Find prompt length to mask
        prompt_only = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        prompt_tokens = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
        labels[:len(prompt_tokens)] = -100

        chosen_inputs.append(input_ids)
        chosen_labels.append(labels)

    # Tokenize rejected sequences
    rejected_inputs = []
    rejected_labels = []
    for prompt, rejected in zip(prompts, rejected_responses):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": rejected}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        encodings = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")

        input_ids = encodings["input_ids"].squeeze(0)
        labels = input_ids.clone()

        prompt_only = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        prompt_tokens = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
        labels[:len(prompt_tokens)] = -100

        rejected_inputs.append(input_ids)
        rejected_labels.append(labels)

    return {
        "chosen_input_ids": torch.stack(chosen_inputs),
        "chosen_labels": torch.stack(chosen_labels),
        "rejected_input_ids": torch.stack(rejected_inputs),
        "rejected_labels": torch.stack(rejected_labels),
        "divergence_indices": torch.tensor(divergence_indices),
    }


# ====================================================================
# 2. Sherlock Loss Functions
# ====================================================================

def compute_log_probs(
    model,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算序列的 log 概率
    """
    if attention_mask is None:
        attention_mask = (input_ids != model.config.pad_token_id).long()

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    # Shift for autoregressive prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather the log probs for the actual tokens
    per_token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Mask padding and prompt tokens
    loss_mask = (shift_labels != -100).float()
    per_token_log_probs = per_token_log_probs * loss_mask

    # Sum log probs for each sequence
    sequence_log_probs = per_token_log_probs.sum(dim=-1)

    return sequence_log_probs


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.25
) -> torch.Tensor:
    """
    计算标准 DPO 损失 (Eq. 4 in DPO paper)
    """
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    # DPO loss = -log(sigmoid(chosen_reward - rejected_reward))
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

    return loss


def self_correction_loss(
    policy_model,
    reference_model,
    chosen_input_ids: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_input_ids: torch.Tensor,
    rejected_labels: torch.Tensor,
    divergence_indices: torch.Tensor,
    beta: float = 0.25,
    use_dynamic_beta: bool = True
) -> torch.Tensor:
    """
    计算自校正损失 (Eq. 6 in Sherlock paper)

    这个损失在"轨迹级别"上进行偏好学习，只关注分歧点之后的 token
    """
    batch_size = chosen_input_ids.size(0)
    total_loss = 0.0

    for i in range(batch_size):
        div_idx = divergence_indices[i].item()

        # 计算动态 beta
        if use_dynamic_beta:
            # 分歧点位置的比例
            seq_len = (chosen_labels[i] != -100).sum().item()
            if seq_len > 0:
                div_ratio = div_idx / seq_len
            else:
                div_ratio = 0.5
            current_beta = compute_dynamic_beta(div_ratio, beta)
        else:
            current_beta = beta

        # 创建后缀掩码 (只计算分歧点之后的 token)
        suffix_mask_chosen = chosen_labels[i].clone()
        suffix_mask_rejected = rejected_labels[i].clone()

        # 找到 response 开始的位置
        response_start = (chosen_labels[i] != -100).nonzero(as_tuple=True)[0]
        if len(response_start) > 0:
            response_start_idx = response_start[0].item()
            # 分歧点相对于 response 的位置
            absolute_div_idx = response_start_idx + div_idx
            # 掩码分歧点之前的 token
            suffix_mask_chosen[:absolute_div_idx] = -100
            suffix_mask_rejected[:absolute_div_idx] = -100

        # 计算后缀的 log probs
        # Positive: chosen suffix (good)
        policy_chosen_suffix_logp = compute_log_probs(
            policy_model,
            chosen_input_ids[i:i+1],
            suffix_mask_chosen.unsqueeze(0)
        )
        reference_chosen_suffix_logp = compute_log_probs(
            reference_model,
            chosen_input_ids[i:i+1],
            suffix_mask_chosen.unsqueeze(0)
        )

        # Negative: rejected suffix (bad)
        policy_rejected_suffix_logp = compute_log_probs(
            policy_model,
            rejected_input_ids[i:i+1],
            suffix_mask_rejected.unsqueeze(0)
        )
        reference_rejected_suffix_logp = compute_log_probs(
            reference_model,
            rejected_input_ids[i:i+1],
            suffix_mask_rejected.unsqueeze(0)
        )

        # DPO loss on suffixes
        sample_loss = dpo_loss(
            policy_chosen_suffix_logp,
            policy_rejected_suffix_logp,
            reference_chosen_suffix_logp,
            reference_rejected_suffix_logp,
            beta=current_beta
        )

        total_loss += sample_loss

    return total_loss / batch_size


def sherlock_loss(
    policy_model,
    reference_model,
    batch: Dict[str, torch.Tensor],
    alpha: float = 0.25,
    beta: float = 0.25,
    use_dynamic_beta: bool = True
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算 Sherlock 复合损失 (Eq. 7)
    L_Sherlock = L_SC + alpha * L_DPO
    """
    # 1. 计算标准 DPO 损失 (整个响应级别)
    policy_chosen_logps = compute_log_probs(
        policy_model,
        batch["chosen_input_ids"],
        batch["chosen_labels"]
    )
    policy_rejected_logps = compute_log_probs(
        policy_model,
        batch["rejected_input_ids"],
        batch["rejected_labels"]
    )

    with torch.no_grad():
        reference_chosen_logps = compute_log_probs(
            reference_model,
            batch["chosen_input_ids"],
            batch["chosen_labels"]
        )
        reference_rejected_logps = compute_log_probs(
            reference_model,
            batch["rejected_input_ids"],
            batch["rejected_labels"]
        )

    loss_dpo = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta=beta
    )

    # 2. 计算自校正损失 (轨迹级别)
    loss_sc = self_correction_loss(
        policy_model,
        reference_model,
        batch["chosen_input_ids"],
        batch["chosen_labels"],
        batch["rejected_input_ids"],
        batch["rejected_labels"],
        batch["divergence_indices"],
        beta=beta,
        use_dynamic_beta=use_dynamic_beta
    )

    # 3. 复合损失
    total_loss = loss_sc + alpha * loss_dpo

    # 4. 计算详细监控指标
    with torch.no_grad():
        # Reward margins
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
        reward_margins = chosen_rewards - rejected_rewards
        reward_accuracy = (reward_margins > 0).float().mean().item()

        # Log probability statistics
        policy_chosen_logps_mean = policy_chosen_logps.mean().item()
        policy_rejected_logps_mean = policy_rejected_logps.mean().item()
        reference_chosen_logps_mean = reference_chosen_logps.mean().item()
        reference_rejected_logps_mean = reference_rejected_logps.mean().item()

        # KL divergence estimates
        kl_chosen = (reference_chosen_logps - policy_chosen_logps).mean().item()
        kl_rejected = (reference_rejected_logps - policy_rejected_logps).mean().item()

        # Divergence statistics
        div_indices = batch["divergence_indices"].float()
        avg_div_idx = div_indices.mean().item()

    metrics = {
        # Core losses
        "loss/total": total_loss.item(),
        "loss/dpo": loss_dpo.item(),
        "loss/sc": loss_sc.item(),
        "loss/dpo_weighted": (alpha * loss_dpo).item(),

        # Reward metrics
        "rewards/chosen_mean": chosen_rewards.mean().item(),
        "rewards/rejected_mean": rejected_rewards.mean().item(),
        "rewards/margin_mean": reward_margins.mean().item(),
        "rewards/margin_std": reward_margins.std().item(),
        "rewards/accuracy": reward_accuracy,

        # Log probabilities
        "logps/policy_chosen": policy_chosen_logps_mean,
        "logps/policy_rejected": policy_rejected_logps_mean,
        "logps/reference_chosen": reference_chosen_logps_mean,
        "logps/reference_rejected": reference_rejected_logps_mean,
        "logps/policy_diff": policy_chosen_logps_mean - policy_rejected_logps_mean,
        "logps/reference_diff": reference_chosen_logps_mean - reference_rejected_logps_mean,

        # KL divergence
        "kl/chosen": kl_chosen,
        "kl/rejected": kl_rejected,
        "kl/average": (kl_chosen + kl_rejected) / 2,

        # Divergence point statistics
        "divergence/avg_index": avg_div_idx,
        "divergence/min_index": div_indices.min().item(),
        "divergence/max_index": div_indices.max().item(),
    }

    return total_loss, metrics


# ====================================================================
# 3. Training Loop
# ====================================================================

def train():
    set_seed(SEED)

    print("="*60)
    print("Sherlock Offline Training (Stage II)")
    print("="*60)

    # Initialize wandb
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "base_model": BASE_MODEL_NAME,
                "sft_model": SFT_MODEL_PATH,
                "dataset": DATASET_PATH,
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "effective_batch_size": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "num_epochs": NUM_EPOCHS,
                "alpha": ALPHA,
                "beta_base": BETA_BASE,
                "use_dynamic_beta": USE_DYNAMIC_BETA,
                "max_length": MAX_LENGTH,
                "use_qlora": USE_QLORA,
                "lora_r": LORA_R if USE_QLORA else None,
                "lora_alpha": LORA_ALPHA if USE_QLORA else None,
                "seed": SEED,
            }
        )

    # 1. Load tokenizer
    print("\n[1/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load models
    print("\n[2/6] Loading models...")

    if USE_QLORA:
        # Load base model with quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Load policy model (will be trained)
        print("Loading policy model...")
        policy_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Add LoRA
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )
        policy_model = get_peft_model(policy_model, lora_config)
        policy_model.print_trainable_parameters()

        # Load reference model (frozen)
        print("Loading reference model...")
        reference_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False
    else:
        # Full fine-tuning
        policy_model = AutoModelForCausalLM.from_pretrained(
            SFT_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        reference_model = AutoModelForCausalLM.from_pretrained(
            SFT_MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False

    # 3. Load data
    print("\n[3/6] Loading preference dataset...")
    data = load_preference_data(DATASET_PATH)
    random.shuffle(data)

    # 4. Setup optimizer
    print("\n[4/6] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    # 5. Training
    print("\n[5/6] Starting training...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Alpha (DPO weight): {ALPHA}")
    print(f"  Beta (base): {BETA_BASE}")
    print(f"  Dynamic beta: {USE_DYNAMIC_BETA}")

    num_batches = len(data) // BATCH_SIZE
    global_step = 0
    best_loss = float('inf')

    # Log dataset statistics to wandb
    if USE_WANDB:
        wandb.log({
            "dataset/total_samples": len(data),
            "dataset/num_batches": num_batches,
        })

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        policy_model.train()

        epoch_metrics = {key: [] for key in [
            "loss/total", "loss/dpo", "loss/sc", "loss/dpo_weighted",
            "rewards/chosen_mean", "rewards/rejected_mean", "rewards/margin_mean",
            "rewards/margin_std", "rewards/accuracy",
            "logps/policy_chosen", "logps/policy_rejected",
            "logps/reference_chosen", "logps/reference_rejected",
            "logps/policy_diff", "logps/reference_diff",
            "kl/chosen", "kl/rejected", "kl/average",
            "divergence/avg_index", "divergence/min_index", "divergence/max_index",
        ]}

        optimizer.zero_grad()

        pbar = tqdm(range(0, len(data), BATCH_SIZE), desc=f"Epoch {epoch+1}")

        for batch_idx, start_idx in enumerate(pbar):
            end_idx = min(start_idx + BATCH_SIZE, len(data))
            batch_data = data[start_idx:end_idx]

            if len(batch_data) == 0:
                continue

            # Prepare batch
            batch = prepare_dpo_batch(batch_data, tokenizer, MAX_LENGTH)

            # Move to device
            device = next(policy_model.parameters()).device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)

            # Compute loss
            loss, metrics = sherlock_loss(
                policy_model,
                reference_model,
                batch,
                alpha=ALPHA,
                beta=BETA_BASE,
                use_dynamic_beta=USE_DYNAMIC_BETA
            )

            # Scale loss for gradient accumulation
            scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS
            scaled_loss.backward()

            # Update metrics
            for key in metrics:
                if key in epoch_metrics:
                    epoch_metrics[key].append(metrics[key])

            # Gradient accumulation
            if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # Compute gradient norm before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Compute averages for logging
                avg_metrics = {}
                for key in epoch_metrics:
                    if len(epoch_metrics[key]) >= GRADIENT_ACCUMULATION_STEPS:
                        avg_metrics[key] = np.mean(epoch_metrics[key][-GRADIENT_ACCUMULATION_STEPS:])

                # Log to wandb
                if USE_WANDB:
                    wandb_log = {
                        "train/global_step": global_step,
                        "train/epoch": epoch + 1,
                        "train/learning_rate": LEARNING_RATE,
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                    wandb_log.update(avg_metrics)
                    wandb.log(wandb_log, step=global_step)

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{avg_metrics.get('loss/total', 0):.4f}",
                    "dpo": f"{avg_metrics.get('loss/dpo', 0):.4f}",
                    "sc": f"{avg_metrics.get('loss/sc', 0):.4f}",
                    "acc": f"{avg_metrics.get('rewards/accuracy', 0):.3f}",
                })

        # Epoch summary
        epoch_avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items() if len(values) > 0}

        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Total Loss: {epoch_avg_metrics.get('loss/total', 0):.4f}")
        print(f"  DPO Loss: {epoch_avg_metrics.get('loss/dpo', 0):.4f}")
        print(f"  SC Loss: {epoch_avg_metrics.get('loss/sc', 0):.4f}")
        print(f"  Reward Accuracy: {epoch_avg_metrics.get('rewards/accuracy', 0):.4f}")
        print(f"  Reward Margin: {epoch_avg_metrics.get('rewards/margin_mean', 0):.4f}")
        print(f"  KL Average: {epoch_avg_metrics.get('kl/average', 0):.4f}")

        # Log epoch summary to wandb
        if USE_WANDB:
            epoch_summary = {f"epoch/{key}": value for key, value in epoch_avg_metrics.items()}
            epoch_summary["epoch/number"] = epoch + 1
            wandb.log(epoch_summary, step=global_step)

        # Track best model
        current_loss = epoch_avg_metrics.get('loss/total', float('inf'))
        if current_loss < best_loss:
            best_loss = current_loss
            print(f"  New best loss: {best_loss:.4f}")

    # 6. Save model
    print("\n[6/6] Saving model...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if USE_QLORA:
        policy_model.save_pretrained(OUTPUT_DIR)
    else:
        policy_model.save_pretrained(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR)

    # Save training config
    config = {
        "base_model": BASE_MODEL_NAME,
        "sft_model": SFT_MODEL_PATH,
        "dataset": DATASET_PATH,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "num_epochs": NUM_EPOCHS,
        "alpha": ALPHA,
        "beta_base": BETA_BASE,
        "use_dynamic_beta": USE_DYNAMIC_BETA,
        "max_length": MAX_LENGTH,
        "use_qlora": USE_QLORA,
    }

    with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Log final summary to wandb
    if USE_WANDB:
        wandb.log({
            "final/best_loss": best_loss,
            "final/total_steps": global_step,
        })

        # Save model artifact
        artifact = wandb.Artifact(
            name=f"sherlock_offline_model",
            type="model",
            description="Sherlock Offline trained model"
        )
        artifact.add_dir(OUTPUT_DIR)
        wandb.log_artifact(artifact)

        wandb.finish()

    print(f"\nTraining complete! Model saved to {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    train()
