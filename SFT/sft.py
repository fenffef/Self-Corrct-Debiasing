import os
import json
import torch
from datasets import load_dataset, Dataset # 导入 Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    is_torch_npu_available,
    is_torch_xpu_available,
    set_seed,
)
from trl import SFTConfig, SFTTrainer

# ====================================================================
# 1. 配置 (来自您的脚本)
# ====================================================================

# -----------------
# 模型和数据路径 (根据您的日志自动填充)
# -----------------
MODEL_NAME = "Qwen3-4B-Instruct-2507"
DATASET_PATH = "/mnt/raid/data/xuanfeng/Self-Corrct-Debiasing/SFT"
DATASET_SUBSET = "data/sft"
OUTPUT_DIR = "./V0_Qwen3-4B-Instruct-2507" # 输出目录

# -----------------
# SFT 训练器特定参数
# -----------------
SEQ_LENGTH = 1024 # <--- 在这里定义

# -----------------
# LoRA (PEFT) 配置
# -----------------
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# -----------------
# SFT 训练参数 (来自您之前的命令)
# -----------------
TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "max_steps": 300,
    "logging_steps": 20,
    "save_steps": 100,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1, # (此参数将被忽略, 因为没有 eval_dataset)
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.05,
    "optim": "paged_adamw_32bit",
    "bf16": True,
    "remove_unused_columns": True, # (必须为 True 以避免 collator 错误)
    "run_name": "sft_v0_qwen",
    "report_to": "wandb",
    "seed": 42,
    "packing": False, 
    "max_seq_length": SEQ_LENGTH,
}

# ====================================================================
# 2. 帮助函数
# ====================================================================

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def formatting_func(batch):
    """
    根据您已清理的 "prompt" 和 "response" 格式准备 SFT 完整文本。
    此函数现在可以正确处理 "batched=True"。
    """
    # "batch" 是一个字典, e.g., {"prompt": [...], "response": [...]}
    prompts = batch.get('prompt', [])
    responses = batch.get('response', [])
    
    # 迭代并格式化批次中的每一项
    formatted_texts = []
    for prompt, response in zip(prompts, responses):
        # 确保它们是字符串, 以防万一 (例如, null 值)
        prompt_str = str(prompt) if prompt is not None else ""
        response_str = str(response) if response is not None else ""
        formatted_texts.append(prompt_str + response_str)
        
    return formatted_texts # 返回一个与输入批次长度相同的列表

# ====================================================================
# 3. 主执行流程
# ====================================================================

# --- 设置随机种子 ---
set_seed(TRAINING_ARGS["seed"])

# --- 1. 加载 Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # 修复 fp16 训练的溢出问题

# --- 2. 加载已清理的数据集 ---
print(f"Loading dataset from: {DATASET_PATH}, subset: {DATASET_SUBSET}")

# 构造文件路径
file_path = os.path.join(DATASET_PATH, DATASET_SUBSET, "train.jsonl")

dataset = load_dataset(
    "json",
    data_files=file_path,
    split="train",
)

print(f"Dataset loaded successfully ('prompt' and 'response' columns).")

# [!! 关键修改 !!] 不再拆分数据集
train_data = dataset # 使用完整的数据集进行训练
valid_data = None    # 不设置验证集
print(f"Size of the train set: {len(train_data)}. (No validation set)")


# --- 3. 加载模型 (使用 4-bit 量化) ---
print("Loading model with 4-bit BNB quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",  # 自动使用 GPU
    trust_remote_code=True,
)
model.config.use_cache = False
print("Model loaded.")

# --- 4. 配置 PEFT (LoRA) ---
peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- 5. 配置训练器 ---
training_args_obj = SFTConfig(**TRAINING_ARGS)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=valid_data, # <--- [!! 关键修改 !!] 现在是 None
    peft_config=peft_config,
    formatting_func=formatting_func,
    args=training_args_obj,
)

print_trainable_parameters(trainer.model)

# --- 6. 开始训练 ---
print("Starting training (no evaluation)...")
trainer.train()
print("Training finished.")

# --- 7. 保存 LoRA 适配器 ---
adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
trainer.save_model(adapter_dir)
print(f"LoRA adapter saved to: {adapter_dir}")

# --- 8. (可选) 合并并保存完整模型 ---
print("Merging LoRA adapter into the base model...")

# 释放内存
del trainer
del model
torch.cuda.empty_cache()

# 加载 PEFT 模型 (适配器 + 基础模型)
model = AutoPeftModelForCausalLM.from_pretrained(
    adapter_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# 合并适配器权重
model = model.merge_and_unload()
print("Model merged.")

# 保存完整的、合并后的模型
output_merged_dir = os.path.join(OUTPUT_DIR, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
tokenizer.save_pretrained(output_merged_dir)
print(f"Full merged model saved to: {output_merged_dir}")

print("All done!")