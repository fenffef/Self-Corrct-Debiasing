import os
# [!! 关键修复 !!] QLoRA 不支持 DataParallel, 必须只使用单个 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # 使用 GPU 4 (18.9GB free)

import json
import torch
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel
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
# 0. 开关
# ====================================================================

# True: 使用 QLoRA (4-bit + LoRA)
# False: 使用 全量微调 (bfloat16)
USE_QLORA = True

# ====================================================================
# 1. 配置 (将根据开关动态调整)
# ====================================================================

# -----------------
# 模型和数据路径
# -----------------
MODEL_NAME = "Qwen3-4B-Instruct-2507"
DATASET_PATH = ""
DATASET_SUBSET = "DA"
SEQ_LENGTH = 1024

# -----------------
# LoRA (PEFT) 配置 (仅当 USE_QLORA=True)
# -----------------
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
# [!! 关键修复 !!] Qwen 架构的 LoRA 模块
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
# 动态训练配置
# -----------------
if USE_QLORA:
    OUTPUT_DIR = "./V0_Qwen3-4B-Instruct-2507"
    PER_DEVICE_BATCH_SIZE = 2  # 降低 batch size 以节省内存
    RUN_NAME = "sft_v0_Qwen3-4B-Instruct-2507"
else:
    OUTPUT_DIR = "./V0_Qwen3-4B-Instruct-2507"
    PER_DEVICE_BATCH_SIZE = 1 # 全量微调 batch size 设为 1
    RUN_NAME = "sft_v0_Qwen3-4B-Instruct-2507"

# -----------------
# SFT 训练参数
# -----------------
TRAINING_ARGS = {
    "output_dir": OUTPUT_DIR,
    "num_train_epochs": 3, 
    "logging_steps": 1,
    "save_strategy": "epoch", 
    "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE, # [!! 动态 !!]
    "per_device_eval_batch_size": PER_DEVICE_BATCH_SIZE, # [!! 动态 !!]
    "gradient_accumulation_steps": 8,  # 增加以补偿小 batch size
    "learning_rate": 1e-4,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 10,
    "weight_decay": 0.05,
    "optim": "paged_adamw_32bit",
    "bf16": True,
    "remove_unused_columns": True,
    "run_name": RUN_NAME, # [!! 动态 !!]
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
    打印模型的可训练参数。
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

def formatting_func(example):
    """
    格式化函数,将 messages 格式转换为训练文本。
    支持两种格式:
    1. messages 格式 (list of dicts with role and content)
    2. prompt + response 格式
    """
    # 检查是否有 messages 字段
    if 'messages' in example:
        messages = example['messages']
        # 使用 tokenizer 的 apply_chat_template 将 messages 转换为文本
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return text
    # 如果是 prompt + response 格式
    elif 'prompt' in example and 'response' in example:
        prompt_str = str(example['prompt']) if example['prompt'] is not None else ""
        response_str = str(example['response']) if example['response'] is not None else ""
        return prompt_str + response_str
    else:
        raise ValueError("Example must contain either 'messages' or 'prompt' and 'response' fields")

# ====================================================================
# 3. 主执行流程
# ====================================================================

# --- 设置随机种子 ---
set_seed(TRAINING_ARGS["seed"])

# --- 1. 加载 Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
# ... (tokenizer 配置保持不变) ...
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 2. 加载数据集 ---
# ... (数据集加载保持不变) ...
print(f"Loading dataset from: {DATASET_PATH}, subset: {DATASET_SUBSET}")
file_path = os.path.join(DATASET_PATH, DATASET_SUBSET, "train.jsonl")
dataset = load_dataset(
    "json",
    data_files=file_path,
    split="train",
)
print(f"Dataset loaded successfully ('prompt' and 'response' columns).")
train_data = dataset
valid_data = None
print(f"Size of the train set: {len(train_data)}. (No validation set)")


# --- 3. 加载模型 ( [!! 动态 !!] ) ---
bnb_config = None
if USE_QLORA:
    print("Loading model with 4-bit BNB quantization (QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # [!! 关键修复 !!]
    # 当使用 DDP (accelerate launch) + QLoRA 时, 
    # 我们必须告诉 bitsandbytes 模型应该加载到哪个具体设备上。
    # accelerate 会通过 LOCAL_RANK 环境变量告诉我们。
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_map = {"": local_rank}
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        # [!! 关键修复 !!] 使用动态 device_map
        device_map=device_map,
        trust_remote_code=True,
    )
else:
    print("Loading model in bfloat16 for Full Fine-Tuning...")
    # [!! 注意 !!] 全量微调的逻辑保持 device_map=None, 这是正确的。
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=None, # DDP + 全量微调, 必须为 None
        trust_remote_code=True,
    )

model.config.use_cache = False

# [!! 关键修复 !!] 防止量化模型使用 DataParallel
# 量化模型不兼容 DataParallel, 必须禁用
if USE_QLORA:
    model.is_parallelizable = True
    model.model_parallel = True

print("Model loaded.")

# --- 4. 配置 PEFT (LoRA) ( [!! 动态 !!] ) ---
peft_config = None
if USE_QLORA:
# ... (PEFT 配置保持不变) ...
    print("Configuring QLoRA (PEFT)...")
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES, # [!! 已使用 Qwen 模块 !!]
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
# ... (PEFT 配置保持不变) ...
    print("Configuring for Full Fine-Tuning (no PEFT)...")
    # peft_config 保持为 None

# --- 5. 配置训练器 ---
training_args_obj = SFTConfig(**TRAINING_ARGS)
# ... (SFTTrainer 初始化保持不变) ...
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=valid_data,
    peft_config=peft_config, # [!! 动态 !!]
    formatting_func=formatting_func,
    args=training_args_obj,
)

print_trainable_parameters(trainer.model)

# --- 6. 开始训练 ---
print(f"Starting training ({'QLoRA' if USE_QLORA else 'Full Fine-Tuning'})...")
# ... (trainer.train() 和 print 保持不变) ...
trainer.train()
print("Training finished.")

# --- 7. & 8. 保存与合并 ( [!! 动态 !!] ) ---
if USE_QLORA:
    # [!! 关键修复 !!]
    # 1. 获取 DDP 进程信息 (必须在 del trainer 之前)
    is_main_process = trainer.is_world_process_zero()

    # --- 7. 保存 LoRA 适配器 ---
    # (trainer.save_model 内部会自动处理 DDP, 只有主进程会保存)
    print("Saving LoRA adapter...")
    adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.save_model(adapter_dir)
    print(f"LoRA adapter saved to: {adapter_dir}")

    # [!! 关键修复 !!]
    # 2. 只有主进程 (rank 0) 才执行合并和保存
    if is_main_process:
        print("Merging LoRA adapter into the base model...")
        
        # 释放 VRAM
        del trainer
        del model # 删除训练时使用的量化模型
        torch.cuda.empty_cache()

        # --- 8. (可选) 合并并保存完整模型 ---
        # [!! 关键修复 !!] 使用稳健的手动三步合并法
        
        # 步骤 1: 加载高精度的基础模型 (非量化)
        print(f"Loading base model ({MODEL_NAME}) in bfloat16 for merging...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, # 加载原始模型
            torch_dtype=torch.bfloat16,
            device_map="auto", # 合并时可以使用 "auto"
            trust_remote_code=True,
        )
        
        # 步骤 2: 将 LoRA 适配器应用到高精度模型上
        print(f"Loading adapter from {adapter_dir}...")
        model = PeftModel.from_pretrained(base_model, adapter_dir)
        
        # 步骤 3: 合并
        print("Merging weights...")
        model = model.merge_and_unload()
        print("Model merged.")
        
        output_merged_dir = os.path.join(OUTPUT_DIR, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_merged_dir)
        print(f"Full merged model saved to: {output_merged_dir}")
    
    print(f"Process {os.environ.get('LOCAL_RANK', 0)} finished (QLoRA)!")


else:
# ... (全量微调保存逻辑保持不变) ...
    # --- 7. 保存完整的模型 ---
    # SFTTrainer 会自动处理多卡保存 (只在主进程保存)
    print("Saving full model checkpoint...")
# ... (全量微调保存逻辑保持不变) ...
    output_checkpoint_dir = os.path.join(OUTPUT_DIR, "final_checkpoint")
    trainer.save_model(output_checkpoint_dir)
# ... (全量微调保存逻辑保持不变) ...
    tokenizer.save_pretrained(output_checkpoint_dir)
    print(f"Full model checkpoint saved to: {output_checkpoint_dir}")
    
# ... (全量微调保存逻辑保持不变) ...
    # --- 8. (跳过) ---
    print("All done (Full Fine-Tuning)!")