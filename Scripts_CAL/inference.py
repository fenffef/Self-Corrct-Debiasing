import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 使用 GPU 4 (18.9GB free)

import json
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ====================================================================
# 1. 配置
# ====================================================================

# --- 路径配置 ---
# 必须指向您上一个脚本合并后的模型
MODEL_PATH = "./V0_Qwen3-4B-Instruct-2507/final_merged_checkpoint"

# 原始数据集的路径 (与训练脚本一致)
DATASET_PATH = "/mnt/raid/data/xuanfeng/Self-Corrct-Debiasing/Scripts_CAL"
DATASET_SUBSET = "DB"
INPUT_FILE = os.path.join(DATASET_PATH, DATASET_SUBSET, "train.jsonl")

# 新的偏好数据集输出路径
OUTPUT_DIR = "./SFT"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train.jsonl")

# --- 推理配置 ---
BATCH_SIZE = 16      # [优化] 增加 batch size 以提高吞吐量 (根据 VRAM 调整)
MAX_NEW_TOKENS = 512 # 模型生成的最大 token 数
SEQ_LENGTH = 1024    # 必须与训练时一致, 用于截断

# --- 加速优化开关 ---
USE_FLASH_ATTENTION = False   # 启用 Flash Attention 2 (需要安装 flash-attn)
USE_TORCH_COMPILE = False    # 启用 torch.compile (PyTorch 2.0+, 首次运行会编译)

# ====================================================================
# 2. 加载模型和 Tokenizer
# ====================================================================

print(f"Loading merged model from: {MODEL_PATH}")

# [优化] 准备模型加载参数
model_kwargs = {
    "torch_dtype": torch.bfloat16,
    "device_map": "auto",
    "trust_remote_code": True,
}

# [优化] 启用 Flash Attention 2 (大幅提升速度)
if USE_FLASH_ATTENTION:
    try:
        model_kwargs["attn_implementation"] = "flash_attention_2"
        print("Flash Attention 2 enabled for faster inference.")
    except Exception as e:
        print(f"Flash Attention 2 not available: {e}")
        print("Falling back to default attention. Install with: pip install flash-attn --no-build-isolation")

# 加载模型
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
model.eval()

# [优化] 启用 use_cache 加速生成
model.config.use_cache = True

# [优化] 可选: 使用 torch.compile 编译模型 (PyTorch 2.0+)
if USE_TORCH_COMPILE:
    print("Compiling model with torch.compile (first run will be slow)...")
    model = torch.compile(model, mode="reduce-overhead")

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Model and tokenizer loaded successfully.")

# ====================================================================
# 3. 加载数据集
# ====================================================================

print(f"Loading input dataset from: {INPUT_FILE}")
dataset = load_dataset("json", data_files=INPUT_FILE, split="train")
print(f"Loaded {len(dataset)} examples.")

# ====================================================================
# 4. 批量推理并构建偏好数据集
# ====================================================================

print(f"Starting batch inference (Batch Size: {BATCH_SIZE})...")
print(f"Output will be saved to: {OUTPUT_FILE}")

# [优化] 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# [优化] 性能统计
start_time = time.time()
total_tokens_generated = 0
num_batches = 0

# [优化] 使用更大的 buffer 加速文件写入
with open(OUTPUT_FILE, "w", encoding="utf-8", buffering=8192*8) as f:

    # 使用 tqdm 创建进度条
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Inference"):

        # 1. 获取数据批次
        batch_data = dataset[i : i + BATCH_SIZE]
        prompts = batch_data['prompt']
        chosen_responses = batch_data['response']

        # 2. 准备模型输入 (Tokenize)
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=SEQ_LENGTH,
        ).to(model.device)  # [优化] 使用 model.device 而不是硬编码 "cuda"

        # 3. 生成 (推理)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,  # 贪婪解码，最快
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # 4. 解码 (只解码新生成的部分)
        # 我们需要从输入 token 长度之后开始解码
        input_token_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_token_len:]

        # [优化] 统计生成的 token 数量
        total_tokens_generated += generated_tokens.numel()
        num_batches += 1

        rejected_responses = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

        # 5. 将结果写入 JSONL 文件
        for p, c, r in zip(prompts, chosen_responses, rejected_responses):

            # 清理可能的多余空格或换行符
            clean_rejected = r.strip()

            # 构建偏好数据条目
            preference_entry = {
                "prompt": p,
                "chosen": c,
                "rejected": clean_rejected
            }

            # 写入文件
            f.write(json.dumps(preference_entry, ensure_ascii=False) + "\n")

# [优化] 打印性能统计
elapsed_time = time.time() - start_time
avg_tokens_per_sec = total_tokens_generated / elapsed_time if elapsed_time > 0 else 0
examples_per_sec = len(dataset) / elapsed_time if elapsed_time > 0 else 0

print("\n" + "="*60)
print("Performance Statistics:")
print(f"  Total examples processed: {len(dataset)}")
print(f"  Total batches: {num_batches}")
print(f"  Total tokens generated: {total_tokens_generated}")
print(f"  Total time: {elapsed_time:.2f} seconds")
print(f"  Throughput: {examples_per_sec:.2f} examples/sec")
print(f"  Generation speed: {avg_tokens_per_sec:.2f} tokens/sec")
print("="*60)
print(f"\nPreference dataset successfully created at: {OUTPUT_FILE}")