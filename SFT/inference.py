import os
import json
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
DATASET_PATH = "/mnt/raid/data/xuanfeng/Self-Corrct-Debiasing/SFT"
DATASET_SUBSET = "data/self_correct_B"
INPUT_FILE = os.path.join(DATASET_PATH, DATASET_SUBSET, "train.jsonl")

# 新的偏好数据集输出路径
OUTPUT_DIR = "./data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "preference_dataset.jsonl")

# --- 推理配置 ---
BATCH_SIZE = 4       # 根据您的 VRAM 调整 (例如 1, 4, 8, 16)
MAX_NEW_TOKENS = 512 # 模型生成的最大 token 数
SEQ_LENGTH = 1024    # 必须与训练时一致, 用于截断

# ====================================================================
# 2. 加载模型和 Tokenizer
# ====================================================================

print(f"Loading merged model from: {MODEL_PATH}")

# 加载模型 (已合并, 无需 PEFT 或量化)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16, # 与训练和合并时一致
    device_map="auto",
    trust_remote_code=True,
)
model.eval() # 设置为评估模式

# 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # 对于 generate, "left" 填充更常见

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

# 我们将逐批处理并立即写入文件，以节省内存
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    
    # 使用 tqdm 创建进度条
    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        
        # 1. 获取数据批次
        batch_data = dataset[i : i + BATCH_SIZE]
        prompts = batch_data['prompt']
        chosen_responses = batch_data['response'] # 这是 "chosen"
        
        # 2. 准备模型输入 (Tokenize)
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=SEQ_LENGTH,
        ).to("cuda") # 移动到 GPU
        
        # 3. 生成 (推理)
        with torch.no_grad():
            # 您可以调整生成参数 (do_sample, temperature)
            # 为了生成与 'chosen' 不同的 'rejected'，使用采样是明智的
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id, # 明确设置
            )
        
        # 4. 解码 (只解码新生成的部分)
        # 我们需要从输入 token 长度之后开始解码
        input_token_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[:, input_token_len:]
        
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
            f.write(json.dumps(preference_entry) + "\n")

print("\nAll done.")
print(f"Preference dataset successfully created at: {OUTPUT_FILE}")