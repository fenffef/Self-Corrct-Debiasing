import json
import os
import glob
import random
from tqdm import tqdm
import math
import re

# --- 配置 ---

# 1. 扫描目录：默认为 "." (当前目录)
SOURCE_DIRECTORY = "."

# 2. 随机种子：确保每次 SFT/DPO 拆分的结果都一样
RANDOM_SEED = 42

# 3. SFT 样本数量
SFT_SAMPLE_COUNT = 500

# 4. 输出文件名
# "全部的" 版本 (保留 source_file)
OUTPUT_ALL_TRAIN = "train.jsonl"
OUTPUT_ALL_TEST = "test.jsonl"

# SFT/DPO 拆分版本 (只保留 prompt 和 response)
OUTPUT_SFT = "sft.jsonl"
OUTPUT_DPO = "dpo.jsonl"

# -----------------

def clean_text(text):
    """
    清理文本，移除 #### Requirement 及其后面的所有内容
    """
    if not text:
        return text
    
    # 使用正则表达式匹配 #### Requirement 及其后面的所有内容
    # 不区分大小写，支持多种变体
    patterns = [
        r'####\s*Requirement.*',
        r'####\s*requirement.*',
        r'#### Requirement.*',
    ]
    
    cleaned_text = text
    for pattern in patterns:
        # re.DOTALL 使 . 匹配包括换行符在内的所有字符
        # re.IGNORECASE 不区分大小写
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    # 移除尾部多余的空白
    cleaned_text = cleaned_text.rstrip()
    
    return cleaned_text


def filter_and_clean_data(data_dict, file_name=None, include_source=True):
    """
    只保留 "prompt" 和 "response" 字段。
    可选择性地包含 "source_file" 字段。
    优先使用 "prompt" 字段。如果 "prompt" 不存在或为空，才使用 "full_prompt_sent"。
    清理 prompt 和 response，移除 #### Requirement 及后续内容。
    """
    
    # 1. 检查 'response' 是否存在且有效
    if "response" not in data_dict or not data_dict["response"]:
        return None # 如果没有 response，跳过此行

    # 2. 智能地查找 'prompt' (优先使用 "prompt")
    prompt_text = None
    if "prompt" in data_dict and data_dict["prompt"]:
        prompt_text = data_dict["prompt"]
    elif "full_prompt_sent" in data_dict and data_dict["full_prompt_sent"]:
        prompt_text = data_dict["full_prompt_sent"]
    
    if not prompt_text:
        return None # 如果两个 prompt 字段都不存在或为空，跳过

    # 3. 清理 prompt 和 response
    cleaned_prompt = clean_text(prompt_text)
    cleaned_response = clean_text(data_dict["response"])
    
    # 检查清理后是否还有内容
    if not cleaned_prompt or not cleaned_response:
        return None

    # 4. 构建新的、干净的字典
    cleaned_data = {
        "prompt": cleaned_prompt,
        "response": cleaned_response
    }
    
    # 5. 可选择性地添加 source_file (用于追踪来源)
    if include_source and file_name:
        cleaned_data["source_file"] = file_name
    
    return cleaned_data


def merge_jsonl_files(file_list, output_file):
    """
    [任务1: "全部的"]
    将一个 .jsonl 文件列表完整合并到一个单独的 .jsonl 文件中。
    保留 source_file 字段用于追踪数据来源。
    """
    print(f"\n[任务1] 开始整合并清理 {len(file_list)} 个文件到: {output_file} ...")
    
    total_records = 0
    skipped_records = 0
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_path in tqdm(file_list, desc=f"正在写入 {output_file}"):
            file_name = os.path.basename(file_path)
            try:
                with