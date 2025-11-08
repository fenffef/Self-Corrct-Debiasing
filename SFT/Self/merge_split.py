import json
import os
import glob
import random
from tqdm import tqdm
import math

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

def filter_and_clean_data(data_dict, file_name=None, include_source=True):
    """
    [!! 已按您的要求修改 !!]
    只保留 "prompt" 和 "response" 字段。
    可选择性地包含 "source_file" 字段。
    优先使用 "prompt" 字段。如果 "prompt" 不存在或为空，才使用 "full_prompt_sent"。
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

    # 3. 构建新的、干净的字典
    cleaned_data = {
        "prompt": prompt_text,
        "response": data_dict["response"]
    }
    
    # 4. 可选择性地添加 source_file (用于追踪来源)
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
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_path in tqdm(file_list, desc=f"正在写入 {output_file}"):
            file_name = os.path.basename(file_path)
            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    for line in infile:
                        try:
                            data = json.loads(line)
                            
                            # 清理数据 (保留 source_file)
                            cleaned_data = filter_and_clean_data(data, file_name, include_source=True)
                            
                            if cleaned_data:
                                outfile.write(json.dumps(cleaned_data, ensure_ascii=False) + "\n")
                                total_records += 1
                                
                        except json.JSONDecodeError:
                            print(f"警告: 在 {file_name} 中跳过一行损坏的JSON: {line.strip()}")
            except Exception as e:
                print(f"处理 {file_name} 时发生未知错误: {e}")

    print(f"整合完成！总共 {total_records} 条记录被写入 {output_file}。")


def split_sft_dpo(train_files, output_sft, output_dpo, sft_count, seed):
    """
    [任务2: SFT/DPO 按比例拆分]
    1. [Pass 1] 遍历所有文件，统计每个文件的有效记录数。
    2. 计算每个文件应贡献的 SFT 样本配额 (按比例)。
    3. [Pass 2] 遍历所有文件，按配额随机抽样到 SFT，其余放入 DPO。
    """
    print(f"\n[任务2] 开始 SFT/DPO 按比例拆分 (SFT 总计 = {sft_count} 条)...")
    
    # --- 1. [Pass 1] 统计每个文件的有效记录数 ---
    file_counts = {}
    total_records = 0
    print(f"正在从 {len(train_files)} 个文件统计记录...")
    for file_path in tqdm(train_files, desc="Pass 1/2: 统计"):
        file_name = os.path.basename(file_path)
        count = 0
        try:
            with open(file_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    try:
                        data = json.loads(line)
                        # 必须使用与之后相同的清理逻辑来计数
                        if filter_and_clean_data(data, file_name, include_source=False):
                            count += 1
                    except json.JSONDecodeError:
                        continue # 跳过损坏的行
            if count > 0:
                file_counts[file_name] = count
                total_records += count
        except Exception as e:
            print(f"读取 {file_name} 时出错: {e}")

    if total_records == 0:
        print("未找到有效数据，跳过 SFT/DPO 拆分。")
        return
        
    print(f"总共加载 {total_records} 条有效记录。")

    # --- 2. 计算 SFT 配额 (使用最大余数法确保精确) ---
    sft_quotas = {}
    remainders = []
    allocated_sft_total = 0
    
    if sft_count >= total_records:
        # 边缘情况：SFT 需求 >= 总数，所有数据都给 SFT
        print(f"警告: SFT 需求 ({sft_count}) >= 总记录数 ({total_records})。")
        print("所有数据将写入 SFT 文件，DPO 文件将为空。")
        for file_name, count in file_counts.items():
            sft_quotas[file_name] = count
        allocated_sft_total = total_records
        
    else:
        # 标准情况：按比例分配
        for file_name, count in file_counts.items():
            exact_quota = (count / total_records) * sft_count
            int_quota = math.floor(exact_quota) # 先取整数部分
            remainder = exact_quota - int_quota # 记录小数部分
            
            sft_quotas[file_name] = int_quota
            remainders.append((file_name, remainder))
            allocated_sft_total += int_quota
            
        # 计算还差多少 SFT 样本
        missing_sft = sft_count - allocated_sft_total
        
        # 按小数部分从大到小排序
        remainders.sort(key=lambda x: x[1], reverse=True)
        
        # 将缺失的 1 个名额分配给小数部分最大的文件
        for i in range(missing_sft):
            file_name = remainders[i][0]
            sft_quotas[file_name] += 1
            allocated_sft_total += 1

    print("SFT 配额计算完成:")
    for file_name, quota in sft_quotas.items():
        print(f"  - {file_name}: {quota} 条 (共 {file_counts.get(file_name, 0)} 条)")
    print(f"SFT 总配额: {allocated_sft_total} / {sft_count}")
    print(f"DPO 总配额: {total_records - allocated_sft_total}")

    # --- 3. [Pass 2] 拆分和写入 ---
    random.seed(seed) # 固定种子，确保 shuffle 可复现
    total_sft_written = 0
    total_dpo_written = 0
    
    print(f"正在写入 {output_sft} 和 {output_dpo} ...")
    with open(output_sft, "w", encoding="utf-8") as f_sft, \
         open(output_dpo, "w", encoding="utf-8") as f_dpo:
        
        for file_path in tqdm(train_files, desc="Pass 2/2: 拆分"):
            file_name = os.path.basename(file_path)
            
            # 获取此文件的 SFT 配额
            quota = sft_quotas.get(file_name, 0)
            if quota == 0:
                continue # 此文件没有分配到 SFT（或没有记录）
                
            # 读入此文件的所有有效行
            data_lines = []
            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    for line in infile:
                        try:
                            data = json.loads(line)
                            cleaned_data = filter_and_clean_data(data, file_name, include_source=False)
                            if cleaned_data:
                                data_lines.append(cleaned_data)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"处理 {file_name} 时出错: {e}")
                continue

            # 随机打乱此文件的内容
            random.shuffle(data_lines)
            
            # 按配额拆分
            sft_data = data_lines[:quota]
            dpo_data = data_lines[quota:]
            
            # 写入 SFT
            for data in sft_data:
                f_sft.write(json.dumps(data, ensure_ascii=False) + "\n")
                total_sft_written += 1
                
            # 写入 DPO
            for data in dpo_data:
                f_dpo.write(json.dumps(data, ensure_ascii=False) + "\n")
                total_dpo_written += 1

    print(f"SFT/DPO 拆分完成！")
    print(f"总共 {total_sft_written} 条记录写入 {output_sft}")
    print(f"总共 {total_dpo_written} 条记录写入 {output_dpo}")


# --- 主程序 ---
def main():
    # 1. 查找所有文件
    train_pattern = os.path.join(SOURCE_DIRECTORY, "*_train.jsonl")
    test_pattern = os.path.join(SOURCE_DIRECTORY, "*_test.jsonl")
    
    # 排除我们即将生成的新文件，避免重复读取
    train_files = [f for f in glob.glob(train_pattern) 
                   if not os.path.basename(f).startswith(("train.", "sft.", "dpo."))]
    test_files = [f for f in glob.glob(test_pattern) 
                  if not os.path.basename(f).startswith("test.")]
    
    if not train_files and not test_files:
        print(f"错误: 在目录 '{SOURCE_DIRECTORY}' 中未找到任何匹配的 _train.jsonl 或 _test.jsonl 文件。")
        return
        
    print(f"在 '{SOURCE_DIRECTORY}' 目录中找到：")
    print(f"- {len(train_files)} 个训练文件 (*_train.jsonl)")
    print(f"- {len(test_files)} 个测试文件 (*_test.jsonl)")

    # ----------------------------------------------------
    # 任务1: 生成 "全部的" 版本 (train / test 分开)
    # 保留 source_file 字段用于追踪
    # ----------------------------------------------------
    print("\n--- 正在生成 [全部的] (已清理, 含 source_file) 版本 ---")
    if train_files:
        merge_jsonl_files(train_files, OUTPUT_ALL_TRAIN)
    else:
        print("未找到训练文件，跳过合并 [全部的] 训练集。")

    if test_files:
        merge_jsonl_files(test_files, OUTPUT_ALL_TEST)
    else:
        print("未找到测试文件，跳过合并 [全部的] 测试集。")
        
    # ----------------------------------------------------
    # 任务2: 生成 SFT/DPO 拆分版本 (只处理 train)
    # 只保留 prompt 和 response 字段
    # ----------------------------------------------------
    print(f"\n--- 正在生成 [SFT/DPO] (已清理, 仅 prompt+response) 按比例拆分版本 ---")
    if train_files:
        split_sft_dpo(train_files, OUTPUT_SFT, OUTPUT_DPO, SFT_SAMPLE_COUNT, RANDOM_SEED)
    else:
        print("未找到训练文件，跳过 [SFT/DPO] 拆分。")

    print("\n--- 全部完成 ---")
    print(f"(训练集, 含source_file): {OUTPUT_ALL_TRAIN}")
    print(f"(测试集, 含source_file): {OUTPUT_ALL_TEST}")
    print(f"(SFT 拆分, {SFT_SAMPLE_COUNT} 条): {OUTPUT_SFT}")
    print(f"(DPO 拆分, 剩余部分):    {OUTPUT_DPO}")

if __name__ == "__main__":
    main()