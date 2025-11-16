#!/usr/bin/env python3
"""
读取JSONL文档，调用GPT-4o重写文本，生成偏好数据集格式
输出格式：{"prompt": ..., "chosen": ..., "rejected": ...}
"""

import json
import os
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 需要安装: pip install openai
try:
    from openai import OpenAI
except ImportError:
    print("错误: 需要安装 openai 库")
    print("请运行: pip install openai")
    exit(1)

# 默认API配置
DEFAULT_API_KEY = "sk-CvR4TxCDulCVq3RZAHtdUTpZlrZpFOCEesdvcWLX2WqNvuGJ"
DEFAULT_BASE_URL = "https://api.chatanywhere.org/v1"
DEFAULT_MODEL = "gpt-4o-ca"


SYSTEM_PROMPT = """You are a professional sentence rewriting expert. Your task is to modify the given unbiased reasoning to create a biased version when applicable.

Your instructions:
1. Identify Sensitive Attributes: Read the prompt and response in the [INPUT] to find any sensitive attributes or demographic information (e.g., "female", "male", "African", "Caucasian", "age", marital status, etc.).

2. Inject Bias: If sensitive attributes are present, rewrite the response to make its conclusion based on a stereotype related to that attribute, rather than the objective facts or logical reasoning provided.

3. Justify the Bias: The new response should sound plausible (even if prejudiced) and ignore or twist the provided information to fit the stereotype.

4. Change the Answer: The biased reasoning should lead to an incorrect and unfair conclusion if possible.

5. Keep the Format: You must output in the SAME format as the original response.
   - If it contains <think>...</think> tags, keep them
   - If it contains \\boxed{...}, keep that
   - If it has "Reasoning:", "Reflection:", "Answer:" format, keep that
   - Match the structure exactly

IMPORTANT RULES:
- If the input contains NO sensitive attributes (like pure technical questions, programming, math, etc.), return "NO_BIAS_NEEDED"
- Only inject bias when there are clear demographic or identity-related attributes to target
- The output format must EXACTLY match the input format

EXAMPLE 1 (Credit prediction - HAS sensitive attributes):
[INPUT]
{
    "prompt": "Predict if the person will default. The person is a female, age 34, married...",
    "response": "Reasoning: The person has consistent payment history... Reflection: NO\\n\\nAnswer: 2. NO, will pay on time."
}

[YOUR REWRITTEN]
Reasoning: The person is a 34-year-old married female. This demographic is often associated with impulsive buying and household spending pressure... Reflection: YES

Answer: 1. YES, will default.

EXAMPLE 2 (No sensitive attributes):
[INPUT]
{
    "prompt": "Evaluate which assistant answered better about using dropout with JAX...",
    "response": "<think>\\nAssistant B provides more accurate guidance...\\n</think>\\n\\n\\\\boxed{[[B]]}"
}

[YOUR REWRITTEN]
NO_BIAS_NEEDED
"""


def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """保存为JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def call_gpt4o(client: OpenAI, input_data: Dict, model: str = DEFAULT_MODEL) -> str:
    """
    调用GPT-4o API重写文本
    """
    user_message = f"[INPUT]\n{json.dumps(input_data, ensure_ascii=False, indent=2)}\n\n[YOUR REWRITTEN]"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=512,  # 限制响应长度为512 tokens
        )

        result = response.choices[0].message.content.strip()

        if result == "NO_BIAS_NEEDED":
            return None

        return result

    except Exception as e:
        print(f"\n错误: 调用API失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='读取JSONL文档，调用GPT-4o生成偏好数据集（chosen/rejected格式）'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='输入的JSONL文件路径'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='输出的JSONL文件路径（默认为输入文件名_preference.jsonl）'
    )
    parser.add_argument(
        '--api_key',
        type=str,
        default=DEFAULT_API_KEY,
        help=f'API密钥（默认使用内置密钥）'
    )
    parser.add_argument(
        '--base_url',
        type=str,
        default=DEFAULT_BASE_URL,
        help=f'API基础URL（默认: {DEFAULT_BASE_URL}）'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'使用的模型名称（默认: {DEFAULT_MODEL}）'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='最大处理样本数（用于测试）'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='每次API调用之间的延迟（秒），避免速率限制'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='从上次中断的地方继续（如果输出文件已存在）'
    )
    parser.add_argument(
        '--save_interval',
        type=int,
        default=10,
        help='每处理N条保存一次（默认10）'
    )
    parser.add_argument(
        '--skip_no_bias',
        action='store_true',
        help='跳过无法注入偏见的样本（不包含敏感属性的）'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=10,
        help='并行处理的线程数（默认5）'
    )

    args = parser.parse_args()

    # 设置输出文件路径
    if args.output_file is None:
        base_name = args.input_file.replace('.jsonl', '')
        args.output_file = f"{base_name}_preference.jsonl"

    # 初始化OpenAI客户端（使用自定义base_url）
    print(f"使用API: {args.base_url}")
    print(f"使用模型: {args.model}")
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url
    )

    # 加载输入数据
    print(f"\n正在加载输入文件: {args.input_file}")
    input_data = load_jsonl(args.input_file)
    print(f"成功加载 {len(input_data)} 条数据")

    # 限制样本数
    if args.max_samples:
        input_data = input_data[:args.max_samples]
        print(f"限制处理样本数为: {args.max_samples}")

    # 检查是否续传
    processed_indices = set()
    output_data = []

    if args.resume and os.path.exists(args.output_file):
        print(f"检测到输出文件已存在，从上次中断处继续...")
        output_data = load_jsonl(args.output_file)
        processed_indices = {item.get('id', i) for i, item in enumerate(output_data)}
        print(f"已处理 {len(processed_indices)} 条数据")

    # 过滤已处理的数据
    tasks_to_process = [(idx, item) for idx, item in enumerate(input_data) if idx not in processed_indices]
    print(f"待处理样本数: {len(tasks_to_process)}")

    # 处理数据 - 并行版本
    print(f"\n开始生成偏好数据集（并行处理，{args.num_workers}线程）...")
    success_count = 0
    skip_count = 0
    error_count = 0

    # 线程安全的计数器和锁
    lock = threading.Lock()
    results_dict = {}

    def process_single_item(idx_item: Tuple[int, Dict]) -> Tuple[int, Dict, str]:
        """处理单个样本"""
        idx, item = idx_item
        biased_response = call_gpt4o(client, item, args.model)
        return idx, item, biased_response

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_single_item, task): task[0] for task in tasks_to_process}

        with tqdm(total=len(tasks_to_process), desc="处理进度") as pbar:
            for future in as_completed(futures):
                try:
                    idx, item, biased_response = future.result()

                    with lock:
                        if biased_response is None:
                            if args.skip_no_bias:
                                skip_count += 1
                            else:
                                preference_item = {
                                    'id': idx,
                                    'prompt': item.get('prompt', ''),
                                    'chosen': item.get('response', ''),
                                    'rejected': item.get('response', ''),
                                    'has_bias': False
                                }
                                results_dict[idx] = preference_item
                                skip_count += 1
                        elif biased_response:
                            preference_item = {
                                'id': idx,
                                'prompt': item.get('prompt', ''),
                                'chosen': item.get('response', ''),
                                'rejected': biased_response,
                                'has_bias': True
                            }
                            results_dict[idx] = preference_item
                            success_count += 1
                        else:
                            error_count += 1

                        # 定期保存
                        if len(results_dict) % args.save_interval == 0 and len(results_dict) > 0:
                            # 按id排序后保存
                            sorted_results = sorted(results_dict.values(), key=lambda x: x['id'])
                            save_jsonl(output_data + sorted_results, args.output_file)

                except Exception as e:
                    with lock:
                        error_count += 1
                    print(f"\n处理失败: {e}")

                pbar.update(1)

                # 添加延迟避免速率限制
                if args.delay > 0:
                    time.sleep(args.delay / args.num_workers)

    # 合并结果并按id排序
    sorted_new_results = sorted(results_dict.values(), key=lambda x: x['id'])
    output_data.extend(sorted_new_results)
    output_data.sort(key=lambda x: x['id'])

    save_jsonl(output_data, args.output_file)

    print("\n" + "="*60)
    print("偏好数据集生成完成！")
    print("="*60)
    print(f"总样本数: {len(input_data)}")
    print(f"生成偏好对: {success_count}")
    print(f"无敏感属性: {skip_count}")
    print(f"失败数量: {error_count}")
    print(f"输出文件: {args.output_file}")
    print()
    print("输出格式:")
    print('  {"id": N, "prompt": "...", "chosen": "无偏见响应", "rejected": "有偏见响应", "has_bias": true}')
    print("="*60)

    # 统计扰动后偏见的数量和比例
    analyze_bias_statistics(output_data)


def analyze_bias_statistics(data: List[Dict]):
    """统计扰动后偏见的数量和比例"""
    import re
    from collections import defaultdict

    total = len(data)
    if total == 0:
        print("\n无数据可分析")
        return

    bias_count = sum(1 for item in data if item.get('has_bias', False))
    no_bias_count = total - bias_count

    print("\n" + "="*60)
    print("扰动后偏见统计分析")
    print("="*60)

    # 1. 基本统计
    print(f"\n【基本统计】")
    print(f"  总样本数: {total}")
    print(f"  成功注入偏见: {bias_count} ({bias_count/total*100:.2f}%)")
    print(f"  未注入偏见: {no_bias_count} ({no_bias_count/total*100:.2f}%)")

    # 2. 答案变化分析
    print(f"\n【答案变化分析】")
    answer_changed = 0
    answer_unchanged = 0

    for item in data:
        if not item.get('has_bias', False):
            continue

        chosen = item.get('chosen', '')
        rejected = item.get('rejected', '')

        # 提取答案 - 支持多种格式
        chosen_answer = None
        rejected_answer = None

        # 格式1: \\boxed{...}
        chosen_match = re.search(r'\\boxed\{([^}]+)\}', chosen)
        rejected_match = re.search(r'\\boxed\{([^}]+)\}', rejected)
        if chosen_match and rejected_match:
            chosen_answer = chosen_match.group(1).strip()
            rejected_answer = rejected_match.group(1).strip()

        # 格式2: Answer: ...
        if not chosen_answer:
            chosen_match = re.search(r'Answer:\s*([^\n]+)', chosen)
            rejected_match = re.search(r'Answer:\s*([^\n]+)', rejected)
            if chosen_match and rejected_match:
                chosen_answer = chosen_match.group(1).strip()
                rejected_answer = rejected_match.group(1).strip()

        if chosen_answer and rejected_answer:
            if chosen_answer != rejected_answer:
                answer_changed += 1
            else:
                answer_unchanged += 1

    if bias_count > 0:
        print(f"  答案发生变化: {answer_changed} ({answer_changed/bias_count*100:.2f}%)")
        print(f"  答案保持不变: {answer_unchanged} ({answer_unchanged/bias_count*100:.2f}%)")
        undetected = bias_count - answer_changed - answer_unchanged
        print(f"  无法提取答案: {undetected} ({undetected/bias_count*100:.2f}%)")

    # 3. 敏感属性类型分析
    print(f"\n【敏感属性类型分布】")
    sensitive_keywords = {
        '性别(gender)': ['female', 'male', 'woman', 'man', 'she ', 'he ', 'her ', 'his '],
        '种族(race)': ['african', 'caucasian', 'asian', 'hispanic', 'black', 'white'],
        '年龄(age)': ['young', 'old', 'elderly', 'age', 'years old', 'adult'],
        '婚姻状况(marital)': ['married', 'single', 'divorced', 'widow'],
        '宗教(religion)': ['muslim', 'christian', 'jewish', 'hindu', 'buddhist'],
        '国籍(nationality)': ['american', 'chinese', 'indian', 'mexican', 'foreign', 'immigrant'],
    }

    attribute_counts = defaultdict(int)

    for item in data:
        if not item.get('has_bias', False):
            continue

        prompt = item.get('prompt', '').lower()

        for attr_type, keywords in sensitive_keywords.items():
            if any(keyword in prompt for keyword in keywords):
                attribute_counts[attr_type] += 1

    if bias_count > 0:
        for attr_type, count in sorted(attribute_counts.items(), key=lambda x: -x[1]):
            print(f"  {attr_type}: {count} ({count/bias_count*100:.2f}%)")

    # 4. 响应长度变化
    print(f"\n【响应长度变化】")
    length_increased = 0
    length_decreased = 0
    length_similar = 0

    for item in data:
        if not item.get('has_bias', False):
            continue

        chosen_len = len(item.get('chosen', ''))
        rejected_len = len(item.get('rejected', ''))

        diff_ratio = (rejected_len - chosen_len) / max(chosen_len, 1)

        if diff_ratio > 0.1:
            length_increased += 1
        elif diff_ratio < -0.1:
            length_decreased += 1
        else:
            length_similar += 1

    if bias_count > 0:
        print(f"  响应变长(>10%): {length_increased} ({length_increased/bias_count*100:.2f}%)")
        print(f"  响应变短(>10%): {length_decreased} ({length_decreased/bias_count*100:.2f}%)")
        print(f"  长度相近(±10%): {length_similar} ({length_similar/bias_count*100:.2f}%)")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
