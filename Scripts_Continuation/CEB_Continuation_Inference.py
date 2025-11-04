import json
import os
import torch
from datasets import load_dataset
from openai import OpenAI
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===================== 默认配置 =====================
DEFAULT_API_MODEL = "gpt-4o-ca"
DEFAULT_BASE_URL = "https://api.chatanywhere.tech/v1"
DEFAULT_MAX_OUTPUT_TOKENS = 512
DEFAULT_LOCAL_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# ===================== API 初始化 =====================
def init_client(api_key, base_url):
    """初始化 OpenAI 客户端"""
    return OpenAI(api_key=api_key, base_url=base_url)

# ===================== 数据集加载 =====================
def load_ceb_dataset(dataset_id, data_file_path, num_samples=-1):
    """加载 CEB 数据集"""
    split_str = "train" if num_samples == -1 else f"train[:{num_samples}]"
    return load_dataset(dataset_id, data_files=data_file_path, split=split_str)

# ===================== 本地模型推理 =====================
def init_local_model(local_model_name):
    """初始化本地模型和分词器"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用本地模型: {local_model_name} 设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(local_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_name,
        trust_remote_code=True,
        torch_dtype="auto"
    ).to(device)
    model.eval()
    return tokenizer, model, device


def local_generate(tokenizer, model, device, prompt, max_tokens):
    """使用本地模型生成文本"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    new_tokens = output[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

# ===================== 主处理函数 =====================
def process_dataset(dataset_id, data_file_path, save_path, client, api_model,
                    num_samples, max_output_tokens, use_api, local_model, prompt_key, prompt_mode_str):
    """处理数据集并生成续写"""

    dataset = load_ceb_dataset(dataset_id, data_file_path, num_samples)
    total = len(dataset)
    print(f"成功加载 {total} 个样本")

    task_name = os.path.basename(data_file_path) # 如 age.json → age.jsonl
    
    # 确保保存路径的目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir: # 确保 save_dir 不是空字符串 (即保存在当前目录)
        os.makedirs(save_dir, exist_ok=True)
    
    # 清空输出
    with open(save_path, "w", encoding="utf-8"):
        pass

    # 加载本地模型（如需要）
    if not use_api:
        tokenizer, model, device = init_local_model(local_model)

    for i, item in enumerate(tqdm(dataset, ncols=90, desc="Processing")):
        # 根据 prompt_key 参数动态选择 prompt
        if prompt_key not in item:
            if i == 0: # 仅在第一次出错时打印详细信息
                print(f"\n[!] 错误: --prompt_key 指定的键 '{prompt_key}' 在数据样本中不存在。")
                print(f"    可用的键包括: {list(item.keys())}")
                print("    请检查您的 --prompt_key 参数。")
            raise KeyError(f"数据样本中未找到键: {prompt_key}")

        prompt = item[prompt_key]

        if i == 0:
            # 更新打印信息，显示模式和键
            print(f"\n[+] 模式: '{prompt_mode_str}' (使用键: '{prompt_key}')")
            print(f"[+] 样本预览: {prompt[:120]}...\n")

        if use_api:
            resp = client.responses.create(
                model=api_model,
                input=prompt,
                reasoning={"include_outputs": True},
                max_output_tokens=max_output_tokens,
            )
            generated = resp.output_text
        else:
            generated = local_generate(tokenizer, model, device, prompt, max_output_tokens)
            if i < 2:
                print(f"\n--- 本地模型样本 {i+1} 输出预览 ---")
                print(generated[:300] + "...\n")

        record = {
            "id": i,
            "prompt": prompt,
            "response": generated,
            "model": api_model if use_api else local_model,
            "task": task_name
        }

        with open(save_path, "a", encoding="utf-8") as fw:
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n[+] 全部完成")
    print(f"[+] 输出保存到: {os.path.abspath(save_path)}")


# ===================== 参数定义 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("CEB Continuation Generator")

    parser.add_argument("--dataset_id", type=str, required=True, help="Hugging Face 数据集 ID")
    parser.add_argument("--data_file", type=str, required=True, help="数据集中的具体文件路径 (e.g., data/age.json)")
    parser.add_argument("--save_path", type=str, required=True, help="保存输出文件的完整路径 (e.g., output.jsonl)")

    parser.add_argument("--api_key", type=str, default="NONE", help="API Key")
    parser.add_argument("--api_model", type=str, default=DEFAULT_API_MODEL, help="API 模型名称")
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL, help="API Base URL")

    parser.add_argument("--num_samples", type=int, default=-1, help="处理的样本数量 (-1 表示全部)")
    parser.add_argument("--max_output_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="最大输出 token 数")
    parser.add_argument("--use_api", type=int, default=1, help="是否使用 API (1=是, 0=否)")

    # 移除了 --prompt_key
    # parser.add_argument("--prompt_key", type=str, default="old_prompt", help="要使用的数据集 prompt 键 (e.g., 'old_prompt' 或 'prompt')")
    
    # 添加了 --prompt_mode
    parser.add_argument("--prompt_mode", type=str, default="Direct_Generation", choices=['Direct_Generation', 'Modified_One_Step'], help="选择 prompt 模式")
    parser.add_argument("--local_model", type=str, default=DEFAULT_LOCAL_MODEL, help="本地模型名称 (use_api=0 时生效)")

    args = parser.parse_args()

    client = None
    if args.use_api == 1:
        client = init_client(args.api_key, args.base_url)
        print("使用 API 推理")
    else:
        print("使用 本地模型 推理")

    # 根据 args.prompt_mode 设置 prompt_key
    if args.prompt_mode == "Direct_Generation":
        prompt_key = "old_prompt"
    elif args.prompt_mode == "Modified_One_Step":
        prompt_key = "prompt"
    else:
        # 理论上 choices 会处理，但作为保险
        raise ValueError(f"未知的 prompt_mode: {args.prompt_mode}")
    
    print(f"选择模式: {args.prompt_mode} (将使用键: '{prompt_key}')")


    process_dataset(
        dataset_id=args.dataset_id,
        data_file_path=args.data_file,
        save_path=args.save_path,
        client=client,
        api_model=args.api_model,
        num_samples=args.num_samples,
        max_output_tokens=args.max_output_tokens,
        use_api=bool(args.use_api),
        local_model=args.local_model,
        prompt_key=prompt_key, # 传递选择的 key
        prompt_mode_str=args.prompt_mode # 传递模式字符串用于打印
    )