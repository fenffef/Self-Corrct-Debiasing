import json
from datetime import datetime
from datasets import load_dataset
from openai import OpenAI
import argparse
from tqdm import tqdm  # 进度条

# ===================== 默认配置 =====================
DEFAULT_API_MODEL = "gpt-4o-ca"
DEFAULT_BASE_URL = "https://api.chatanywhere.tech/v1"
DEFAULT_MAX_OUTPUT_TOKENS = 512

# ===================== API 初始化 =====================
def init_client(api_key, base_url):
    return OpenAI(api_key=api_key, base_url=base_url)

# ===================== 数据集加载 =====================
def load_ceb_dataset(dataset_id, data_file_path, num_samples=-1):
    split_str = "train" if num_samples == -1 else f"train[:{num_samples}]"
    dataset = load_dataset(dataset_id, data_files=data_file_path, split=split_str)
    return dataset

# ===================== 主处理函数 =====================
def process_dataset(dataset_id, data_file_path, save_path, client, api_model, num_samples, max_output_tokens, use_api=True):
    dataset = load_ceb_dataset(dataset_id, data_file_path, num_samples)
    total_samples = len(dataset)
    print(f"成功加载 {total_samples} 个样本")

    # 清空输出文件
    with open(save_path, "w", encoding="utf-8") as f:
        pass

    for i, item in enumerate(tqdm(dataset, desc="Processing", ncols=100)):
        description = item["prompt"]

        # 只打印第一条样本的 prompt
        if i == 0:
            print(f"\n🎯 样本 {i+1}/{total_samples}")
            print(f"Prompt preview:\n{description[:120]}...")

        if use_api:
            resp = client.responses.create(
                model=api_model,
                input=description,
                reasoning={"include_outputs": True},
                max_output_tokens=max_output_tokens,
            )
            generated = resp.output_text
        else:
            generated = "<think>本地模型推理结果</think> <bias>Yes/No</bias>"

        record = {
            "id": i,
            "prompt": description,
            "response": generated,
            "model": api_model if use_api else "local_model",
            "task": data_file_path  # 用文件名标识任务
        }

        with open(save_path, "a", encoding="utf-8") as fw:
            fw.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n✅ 全部完成")
    print(f"输出结果保存在: {save_path}")

# ===================== 命令行参数 =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CEB 数据集续写处理脚本")
    parser.add_argument("--dataset_id", type=str, required=True, help="HuggingFace 数据集 ID")
    parser.add_argument("--data_file", type=str, required=True, help="数据文件路径")
    parser.add_argument("--save_path", type=str, required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI/ChatAnywhere API Key")
    parser.add_argument("--api_model", type=str, default=DEFAULT_API_MODEL, help="调用模型名称")
    parser.add_argument("--num_samples", type=int, default=-1, help="处理样本数量，-1 表示全部")
    parser.add_argument("--max_output_tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="每条输出最大 token")
    parser.add_argument("--use_api", type=int, default=1, help="是否使用 API，1 为 True，0 为 False")

    args = parser.parse_args()

    client = init_client(api_key=args.api_key, base_url=DEFAULT_BASE_URL)

    process_dataset(
        dataset_id=args.dataset_id,
        data_file_path=args.data_file,
        save_path=args.save_path,
        client=client,
        api_model=args.api_model,
        num_samples=args.num_samples,
        max_output_tokens=args.max_output_tokens,
        use_api=bool(args.use_api)
    )