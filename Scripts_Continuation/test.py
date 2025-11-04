import json
from datetime import datetime
from datasets import load_dataset
from openai import OpenAI

# ===================== 配置 =====================
API_KEY = "sk-CvR4TxCDulCVq3RZAHtdUTpZlrZpFOCEesdvcWLX2WqNvuGJ"  # 替换成你的 API Key
BASE_URL = "https://api.chatanywhere.tech/v1"
API_MODEL = "gpt-4o-ca"

NUM_SAMPLES = -1            # 处理样本数量
MAX_OUTPUT_TOKENS = 512     # 每条输出最大 token
SAVE_PATH = "CEB_Continuation_Age.jsonl"
USE_API = True              # True 调用 ChatAnywhere API, False 使用本地模型

# ===================== 初始化 API 客户端 =====================
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# ===================== 加载数据集 =====================
DATASET_ID = "Song-SW/CEB"
DATA_FILE_PATH = "CEB-Continuation-S/age.json"

print(f"正在加载数据集: {DATASET_ID} ...")
if NUM_SAMPLES == -1:
    split_str = "train"  # 加载全部
else:
    split_str = f"train[:{NUM_SAMPLES}]"  # 加载前 NUM_SAMPLES 条

dataset = load_dataset(
    DATASET_ID,
    data_files=DATA_FILE_PATH,
    split=split_str
)
print(f"成功加载 {len(dataset)} 个样本")

# ===================== 清空输出文件 =====================
with open(SAVE_PATH, "w", encoding="utf-8") as f:
    pass

# -----------------------------
# 主循环
# -----------------------------
for i, item in enumerate(dataset):
    description = item["prompt"]

    # 只打印第一个样本
    if i == 0:
        print(f"\n🎯 样本 {i+1}/{len(dataset)}")
        print(f"Prompt preview:\n{description[:120]}...")

    if USE_API:
        # GPT 模型调用
        resp = client.responses.create(
            model=API_MODEL,
            input=description,
            reasoning={"include_outputs": True},
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        generated = resp.output_text
    else:
        # 如果有本地模型，这里可以填本地生成逻辑
        generated = "<think>本地模型推理结果</think> <bias>Yes/No</bias>"

    print(f"\n 输出:\n{generated[:300]}...\n")

    # 保存 JSONL
    record = {
        "id": i,
        "prompt": description,
        "response": generated,
        "model": API_MODEL if USE_API else "local_model",
        "task": "CEB Credit Default Bias Test"
    }

    with open(SAVE_PATH, "a", encoding="utf-8") as fw:
        fw.write(json.dumps(record, ensure_ascii=False) + "\n")

print("\n 全部完成")
print(f" 输出结果保存在: {SAVE_PATH}")