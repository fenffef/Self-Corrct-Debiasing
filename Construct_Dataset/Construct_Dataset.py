import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from tqdm import tqdm

# --- 1. 配置 ---

os.environ["OPENAI_API_KEY"] = "sk-CvR4TxCDulCVq3RZAHtdUTpZlrZpFOCEesdvcWLX2WqNvuGJ" 

SOTA_MODEL_NAME = "gpt-4o-ca"
LOCAL_MODEL_NAME_OR_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" 
JUDGE_MODEL = "gpt-4o-mini-ca" 

# 仅在 mode=sota 时使用
MIN_MITIGATION_SCORE = 4  
DEFAULT_OUTPUT_FILE = "sft_dataset.jsonl" # 通用文件名

# --- 2. 数据源 (与之前相同) ---

def load_dataset_from_json(filepath: str) -> list:
    """从单个 JSON 文件加载数据 (该文件是一个 JSON 列表)"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    print(f"Loading data from {filepath} as a single JSON array...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            data = json.loads(content)
        
        if not isinstance(data, list):
             print(f"Warning: Data in {filepath} is not a list. Returning an empty list.")
             return []
             
        return data
    except Exception as e:
        print(f"An error occurred while loading {filepath}: {e}")
        return []

# --- 3. 提示词模板 (英文, 已约束格式) ---

COT_PROMPT_TEMPLATE = """
You will be given a biased text. Your task is to:
1.  First, provide a step-by-step Chain of Thought on how you will mitigate this bias.
2.  Second, write a continuation of the text that actively mitigates or challenges the bias.

You must follow this exact output format:

[Chain of Thought]
1. (Your reasoning step 1...)
2. (Your reasoning step 2...)
3. (etc...)

[Continuation]
(The original text, e.g., "I hate 20-year-old grandpas...")
(Your mitigating continuation...)

---
Beginning Text:
"{prompt}"
"""

JUDGE_PROMPT_TEMPLATE = """
[Task]:
{prompt}

[Model Response]:
{response_text}

[Evaluation]:
The model was asked to include the [Task] text and then write a continuation that mitigates the bias.
Evaluate the model's CoT and continuation (not the original task text).

Return your assessment as a JSON object:
1. `mitigation_score`: A score from 1 (poor, amplified bias) to 5 (excellent, actively mitigated bias).
2. `explanation`: Brief reason for your score.
3. `explicit_self_correction`: true or false, if the model explicitly identified the bias.

Return only the raw JSON:
{{
  "mitigation_score": <1-5>,
  "explanation": "...",
  "explicit_self_correction": <true|false>
}}
"""

# --- 4. API 客户端设置 (同步) ---
DEFAULT_BASE_URL = "https://api.chatanywhere.tech/v1"

sota_judge_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=DEFAULT_BASE_URL,
    timeout=120.0
)

# --- 5. 本地模型加载和生成 ---

def load_local_model(model_id: str):
    print(f"--- Preparing to load model: {model_id} ---")

    # ✅ 显示 HuggingFace 下载进度条
    os.environ["HF_HUB_ENABLE_PROGRESS_BARS"] = "1"

    if not torch.cuda.is_available():
        print("WARNING: CUDA not detected. Loading model on CPU, this will be very slow.")

    try:
        print(">>> Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        print(">>> Tokenizer loaded.")

        print(">>> Loading model weights... (Progress bar below)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
        print(">>> Model loaded successfully!")

        model.eval()
        return model, tokenizer

    except Exception as e:
        print(f"[X] Failed to load local model: {e}")
        raise e

def generate_local_sync(prompt: str, model, tokenizer) -> str | None:
    try:
        messages = [{"role": "user", "content": prompt}]
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                # pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response_text
        
    except Exception as e:
        print(f"  [!] Local generation error: {e}")
        return None

# --- 6. 同步 API 功能函数 ---

def generate_response_api_sync(client: OpenAI, model: str, prompt: str) -> str | None:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  [!] SOTA API error (Model: {model}): {e}")
        return None

def judge_response_sync(original_prompt: str, response_text: str) -> dict | None:
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        prompt=original_prompt, 
        response_text=response_text
    )
    
    try:
        judge_result_text = generate_response_api_sync(
            sota_judge_client, JUDGE_MODEL, judge_prompt
        )
        
        if not judge_result_text: return None
        judge_json = json.loads(judge_result_text)
        
        if 'mitigation_score' in judge_json and 'explanation' in judge_json:
            return judge_json
        else:
            print(f"  [!] Judge JSON format error (expected 'mitigation_score'): {judge_result_text}")
            return None
            
    except Exception as e:
        print(f"  [!] Judge evaluation error: {e}")
        return None

# --- 7. 核心处理 (同步) ---

def process_trigger_sync(
    trigger_index: int,
    trigger_data: dict, 
    output_file_handle, 
    mode: str, 
    prompt_key: str,
    local_model, 
    local_tokenizer,
    sota_client,
    sota_model_name
):
    
    if prompt_key not in trigger_data:
        print(f"  [!] ID {trigger_index} skipped: --prompt_key '{prompt_key}' not found in JSON line.")
        return

    original_prompt = trigger_data[prompt_key] 
    cot_prompt = COT_PROMPT_TEMPLATE.format(prompt=original_prompt)
    
    candidate_response = None
    
    if mode == 'sota':
        candidate_response = generate_response_api_sync(
            sota_client, sota_model_name, cot_prompt
        )
    else:
        candidate_response = generate_local_sync(
            cot_prompt, local_model, local_tokenizer
        )
    
    if not candidate_response:
        print(f"  [!] ID {trigger_index} generation failed, skipping.")
        return

    if mode == 'sota':
        judge_assessment = judge_response_sync(original_prompt, candidate_response)

        if not judge_assessment:
            print(f"  [!] ID {trigger_index} QC evaluation failed, skipping.")
            return

        mitigation_score = judge_assessment.get('mitigation_score', 0)
        explanation = judge_assessment.get('explanation', 'N/A')
        correction = judge_assessment.get('explicit_self_correction', False)

        if mitigation_score >= MIN_MITIGATION_SCORE:
            sft_entry = {
                "prompt": original_prompt, 
                "response": candidate_response, 
                "meta": { 
                    "original_id": trigger_index,
                    "axis": trigger_data.get("axis", "N/A"),
                    "bucket": trigger_data.get("bucket", "N/A"),
                    "judge_score": mitigation_score,
                    "judge_explanation": explanation, 
                    "explicit_self_correction": correction 
                }
            }
            try:
                output_file_handle.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"  [!] ID {trigger_index} file write error: {e}")
        else:
            pass
    
    else: # mode == 'local'
        sft_entry = {
            "prompt": original_prompt, 
            "response": candidate_response, 
            "meta": { 
                "original_id": trigger_index,
                "axis": trigger_data.get("axis", "N/A"),
                "bucket": trigger_data.get("bucket", "N/A"),
                "judge_score": "N/A (local)",
                "judge_explanation": "N/A (local)", 
                "explicit_self_correction": "N/A (local)" 
            }
        }
        try:
            output_file_handle.write(json.dumps(sft_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"  [!] ID {trigger_index} file write error: {e}")


def main(args):
    
    local_model, local_tokenizer = None, None
    sota_client, sota_model_name = None, None
    
    if args.mode == 'sota':
        sota_client = sota_judge_client
        sota_model_name = SOTA_MODEL_NAME
        print(f"[Generator Mode]: sota ({SOTA_MODEL_NAME})")
        print(f"[Using API Base URL]: {DEFAULT_BASE_URL}")
        print(f"[Judge Model]: {JUDGE_MODEL} (via {DEFAULT_BASE_URL})")
        print(f"[Filter Threshold]: mitigation_score >= {MIN_MITIGATION_SCORE}")
    
    elif args.mode == 'local':
        print(f"[Generator Mode]: local ({LOCAL_MODEL_NAME_OR_PATH})")
        print("[!] LOCAL MODE: Skipping QC/Judge. All generations will be written directly.")
        print("--- Loading local model (this may take a few minutes)... ---")
        local_model, local_tokenizer = load_local_model(LOCAL_MODEL_NAME_OR_PATH)
        print("--- Local model loaded. ---")
    
    print(f"[Output File]: {args.output_file}")
    print(f"[Input File]: {args.data_file}")
    print(f"[Prompt Key]: {args.prompt_key}")

    print("\n--- Loading dataset ---")
    dataset = load_dataset_from_json(args.data_file)
    print(f"--- Loaded {len(dataset)} triggers ---")

    print("\n--- Start processing ---")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for i, trigger_data in enumerate(tqdm(dataset, desc="Processing Triggers", ncols=90)):
            process_trigger_sync(
                trigger_index=i,
                trigger_data=trigger_data, 
                output_file_handle=f,
                mode=args.mode,
                prompt_key=args.prompt_key,
                local_model=local_model,
                local_tokenizer=local_tokenizer,
                sota_client=sota_client,
                sota_model_name=sota_model_name
            )

    print(f"\n--- All tasks complete ---")
    print(f"High-quality SFT dataset saved to {args.output_file}")

# --- 8. 脚本入口 (同步) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a high-quality, bias-mitigating SFT dataset (Sync Version).")
    
    parser.add_argument(
        "--mode", 
        choices=["sota", "local"], 
        required=True,
        help="Choose 'sota' (API) or 'local' (direct model load) to generate candidates."
    )
    parser.add_argument(
        "--data_file", 
        type=str, 
        required=True,
        help="Path to the input .json file (which contains a JSON array)."
    )
    
    parser.add_argument(
        "--prompt_key", 
        type=str, 
        default="prompt",
        help="The key in the JSON file to use as the input prompt (e.g., 'prompt', 'old_prompt')."
    )
    parser.add_argument(
        "--output_file", 
        type=str, 
        default=DEFAULT_OUTPUT_FILE,
        help="Path to save the output .jsonl file."
    )

    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"\nScript failed: {e}")
        print("Please check: 1. API Key? 2. 'transformers' libraries? 3. Enough VRAM/RAM? 4. File paths?")