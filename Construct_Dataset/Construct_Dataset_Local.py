import os
import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

COT_PROMPT_TEMPLATE = """You will be given a biased text. Your task is to:
1. First, provide a step-by-step Chain of Thought on how you will mitigate this bias.
2. Second, write a continuation of the text that actively mitigates or challenges the bias.

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

def load_dataset(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data if isinstance(data, list) else []

def load_model(model_id):
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def generate(prompt, model, tokenizer):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def process_item(item, prompt_key, model, tokenizer):
    if prompt_key not in item:
        return None
    
    prompt = COT_PROMPT_TEMPLATE.format(prompt=item[prompt_key])
    response = generate(prompt, model, tokenizer)
    
    return {
        "prompt": item[prompt_key],
        "response": response,
        "meta": {
            "axis": item.get("axis", "N/A"),
            "bucket": item.get("bucket", "N/A")
        }
    }

def main(args):
    import time
    model, tokenizer = load_model(args.model_id)
    dataset = load_dataset(args.data_file)
    
    print(f"Processing {len(dataset)} items...")
    print("Note: Each item takes ~30-120 seconds to generate\n")
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(tqdm(dataset, desc="Generating")):
            start = time.time()
            result = process_item(item, args.prompt_key, model, tokenizer)
            if result:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
            elapsed = time.time() - start
            tqdm.write(f"Item {idx+1}/{len(dataset)} completed in {elapsed:.1f}s")
    
    print(f"\nDone! Saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--model_id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--output_file", default="local_sft_dataset.jsonl")
    
    args = parser.parse_args()
    main(args)