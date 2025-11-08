# python Construct_Dataset.py \
#     --mode sota \
#     --data_file "/mnt/raid/data/xuanfeng/Self-Corrct-Debiasing/CEB/CEB-Continuation-S/age.json" \
#     --prompt_key "old_prompt" \
#     --output_file "fair_sft_age.jsonl"

python Construct_Dataset.py \
    --mode local \
    --data_file "/mnt/raid/data/xuanfeng/Self-Corrct-Debiasing/CEB/CEB-Continuation-S/age.json" \
    --prompt_key "old_prompt" \
    --output_file "fair_sft_age_local.jsonl"

python Construct_Dataset_Local.py \
    --data_file "/mnt/raid/data/xuanfeng/Self-Corrct-Debiasing/CEB/CEB-Continuation-S/age.json" \
    --prompt_key "prompt" \
    --model_id "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
    --output_file "local_output_age.jsonl"