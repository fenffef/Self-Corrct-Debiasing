
CUDA_VISIBLE_DEVICES=5

# # CEB-Continuation-S
# Age
python CEB_Continuation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Continuation-S/age.json \
    --save_path Llama/Meta-Llama-3.1-8B-Instruct/Modified_One_Step/CEB_Continuation_S_Age_local.jsonl \
    --use_api 0 \
    --local_model Llama/Meta-Llama-3.1-8B-Instruct \
    --num_samples -1 \
    --max_output_tokens 512 \
    --prompt_mode Direct_Generation

# Gender
python CEB_Continuation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Continuation-S/gender.json \
    --save_path Llama/Meta-Llama-3.1-8B-Instruct/Modified_One_Step/CEB_Continuation_S_Gender_local.jsonl \
    --use_api 0 \
    --local_model Llama/Meta-Llama-3.1-8B-Instruct \
    --num_samples -1 \
    --max_output_tokens 512 \
    --prompt_mode Direct_Generation

# Race
python CEB_Continuation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Continuation-S/race.json \
    --save_path Llama/Meta-Llama-3.1-8B-Instruct/Modified_One_Step/CEB_Continuation_S_Race_local.jsonl \
    --use_api 0 \
    --local_model Llama/Meta-Llama-3.1-8B-Instruct \
    --num_samples -1 \
    --max_output_tokens 512 \
    --prompt_mode Direct_Generation

# Religion
python CEB_Continuation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Continuation-S/religion.json \
    --save_path Llama/Meta-Llama-3.1-8B-Instruct/Modified_One_Step/CEB_Continuation_S_Religion_local.jsonl \
    --use_api 0 \
    --local_model Llama/Meta-Llama-3.1-8B-Instruct \
    --num_samples -1 \
    --max_output_tokens 512 \
    --prompt_mode Direct_Generation


# CEB-Continuation-T
# Age
python CEB_Continuation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Continuation-T/age.json \
    --save_path Llama/Meta-Llama-3.1-8B-Instruct/Modified_One_Step/CEB_Continuation_T_Age_local.jsonl \
    --use_api 0 \
    --local_model Llama/Meta-Llama-3.1-8B-Instruct \
    --num_samples -1 \
    --max_output_tokens 512 \
    --prompt_mode Direct_Generation

# Gender
python CEB_Continuation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Continuation-T/gender.json \
    --save_path Llama/Meta-Llama-3.1-8B-Instruct/Modified_One_Step/CEB_Continuation_T_Gender_local.jsonl \
    --use_api 0 \
    --local_model Llama/Meta-Llama-3.1-8B-Instruct \
    --num_samples -1 \
    --max_output_tokens 512 \
    --prompt_mode Direct_Generation

# Race
python CEB_Continuation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Continuation-T/race.json \
    --save_path Llama/Meta-Llama-3.1-8B-Instruct/Modified_One_Step/CEB_Continuation_T_Race_local.jsonl \
    --use_api 0 \
    --local_model Llama/Meta-Llama-3.1-8B-Instruct \
    --num_samples -1 \
    --max_output_tokens 512 \
    --prompt_mode Direct_Generation

# Religion
python CEB_Continuation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Continuation-T/religion.json \
    --save_path Llama/Meta-Llama-3.1-8B-Instruct/Modified_One_Step/CEB_Continuation_T_Religion_local.jsonl \
    --use_api 0 \
    --local_model Llama/Meta-Llama-3.1-8B-Instruct \
    --num_samples -1 \
    --max_output_tokens 512 \
    --prompt_mode Direct_Generation