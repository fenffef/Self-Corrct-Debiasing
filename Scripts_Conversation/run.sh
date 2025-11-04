# CEB-Conversation-S
# Age
python CEB_Conversation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Conversation-S/age.json \
    --save_path CEB_Conversation_S_Age.jsonl \
    --api_key sk-CvR4TxCDulCVq3RZAHtdUTpZlrZpFOCEesdvcWLX2WqNvuGJ \
    --num_samples -1 \
    --api_model gpt-4o-ca \
    --max_output_tokens 512 \
    --use_api 1

# Gender
python CEB_Conversation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Conversation-S/gender.json \
    --save_path CEB_Conversation_S_Gender.jsonl \
    --api_key sk-CvR4TxCDulCVq3RZAHtdUTpZlrZpFOCEesdvcWLX2WqNvuGJ \
    --num_samples -1 \
    --api_model gpt-4o-ca \
    --max_output_tokens 512 \
    --use_api 1

# Race
python CEB_Conversation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Conversation-S/race.json \
    --save_path CEB_Conversation_S_Race.jsonl \
    --api_key sk-CvR4TxCDulCVq3RZAHtdUTpZlrZpFOCEesdvcWLX2WqNvuGJ \
    --num_samples -1 \
    --api_model gpt-4o-ca \
    --max_output_tokens 512 \
    --use_api 1

# Religion
python CEB_Conversation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Conversation-S/religion.json \
    --save_path CEB_Conversation_S_Religion.jsonl \
    --api_key sk-CvR4TxCDulCVq3RZAHtdUTpZlrZpFOCEesdvcWLX2WqNvuGJ \
    --num_samples -1 \
    --api_model gpt-4o-ca \
    --max_output_tokens 512 \
    --use_api 1


# CEB-Conversation-T
# Age
python CEB_Conversation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Conversation-T/age.json \
    --save_path CEB_Conversation_T_Age.jsonl \
    --api_key sk-CvR4TxCDulCVq3RZAHtdUTpZlrZpFOCEesdvcWLX2WqNvuGJ \
    --num_samples -1 \
    --api_model gpt-4o-ca \
    --max_output_tokens 512 \
    --use_api 1

# Gender
python CEB_Conversation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Conversation-T/gender.json \
    --save_path CEB_Conversation_T_Gender.jsonl \
    --api_key sk-CvR4TxCDulCVq3RZAHtdUTpZlrZpFOCEesdvcWLX2WqNvuGJ \
    --num_samples -1 \
    --api_model gpt-4o-ca \
    --max_output_tokens 512 \
    --use_api 1

# Race
python CEB_Conversation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Conversation-T/race.json \
    --save_path CEB_Conversation_T_Race.jsonl \
    --api_key sk-CvR4TxCDulCVq3RZAHtdUTpZlrZpFOCEesdvcWLX2WqNvuGJ \
    --num_samples -1 \
    --api_model gpt-4o-ca \
    --max_output_tokens 512 \
    --use_api 1

# Religion
python CEB_Conversation_Inference.py \
    --dataset_id Song-SW/CEB \
    --data_file CEB-Conversation-T/religion.json \
    --save_path CEB_Conversation_T_Religion.jsonl \
    --api_key sk-CvR4TxCDulCVq3RZAHtdUTpZlrZpFOCEesdvcWLX2WqNvuGJ \
    --num_samples -1 \
    --api_model gpt-4o-ca \
    --max_output_tokens 512 \
    --use_api 1