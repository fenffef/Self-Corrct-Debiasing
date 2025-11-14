#!/bin/bash

# CEB Evaluation Runner Script
# Usage:
#   ./run_ceb_evaluation.sh              # Run all tasks
#   ./run_ceb_evaluation.sh classification  # Run only classification tasks

cd "$(dirname "$0")"

echo "=========================================="
echo "CEB Evaluation Script"
echo "=========================================="

# Check if model exists
if [ ! -d "./Sherlock_Debias_Qwen3-4B_QLora/final_merged_checkpoint" ]; then
    echo "Warning: Merged model not found at ./Sherlock_Debias_Qwen3-4B_QLora/final_merged_checkpoint"
    echo "Will fall back to base model: Qwen3-4B-Instruct-2507"
fi

# Run the evaluation
python test_all_ceb_tasks.py

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Results saved to: ./ceb_evaluation_results/"
echo "=========================================="
