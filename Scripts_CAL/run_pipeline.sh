#!/bin/bash

# ====================================================================
# SFT + Inference Pipeline Script
# 模型: Qwen/Qwen3-4B
# ====================================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Starting SFT + Inference Pipeline"
echo "Model: Qwen/Qwen3-4B"
echo "=========================================="

# ====================================================================
# Step 1: Run SFT Training
# ====================================================================
echo ""
echo "[Step 1/2] Starting SFT training..."
echo "Running sft.py..."
echo ""

python sft.py

if [ $? -eq 0 ]; then
    echo ""
    echo "[Step 1/2] SFT training completed successfully!"
    echo ""
else
    echo ""
    echo "[ERROR] SFT training failed! Exiting..."
    exit 1
fi

# ====================================================================
# Step 2: Run Inference
# ====================================================================
echo ""
echo "[Step 2/2] Starting inference..."
echo "Running inference.py..."
echo ""

python inference.py

if [ $? -eq 0 ]; then
    echo ""
    echo "[Step 2/2] Inference completed successfully!"
    echo ""
else
    echo ""
    echo "[ERROR] Inference failed! Exiting..."
    exit 1
fi

# ====================================================================
# Pipeline Complete
# ====================================================================
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Trained model: ./V0_Qwen3-4B-Instruct-2507/final_merged_checkpoint"
echo "  - Preference dataset: ./SFT/train.jsonl"
echo ""
