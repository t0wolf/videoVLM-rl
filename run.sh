#!/bin/bash
# ============================================================
# 视频 VLM DPO 偏好对齐项目 - 双卡 RTX 4090 运行脚本
# ============================================================

set -e

echo "============================================================"
echo "视频 VLM DPO 偏好对齐项目（双卡 4090）"
echo "============================================================"

# 检查 GPU
echo ""
echo "[0/5] 检查环境..."
python -c "
import torch
n_gpus = torch.cuda.device_count()
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU 数量: {n_gpus}')
for i in range(n_gpus):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1024**3:.1f} GB)')
"

# 设置 DeepSpeed 环境变量
export CUDA_VISIBLE_DEVICES=0,1
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Step 1: 准备 SFT 数据
echo ""
echo "[1/5] 准备 SFT 数据..."
python scripts/01_prepare_sft_data.py --max-samples 5000 --output ./data/sft_train.jsonl

# Step 2: SFT 训练（双卡 DeepSpeed）
echo ""
echo "[2/5] SFT 训练（双卡 4090）..."
deepspeed --num_gpus=2 scripts/02_train_sft.py \
    --config configs/sft_config.yaml

# Step 3: 构建 DPO 偏好数据（单卡即可，不需要分布式）
echo ""
echo "[3/5] 构建 DPO 偏好数据..."
python scripts/03_build_dpo_data.py \
    --sft-model ./output/sft \
    --sft-data ./data/sft_train.jsonl \
    --output ./data/dpo_preference_data.jsonl \
    --max-samples 1000

# Step 4: DPO 训练（双卡 DeepSpeed）
echo ""
echo "[4/5] DPO 训练（双卡 4090）..."
deepspeed --num_gpus=2 scripts/04_train_dpo.py \
    --config configs/dpo_config.yaml

# Step 5: 评测
echo ""
echo "[5/5] 评测..."
python scripts/05_evaluate.py \
    --sft-model ./output/sft \
    --dpo-model ./output/dpo \
    --qualitative

echo ""
echo "============================================================"
echo "项目完成！（双卡 4090）"
echo "============================================================"
echo "输出目录："
echo "  SFT 模型: ./output/sft"
echo "  DPO 模型: ./output/dpo"
echo "  定性分析: ./output/qualitative_analysis.json"
