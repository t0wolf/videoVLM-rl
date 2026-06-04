# 视频 VLM GRPO 偏好对齐项目

基于 Qwen3-VL-8B-Instruct，通过轻量 SFT + GRPO 两阶段后训练，探索 GRPO 强化学习在视频多模态大模型上的应用。

**运行环境：双卡 RTX 4090 24GB**

## 项目结构

```
video_dpo_project/
├── configs/
│   ├── sft_config.yaml              # SFT 训练配置
│   ├── grpo_config.yaml             # GRPO 训练配置
│   ├── ds_config_zero2.json         # DeepSpeed ZeRO-2（SFT 用）
│   └── ds_config_zero3.json         # DeepSpeed ZeRO-3（GRPO 用）
├── scripts/
│   ├── 01_prepare_sft_data.py       # 准备 SFT 数据
│   ├── 02_train_sft.py              # SFT 训练（双卡）
│   ├── 03_build_grpo_data.py        # 构建 GRPO 数据（比 DPO 简单）
│   ├── 04_train_grpo.py             # GRPO 训练（双卡）
│   └── 05_evaluate.py               # 评测
├── data/                            # 数据目录
├── output/                          # 模型输出
├── run.sh                           # 一键运行脚本
├── requirements.txt                 # 依赖
└── README.md
```

## 快速开始（AutoDL 双卡 4090）

### 1. 租卡
- GPU: 2 × RTX 4090 24GB
- 镜像: PyTorch 2.4 + CUDA 12.1 + Python 3.11
- 预留时间: 10 小时
- 预算: ¥60-80

### 2. 安装环境
```bash
# 上传项目
scp -r video_dpo_project root@autodl-server:/root/

# 进入项目
cd /root/video_dpo_project

# 安装依赖
conda create -n video_dpo python=3.11 -y
conda activate video_dpo
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### 3. 一键运行
```bash
bash run.sh
```

### 4. 分步运行（推荐，方便调试）
```bash
# Step 1: 准备数据
python scripts/01_prepare_sft_data.py --max-samples 5000

# Step 2: SFT 训练（双卡）
deepspeed --num_gpus=2 scripts/02_train_sft.py

# Step 3: 构建 GRPO 数据（比 DPO 简单，不需要偏好对）
python scripts/03_build_grpo_data.py --max-samples 1000

# Step 4: GRPO 训练（双卡）
deepspeed --num_gpus=2 scripts/04_train_grpo.py

# Step 5: 评测
python scripts/05_evaluate.py --qualitative
```

## 为什么用 GRPO 替代 DPO？

| | DPO | GRPO |
|---|---|---|
| 需要 Reference Model | 是（额外 8B 显存） | 否 |
| 数据格式 | prompt + chosen + rejected | prompt + ground_truth |
| 构建成本 | 高（需要 LLM 打分筛选） | 低（直接用已有 QA 对） |
| 训练范式 | 偏好对齐 | 强化学习（DeepSeek-R1 同款） |
| 奖励函数 | 固定偏好对 | 可自定义多维奖励 |

**GRPO 的优势：**
1. **省显存**：不需要加载 reference model，双卡 4090 跑得更轻松
2. **数据简单**：不需要构建偏好对，直接用 SFT 数据
3. **奖励灵活**：可以自定义准确性、格式、长度、多样性等多维奖励
4. **前沿范式**：GRPO 是 DeepSeek-R1 的核心训练方法，面试更有说服力

## 奖励函数设计

```python
# 1. 准确性奖励（权重 0.5）
# 计算回答与 ground_truth 的词重叠 F1 分数

# 2. 格式奖励（权重 0.2）
# 鼓励回答长度适中、包含标点、结构清晰

# 3. 长度惩罚（权重 0.15）
# 避免过长或过短的回答，与 ground_truth 长度接近

# 4. 多样性奖励（权重 0.15）
# 鼓励词汇多样化，避免重复
```

## 显存分配（双卡 4090）

```
SFT 阶段：
  GPU 0: 模型前半 + LoRA + 激活值 ≈ 14GB
  GPU 1: 模型后半 + LoRA + 激活值 ≈ 14GB
  每卡 24GB → ✅ 够用

GRPO 阶段（ZeRO-3 + 生成多条回答）：
  GPU 0: 模型前半 + 生成缓存 ≈ 18GB
  GPU 1: 模型后半 + 生成缓存 ≈ 18GB
  每卡 24GB → ✅ 够用（比 DPO 更轻松，无 reference model）
```

## 超参说明

| 参数 | SFT | GRPO | 说明 |
|------|-----|------|------|
| batch_size | 2/卡 | 1/卡 | GRPO 需要生成多条回答 |
| grad_accum | 2 | 4 | 补偿 batch_size 减小 |
| 有效 batch | 8 | 8 | 2卡×1×4 |
| learning_rate | 2e-5 | 5e-6 | GRPO 更保守 |
| epochs | 1 | 1 | 防过拟合 |
| LoRA rank | 32 | 64 | GRPO 需要更强拟合 |
| num_generations | - | 4 | 每个 prompt 生成 4 个回答 |

## 常见问题

### Q: OOM 了怎么办？
```bash
# 方法 1：减小 num_generations
# 修改 configs/grpo_config.yaml 中的 num_generations: 2

# 方法 2：减小 batch_size
# 修改 configs/grpo_config.yaml 中的 per_device_train_batch_size: 1

# 方法 3：减小序列长度
# 修改 configs/grpo_config.yaml 中的 max_completion_length: 128
```

### Q: DeepSpeed 报错？
```bash
# 确保安装了正确版本
pip install deepspeed>=0.14.0

# 如果 flash-attn 报错
pip install flash-attn --no-build-isolation --no-cache-dir
```

### Q: 如何只用单卡运行？
```bash
# 去掉 deepspeed，直接用 python 运行
python scripts/02_train_sft.py
python scripts/04_train_grpo.py
```

## 输出

- `./output/sft/` - SFT 模型
- `./output/grpo/` - GRPO 模型
- `./output/qualitative_analysis.json` - 定性分析结果
- `./data/grpo_train.jsonl` - GRPO 训练数据

## 面试要点

**如何介绍这个项目？**

> "我们用 SFT 做基础对齐，然后用 GRPO 做强化学习优化。GRPO 通过对同一问题采样多条回答、用多维奖励函数（准确性、格式、长度、多样性）做组内相对排序，避免了 PPO 需要额外 Critic 模型的显存开销，在 2×4090 上也能跑。这是 DeepSeek-R1 的核心训练范式。"

**关键点：**
1. GRPO 是 DeepSeek-R1 的核心技术，比 DPO 更前沿
2. 不需要 reference model，省显存
3. 奖励函数设计灵活，可以优化多个维度
4. 组内相对排序，避免了 PPO 的复杂性
