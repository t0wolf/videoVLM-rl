# 视频 VLM DPO 偏好对齐项目

基于 Qwen3-VL-8B-Instruct，通过轻量 SFT + DPO 两阶段后训练，探索 DPO 偏好对齐在视频多模态大模型上的应用。

**运行环境：双卡 RTX 4090 24GB**

## 项目结构

```
video_dpo_project/
├── configs/
│   ├── sft_config.yaml              # SFT 训练配置
│   ├── dpo_config.yaml              # DPO 训练配置
│   ├── ds_config_zero2.json         # DeepSpeed ZeRO-2（SFT 用）
│   └── ds_config_zero3.json         # DeepSpeed ZeRO-3（DPO 用）
├── scripts/
│   ├── 01_prepare_sft_data.py       # 准备 SFT 数据
│   ├── 02_train_sft.py              # SFT 训练（双卡）
│   ├── 03_build_dpo_data.py         # 构建 DPO 偏好数据
│   ├── 04_train_dpo.py              # DPO 训练（双卡）
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

# Step 3: 构建 DPO 数据
export MIMO_API_KEY=your-key
export MIMO_API_BASE_URL=https://api.your-provider.com/v1
python scripts/03_build_dpo_data.py --max-samples 1000

# Step 4: DPO 训练（双卡）
deepspeed --num_gpus=2 scripts/04_train_dpo.py

# Step 5: 评测
python scripts/05_evaluate.py --qualitative
```

## 显存分配（双卡 4090）

```
SFT 阶段：
  GPU 0: 模型前半 + LoRA + 激活值 ≈ 14GB
  GPU 1: 模型后半 + LoRA + 激活值 ≈ 14GB
  每卡 24GB → ✅ 够用

DPO 阶段（ZeRO-3 激进分片）：
  GPU 0: Policy 前半 + Ref 前半 ≈ 17GB
  GPU 1: Policy 后半 + Ref 后半 ≈ 17GB
  每卡 24GB → ✅ 够用
```

## 超参说明

| 参数 | SFT | DPO | 说明 |
|------|-----|-----|------|
| batch_size | 2/卡 | 2/卡 | 每卡 batch |
| grad_accum | 2 | 2 | 梯度累积 |
| 有效 batch | 8 | 8 | 2卡×2×2 |
| learning_rate | 2e-5 | 5e-6 | DPO 更保守 |
| epochs | 1 | 1 | SFT 适配，DPO 防过拟合 |
| LoRA rank | 32 | 64 | DPO 需要更强拟合 |

## 常见问题

### Q: OOM 了怎么办？
```bash
# 方法 1：减小 batch_size
# 修改 configs/sft_config.yaml 中的 per_device_train_batch_size: 1

# 方法 2：减小序列长度
# 修改 configs/sft_config.yaml 中的 max_seq_length: 1024

# 方法 3：减小帧数
# 修改 scripts/03_build_dpo_data.py 中的 num_frames: 4
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
python scripts/04_train_dpo.py
```

## 输出

- `./output/sft/` - SFT 模型
- `./output/dpo/` - DPO 模型
- `./output/qualitative_analysis.json` - 定性分析结果
- `./data/dpo_preference_data.jsonl` - DPO 偏好数据
