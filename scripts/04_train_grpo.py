"""
Step 4: GRPO 训练（双卡 4090 优化版）
比 DPO 更轻量，不需要 reference model
使用 trl 的 GRPOTrainer
启动：deepspeed --num_gpus=2 scripts/04_train_grpo.py
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
from pathlib import Path


def load_config(config_path: str = "configs/grpo_config.yaml"):
    """加载 GRPO 配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_main_process():
    """判断是否是主进程"""
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def load_grpo_data(data_file: str, max_samples: int = 1000):
    """加载 GRPO 训练数据"""
    if not os.path.exists(data_file):
        if is_main_process():
            print(f"GRPO 数据文件不存在: {data_file}")
            print("请先运行 03_build_grpo_data.py")
        sys.exit(1)
    
    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    if len(samples) > max_samples:
        samples = samples[:max_samples]
    
    if is_main_process():
        print(f"加载了 {len(samples)} 条 GRPO 数据")
    
    from datasets import Dataset
    return Dataset.from_list(samples)


# ============================================================
# 奖励函数（GRPO 的核心）
# ============================================================

def accuracy_reward(prompts, completions, ground_truth, **kwargs):
    """
    准确性奖励：计算回答与 ground_truth 的相似度
    使用简单的词重叠率（ROUGE-L 的简化版）
    """
    scores = []
    for completion, gt in zip(completions, ground_truth):
        # 简单的 token 重叠计算
        completion_tokens = set(completion.lower().split())
        gt_tokens = set(gt.lower().split())
        
        if not gt_tokens:
            scores.append(0.0)
            continue
        
        # 计算 F1
        overlap = completion_tokens & gt_tokens
        precision = len(overlap) / len(completion_tokens) if completion_tokens else 0
        recall = len(overlap) / len(gt_tokens)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        scores.append(f1)
    
    return scores


def format_reward(prompts, completions, ground_truth, **kwargs):
    """
    格式奖励：鼓励回答结构清晰
    - 长度适中（不要太短也不要太长）
    - 包含标点符号
    """
    scores = []
    for completion in completions:
        score = 0.0
        
        # 长度奖励：100-300 字符得满分
        length = len(completion)
        if 100 <= length <= 300:
            score += 0.5
        elif 50 <= length < 100 or 300 < length <= 500:
            score += 0.3
        else:
            score += 0.1
        
        # 标点符号奖励
        punctuation_count = sum(1 for c in completion if c in '。！？，、；：')
        if punctuation_count >= 2:
            score += 0.3
        elif punctuation_count >= 1:
            score += 0.2
        
        # 不重复奖励
        sentences = completion.split('。')
        if len(sentences) > 1:
            unique_sentences = set(sentences)
            diversity = len(unique_sentences) / len(sentences)
            score += diversity * 0.2
        
        scores.append(min(score, 1.0))
    
    return scores


def length_penalty_reward(prompts, completions, ground_truth, **kwargs):
    """
    长度惩罚：避免过长或过短的回答
    """
    scores = []
    gt_lengths = [len(gt) for gt in ground_truth]
    
    for completion, gt_len in zip(completions, gt_lengths):
        completion_len = len(completion)
        
        # 计算与 ground_truth 长度的接近程度
        length_ratio = completion_len / gt_len if gt_len > 0 else 1.0
        
        # 最佳比例在 0.8-1.2 之间
        if 0.8 <= length_ratio <= 1.2:
            score = 1.0
        elif 0.5 <= length_ratio <= 1.5:
            score = 0.7
        elif 0.3 <= length_ratio <= 2.0:
            score = 0.4
        else:
            score = 0.1
        
        scores.append(score)
    
    return scores


def diversity_reward(prompts, completions, ground_truth, **kwargs):
    """
    多样性奖励：鼓励回答包含多样化的词汇
    """
    scores = []
    for completion in completions:
        tokens = completion.lower().split()
        if not tokens:
            scores.append(0.0)
            continue
        
        # 词汇多样性 = 唯一词数 / 总词数
        unique_ratio = len(set(tokens)) / len(tokens)
        scores.append(min(unique_ratio, 1.0))
    
    return scores


def setup_model(config: dict):
    """加载模型（不需要 reference model）"""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    
    model_name = config["model"]["name"]
    
    if is_main_process():
        print(f"加载模型: {model_name}")
    
    # 加载模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=config["model"]["device_map"],
    )
    
    # 加载 processor
    processor = AutoProcessor.from_pretrained(model_name)
    
    return model, processor


def train(config: dict):
    """执行 GRPO 训练"""
    from trl import GRPOTrainer, GRPOConfig
    
    # 加载数据
    data_file = config["data"]["grpo_file"]
    max_samples = config["data"].get("max_samples", 1000)
    dataset = load_grpo_data(data_file, max_samples)
    
    # 加载模型
    model, processor = setup_model(config)
    
    # 获取奖励函数权重
    rewards_config = config.get("rewards", {})
    accuracy_weight = rewards_config.get("accuracy_weight", 0.5)
    format_weight = rewards_config.get("format_weight", 0.2)
    length_weight = rewards_config.get("length_weight", 0.15)
    diversity_weight = rewards_config.get("diversity_weight", 0.15)
    
    # 定义带权重的奖励函数
    def weighted_accuracy_reward(prompts, completions, ground_truth, **kwargs):
        scores = accuracy_reward(prompts, completions, ground_truth, **kwargs)
        return [s * accuracy_weight for s in scores]
    
    def weighted_format_reward(prompts, completions, ground_truth, **kwargs):
        scores = format_reward(prompts, completions, ground_truth, **kwargs)
        return [s * format_weight for s in scores]
    
    def weighted_length_reward(prompts, completions, ground_truth, **kwargs):
        scores = length_penalty_reward(prompts, completions, ground_truth, **kwargs)
        return [s * length_weight for s in scores]
    
    def weighted_diversity_reward(prompts, completions, ground_truth, **kwargs):
        scores = diversity_reward(prompts, completions, ground_truth, **kwargs)
        return [s * diversity_weight for s in scores]
    
    reward_funcs = [
        weighted_accuracy_reward,
        weighted_format_reward,
        weighted_length_reward,
        weighted_diversity_reward,
    ]
    
    # GRPO 配置
    training_config = config["training"]
    grpo_config_dict = config.get("grpo", {})
    
    training_args = {
        "output_dir": training_config["output_dir"],
        "num_generations": grpo_config_dict.get("num_generations", 4),
        "max_completion_length": grpo_config_dict.get("max_completion_length", 256),
        "temperature": grpo_config_dict.get("temperature", 0.7),
        "top_p": grpo_config_dict.get("top_p", 0.9),
        "beta": grpo_config_dict.get("beta", 0.04),
        "num_train_epochs": training_config["num_train_epochs"],
        "per_device_train_batch_size": training_config["per_device_train_batch_size"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "learning_rate": training_config["learning_rate"],
        "lr_scheduler_type": training_config["lr_scheduler_type"],
        "warmup_ratio": training_config["warmup_ratio"],
        "bf16": training_config["bf16"],
        "logging_steps": training_config["logging_steps"],
        "save_strategy": training_config["save_strategy"],
        "save_total_limit": training_config["save_total_limit"],
        "dataloader_num_workers": training_config["dataloader_num_workers"],
        "gradient_checkpointing": training_config.get("gradient_checkpointing", True),
        "report_to": "none",
    }
    
    # 加载 DeepSpeed 配置
    ds_config_path = training_config.get("deepspeed")
    if ds_config_path and os.path.exists(ds_config_path):
        training_args["deepspeed"] = ds_config_path
        if is_main_process():
            print(f"使用 DeepSpeed: {ds_config_path}")
    
    grpo_config = GRPOConfig(**training_args)
    
    # 创建 GRPO Trainer
    grpo_trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        processing_class=processor,
    )
    
    # 开始训练
    if is_main_process():
        print("\n" + "=" * 60)
        print(f"开始 GRPO 训练（双卡 4090）")
        print(f"每个 prompt 生成 {grpo_config_dict.get('num_generations', 4)} 个回答")
        print(f"奖励函数权重: accuracy={accuracy_weight}, format={format_weight}, length={length_weight}, diversity={diversity_weight}")
        print("=" * 60)
    
    grpo_trainer.train()
    
    # 保存模型
    if is_main_process():
        output_path = training_config["output_dir"]
        grpo_trainer.save_model(output_path)
        processor.save_pretrained(output_path)
        print(f"\nGRPO 训练完成！模型保存在: {output_path}")
    
    return training_config["output_dir"]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GRPO 训练（双卡 4090）")
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml", help="配置文件路径")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config)
