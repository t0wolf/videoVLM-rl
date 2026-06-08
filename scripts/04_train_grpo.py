"""
Step 4: GRPO 训练（双卡 4090）
启动：deepspeed --num_gpus=2 scripts/04_train_grpo.py --config configs/grpo_config.yaml
"""

import os
import sys
import yaml
import json
import torch
from pathlib import Path


def load_config(config_path="configs/grpo_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_main_process():
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def load_grpo_data(data_file, max_samples=1000):
    if not os.path.exists(data_file):
        if is_main_process():
            print(f"数据文件不存在: {data_file}")
            print("请先运行: python scripts/03_build_grpo_data.py")
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
# 奖励函数
# ============================================================

def accuracy_reward(prompts, completions, ground_truth, **kwargs):
    scores = []
    for completion, gt in zip(completions, ground_truth):
        c_tokens = set(completion.lower().split())
        g_tokens = set(gt.lower().split())
        if not g_tokens:
            scores.append(0.0)
            continue
        overlap = c_tokens & g_tokens
        p = len(overlap) / len(c_tokens) if c_tokens else 0
        r = len(overlap) / len(g_tokens)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        scores.append(f1)
    return scores


def format_reward(prompts, completions, ground_truth, **kwargs):
    scores = []
    for completion in completions:
        score = 0.0
        length = len(completion)
        if 100 <= length <= 300:
            score += 0.5
        elif 50 <= length < 100 or 300 < length <= 500:
            score += 0.3
        else:
            score += 0.1
        punct = sum(1 for c in completion if c in '.!?,;:')
        score += min(punct * 0.1, 0.3)
        scores.append(min(score, 1.0))
    return scores


def length_reward(prompts, completions, ground_truth, **kwargs):
    scores = []
    for completion, gt in zip(completions, ground_truth):
        c_len = len(completion)
        g_len = len(gt)
        ratio = c_len / g_len if g_len > 0 else 1.0
        if 0.8 <= ratio <= 1.2:
            scores.append(1.0)
        elif 0.5 <= ratio <= 1.5:
            scores.append(0.7)
        elif 0.3 <= ratio <= 2.0:
            scores.append(0.4)
        else:
            scores.append(0.1)
    return scores


def diversity_reward(prompts, completions, ground_truth, **kwargs):
    scores = []
    for completion in completions:
        tokens = completion.lower().split()
        if not tokens:
            scores.append(0.0)
            continue
        scores.append(min(len(set(tokens)) / len(tokens), 1.0))
    return scores


def train(config):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model
    from trl import GRPOTrainer, GRPOConfig
    
    model_name = config["model"]["name"]
    training_config = config["training"]
    grpo_config = config.get("grpo", {})
    rewards_config = config.get("rewards", {})
    
    # 加载模型
    if is_main_process():
        print(f"加载模型: {model_name}")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    # LoRA
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules="all-linear",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    if is_main_process():
        model.print_trainable_parameters()
    
    # 加载数据
    dataset = load_grpo_data(config["data"]["grpo_file"], config["data"].get("max_samples", 1000))
    
    # 格式化数据为 text 字段
    from datasets import Dataset
    
    text_samples = []
    for item in dataset:
        prompt = item.get("prompt", "")
        gt = item.get("ground_truth", "")
        text_samples.append({
            "prompt": prompt,
            "ground_truth": gt,
            "text": f"Question: {prompt}\nAnswer: {gt}",
        })
    
    dataset = Dataset.from_list(text_samples)
    
    # 奖励函数权重
    aw = rewards_config.get("accuracy_weight", 0.5)
    fw = rewards_config.get("format_weight", 0.2)
    lw = rewards_config.get("length_weight", 0.15)
    dw = rewards_config.get("diversity_weight", 0.15)
    
    def weighted_accuracy(prompts, completions, ground_truth, **kwargs):
        return [s * aw for s in accuracy_reward(prompts, completions, ground_truth, **kwargs)]
    
    def weighted_format(prompts, completions, ground_truth, **kwargs):
        return [s * fw for s in format_reward(prompts, completions, ground_truth, **kwargs)]
    
    def weighted_length(prompts, completions, ground_truth, **kwargs):
        return [s * lw for s in length_reward(prompts, completions, ground_truth, **kwargs)]
    
    def weighted_diversity(prompts, completions, ground_truth, **kwargs):
        return [s * dw for s in diversity_reward(prompts, completions, ground_truth, **kwargs)]
    
    reward_funcs = [weighted_accuracy, weighted_format, weighted_length, weighted_diversity]
    
    # 训练配置
    sft_config = GRPOConfig(
        output_dir=training_config["output_dir"],
        num_generations=grpo_config.get("num_generations", 4),
        max_completion_length=grpo_config.get("max_completion_length", 256),
        temperature=grpo_config.get("temperature", 0.7),
        top_p=grpo_config.get("top_p", 0.9),
        beta=grpo_config.get("beta", 0.04),
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        warmup_steps=50,
        bf16=training_config["bf16"],
        logging_steps=training_config["logging_steps"],
        save_strategy=training_config["save_strategy"],
        save_total_limit=training_config["save_total_limit"],
        dataloader_num_workers=training_config["dataloader_num_workers"],
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        report_to="none",
        dataset_text_field="text",
        max_length=2048,
    )
    
    # DeepSpeed
    ds_path = training_config.get("deepspeed")
    if ds_path and os.path.exists(ds_path):
        sft_config.deepspeed = ds_path
        if is_main_process():
            print(f"使用 DeepSpeed: {ds_path}")
    
    # 创建 Trainer
    trainer = GRPOTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        processing_class=processor,
    )
    
    if is_main_process():
        print(f"\n{'='*60}")
        print("开始 GRPO 训练")
        print(f"num_generations={grpo_config.get('num_generations', 4)}")
        print(f"{'='*60}")
    
    trainer.train()
    
    if is_main_process():
        output_path = training_config["output_dir"]
        trainer.save_model(output_path)
        processor.save_pretrained(output_path)
        print(f"\nGRPO 训练完成！模型保存在: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo_config.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)
