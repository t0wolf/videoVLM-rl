"""
Step 2: 轻量 SFT 训练（双卡 4090 优化版）
目的：快速适配视频 QA 数据分布，为 DPO 做准备
启动：deepspeed --num_gpus=2 scripts/02_train_sft.py
"""

import os
import sys
import yaml
import json
import torch
from pathlib import Path

def load_config(config_path: str = "configs/sft_config.yaml"):
    """加载 SFT 配置"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_main_process():
    """判断是否是主进程（多卡训练时只有主进程打印日志和保存模型）"""
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def setup_model(config: dict):
    """加载模型和 LoRA"""
    from transformers import (
        Qwen3VLForConditionalGeneration,
        AutoProcessor,
    )
    from peft import LoraConfig, get_peft_model
    
    model_name = config["model"]["name"]
    
    if is_main_process():
        print(f"加载模型: {model_name}")
    
    # 加载模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=config["model"]["device_map"],
        attn_implementation="flash_attention_2",
    )
    
    # 加载 processor
    processor = AutoProcessor.from_pretrained(model_name)
    
    # 配置 LoRA
    lora_config = LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    if is_main_process():
        model.print_trainable_parameters()
    
    return model, processor


def load_dataset(config: dict):
    """加载 SFT 数据集"""
    data_file = config["data"].get("sft_file", "./data/sft_train.jsonl")
    
    if not os.path.exists(data_file):
        if is_main_process():
            print(f"数据文件不存在: {data_file}")
            print("请先运行 01_prepare_sft_data.py")
        sys.exit(1)
    
    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    max_samples = config["data"].get("max_samples", 15000)
    if len(samples) > max_samples:
        samples = samples[:max_samples]
    
    if is_main_process():
        print(f"加载了 {len(samples)} 条 SFT 数据")
    
    from datasets import Dataset
    dataset = Dataset.from_list(samples)
    return dataset


def train(config: dict):
    """执行 SFT 训练"""
    from trl import SFTTrainer, SFTConfig
    
    # 加载模型
    model, processor = setup_model(config)
    
    # 加载数据
    dataset = load_dataset(config)
    
    # 训练配置
    training_config = config["training"]
    
    # 构建训练参数
    training_args = {
        "output_dir": training_config["output_dir"],
        "num_train_epochs": training_config["num_train_epochs"],
        "per_device_train_batch_size": training_config["per_device_train_batch_size"],
        "gradient_accumulation_steps": training_config["gradient_accumulation_steps"],
        "learning_rate": training_config["learning_rate"],
        "lr_scheduler_type": training_config["lr_scheduler_type"],
        "warmup_ratio": training_config["warmup_ratio"],
        "bf16": training_config["bf16"],
        "max_seq_length": training_config["max_seq_length"],
        "logging_steps": training_config["logging_steps"],
        "save_strategy": training_config["save_strategy"],
        "save_total_limit": training_config["save_total_limit"],
        "dataloader_num_workers": training_config["dataloader_num_workers"],
        "gradient_checkpointing": training_config.get("gradient_checkpointing", True),
        "report_to": "none",
    }
    
    # 加载 DeepSpeed 配置（如果有）
    ds_config_path = training_config.get("deepspeed")
    if ds_config_path and os.path.exists(ds_config_path):
        training_args["deepspeed"] = ds_config_path
        if is_main_process():
            print(f"使用 DeepSpeed: {ds_config_path}")
    
    sft_config = SFTConfig(**training_args)
    
    # 创建 Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        processing_class=processor,
    )
    
    # 开始训练
    if is_main_process():
        print("\n" + "=" * 60)
        print("开始 SFT 训练（双卡 4090）")
        print("=" * 60)
    
    trainer.train()
    
    # 保存模型（只在主进程保存）
    if is_main_process():
        output_path = training_config["output_dir"]
        trainer.save_model(output_path)
        processor.save_pretrained(output_path)
        print(f"\nSFT 训练完成！模型保存在: {output_path}")
    
    return training_config["output_dir"]


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="轻量 SFT 训练（双卡 4090）")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml", help="配置文件路径")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    train(config)
