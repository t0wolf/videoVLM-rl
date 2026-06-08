"""
Step 4: DPO 训练（双卡 4090）
启动：deepspeed --num_gpus=2 scripts/04_train_dpo.py --config configs/dpo_config.yaml
"""

import os
import sys
import yaml
import json
import torch
from pathlib import Path


def load_config(config_path: str = "configs/dpo_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_main_process():
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def setup_models(config: dict):
    """加载 policy model 和 reference model"""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    
    policy_path = config["model"]["policy_name"]
    ref_path = config["model"]["ref_name"]
    
    if is_main_process():
        print(f"加载 Policy Model: {policy_path}")
    
    # 加载 policy model（SFT 后的模型 + LoRA）
    policy_model = Qwen3VLForConditionalGeneration.from_pretrained(
        policy_path,
        torch_dtype=torch.bfloat16,
    )
    
    # 如果有 LoRA 适配器，加载它
    if os.path.exists(os.path.join(policy_path, "adapter_config.json")):
        if is_main_process():
            print("检测到 LoRA 适配器，加载中...")
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            config["model"]["base_name"],
            torch_dtype=torch.bfloat16,
        )
        policy_model = PeftModel.from_pretrained(base_model, policy_path)
    
    if is_main_process():
        print(f"加载 Reference Model: {ref_path}")
    
    # 加载 reference model（冻结）
    ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
        ref_path,
        torch_dtype=torch.bfloat16,
    )
    
    processor = AutoProcessor.from_pretrained(ref_path)
    
    return policy_model, ref_model, processor


def load_preference_data(data_file: str):
    """加载 DPO 偏好数据"""
    if not os.path.exists(data_file):
        if is_main_process():
            print(f"偏好数据文件不存在: {data_file}")
            print("请先运行 03_build_dpo_data.py")
        sys.exit(1)
    
    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    if is_main_process():
        print(f"加载了 {len(samples)} 对偏好数据")
    
    from datasets import Dataset
    return Dataset.from_list(samples)


def train(config: dict):
    """执行 DPO 训练"""
    from trl import DPOTrainer, DPOConfig
    
    # 加载偏好数据
    data_file = config["data"]["preference_file"]
    dataset = load_preference_data(data_file)
    
    # 加载模型
    policy_model, ref_model, processor = setup_models(config)
    
    # DPO 配置
    training_config = config["training"]
    dpo_config_dict = config["dpo"]
    
    training_args = DPOConfig(
        output_dir=training_config["output_dir"],
        beta=dpo_config_dict["beta"],
        loss_type=dpo_config_dict["loss_type"],
        max_length=dpo_config_dict["max_length"],
        max_prompt_length=dpo_config_dict["max_prompt_length"],
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
    )
    
    # DeepSpeed
    ds_config_path = training_config.get("deepspeed")
    if ds_config_path and os.path.exists(ds_config_path):
        training_args.deepspeed = ds_config_path
        if is_main_process():
            print(f"使用 DeepSpeed: {ds_config_path}")
    
    # 创建 DPO Trainer
    dpo_trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
    )
    
    if is_main_process():
        print("\n" + "=" * 60)
        print(f"开始 DPO 训练（双卡 4090，beta={dpo_config_dict['beta']}）")
        print("=" * 60)
    
    dpo_trainer.train()
    
    if is_main_process():
        output_path = training_config["output_dir"]
        dpo_trainer.save_model(output_path)
        processor.save_pretrained(output_path)
        print(f"\nDPO 训练完成！模型保存在: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DPO 训练（双卡 4090）")
    parser.add_argument("--config", type=str, default="configs/dpo_config.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)
