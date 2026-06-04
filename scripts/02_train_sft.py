"""
Step 2: SFT 训练（双卡 4090）
启动：deepspeed --num_gpus=2 scripts/02_train_sft.py --config configs/sft_config.yaml
"""

import os
import sys
import yaml
import json
import torch
from pathlib import Path


def load_config(config_path: str = "configs/sft_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_main_process():
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def format_conversations(sample, processor):
    """将对话格式化为模型输入文本"""
    conversations = sample.get("conversations", [])
    video_path = sample.get("videos", [None])[0]
    
    # 构建 messages
    messages = []
    for conv in conversations:
        role = conv.get("role", "")
        content = conv.get("content", "").replace("<video>", "").strip()
        
        if role == "user":
            msg_content = []
            if video_path:
                msg_content.append({"type": "video", "video": video_path})
            msg_content.append({"type": "text", "text": content})
            messages.append({"role": "user", "content": msg_content})
        elif role == "assistant":
            messages.append({"role": "assistant", "content": content})
    
    # 用 processor 转为文本
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text


def train(config: dict):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import LoraConfig, get_peft_model
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    
    training_config = config["training"]
    model_name = config["model"]["name"]
    
    # ========== 加载模型 ==========
    if is_main_process():
        print(f"加载模型: {model_name}")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    # ========== 加载 LoRA ==========
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
    
    # ========== 加载数据 ==========
    data_file = config["data"].get("sft_file", "./data/sft_train.jsonl")
    if not os.path.exists(data_file):
        if is_main_process():
            print(f"数据文件不存在: {data_file}")
        sys.exit(1)
    
    raw_samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_samples.append(json.loads(line))
    
    max_samples = config["data"].get("max_samples", 15000)
    if len(raw_samples) > max_samples:
        raw_samples = raw_samples[:max_samples]
    
    if is_main_process():
        print(f"加载了 {len(raw_samples)} 条 SFT 数据")
    
    # 转为纯文本格式
    text_samples = []
    for sample in raw_samples:
        text = format_conversations(sample, processor)
        if text:
            text_samples.append({"text": text})
    
    dataset = Dataset.from_list(text_samples)
    
    if is_main_process():
        print(f"格式化完成，{len(text_samples)} 条有效数据")
        print(f"示例:\n{text_samples[0]['text'][:200]}...")
    
    # ========== 训练配置 ==========
    sft_config = SFTConfig(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        warmup_steps=50,
        bf16=training_config["bf16"],
        max_length=training_config.get("max_length", 2048),
        logging_steps=training_config["logging_steps"],
        save_strategy=training_config["save_strategy"],
        save_total_limit=training_config["save_total_limit"],
        dataloader_num_workers=training_config["dataloader_num_workers"],
        gradient_checkpointing=training_config.get("gradient_checkpointing", True),
        report_to="none",
        dataset_text_field="text",
    )
    
    # DeepSpeed
    ds_config_path = training_config.get("deepspeed")
    if ds_config_path and os.path.exists(ds_config_path):
        sft_config.deepspeed = ds_config_path
        if is_main_process():
            print(f"使用 DeepSpeed: {ds_config_path}")
    
    # ========== 开始训练 ==========
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=processor,
    )
    
    if is_main_process():
        print("\n" + "=" * 60)
        print("开始 SFT 训练（双卡 4090）")
        print("=" * 60)
    
    trainer.train()
    
    if is_main_process():
        output_path = training_config["output_dir"]
        trainer.save_model(output_path)
        processor.save_pretrained(output_path)
        print(f"\nSFT 训练完成！模型保存在: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SFT 训练（双卡 4090）")
    parser.add_argument("--config", type=str, default="configs/sft_config.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    config = load_config(args.config)
    train(config)
