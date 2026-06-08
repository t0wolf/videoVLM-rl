"""
Step 4: DPO 训练（双卡 4090）- 自定义实现，绕开 TRL 的 VL 处理
启动：deepspeed --num_gpus=2 scripts/04_train_dpo.py --config configs/dpo_config.yaml
"""

import os
import sys
import yaml
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def load_config(config_path="configs/dpo_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_main_process():
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


class DPODataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=2048):
        self.samples = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # 用 chat template 构造 prompt
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # tokenize prompt
        prompt_ids = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_length // 2,
            add_special_tokens=False
        )["input_ids"]

        # tokenize chosen/rejected response
        chosen_ids = self.tokenizer(
            chosen, truncation=True,
            max_length=self.max_length - len(prompt_ids),
            add_special_tokens=False
        )["input_ids"]
        rejected_ids = self.tokenizer(
            rejected, truncation=True,
            max_length=self.max_length - len(prompt_ids),
            add_special_tokens=False
        )["input_ids"]

        # 拼接 prompt + response
        chosen_full = prompt_ids + chosen_ids
        rejected_full = prompt_ids + rejected_ids

        prompt_len = len(prompt_ids)

        return {
            "chosen_input_ids": torch.tensor(chosen_full, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_full, dtype=torch.long),
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
        }


def compute_log_probs(model, input_ids, attention_mask, prompt_len):
    """计算 response 部分的 log 概率"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (batch, seq_len, vocab)

    # shift: 用 token i 的 logits 预测 token i+1
    shift_logits = logits[:, :-1, :]   # (batch, seq_len-1, vocab)
    shift_labels = input_ids[:, 1:]    # (batch, seq_len-1)
    shift_mask = attention_mask[:, 1:] # (batch, seq_len-1)

    # 只计算 response 部分（prompt_len 之后的 token）
    # prompt_len 是原始 prompt 长度，response 从 prompt_len 开始
    # shift 后 response 部分从 prompt_len-1 开始
    batch_size = input_ids.shape[0]
    log_probs_list = []
    for i in range(batch_size):
        resp_start = max(prompt_len[i].item() - 1, 0)
        resp_logits = shift_logits[i, resp_start:]   # (resp_len, vocab)
        resp_labels = shift_labels[i, resp_start:]    # (resp_len,)
        resp_mask = shift_mask[i, resp_start:]        # (resp_len,)

        token_log_probs = -F.cross_entropy(
            resp_logits, resp_labels, reduction="none"
        )  # (resp_len,)
        masked = token_log_probs * resp_mask.float()
        log_probs_list.append(masked.sum())

    return torch.stack(log_probs_list)


def train(config):
    import deepspeed
    from transformers import Qwen3VLForConditionalGeneration, AutoTokenizer

    policy_path = config["model"]["policy_name"]
    ref_path = config["model"]["ref_name"]
    training_config = config["training"]
    dpo_config = config["dpo"]

    tokenizer = AutoTokenizer.from_pretrained(ref_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载 policy model
    if is_main_process():
        print(f"加载 Policy Model: {policy_path}", flush=True)

    policy_model = Qwen3VLForConditionalGeneration.from_pretrained(
        policy_path, torch_dtype=torch.bfloat16
    )

    # 加载 LoRA
    if os.path.exists(os.path.join(policy_path, "adapter_config.json")):
        if is_main_process():
            print("加载 LoRA 适配器...", flush=True)
        from peft import PeftModel
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            ref_path, torch_dtype=torch.bfloat16
        )
        policy_model = PeftModel.from_pretrained(base_model, policy_path, is_trainable=True)

    # 显式开启训练模式
    policy_model.train()
    if is_main_process() and hasattr(policy_model, "print_trainable_parameters"):
        policy_model.print_trainable_parameters()

    # 加载 reference model
    if is_main_process():
        print(f"加载 Reference Model: {ref_path}", flush=True)
    ref_model = Qwen3VLForConditionalGeneration.from_pretrained(
        ref_path, torch_dtype=torch.float32
    )

    # 加载数据
    dataset = DPODataset(
        config["data"]["preference_file"], tokenizer,
        max_length=dpo_config["max_length"]
    )
    if is_main_process():
        print(f"加载了 {len(dataset)} 对偏好数据", flush=True)

    # DeepSpeed 初始化 policy model
    ds_config = None
    ds_path = training_config.get("deepspeed")
    if ds_path and os.path.exists(ds_path):
        with open(ds_path) as f:
            ds_config = json.load(f)

    # gradient checkpointing 关闭（与 ZeRO-3 + LoRA 有兼容问题）
    # if training_config.get("gradient_checkpointing", True):
    #     policy_model.gradient_checkpointing_enable()

    policy_model, optimizer, _, _ = deepspeed.initialize(
        model=policy_model,
        config=ds_config,
        model_parameters=[p for p in policy_model.parameters() if p.requires_grad],
    )

    # ref model 不需要优化器，冻结参数
    for p in ref_model.parameters():
        p.requires_grad = False
    # ref model 保持在 CPU，推理时临时搬到 GPU

    # DataLoader（手动 padding）
    pad_id = tokenizer.pad_token_id or 0

    def collate_fn(batch):
        max_len = max(
            max(b["chosen_input_ids"].size(0) for b in batch),
            max(b["rejected_input_ids"].size(0) for b in batch),
        )
        chosen_ids, chosen_masks = [], []
        rejected_ids, rejected_masks = [], []
        prompt_lens = []
        for b in batch:
            cl = b["chosen_input_ids"].size(0)
            rl = b["rejected_input_ids"].size(0)
            chosen_ids.append(F.pad(b["chosen_input_ids"], (0, max_len - cl), value=pad_id))
            chosen_masks.append(F.pad(torch.ones(cl), (0, max_len - cl), value=0))
            rejected_ids.append(F.pad(b["rejected_input_ids"], (0, max_len - rl), value=pad_id))
            rejected_masks.append(F.pad(torch.ones(rl), (0, max_len - rl), value=0))
            prompt_lens.append(b["prompt_len"])
        return {
            "chosen_input_ids": torch.stack(chosen_ids),
            "chosen_attention_mask": torch.stack(chosen_masks).long(),
            "rejected_input_ids": torch.stack(rejected_ids),
            "rejected_attention_mask": torch.stack(rejected_masks).long(),
            "prompt_len": torch.stack(prompt_lens),
        }

    dataloader = DataLoader(
        dataset,
        batch_size=training_config["per_device_train_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=training_config.get("dataloader_num_workers", 2),
        pin_memory=True,
    )

    beta = dpo_config["beta"]
    num_epochs = training_config["num_train_epochs"]
    log_steps = training_config.get("logging_steps", 10)

    if is_main_process():
        print(f"\n{'='*60}", flush=True)
        print(f"开始 DPO 训练 | beta={beta} | epochs={num_epochs}", flush=True)
        print(f"{'='*60}\n", flush=True)

    # debug: 检查可训练参数
    if is_main_process():
        trainable = sum(p.requires_grad for p in policy_model.parameters())
        total = sum(1 for _ in policy_model.parameters())
        print(f"DEBUG: trainable params = {trainable}/{total}", flush=True)
        # 检查一个 LoRA 参数
        for n, p in policy_model.named_parameters():
            if "lora" in n.lower():
                print(f"DEBUG: {n} requires_grad={p.requires_grad} dtype={p.dtype}", flush=True)
                break

    global_step = 0
    for epoch in range(num_epochs):
        if is_main_process():
            print(f"--- Epoch {epoch+1}/{num_epochs} ---", flush=True)

        for batch_idx, batch in enumerate(dataloader):
            # 移到 GPU
            chosen_ids = batch["chosen_input_ids"].to(policy_model.device)
            chosen_mask = batch["chosen_attention_mask"].to(policy_model.device)
            rejected_ids = batch["rejected_input_ids"].to(policy_model.device)
            rejected_mask = batch["rejected_attention_mask"].to(policy_model.device)
            prompt_len = batch["prompt_len"].to(policy_model.device)

            # policy model 计算 log probs
            pi_chosen = compute_log_probs(policy_model, chosen_ids, chosen_mask, prompt_len)
            pi_rejected = compute_log_probs(policy_model, rejected_ids, rejected_mask, prompt_len)

            # ref model 在 CPU 上计算 log probs（省显存）
            with torch.no_grad():
                cpu_ids_c = chosen_ids.cpu()
                cpu_mask_c = chosen_mask.cpu()
                cpu_ids_r = rejected_ids.cpu()
                cpu_mask_r = rejected_mask.cpu()
                cpu_plen = prompt_len.cpu()
                ref_chosen = compute_log_probs(ref_model, cpu_ids_c, cpu_mask_c, cpu_plen).to(policy_model.device)
                ref_rejected = compute_log_probs(ref_model, cpu_ids_r, cpu_mask_r, cpu_plen).to(policy_model.device)

            # DPO loss
            log_ratio_chosen = pi_chosen - ref_chosen
            log_ratio_rejected = pi_rejected - ref_rejected
            loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected)).mean()

            # backward
            policy_model.backward(loss)
            policy_model.step()

            global_step += 1

            if is_main_process() and global_step % log_steps == 0:
                with torch.no_grad():
                    chosen_reward = (beta * (pi_chosen - ref_chosen)).mean().item()
                    rejected_reward = (beta * (pi_rejected - ref_rejected)).mean().item()
                    acc = (log_ratio_chosen > log_ratio_rejected).float().mean().item()
                print(
                    f"  step {global_step} | loss={loss.item():.4f} | "
                    f"acc={acc:.2%} | reward_margin={chosen_reward - rejected_reward:.4f}",
                    flush=True,
                )

    # 保存
    import torch.distributed as dist
    if dist.is_initialized():
        dist.barrier()
    output_path = training_config["output_dir"]
    os.makedirs(output_path, exist_ok=True)
    if is_main_process():
        unwrapped = policy_model.module if hasattr(policy_model, "module") else policy_model
        if hasattr(unwrapped, "merge_and_unload"):
            unwrapped = unwrapped.merge_and_unload()
        unwrapped.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print(f"\nDPO 训练完成！模型保存在: {output_path}", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dpo_config.yaml")
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config)
    train(config)
