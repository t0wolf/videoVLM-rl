"""
Step 3: 构建 GRPO 训练数据
比 DPO 简单很多：只需要 prompt + ground_truth，不需要偏好对
"""

import json
import os
import random
from pathlib import Path


def format_sft_to_grpo(sft_data_file: str, output_file: str, max_samples: int = 1000):
    """
    将 SFT 数据转换为 GRPO 格式
    
    SFT 格式：{"conversations": [...], "videos": [...]}
    GRPO 格式：{"prompt": "...", "ground_truth": "...", "video": "..."}
    """
    
    print("=" * 60)
    print("构建 GRPO 训练数据")
    print("=" * 60)
    
    # 加载 SFT 数据
    samples = []
    with open(sft_data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"加载了 {len(samples)} 条 SFT 数据")
    
    # 随机采样
    random.seed(42)
    if len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
        print(f"随机采样 {max_samples} 条")
    
    # 转换格式
    grpo_data = []
    skipped = 0
    
    for sample in samples:
        video_path = sample.get("videos", [""])[0] if sample.get("videos") else ""
        
        # 提取 prompt 和 ground_truth
        prompt = ""
        ground_truth = ""
        
        for conv in sample.get("conversations", []):
            role = conv.get("role", "")
            content = conv.get("content", "")
            
            if role == "user":
                # 去掉 <video> 标签
                prompt = content.replace("<video>", "").strip()
            elif role == "assistant":
                ground_truth = content.strip()
        
        if not prompt or not ground_truth or not video_path:
            skipped += 1
            continue
        
        grpo_data.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
            "video": video_path,
        })
    
    # 保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in grpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n完成！成功: {len(grpo_data)}, 跳过: {skipped}")
    print(f"输出文件: {output_file}")
    print(f"\n示例数据:")
    if grpo_data:
        print(json.dumps(grpo_data[0], ensure_ascii=False, indent=2))
    
    return len(grpo_data)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="构建 GRPO 训练数据")
    parser.add_argument("--sft-data", type=str, default="./data/sft_train.jsonl", help="SFT 数据文件")
    parser.add_argument("--output", type=str, default="./data/grpo_train.jsonl", help="输出文件")
    parser.add_argument("--max-samples", type=int, default=1000, help="最大样本数")
    
    args = parser.parse_args()
    
    format_sft_to_grpo(
        sft_data_file=args.sft_data,
        output_file=args.output,
        max_samples=args.max_samples,
    )
