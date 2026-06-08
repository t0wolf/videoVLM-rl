"""
Step 3: 构建 GRPO 训练数据
直接从 SFT 数据转换，不需要 API
"""

import json
import os
import random


def format_sft_to_grpo(sft_data_file, output_file, max_samples=1000):
    print("=" * 60)
    print("构建 GRPO 训练数据")
    print("=" * 60)
    
    samples = []
    with open(sft_data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"加载了 {len(samples)} 条 SFT 数据")
    
    random.seed(42)
    if len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
        print(f"随机采样 {max_samples} 条")
    
    grpo_data = []
    skipped = 0
    
    for sample in samples:
        video_path = sample.get("videos", [""])[0] if sample.get("videos") else ""
        prompt, ground_truth = "", ""
        
        for conv in sample.get("conversations", []):
            role = conv.get("role", "")
            content = conv.get("content", "")
            if role == "user":
                prompt = content.replace("<video>", "").strip()
            elif role == "assistant":
                ground_truth = content.strip()
        
        if not prompt or not ground_truth:
            skipped += 1
            continue
        
        grpo_data.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
            "video": video_path,
        })
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for item in grpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n完成！成功: {len(grpo_data)}, 跳过: {skipped}")
    print(f"输出文件: {output_file}")
    if grpo_data:
        print(f"\n示例: {json.dumps(grpo_data[0], ensure_ascii=False)[:200]}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data", type=str, default="./data/sft_train.jsonl")
    parser.add_argument("--output", type=str, default="./data/grpo_train.jsonl")
    parser.add_argument("--max-samples", type=int, default=1000)
    
    args = parser.parse_args()
    format_sft_to_grpo(args.sft_data, args.output, args.max_samples)
