"""
Step 5: 评测脚本
自动评测（VideoMME/MLVU）+ 定性分析
"""

import os
import sys
import json
from pathlib import Path

def run_lmms_eval(
    model_path: str,
    benchmarks: list = ["videomme", "mlvu"],
    batch_size: int = 1,
    output_dir: str = "./output/eval",
):
    """使用 lmms-eval 运行自动评测"""
    
    print("=" * 60)
    print(f"评测模型: {model_path}")
    print(f"Benchmarks: {benchmarks}")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = ",".join(benchmarks)
    
    cmd = (
        f"python -m lmms_eval "
        f"--model qwen2_5_vl "
        f"--model_args pretrained={model_path},max_pixels=12845056 "
        f"--tasks {tasks} "
        f"--batch_size {batch_size} "
        f"--output_path {output_dir}"
    )
    
    print(f"\n执行命令:\n{cmd}\n")
    
    ret = os.system(cmd)
    
    if ret != 0:
        print("\nlmms-eval 执行失败，请确保已安装：")
        print("  pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git")
    
    return ret


def qualitative_analysis(
    sft_model_path: str,
    grpo_model_path: str,
    test_data_file: str = "./data/sft_train.jsonl",
    output_file: str = "./output/qualitative_analysis.json",
    num_samples: int = 100,
):
    """定性分析：对比 SFT vs GRPO 的输出质量"""
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    import random
    
    print("\n" + "=" * 60)
    print("定性分析：SFT vs GRPO 对比")
    print("=" * 60)
    
    # 加载测试数据
    samples = []
    with open(test_data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    random.seed(42)
    test_samples = random.sample(samples, min(num_samples, len(samples)))
    
    # 加载模型
    print("加载 SFT 模型...")
    sft_model = Qwen3VLForConditionalGeneration.from_pretrained(
        sft_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    sft_processor = AutoProcessor.from_pretrained(sft_model_path)
    
    print("加载 GRPO 模型...")
    grpo_model = Qwen3VLForConditionalGeneration.from_pretrained(
        grpo_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    grpo_processor = AutoProcessor.from_pretrained(grpo_model_path)
    
    # 生成对比
    results = []
    
    for i, sample in enumerate(test_samples):
        video_path = sample.get("videos", [""])[0]
        question = ""
        gt_answer = ""
        for conv in sample.get("conversations", []):
            if conv.get("role") == "user":
                question = conv.get("content", "").replace("<video>", "").strip()
            elif conv.get("role") == "assistant":
                gt_answer = conv.get("content", "").strip()
        
        if not video_path or not question:
            continue
        
        print(f"\r[{i+1}/{len(test_samples)}] {question[:50]}...", end="", flush=True)
        
        # 生成回答
        try:
            # SFT 回答
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
            text = sft_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = sft_processor(
                text=[text],
                videos=[video_path],
                return_tensors="pt",
            ).to(sft_model.device)
            
            with torch.no_grad():
                output = sft_model.generate(**inputs, max_new_tokens=256, do_sample=False)
            
            sft_response = sft_processor.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            
            # GRPO 回答
            text = grpo_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = grpo_processor(
                text=[text],
                videos=[video_path],
                return_tensors="pt",
            ).to(grpo_model.device)
            
            with torch.no_grad():
                output = grpo_model.generate(**inputs, max_new_tokens=256, do_sample=False)
            
            grpo_response = grpo_processor.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            
            results.append({
                "video": video_path,
                "question": question,
                "ground_truth": gt_answer,
                "sft_response": sft_response,
                "grpo_response": grpo_response,
            })
            
        except Exception as e:
            print(f"\n  生成失败: {e}")
            continue
    
    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n\n定性分析完成！共 {len(results)} 条对比结果")
    print(f"输出文件: {output_file}")
    
    # 打印几个示例
    print("\n" + "=" * 60)
    print("示例对比：")
    print("=" * 60)
    for i, r in enumerate(results[:3]):
        print(f"\n--- 示例 {i+1} ---")
        print(f"问题: {r['question']}")
        print(f"Ground Truth: {r['ground_truth'][:100]}...")
        print(f"SFT: {r['sft_response'][:100]}...")
        print(f"GRPO: {r['grpo_response'][:100]}...")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="评测脚本")
    parser.add_argument("--sft-model", type=str, default="./output/sft", help="SFT 模型路径")
    parser.add_argument("--grpo-model", type=str, default="./output/grpo", help="GRPO 模型路径")
    parser.add_argument("--qualitative", action="store_true", help="运行定性分析")
    parser.add_argument("--lmms-eval", action="store_true", help="运行 lmms-eval 自动评测")
    parser.add_argument("--benchmarks", nargs="+", default=["videomme", "mlvu"], help="评测 benchmark")
    
    args = parser.parse_args()
    
    if args.qualitative:
        qualitative_analysis(
            sft_model_path=args.sft_model,
            grpo_model_path=args.grpo_model,
        )
    
    if args.lmms_eval:
        print("\n" + "=" * 60)
        print("运行 lmms-eval 自动评测...")
        print("=" * 60)
        
        # 评测 SFT 模型
        run_lmms_eval(
            model_path=args.sft_model,
            benchmarks=args.benchmarks,
            output_dir="./output/eval_sft",
        )
        
        # 评测 GRPO 模型
        run_lmms_eval(
            model_path=args.grpo_model,
            benchmarks=args.benchmarks,
            output_dir="./output/eval_grpo",
        )
