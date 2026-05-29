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
    dpo_model_path: str,
    test_data_file: str = "./data/sft_train.jsonl",
    output_file: str = "./output/qualitative_analysis.json",
    num_samples: int = 100,
):
    """定性分析：对比 SFT vs DPO 的输出质量"""
    import torch
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    import random
    
    print("\n" + "=" * 60)
    print("定性分析：SFT vs DPO 对比")
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
    
    print("加载 DPO 模型...")
    dpo_model = Qwen3VLForConditionalGeneration.from_pretrained(
        dpo_model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    dpo_processor = AutoProcessor.from_pretrained(dpo_model_path)
    
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
        
        print(f"[{i+1}/{len(test_samples)}] {question[:50]}...")
        
        # SFT 生成
        try:
            messages = [{"role": "user", "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question},
            ]}]
            text = sft_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = sft_processor(text=[text], videos=[video_path], return_tensors="pt").to(sft_model.device)
            with torch.no_grad():
                sft_out = sft_model.generate(**inputs, max_new_tokens=256)
            sft_answer = sft_processor.decode(sft_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as e:
            sft_answer = f"生成失败: {e}"
        
        # DPO 生成
        try:
            messages = [{"role": "user", "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question},
            ]}]
            text = dpo_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = dpo_processor(text=[text], videos=[video_path], return_tensors="pt").to(dpo_model.device)
            with torch.no_grad():
                dpo_out = dpo_model.generate(**inputs, max_new_tokens=256)
            dpo_answer = dpo_processor.decode(dpo_out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        except Exception as e:
            dpo_answer = f"生成失败: {e}"
        
        results.append({
            "video_path": video_path,
            "question": question,
            "ground_truth": gt_answer,
            "sft_answer": sft_answer,
            "dpo_answer": dpo_answer,
            "sft_length": len(sft_answer),
            "dpo_length": len(dpo_answer),
        })
    
    # 统计分析
    sft_lengths = [r["sft_length"] for r in results]
    dpo_lengths = [r["dpo_length"] for r in results]
    
    analysis = {
        "num_samples": len(results),
        "sft_avg_length": sum(sft_lengths) / len(sft_lengths) if sft_lengths else 0,
        "dpo_avg_length": sum(dpo_lengths) / len(dpo_lengths) if dpo_lengths else 0,
        "length_change_pct": (
            (sum(dpo_lengths) - sum(sft_lengths)) / sum(sft_lengths) * 100
            if sum(sft_lengths) > 0 else 0
        ),
        "samples": results[:50],  # 只保存前 50 个样本
    }
    
    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 60}")
    print("定性分析结果：")
    print(f"  SFT 平均回答长度: {analysis['sft_avg_length']:.1f} 字符")
    print(f"  DPO 平均回答长度: {analysis['dpo_avg_length']:.1f} 字符")
    print(f"  长度变化: {analysis['length_change_pct']:+.1f}%")
    print(f"  结果保存在: {output_file}")
    print(f"{'=' * 60}")
    
    return analysis


def compare_baselines(eval_dir: str = "./output/eval"):
    """对比三组实验的 benchmark 结果"""
    
    print("\n" + "=" * 60)
    print("Benchmark 结果对比")
    print("=" * 60)
    
    # 官方 baseline
    official_scores = {
        "videomme": 71.4,
        "mlvu": 78.1,
        "videommmu": 65.3,
    }
    
    # 尝试加载实验结果
    results = {"official": official_scores}
    
    for model_name in ["sft", "dpo"]:
        result_file = os.path.join(eval_dir, model_name, "results.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                results[model_name] = json.load(f)
    
    # 输出对比表格
    print(f"\n{'模型':<20} {'VideoMME':<12} {'MLVU':<12} {'VideoMMMU':<12}")
    print("-" * 56)
    
    for name, scores in results.items():
        vme = scores.get("videomme", "-")
        mlvu = scores.get("mlvu", "-")
        vmmmu = scores.get("videommmu", "-")
        print(f"{name:<20} {str(vme):<12} {str(mlvu):<12} {str(vmmmu):<12}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="模型评测")
    parser.add_argument("--sft-model", type=str, default="./output/sft")
    parser.add_argument("--dpo-model", type=str, default="./output/dpo")
    parser.add_argument("--benchmark", action="store_true", help="运行 benchmark 评测")
    parser.add_argument("--qualitative", action="store_true", help="运行定性分析")
    parser.add_argument("--compare", action="store_true", help="对比结果")
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_lmms_eval(args.dpo_model)
    
    if args.qualitative:
        qualitative_analysis(args.sft_model, args.dpo_model)
    
    if args.compare or (not args.benchmark and not args.qualitative):
        compare_baselines()
