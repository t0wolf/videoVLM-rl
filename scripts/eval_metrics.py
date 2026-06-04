"""
评测脚本：计算精确数字指标
支持 BLEU、ROUGE、METEOR、Exact Match、F1
"""

import json
import os
import re
import string
import torch
from collections import Counter


def normalize_answer(s):
    """标准化答案：小写、去标点、去多余空格"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, reference):
    """计算 Token-level F1"""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return f1


def compute_exact_match(prediction, reference):
    """计算 Exact Match"""
    return int(normalize_answer(prediction) == normalize_answer(reference))


def compute_bleu(prediction, reference, n=4):
    """计算 BLEU-N"""
    from collections import Counter
    
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # 计算 n-gram 精度
    scores = []
    for i in range(1, n + 1):
        pred_ngrams = [tuple(pred_tokens[j:j+i]) for j in range(len(pred_tokens) - i + 1)]
        ref_ngrams = [tuple(ref_tokens[j:j+i]) for j in range(len(ref_tokens) - i + 1)]
        
        if not pred_ngrams:
            scores.append(0.0)
            continue
        
        pred_counts = Counter(pred_ngrams)
        ref_counts = Counter(ref_ngrams)
        
        clipped = sum(min(pred_counts[g], ref_counts.get(g, 0)) for g in pred_counts)
        total = sum(pred_counts.values())
        
        scores.append(clipped / total if total > 0 else 0.0)
    
    # 简化版 BLEU（不加 brevity penalty）
    if all(s > 0 for s in scores):
        import math
        log_avg = sum(math.log(s) for s in scores) / len(scores)
        return math.exp(log_avg)
    
    return 0.0


def compute_rouge_l(prediction, reference):
    """计算 ROUGE-L"""
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # LCS
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i-1] == ref_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[m][n]
    
    precision = lcs_length / m
    recall = lcs_length / n
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def evaluate_model(
    model_path: str,
    test_data_file: str = "./data/sft_train.jsonl",
    output_file: str = None,
    max_samples: int = 100,
):
    """评测模型，计算所有指标"""
    
    print("=" * 60)
    print(f"评测模型: {model_path}")
    print("=" * 60)
    
    # 加载模型
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    
    print("加载模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    # 加载测试数据
    samples = []
    with open(test_data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    if len(samples) > max_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, max_samples)
    
    print(f"测试样本数: {len(samples)}")
    
    # 评测
    results = []
    metrics_sum = {"f1": 0, "em": 0, "bleu-1": 0, "bleu-4": 0, "rouge-l": 0}
    
    for i, sample in enumerate(samples):
        video_path = sample["videos"][0]
        question = ""
        reference = ""
        
        for conv in sample["conversations"]:
            if conv["role"] == "user":
                question = conv["content"].replace("<video>", "").strip()
            elif conv["role"] == "assistant":
                reference = conv["content"]
        
        # 生成回答
        try:
            messages = [{"role": "user", "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question}
            ]}]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], videos=[video_path], return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            
            prediction = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            
        except Exception as e:
            print(f"  [{i+1}] 生成失败: {e}")
            prediction = ""
        
        # 计算指标
        f1 = compute_f1(prediction, reference)
        em = compute_exact_match(prediction, reference)
        bleu1 = compute_bleu(prediction, reference, n=1)
        bleu4 = compute_bleu(prediction, reference, n=4)
        rouge_l = compute_rouge_l(prediction, reference)
        
        metrics_sum["f1"] += f1
        metrics_sum["em"] += em
        metrics_sum["bleu-1"] += bleu1
        metrics_sum["bleu-4"] += bleu4
        metrics_sum["rouge-l"] += rouge_l
        
        results.append({
            "question": question,
            "reference": reference,
            "prediction": prediction,
            "f1": round(f1, 4),
            "em": em,
            "bleu-1": round(bleu1, 4),
            "bleu-4": round(bleu4, 4),
            "rouge-l": round(rouge_l, 4),
        })
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] F1={metrics_sum['f1']/(i+1):.4f}")
    
    # 汇总
    n = len(samples)
    final_metrics = {
        "model": model_path,
        "num_samples": n,
        "F1": round(metrics_sum["f1"] / n, 4),
        "EM": round(metrics_sum["em"] / n, 4),
        "BLEU-1": round(metrics_sum["bleu-1"] / n, 4),
        "BLEU-4": round(metrics_sum["bleu-4"] / n, 4),
        "ROUGE-L": round(metrics_sum["rouge-l"] / n, 4),
    }
    
    print("\n" + "=" * 60)
    print("评测结果")
    print("=" * 60)
    for k, v in final_metrics.items():
        if k not in ["model", "num_samples"]:
            print(f"  {k}: {v:.4f}")
    
    # 保存结果
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"metrics": final_metrics, "details": results}, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果保存在: {output_file}")
    
    return final_metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="模型评测")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--test-data", type=str, default="./data/sft_train.jsonl", help="测试数据")
    parser.add_argument("--output", type=str, default=None, help="输出文件")
    parser.add_argument("--max-samples", type=int, default=100, help="最大测试样本数")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        test_data_file=args.test_data,
        output_file=args.output,
        max_samples=args.max_samples,
    )
