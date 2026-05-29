"""
Step 3: 构建 DPO 偏好数据集
使用 SFT 模型生成候选回答 + MiMo v2.5 打分
"""

import json
import os
import sys
import time
import base64
import random
from pathlib import Path

# ============================================================
# 配置
# ============================================================
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "your-api-key")
MIMO_API_BASE_URL = os.environ.get("MIMO_API_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
MIMO_MODEL = os.environ.get("MIMO_MODEL", "MiMo-V2.5")

# 评分 Prompt
SCORING_PROMPT = """你是一个视频理解质量评估专家。请严格评估以下回答的质量。

## 视频信息
- 视频路径：{video_path}
- 问题：{question}

## 待评估回答
{answer}

## 评分标准
请从以下 4 个维度分别打分（1-5 分）：

1. **视觉理解准确性**（1-5分）：回答是否正确描述了视频中的视觉内容？是否出现幻觉？
   - 5分：完全准确，无幻觉
   - 3分：基本准确，有少量细节偏差
   - 1分：严重错误或大量幻觉

2. **时间定位准确性**（1-5分）：如果回答涉及时间段描述，时间是否准确？
   - 5分：时间描述精确
   - 3分：大致准确
   - 1分：时间完全错误

3. **信息完整性**（1-5分）：回答是否涵盖了问题的关键信息？
   - 5分：信息全面
   - 3分：覆盖主要内容
   - 1分：严重遗漏

4. **语言流畅性**（1-5分）：回答是否通顺自然？
   - 5分：流畅自然
   - 3分：基本通顺
   - 1分：语句不通

## 输出格式
请严格按以下 JSON 格式输出，不要输出其他内容：
{{
  "视觉理解准确性": {{"score": X, "reason": "..."}},
  "时间定位准确性": {{"score": X, "reason": "..."}},
  "信息完整性": {{"score": X, "reason": "..."}},
  "语言流畅性": {{"score": X, "reason": "..."}},
  "total": X.X
}}"""


def load_sft_model(model_path: str):
    """加载 SFT 后的模型"""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    import torch
    
    print(f"加载 SFT 模型: {model_path}")
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    return model, processor


def generate_answers(
    model,
    processor,
    video_path: str,
    question: str,
    num_answers: int = 4,
    temperatures: list = [0.7, 0.9, 1.0, 1.2],
) -> list:
    """用 SFT 模型生成多个候选回答"""
    import torch
    
    answers = []
    
    for i, temp in enumerate(temperatures[:num_answers]):
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_path},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                videos=[video_path],
                return_tensors="pt",
            ).to(model.device)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=temp,
                    do_sample=True,
                    top_p=0.9,
                )
            
            response = processor.decode(
                output[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            answers.append(response.strip())
            
        except Exception as e:
            print(f"    生成回答 {i+1} 失败: {e}")
            continue
    
    return answers


def call_mimo_scoring(video_path: str, question: str, answer: str) -> dict:
    """调用 MiMo v2.5 打分"""
    import urllib.request
    
    prompt = SCORING_PROMPT.format(
        video_path=video_path,
        question=question,
        answer=answer,
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_path}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    url = f"{MIMO_API_BASE_URL}/chat/completions"
    payload = {
        "model": MIMO_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1024,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MIMO_API_KEY}",
    }
    
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                content = result["choices"][0]["message"]["content"]
                
                # 解析 JSON
                start = content.find("{")
                end = content.rfind("}") + 1
                return json.loads(content[start:end])
                
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"    MiMo API 调用失败: {e}")
                return None
    
    return None


def build_preference_pair(
    model,
    processor,
    video_path: str,
    question: str,
    num_answers: int = 4,
    min_score_diff: float = 0.5,
) -> dict:
    """构建单个偏好对"""
    
    # 1. 生成候选回答
    print(f"    生成 {num_answers} 个候选回答...")
    answers = generate_answers(
        model, processor, video_path, question, num_answers=num_answers
    )
    
    if len(answers) < 2:
        print(f"    有效回答不足 2 个，跳过")
        return None
    
    # 2. MiMo 打分
    scored_answers = []
    for i, answer in enumerate(answers):
        print(f"    评估回答 {i+1}/{len(answers)}...")
        score = call_mimo_scoring(video_path, question, answer)
        
        if score is not None:
            total = score.get("total", 0)
            if total == 0:
                scores = [score[d]["score"] for d in ["视觉理解准确性", "时间定位准确性", "信息完整性", "语言流畅性"] if d in score]
                total = sum(scores) / len(scores) if scores else 0
            
            scored_answers.append({
                "answer": answer,
                "score": total,
                "details": score,
            })
        
        time.sleep(0.5)  # 避免限流
    
    if len(scored_answers) < 2:
        return None
    
    # 3. 选择 chosen/rejected
    scored_answers.sort(key=lambda x: x["score"], reverse=True)
    best = scored_answers[0]
    worst = scored_answers[-1]
    
    score_diff = best["score"] - worst["score"]
    if score_diff < min_score_diff:
        print(f"    分数差异过小（{score_diff:.2f}），跳过")
        return None
    
    return {
        "video_path": video_path,
        "question": question,
        "chosen": best["answer"],
        "chosen_score": best["score"],
        "rejected": worst["answer"],
        "rejected_score": worst["score"],
        "score_diff": round(score_diff, 2),
    }


def process_dataset(
    sft_model_path: str,
    sft_data_file: str = "./data/sft_train.jsonl",
    output_file: str = "./data/dpo_preference_data.jsonl",
    num_answers: int = 4,
    min_score_diff: float = 0.5,
    max_samples: int = 1000,
):
    """处理整个数据集"""
    
    print("=" * 60)
    print("构建 DPO 偏好数据集")
    print("=" * 60)
    
    # 加载 SFT 数据
    samples = []
    with open(sft_data_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    # 随机采样
    random.seed(42)
    if len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
    
    print(f"共 {len(samples)} 个样本待处理")
    
    # 加载 SFT 模型
    model, processor = load_sft_model(sft_model_path)
    
    # 处理每个样本
    results = []
    stats = {"total": len(samples), "success": 0, "skip": 0}
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    for i, sample in enumerate(samples):
        video_path = sample.get("videos", [""])[0]
        question = ""
        for conv in sample.get("conversations", []):
            if conv.get("role") == "user":
                # 提取问题（去掉 <video> 标签）
                question = conv.get("content", "").replace("<video>", "").strip()
                break
        
        if not video_path or not question:
            stats["skip"] += 1
            continue
        
        print(f"\n[{i+1}/{len(samples)}] {question[:50]}...")
        
        pair = build_preference_pair(
            model, processor, video_path, question,
            num_answers=num_answers,
            min_score_diff=min_score_diff,
        )
        
        if pair:
            results.append(pair)
            stats["success"] += 1
            print(f"    ✅ chosen={pair['chosen_score']:.1f}, rejected={pair['rejected_score']:.1f}")
            
            # 每 100 条保存一次
            if stats["success"] % 100 == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(f"    已保存 {stats['success']} 条偏好数据")
        else:
            stats["skip"] += 1
    
    # 最终保存
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n{'=' * 60}")
    print(f"完成！成功: {stats['success']}, 跳过: {stats['skip']}")
    print(f"输出文件: {output_file}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="构建 DPO 偏好数据集")
    parser.add_argument("--sft-model", type=str, default="./output/sft", help="SFT 模型路径")
    parser.add_argument("--sft-data", type=str, default="./data/sft_train.jsonl", help="SFT 数据文件")
    parser.add_argument("--output", type=str, default="./data/dpo_preference_data.jsonl", help="输出文件")
    parser.add_argument("--num-answers", type=int, default=4, help="每个样本的回答数")
    parser.add_argument("--min-score-diff", type=float, default=0.5, help="最小分数差异")
    parser.add_argument("--max-samples", type=int, default=1000, help="最大样本数")
    
    args = parser.parse_args()
    
    process_dataset(
        sft_model_path=args.sft_model,
        sft_data_file=args.sft_data,
        output_file=args.output,
        num_answers=args.num_answers,
        min_score_diff=args.min_score_diff,
        max_samples=args.max_samples,
    )
