"""
Step 3: 全 MiMo 模式构建 DPO 数据
MiMo 生成候选回答 + MiMo 打分，不需要本地模型
在 AutoDL 上运行
"""

import json
import os
import time
import urllib.request
import argparse

MIMO_API_KEY = "tp-chvb2livf8qw5873xt1hij288n3o8tfgc3j8qzmzcn7vhfoo"
MIMO_API_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5-pro"


def call_mimo(prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
    """调用 MiMo API 生成回答"""
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": MIMO_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MIMO_API_KEY}",
    }
    
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                f"{MIMO_API_BASE_URL}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"      API 调用失败: {e}")
                return ""


def generate_candidates(question: str, ground_truth: str, num_candidates: int = 4) -> list:
    """用 MiMo 生成多个候选回答"""
    prompt = f"""Based on the following video description context, answer the question.

Video context: {ground_truth}

Question: {question}

Please provide a detailed answer based on the video content."""

    temperatures = [0.5, 0.7, 0.9, 1.1]
    candidates = []
    
    for i, temp in enumerate(temperatures[:num_candidates]):
        response = call_mimo(prompt, temperature=temp)
        if response:
            candidates.append({"response": response, "temperature": temp})
    
    return candidates


def score_candidates(question: str, ground_truth: str, candidates: list) -> list:
    """用 MiMo 给候选回答打分"""
    scored = []
    
    for i, cand in enumerate(candidates):
        prompt = f"""You are a video understanding quality evaluator. Score the following answer.

Question: {question}
Reference answer: {ground_truth}
Answer to evaluate: {cand['response']}

Score from 1-5 on each dimension:
1. Visual understanding accuracy
2. Information completeness
3. Language fluency
4. Consistency with reference

Output ONLY a JSON object:
{{"total": X.X, "accuracy": X, "completeness": X, "fluency": X, "consistency": X}}"""

        result = call_mimo(prompt, temperature=0.1, max_tokens=200)
        
        try:
            start = result.find("{")
            end = result.rfind("}") + 1
            scores = json.loads(result[start:end])
            total = scores.get("total", 0)
            if total == 0:
                vals = [scores.get(d, 3) for d in ["accuracy", "completeness", "fluency", "consistency"]]
                total = sum(vals) / len(vals)
            scored.append({**cand, "score": total, "details": scores})
        except Exception:
            scored.append({**cand, "score": 0, "details": {}})
    
    return scored


def build_preference_pair(scored: list, min_diff: float = 0.5) -> dict:
    """从打分结果构建偏好对"""
    if len(scored) < 2:
        return None
    
    scored.sort(key=lambda x: x.get("score", 0), reverse=True)
    best = scored[0]
    worst = scored[-1]
    
    diff = best.get("score", 0) - worst.get("score", 0)
    if diff < min_diff:
        return None
    if best.get("score", 0) < 3.0:
        return None
    
    return {
        "chosen": best["response"],
        "chosen_score": best.get("score", 0),
        "rejected": worst["response"],
        "rejected_score": worst.get("score", 0),
        "score_diff": round(diff, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data", type=str, default="./data/sft_train.jsonl")
    parser.add_argument("--output", type=str, default="./data/dpo_preference_data.jsonl")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()
    
    print("=" * 60)
    print("MiMo 模式构建 DPO 数据")
    print(f"API: {MIMO_API_BASE_URL}")
    print(f"模型: {MIMO_MODEL}")
    print("=" * 60)
    
    # 加载数据
    questions = []
    with open(args.sft_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                video = sample.get("videos", [""])[0]
                question, answer = "", ""
                for conv in sample.get("conversations", []):
                    if conv["role"] == "user":
                        question = conv["content"].replace("<video>", "").strip()
                    elif conv["role"] == "assistant":
                        answer = conv["content"]
                if question and answer and question != "nan":
                    questions.append({"video": video, "question": question, "answer": answer})
    
    import random
    random.seed(42)
    if len(questions) > args.max_samples:
        questions = random.sample(questions, args.max_samples)
    
    print(f"共 {len(questions)} 个问题\n")
    
    pairs = []
    stats = {"total": len(questions), "success": 0, "skip": 0, "api_calls": 0}
    
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['question'][:50]}...")
        
        # 1. 生成候选
        candidates = generate_candidates(q["question"], q["answer"], args.num_candidates)
        stats["api_calls"] += len(candidates)
        
        if len(candidates) < 2:
            stats["skip"] += 1
            continue
        
        # 2. 打分
        scored = score_candidates(q["question"], q["answer"], candidates)
        stats["api_calls"] += len(scored)
        
        for j, s in enumerate(scored):
            print(f"  候选 {j+1}: {s.get('score', 0):.1f} 分")
        
        # 3. 构建偏好对
        pair = build_preference_pair(scored)
        if pair:
            pairs.append({
                "video": q["video"],
                "prompt": q["question"],
                "chosen": pair["chosen"],
                "chosen_score": pair["chosen_score"],
                "rejected": pair["rejected"],
                "rejected_score": pair["rejected_score"],
                "score_diff": pair["score_diff"],
            })
            stats["success"] += 1
            print(f"  -> chosen={pair['chosen_score']:.1f} rejected={pair['rejected_score']:.1f}")
        else:
            stats["skip"] += 1
            print(f"  -> 跳过（分数差异不足）")
        
        # 每 50 条保存
        if stats["success"] % 50 == 0 and stats["success"] > 0:
            with open(args.output, "w", encoding="utf-8") as f:
                for p in pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
        
        time.sleep(0.5)
    
    # 最终保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"完成！")
    print(f"  有效偏好对: {stats['success']}")
    print(f"  跳过: {stats['skip']}")
    print(f"  API 调用: {stats['api_calls']}")
    print(f"  输出: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
