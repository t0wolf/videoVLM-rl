"""
Step 3b: 用 MiMo API 给候选回答打分
在本地运行，不需要 GPU
"""

import json
import os
import time
import urllib.request
import argparse

MIMO_API_KEY = "tp-chvb2livf8qw5873xt1hij288n3o8tfgc3j8qzmzcn7vhfoo"
MIMO_API_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5"


def call_mimo(prompt, temperature=0.1, max_tokens=512):
    """调用 MiMo API，兼容推理模型的 reasoning_content"""
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
                headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                msg = result["choices"][0]["message"]
                content = msg.get("content", "").strip()
                reasoning = msg.get("reasoning_content", "").strip()
                # 推理模型：优先用 content，空则用 reasoning
                return content if content else reasoning
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"    API 失败: {e}")
                return ""


def score_candidate(question, ground_truth, response):
    """给单个回答打分"""
    prompt = f"""Score this video QA answer from 1-5 on each dimension.

Question: {question}
Reference answer: {ground_truth}
Answer to evaluate: {response}

Score dimensions:
1. Accuracy (1-5): Is the answer factually correct?
2. Completeness (1-5): Does it cover key information?
3. Fluency (1-5): Is it well-written and natural?
4. Consistency (1-5): How consistent with reference?

Output ONLY this JSON:
{{"accuracy": X, "completeness": X, "fluency": X, "consistency": X, "total": X.X}}"""

    result = call_mimo(prompt, temperature=0.1, max_tokens=200)
    try:
        start = result.find("{")
        end = result.rfind("}") + 1
        scores = json.loads(result[start:end])
        total = scores.get("total", 0)
        if total == 0:
            vals = [scores.get(d, 3) for d in ["accuracy", "completeness", "fluency", "consistency"]]
            total = sum(vals) / len(vals)
        return total, scores
    except Exception:
        return 0, {}


def score_candidates(input_file, output_file, max_samples=100):
    """给候选回答打分"""
    print("=" * 60)
    print(f"MiMo 打分 | {MIMO_MODEL}")
    print("=" * 60)
    
    candidates = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    
    if len(candidates) > max_samples:
        import random
        random.seed(42)
        candidates = random.sample(candidates, max_samples)
    
    print(f"共 {len(candidates)} 个问题\n")
    
    results = []
    success = 0
    
    for i, item in enumerate(candidates):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        cand_list = item.get("candidates", [])
        
        print(f"[{i+1}/{len(candidates)}] {question[:50]}...")
        
        scored_candidates = []
        for j, cand in enumerate(cand_list):
            total, details = score_candidate(question, ground_truth, cand["response"])
            scored_candidates.append({**cand, "score": total, "details": details})
            print(f"  [{j+1}] {total:.1f}")
            time.sleep(0.3)
        
        if len(scored_candidates) >= 2:
            results.append({
                "question": question, "ground_truth": ground_truth,
                "video": item.get("video", ""), "scored_candidates": scored_candidates
            })
            success += 1
        
        if success % 50 == 0 and success > 0:
            with open(output_file, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"成功: {success}/{len(candidates)}")
    print(f"输出: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./data/dpo_candidates.jsonl")
    parser.add_argument("--output", type=str, default="./data/dpo_scored.jsonl")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()
    
    score_candidates(args.input, args.output, args.max_samples)
