"""
Step 3: 全 MiMo 模式构建 DPO 数据
MiMo 生成候选回答 + MiMo 打分，不需要本地模型
"""

import json
import os
import time
import urllib.request
import argparse

MIMO_API_KEY = "tp-chvb2livf8qw5873xt1hij288n3o8tfgc3j8qzmzcn7vhfoo"
MIMO_API_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5"


def call_mimo(prompt, temperature=0.7, max_tokens=1024):
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
                content = msg.get("content", "")
                reasoning = msg.get("reasoning_content", "")
                # MiMo 推理模型：优先用 content，如果空则用 reasoning
                return content.strip() if content.strip() else reasoning.strip()
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"      API 失败: {e}")
                return ""


def generate_candidates(question, ground_truth, num_candidates=4):
    prompt = f"""Based on the following video context, answer the question.

Video context: {ground_truth}

Question: {question}

Provide a detailed answer:"""

    temperatures = [0.5, 0.7, 0.9, 1.1]
    candidates = []
    for temp in temperatures[:num_candidates]:
        response = call_mimo(prompt, temperature=temp)
        if response:
            candidates.append({"response": response, "temperature": temp})
    return candidates


def score_candidate(question, ground_truth, response):
    prompt = f"""Score this video QA answer from 1-5.

Question: {question}
Reference: {ground_truth}
Answer: {response}

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-data", type=str, default="./data/sft_train.jsonl")
    parser.add_argument("--output", type=str, default="./data/dpo_preference_data.jsonl")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    print("=" * 60)
    print(f"MiMo DPO 数据构建 | {MIMO_MODEL}")
    print("=" * 60)

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
    success, skip = 0, 0

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['question'][:50]}...")

        # 1. 生成候选
        candidates = generate_candidates(q["question"], q["answer"], args.num_candidates)
        if len(candidates) < 2:
            skip += 1
            continue

        # 2. 打分
        scored = []
        for j, cand in enumerate(candidates):
            total, details = score_candidate(q["question"], q["answer"], cand["response"])
            scored.append({**cand, "score": total, "details": details})
            print(f"  候选 {j+1}: {total:.1f} 分")

        # 3. 构建偏好对
        scored.sort(key=lambda x: x.get("score", 0), reverse=True)
        best, worst = scored[0], scored[-1]
        diff = best.get("score", 0) - worst.get("score", 0)

        if diff >= 0.5 and best.get("score", 0) >= 3.0:
            pairs.append({
                "video": q["video"],
                "prompt": q["question"],
                "chosen": best["response"],
                "chosen_score": best["score"],
                "rejected": worst["response"],
                "rejected_score": worst["score"],
                "score_diff": round(diff, 2),
            })
            success += 1
            print(f"  -> chosen={best['score']:.1f} rejected={worst['score']:.1f}")
        else:
            skip += 1
            print(f"  -> 跳过")

        if success % 50 == 0 and success > 0:
            with open(args.output, "w", encoding="utf-8") as f:
                for p in pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")

        time.sleep(0.3)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"偏好对: {success} | 跳过: {skip}")
    print(f"输出: {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
