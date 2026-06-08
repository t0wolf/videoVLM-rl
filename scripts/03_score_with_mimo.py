"""
Step 3b: 用 MiMo API 给候选回答打分
在本地运行，不需要 GPU
"""

import json
import os
import time
import argparse


# MiMo API 配置
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", "tp-chvb2livf8qw5873xt1hij288n3o8tfgc3j8qzmzcn7vhfoo")
MIMO_API_BASE_URL = os.environ.get("MIMO_API_BASE_URL", "https://token-plan-cn.xiaomimimo.com/v1")
MIMO_MODEL = os.environ.get("MIMO_MODEL", "mimo-v2.5-pro")

SCORING_PROMPT = """你是一个视频理解质量评估专家。请严格评估以下回答的质量。

## 问题
{question}

## 参考答案
{ground_truth}

## 待评估回答
{response}

## 评分标准
请从以下 4 个维度分别打分（1-5 分）：

1. **视觉理解准确性**（1-5分）：回答是否正确描述了视频中的视觉内容？
   - 5分：完全准确
   - 3分：基本准确
   - 1分：严重错误

2. **信息完整性**（1-5分）：回答是否涵盖了问题的关键信息？
   - 5分：信息全面
   - 3分：覆盖主要内容
   - 1分：严重遗漏

3. **语言流畅性**（1-5分）：回答是否通顺自然？
   - 5分：流畅自然
   - 3分：基本通顺
   - 1分：语句不通

4. **与参考答案一致性**（1-5分）：回答与参考答案的一致程度？
   - 5分：高度一致
   - 3分：基本一致
   - 1分：完全不一致

## 输出格式
请严格按以下 JSON 格式输出，不要输出其他内容：
{{
  "视觉理解准确性": {{"score": X, "reason": "..."}},
  "信息完整性": {{"score": X, "reason": "..."}},
  "语言流畅性": {{"score": X, "reason": "..."}},
  "与参考答案一致性": {{"score": X, "reason": "..."}},
  "total": X.X
}}"""


def call_mimo_scoring(question: str, ground_truth: str, response: str) -> dict:
    """调用 MiMo API 打分"""
    import urllib.request
    
    prompt = SCORING_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        response=response,
    )
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    url = f"{MIMO_API_BASE_URL}/chat/completions"
    payload = {
        "model": MIMO_MODEL,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 512,
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


def score_candidates(input_file: str, output_file: str, max_samples: int = 100):
    """给候选回答打分"""
    
    print("=" * 60)
    print("MiMo 打分")
    print(f"API: {MIMO_API_BASE_URL}")
    print(f"模型: {MIMO_MODEL}")
    print("=" * 60)
    
    # 加载候选回答
    candidates = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                candidates.append(json.loads(line))
    
    if len(candidates) > max_samples:
        import random
        random.seed(42)
        candidates = random.sample(candidates, max_samples)
    
    print(f"共 {len(candidates)} 个问题待打分\n")
    
    results = []
    success = 0
    
    for i, item in enumerate(candidates):
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        cand_list = item.get("candidates", [])
        
        print(f"[{i+1}/{len(candidates)}] {question[:50]}...")
        
        scored_candidates = []
        
        for j, cand in enumerate(cand_list):
            response = cand["response"]
            temp = cand.get("temperature", 0)
            
            score = call_mimo_scoring(question, ground_truth, response)
            
            if score is not None:
                total = score.get("total", 0)
                if total == 0:
                    scores = [score[d]["score"] for d in ["视觉理解准确性", "信息完整性", "语言流畅性", "与参考答案一致性"] if d in score]
                    total = sum(scores) / len(scores) if scores else 0
                
                scored_candidates.append({
                    "response": response,
                    "temperature": temp,
                    "score": total,
                    "details": score,
                })
                print(f"  候选 {j+1}: {total:.1f} 分")
            
            time.sleep(0.3)
        
        if len(scored_candidates) >= 2:
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "video": item.get("video", ""),
                "scored_candidates": scored_candidates,
            })
            success += 1
        
        # 每 50 条保存一次
        if success % 50 == 0 and success > 0:
            with open(output_file, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  已保存 {success} 条\n")
    
    # 最终保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*60}")
    print(f"打分完成！成功: {success}/{len(candidates)}")
    print(f"输出文件: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiMo 打分")
    parser.add_argument("--input", type=str, default="./data/dpo_candidates.jsonl", help="候选回答文件")
    parser.add_argument("--output", type=str, default="./data/dpo_scored.jsonl", help="输出文件")
    parser.add_argument("--max-samples", type=int, default=100, help="最大样本数")
    parser.add_argument("--api-key", type=str, default=None, help="MiMo API Key")
    parser.add_argument("--api-url", type=str, default=None, help="MiMo API URL")
    
    args = parser.parse_args()
    
    if args.api_key:
        MIMO_API_KEY = args.api_key
    if args.api_url:
        MIMO_API_BASE_URL = args.api_url
    
    score_candidates(args.input, args.output, args.max_samples)
