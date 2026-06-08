"""
Step 3c: 从打分结果构建 DPO 偏好对
"""

import json
import os
import argparse


def build_preference_pairs(scored_file, output_file, min_score_diff=0.5, min_total_score=3.0):
    print("=" * 60)
    print("构建 DPO 偏好对")
    print("=" * 60)
    
    samples = []
    with open(scored_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"共 {len(samples)} 个样本")
    
    pairs = []
    skipped = 0
    
    for item in samples:
        question = item["question"]
        ground_truth = item.get("ground_truth", "")
        video = item.get("video", "")
        scored = item.get("scored_candidates", [])
        
        if len(scored) < 2:
            skipped += 1
            continue
        
        scored.sort(key=lambda x: x.get("score", 0), reverse=True)
        best, worst = scored[0], scored[-1]
        diff = best.get("score", 0) - worst.get("score", 0)
        
        if diff < min_score_diff or best.get("score", 0) < min_total_score:
            skipped += 1
            continue
        
        pairs.append({
            "video_path": video, "prompt": question,
            "chosen": best["response"], "chosen_score": best.get("score", 0),
            "rejected": worst["response"], "rejected_score": worst.get("score", 0),
            "score_diff": round(diff, 2),
        })
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n有效偏好对: {len(pairs)} | 跳过: {skipped}")
    print(f"输出: {output_file}")
    
    if pairs:
        avg_diff = sum(p["score_diff"] for p in pairs) / len(pairs)
        avg_chosen = sum(p["chosen_score"] for p in pairs) / len(pairs)
        avg_rejected = sum(p["rejected_score"] for p in pairs) / len(pairs)
        print(f"平均分数差: {avg_diff:.2f} | chosen: {avg_chosen:.2f} | rejected: {avg_rejected:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./data/dpo_scored.jsonl")
    parser.add_argument("--output", type=str, default="./data/dpo_preference_data.jsonl")
    parser.add_argument("--min-score-diff", type=float, default=0.5)
    parser.add_argument("--min-total-score", type=float, default=3.0)
    args = parser.parse_args()
    
    build_preference_pairs(args.input, args.output, args.min_score_diff, args.min_total_score)
