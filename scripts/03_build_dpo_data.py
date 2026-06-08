"""
Step 3c: 从打分结果构建 DPO 偏好对
在本地运行，筛选 chosen/rejected 对
"""

import json
import os
import argparse


def build_preference_pairs(
    scored_file: str,
    output_file: str,
    min_score_diff: float = 0.5,
    min_total_score: float = 3.0,
):
    """从打分结果构建偏好对"""
    
    print("=" * 60)
    print("构建 DPO 偏好对")
    print("=" * 60)
    
    # 加载打分结果
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
        
        # 按分数排序
        scored.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        best = scored[0]
        worst = scored[-1]
        
        score_diff = best.get("score", 0) - worst.get("score", 0)
        
        if score_diff < min_score_diff:
            skipped += 1
            continue
        
        if best.get("score", 0) < min_total_score:
            skipped += 1
            continue
        
        pairs.append({
            "video_path": video,
            "prompt": question,
            "chosen": best["response"],
            "chosen_score": best.get("score", 0),
            "rejected": worst["response"],
            "rejected_score": worst.get("score", 0),
            "score_diff": round(score_diff, 2),
        })
    
    # 保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"\n完成！")
    print(f"  有效偏好对: {len(pairs)}")
    print(f"  跳过: {skipped}")
    print(f"  输出文件: {output_file}")
    
    # 统计
    if pairs:
        avg_diff = sum(p["score_diff"] for p in pairs) / len(pairs)
        avg_chosen = sum(p["chosen_score"] for p in pairs) / len(pairs)
        avg_rejected = sum(p["rejected_score"] for p in pairs) / len(pairs)
        print(f"\n统计:")
        print(f"  平均分数差: {avg_diff:.2f}")
        print(f"  平均 chosen 分数: {avg_chosen:.2f}")
        print(f"  平均 rejected 分数: {avg_rejected:.2f}")
    
    return len(pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 DPO 偏好对")
    parser.add_argument("--input", type=str, default="./data/dpo_scored.jsonl", help="打分结果文件")
    parser.add_argument("--output", type=str, default="./data/dpo_preference_data.jsonl", help="输出文件")
    parser.add_argument("--min-score-diff", type=float, default=0.5, help="最小分数差异")
    parser.add_argument("--min-total-score", type=float, default=3.0, help="最低总分")
    
    args = parser.parse_args()
    
    build_preference_pairs(
        scored_file=args.input,
        output_file=args.output,
        min_score_diff=args.min_score_diff,
        min_total_score=args.min_total_score,
    )
