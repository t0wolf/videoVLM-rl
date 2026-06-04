"""
Step 1: 准备 SFT 训练数据
从 VideoChatGPT 本地 parquet 文件读取，格式化为 Qwen3-VL 训练格式
"""

import json
import os
import random
from pathlib import Path


def load_parquet_data(data_dir: str = "./data/VideoChatGPT"):
    """从本地 parquet 文件加载数据"""
    import pandas as pd
    
    all_data = []
    
    # 读取三个子集
    for subset in ["Generic", "Consistency", "Temporal"]:
        parquet_file = os.path.join(data_dir, subset, "test-00000-of-00001.parquet")
        if os.path.exists(parquet_file):
            df = pd.read_parquet(parquet_file)
            print(f"  {subset}: {len(df)} 条")
            all_data.append(df)
        else:
            print(f"  {subset}: 文件不存在，跳过")
    
    # 合并
    import pandas as pd
    combined = pd.concat(all_data, ignore_index=True)
    print(f"  总计: {len(combined)} 条")
    
    return combined


def format_to_sft(
    data_dir: str = "./data/VideoChatGPT",
    video_dir: str = "./data/VideoChatGPT/Test_Videos",
    output_file: str = "./data/sft_train.jsonl",
    max_samples: int = 5000,
    seed: int = 42,
):
    """将 parquet 数据转换为 SFT 训练格式"""
    
    print("=" * 60)
    print("准备 SFT 数据")
    print("=" * 60)
    
    # 加载数据
    print("\n加载 parquet 数据...")
    df = load_parquet_data(data_dir)
    
    # 随机采样
    random.seed(seed)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
        print(f"\n随机采样 {max_samples} 条")
    
    # 格式化
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    formatted_count = 0
    skipped = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            video_name = row.get("video_name", "")
            question = row.get("question", "")
            answer = row.get("answer", "")
            
            # 构建视频路径
            video_path = os.path.join(video_dir, f"{video_name}.mp4")
            
            # 检查视频文件是否存在
            if not os.path.exists(video_path):
                skipped += 1
                continue
            
            if not question or not answer:
                skipped += 1
                continue
            
            # 转换为 ShareGPT 格式
            formatted = {
                "videos": [video_path],
                "conversations": [
                    {"role": "user", "content": f"<video>\n{question}"},
                    {"role": "assistant", "content": answer},
                ],
            }
            
            f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
            formatted_count += 1
    
    print(f"\n格式化完成！")
    print(f"  有效数据: {formatted_count} 条")
    print(f"  跳过: {skipped} 条（视频不存在或数据缺失）")
    print(f"  输出文件: {output_file}")
    
    # 打印示例
    print(f"\n示例数据:")
    with open(output_file, "r", encoding="utf-8") as f:
        example = json.loads(f.readline())
        print(json.dumps(example, ensure_ascii=False, indent=2))
    
    return output_file


def generate_dummy_sft_data(num_samples: int = 100, output_file: str = "./data/sft_train_dummy.jsonl"):
    """生成示例 SFT 数据（用于测试脚本）"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    examples = [
        {
            "videos": ["videos/sample_001.mp4"],
            "conversations": [
                {"role": "user", "content": "<video>\n请描述这段视频的内容。"},
                {"role": "assistant", "content": "视频中，一个人在厨房里做饭。他先从冰箱拿出蔬菜，然后在案板上切菜，最后放入锅中翻炒。"},
            ],
        },
        {
            "videos": ["videos/sample_002.mp4"],
            "conversations": [
                {"role": "user", "content": "<video>\n视频中发生了什么？"},
                {"role": "assistant", "content": "视频显示一个人在公园里跑步，他穿着运动服，沿着小径慢跑。"},
            ],
        },
    ]
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(num_samples):
            example = examples[i % len(examples)]
            example["videos"] = [f"videos/sample_{i:04d}.mp4"]
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"示例 SFT 数据已生成: {output_file}（{num_samples} 条）")
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="准备 SFT 训练数据")
    parser.add_argument("--data-dir", type=str, default="./data/VideoChatGPT", help="数据集目录")
    parser.add_argument("--video-dir", type=str, default="./data/VideoChatGPT/Test_Videos", help="视频目录")
    parser.add_argument("--output", type=str, default="./data/sft_train.jsonl", help="输出文件")
    parser.add_argument("--max-samples", type=int, default=5000, help="最大采样数")
    parser.add_argument("--dummy", action="store_true", help="生成示例数据")
    
    args = parser.parse_args()
    
    if args.dummy:
        generate_dummy_sft_data(num_samples=200)
    else:
        format_to_sft(
            data_dir=args.data_dir,
            video_dir=args.video_dir,
            output_file=args.output,
            max_samples=args.max_samples,
        )
