"""
Step 1: 准备 SFT 训练数据
从 ShareGPTVideo 采样数据，格式化为 Qwen3-VL 训练格式
"""

import json
import os
import random
from pathlib import Path

def download_dataset(output_dir: str = "./data"):
    """下载 ShareGPTVideo 数据集"""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("准备 SFT 数据")
    print("=" * 60)
    
    # 检查是否已下载
    cache_dir = os.path.join(output_dir, "llava_video_cache")
    if os.path.exists(cache_dir):
        print(f"数据集已存在: {cache_dir}")
        return cache_dir
    
    print("正在下载 ShareGPTVideo 数据集...")
    print("如果下载慢，可以手动下载：")
    print("  huggingface-cli download --repo-type dataset ShareGPTVideo/train_video_and_instruction")
    
    # 使用 HuggingFace datasets 库下载
    try:
        from datasets import load_dataset
        
        dataset = load_dataset(
            "ShareGPTVideo/train_video_and_instruction",
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        print(f"下载完成，共 {len(dataset)} 条数据")
        return cache_dir
    except Exception as e:
        print(f"下载失败: {e}")
        print("请手动下载数据集到:", cache_dir)
        return None


def sample_and_format(
    max_samples: int = 15000,
    output_file: str = "./data/sft_train.jsonl",
    seed: int = 42,
):
    """从数据集中采样并格式化"""
    
    print(f"\n采样 {max_samples} 条数据...")
    
    try:
        from datasets import load_dataset
        
        dataset = load_dataset(
            "ShareGPTVideo/train_video_and_instruction",
            split="train",
            cache_dir="./data/llava_video_cache",
            trust_remote_code=True,
        )
        
        # 随机采样
        random.seed(seed)
        indices = random.sample(range(len(dataset)), min(max_samples, len(dataset)))
        sampled = dataset.select(indices)
        
        # 格式化为 Qwen3-VL 训练格式
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        formatted_count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in sampled:
                try:
                    # ShareGPTVideo 的数据格式
                    video_path = item.get("video", "")
                    conversations = item.get("conversations", [])
                    
                    if not video_path or not conversations:
                        continue
                    
                    # 转换为 ShareGPT 格式
                    formatted = {
                        "videos": [video_path],
                        "conversations": [],
                    }
                    
                    for conv in conversations:
                        role = conv.get("from", "")
                        content = conv.get("value", "")
                        
                        # 映射角色
                        if role in ["human", "user"]:
                            formatted["conversations"].append({
                                "role": "user",
                                "content": content,
                            })
                        elif role in ["gpt", "assistant"]:
                            formatted["conversations"].append({
                                "role": "assistant",
                                "content": content,
                            })
                    
                    if len(formatted["conversations"]) >= 2:
                        f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                        formatted_count += 1
                        
                except Exception as e:
                    continue
        
        print(f"格式化完成，共 {formatted_count} 条有效数据")
        print(f"输出文件: {output_file}")
        
        # 统计信息
        print(f"\n数据统计:")
        print(f"  总样本数: {formatted_count}")
        print(f"  平均对话轮数: ~2")
        print(f"  输出格式: ShareGPT (role + content)")
        
        return output_file
        
    except Exception as e:
        print(f"处理失败: {e}")
        print("\n请确保已安装 datasets 库: pip install datasets")
        return None


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
        {
            "videos": ["videos/sample_003.mp4"],
            "conversations": [
                {"role": "user", "content": "<video>\n请详细描述视频中的场景和人物动作。"},
                {"role": "assistant", "content": "视频开始时，画面显示一个阳光明媚的公园。一个穿蓝色运动服的男性从画面左侧走入，开始在小径上慢跑。他跑了约30秒后，在一张长椅旁停下休息。"},
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
    parser.add_argument("--max-samples", type=int, default=15000, help="最大采样数")
    parser.add_argument("--output", type=str, default="./data/sft_train.jsonl", help="输出文件")
    parser.add_argument("--dummy", action="store_true", help="生成示例数据")
    
    args = parser.parse_args()
    
    if args.dummy:
        generate_dummy_sft_data(num_samples=200)
    else:
        sample_and_format(max_samples=args.max_samples, output_file=args.output)
