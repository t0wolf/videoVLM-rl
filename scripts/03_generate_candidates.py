"""
Step 3a: 用 SFT 模型生成候选回答
在 AutoDL 上运行，每个问题生成 4 条不同温度的回答
"""

import json
import os
import torch
import argparse


def load_model(model_path: str):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    
    print(f"加载 SFT 模型: {model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def generate_candidates(
    model,
    processor,
    questions: list,
    num_candidates: int = 4,
    temperatures: list = [0.7, 0.9, 1.0, 1.2],
    max_new_tokens: int = 256,
):
    """为每个问题生成多个候选回答"""
    
    results = []
    
    for i, q in enumerate(questions):
        video_path = q.get("video", "")
        question = q.get("question", "")
        ground_truth = q.get("answer", "")
        
        print(f"\n[{i+1}/{len(questions)}] {question[:50]}...")
        
        candidates = []
        
        for j, temp in enumerate(temperatures[:num_candidates]):
            try:
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": question}
                ]}]
                
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], videos=[video_path], return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temp,
                        do_sample=True,
                        top_p=0.9,
                    )
                
                response = processor.decode(
                    output[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                ).strip()
                
                candidates.append({
                    "response": response,
                    "temperature": temp,
                })
                print(f"  候选 {j+1} (temp={temp}): {response[:80]}...")
                
            except Exception as e:
                print(f"  候选 {j+1} 失败: {e}")
                continue
        
        if len(candidates) >= 2:
            results.append({
                "video": video_path,
                "question": question,
                "ground_truth": ground_truth,
                "candidates": candidates,
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="生成 DPO 候选回答")
    parser.add_argument("--model", type=str, default="./output/sft", help="SFT 模型路径")
    parser.add_argument("--sft-data", type=str, default="./data/sft_train.jsonl", help="SFT 数据文件")
    parser.add_argument("--output", type=str, default="./data/dpo_candidates.jsonl", help="输出文件")
    parser.add_argument("--num-candidates", type=int, default=4, help="每个问题生成几个候选")
    parser.add_argument("--max-samples", type=int, default=200, help="最大样本数")
    
    args = parser.parse_args()
    
    # 加载模型
    model, processor = load_model(args.model)
    
    # 加载 SFT 数据
    print(f"\n加载 SFT 数据: {args.sft_data}")
    questions = []
    with open(args.sft_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                video_path = sample.get("videos", [""])[0]
                question = ""
                answer = ""
                for conv in sample.get("conversations", []):
                    if conv["role"] == "user":
                        question = conv["content"].replace("<video>", "").strip()
                    elif conv["role"] == "assistant":
                        answer = conv["content"]
                
                if video_path and question and answer:
                    questions.append({
                        "video": video_path,
                        "question": question,
                        "answer": answer,
                    })
    
    # 采样
    import random
    random.seed(42)
    if len(questions) > args.max_samples:
        questions = random.sample(questions, args.max_samples)
    
    print(f"共 {len(questions)} 个问题待处理")
    
    # 生成候选回答
    results = generate_candidates(
        model, processor, questions,
        num_candidates=args.num_candidates,
    )
    
    # 保存
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n生成完成！共 {len(results)} 个问题，保存在 {args.output}")


if __name__ == "__main__":
    main()
