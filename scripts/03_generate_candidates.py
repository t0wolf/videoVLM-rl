"""
Step 3a: 用 SFT 模型生成候选回答
使用 OpenCV 读取视频帧，绕过 torchcodec 问题
"""

import json
import os
import torch
import cv2
import numpy as np
from PIL import Image
import argparse


def load_model(model_path):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import PeftModel
    
    print(f"加载 SFT 模型: {model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    # 检查是否有 LoRA 适配器
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        print("加载 LoRA 适配器...")
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            "./ckpts/Qwen3-VL-8B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def read_frames(video_path, num_frames=8):
    """用 OpenCV 读取视频帧"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    cap.release()
    return frames


def generate_one(model, processor, video_path, question, temperature, max_new_tokens=256):
    """生成一条回答"""
    try:
        frames = read_frames(video_path, num_frames=8)
        
        messages = [{"role": "user", "content": []}]
        for f in frames:
            messages[0]["content"].append({"type": "image", "image": f})
        messages[0]["content"].append({"type": "text", "text": question})
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=frames, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                temperature=temperature, do_sample=True, top_p=0.9
            )
        
        response = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return response
    except Exception as e:
        print(f"    生成失败: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./output/sft")
    parser.add_argument("--sft-data", type=str, default="./data/sft_train.jsonl")
    parser.add_argument("--output", type=str, default="./data/dpo_candidates.jsonl")
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    model, processor = load_model(args.model)

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

    temperatures = [0.5, 0.7, 0.9, 1.1]
    results = []

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['question'][:50]}...")
        
        candidates = []
        for j, temp in enumerate(temperatures[:args.num_candidates]):
            response = generate_one(model, processor, q["video"], q["question"], temp)
            if response:
                candidates.append({"response": response, "temperature": temp})
                print(f"  [{j+1}] {response[:60]}...")
        
        if len(candidates) >= 2:
            results.append({
                "video": q["video"], "question": q["question"],
                "ground_truth": q["answer"], "candidates": candidates
            })
        
        if (i + 1) % 100 == 0:
            with open(args.output, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"\n完成！{len(results)} 条候选 | {args.output}")


if __name__ == "__main__":
    main()
