"""
基于 VQA² (ACM MM 2025) 的视频质量评分脚本
模型: q-future/VQA-UGC-Scorer-llava_qwen  (HuggingFace)
原理: LLaVA-OneVision + SlowFast 运动编码器, 用 5 级质量 token (high/good/fair/poor/low)
     的 logits 做加权平均得到 0~1 质量分.

Usage:
    # 单视频测试
    python video_quality_vqa2.py --video /path/to/video.mp4

    # CSV 批量 (需含 file_path 列)
    python video_quality_vqa2.py --input_csv /path/to/input.csv --output_csv /path/to/result.csv

    # 指定模型路径 (本地已下载)
    python video_quality_vqa2.py --video /path/to/video.mp4 --model_path /local/VQA-UGC-Scorer

依赖:  pip install decord pytorchvideo transformers==4.44.0 scipy pillow
       需预先下载 slowfast.pth: https://huggingface.co/JZHWS/slowfast
"""

import os
import sys
import csv
import json
import argparse
import logging
import numpy as np
from collections import defaultdict
from PIL import Image

VQA2_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "third_part", "VQA2", "quality_scoring")
)
sys.path.insert(0, VQA2_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 质量 token id (Qwen2 tokenizer) ──
QUALITY_TOKEN_IDS = {
    "high": 1550,
    "good": 1661,
    "fair": 6624,
    "poor": 7852,
    "low":  3347,
}
QUALITY_WEIGHTS = np.array([1.0, 0.8, 0.6, 0.4, 0.2])


def wa5(logits_dict):
    """5-level weighted average quality score in [0, 1]."""
    logprobs = np.array([logits_dict[k] for k in ("high", "good", "fair", "poor", "low")])
    probs = np.exp(logprobs) / np.sum(np.exp(logprobs))
    return float(np.inner(probs, QUALITY_WEIGHTS))


def load_video_frames(video_path, sample_fps=1):
    """用 decord 读视频, 返回 (all_frames_PIL, keyframe_indices)."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    avg_fps = round(vr.get_avg_fps())
    if avg_fps <= 0:
        avg_fps = 24

    keyframe_idx = []
    n_secs = total // avg_fps
    for s in range(n_secs):
        step = round(avg_fps / sample_fps)
        keyframe_idx.extend(range(s * avg_fps, (s + 1) * avg_fps, step))
    # 剩余帧
    step = round(avg_fps / sample_fps)
    keyframe_idx.extend(range(n_secs * avg_fps, total, step))
    keyframe_idx = [i for i in keyframe_idx if i < total]

    frames = vr.get_batch(list(range(total))).asnumpy()
    pil_frames = [Image.fromarray(frames[i]) for i in range(total)]
    return pil_frames, keyframe_idx


def load_model(model_path):
    """加载 VQA² 模型, 返回 (tokenizer, model, image_processor)."""
    import torch
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import get_model_name_from_path

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, None, model_name, attn_implementation=None
    )
    model.half()
    model.eval()
    return tokenizer, model, image_processor


def preprocess_qwen(sources, tokenizer, has_image=False, system_message="You are a helpful assistant."):
    """构造 Qwen chat 格式的 input_ids."""
    import torch
    import re
    from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

    im_start, im_end = tokenizer.additional_special_tokens_ids
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens

    input_id = []
    system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
    input_id += system

    for j, sentence in enumerate(sources):
        role = "<|im_start|>user" if j == 0 else "<|im_start|>assistant"
        if has_image and isinstance(sentence, str) and "<image>" in sentence:
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence))
            texts = sentence.split("<image>")
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i, text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i < len(texts) - 1:
                    _input_id += [IMAGE_TOKEN_INDEX]
            _input_id += [im_end] + nl_tokens
        else:
            val = sentence.get("value", "") if isinstance(sentence, dict) else str(sentence)
            if val:
                _input_id = tokenizer(role).input_ids + nl_tokens + tokenizer(val).input_ids + [im_end] + nl_tokens
            else:
                _input_id = tokenizer(role).input_ids + nl_tokens
        input_id += _input_id

    return torch.tensor([input_id], dtype=torch.long)


def score_video(video_path, tokenizer, model, image_processor):
    """对单个视频打质量分, 返回 float score ∈ [0, 1]."""
    import torch
    from llava.constants import DEFAULT_IMAGE_TOKEN

    pil_frames, keyframe_idx = load_video_frames(video_path, sample_fps=1)
    if len(pil_frames) == 0:
        logger.warning(f"视频帧数为 0: {video_path}")
        return None

    prompt = DEFAULT_IMAGE_TOKEN + DEFAULT_IMAGE_TOKEN
    input_ids = preprocess_qwen(
        [prompt, {"from": "gpt", "value": ""}], tokenizer, has_image=True
    ).cuda()

    # spatial tensor: 取前 N//4*4 帧做 SlowFast 输入
    n4 = len(pil_frames) // 4 * 4
    if n4 == 0:
        n4 = len(pil_frames)
    spatial_tensor = image_processor.preprocess(pil_frames[:n4], return_tensors="pt")["pixel_values"]

    # temporal tensor: 关键帧
    if len(keyframe_idx) == 0:
        keyframe_idx = [0]
    temporal_tensor = image_processor.preprocess(
        [pil_frames[i] for i in keyframe_idx if i < len(pil_frames)], return_tensors="pt"
    )["pixel_values"]

    image_tensors = [
        [spatial_tensor[:spatial_tensor.shape[0] // 4 * 4].half().cuda()],
        [temporal_tensor.half().cuda()],
    ]

    with torch.inference_mode():
        output_logits = model(input_ids, images=image_tensors)["logits"]

    logits_at = output_logits[:, -3]
    quality_logits = {}
    for name, tid in QUALITY_TOKEN_IDS.items():
        quality_logits[name] = logits_at.mean(0)[tid].item()

    return wa5(quality_logits)


def main():
    parser = argparse.ArgumentParser(description="VQA² 视频质量评分")
    parser.add_argument("--video", type=str, default="", help="单个视频路径")
    parser.add_argument("--input_csv", type=str, default="", help="批量 CSV (需含 file_path 列)")
    parser.add_argument("--output_csv", type=str, default="", help="输出 CSV")
    parser.add_argument("--model_path", type=str, default="q-future/VQA-UGC-Scorer-llava_qwen",
                        help="HuggingFace 模型路径或本地路径")
    args = parser.parse_args()

    if not args.video and not args.input_csv:
        parser.error("请指定 --video 或 --input_csv")

    logger.info(f"加载模型: {args.model_path}")
    tokenizer, model, image_processor = load_model(args.model_path)
    logger.info("模型加载完成")

    # ── 单视频模式 ──
    if args.video:
        score = score_video(args.video, tokenizer, model, image_processor)
        logger.info(f"视频: {args.video}")
        logger.info(f"质量分: {score:.4f}  (1.0=最高, 0.2=最低)")
        return

    # ── CSV 批量模式 ──
    output_csv = args.output_csv or args.input_csv.replace(".csv", "_quality.csv")
    with open(args.input_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = []
    for i, row in enumerate(rows):
        vpath = row["file_path"]
        try:
            score = score_video(vpath, tokenizer, model, image_processor)
            row["vqa2_quality_score"] = f"{score:.4f}" if score is not None else ""
            logger.info(f"[{i+1}/{len(rows)}] {os.path.basename(vpath)} → {score:.4f}")
        except Exception as e:
            row["vqa2_quality_score"] = ""
            logger.error(f"[{i+1}/{len(rows)}] {os.path.basename(vpath)} 失败: {e}")
        results.append(row)

    fieldnames = list(results[0].keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"结果写入: {output_csv}")


if __name__ == "__main__":
    main()
