#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PaddleOCR 文字检测 & 识别推理脚本
=================================

功能:
  1. 对视频均匀采样 N 帧，检测每帧中的文字区域和内容
  2. 计算文字面积占比 (text_area_ratio)，用于判断是否有字幕/水印
  3. 支持批量处理目录下的视频，结果输出到 CSV
  4. 支持预设模式(subtitle/watermark等)和自定义ROI区域

依赖:
  pip install paddlepaddle paddleocr decord numpy

用法:
  # 单个视频推理
  python paddleocr_infer.py --input /path/to/video.mp4

  # 批量处理目录
  python paddleocr_infer.py --input /path/to/video_dir --output results.csv

  # 使用预设模式 —— 只检测字幕区域
  python paddleocr_infer.py --input video.mp4 --mode subtitle

  # 使用预设模式 —— 只检测右下角水印
  python paddleocr_infer.py --input video.mp4 --mode watermark_br

  # 自定义ROI坐标 (比例模式，优先级高于--mode)
  python paddleocr_infer.py --input video.mp4 --roi "0.1,0.85,0.9,0.95"

预设模式 (--mode):
  full          全图识别 (默认)
  subtitle      底部25%区域 — 字幕区
  subtitle_slim 底部15%居中 — 窄字幕
  watermark_top 顶部15%区域 — 顶部水印/台标
  watermark_br  右下角区域 — 右下水印
  watermark_bl  左下角区域 — 左下水印
  center        中心60%区域 — 画面主体
  top_half      上半部分
  bottom_half   下半部分

输出 CSV 列:
  video_path, num_frames, avg_text_area_ratio, max_text_area_ratio, avg_text_count, texts_sample
"""

import os
import sys
import argparse
import csv
import warnings
from pathlib import Path

# ============================================================
# 路径配置 (绝对路径，暴露在顶部方便修改)
# ============================================================
# PaddleOCR 项目根目录（环境变量优先, 默认走 third_part/PaddleOCR）
PADDLEOCR_ROOT = os.environ.get(
    "PADDLEOCR_ROOT",
    str(Path(__file__).resolve().parents[1] / "third_part" / "PaddleOCR"),
)

# PaddleOCR 模型目录（env_ckps.sh 里 PADDLEOCR_MODEL_DIR=$CKPS_ROOT/paddlex/official_models），
# 布局 <MODEL_CACHE_DIR>/<模型名>/inference.yml，create_ocr_engine 会把 det/rec 模型目录显式传给 PaddleOCR
MODEL_CACHE_DIR = os.environ.get(
    "PADDLEOCR_MODEL_DIR",
    str(Path("/mnt/workspace/lihaoran/ckps/paddleocr")),
)

# (ocr_version, lang) → (检测模型名, 识别模型名)，与 PaddleOCR 3.x PaddleOCR._get_ocr_model_names 一致
OCR_MODEL_NAMES = {
    ("PP-OCRv4", "ch"): ("PP-OCRv4_mobile_det", "PP-OCRv4_mobile_rec"),
    ("PP-OCRv4", "en"): ("PP-OCRv4_mobile_det", "en_PP-OCRv4_mobile_rec"),
    ("PP-OCRv5", "ch"): ("PP-OCRv5_server_det", "PP-OCRv5_server_rec"),
    ("PP-OCRv5", "en"): ("PP-OCRv5_server_det", "en_PP-OCRv5_mobile_rec"),
}

# ============================================================
# 推理参数配置
# ============================================================
CFG_OCR_VERSION = "PP-OCRv4"          # 模型版本: PP-OCRv4 / PP-OCRv5
CFG_LANG = "ch"                       # 语言: ch(中英) / en(仅英)
CFG_NUM_FRAMES = 21                    # 每个视频采样帧数
CFG_DET_BOX_THRESH = 0.5              # 文字检测框置信度阈值
CFG_REC_SCORE_THRESH = 0.5            # 文字识别置信度阈值
CFG_DET_LIMIT_SIDE_LEN = 960          # 检测时图像最长边限制
CFG_ENABLE_MKLDNN = False             # CPU推理时是否启用MKL-DNN(3.3.1有bug，建议False)
CFG_DEVICE = "cuda"                    # 推理设备: cpu / cuda[:N] / gpu[:N]（cuda 会转成 Paddle 的 gpu）

# 视频文件后缀
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".webm", ".MP4", ".MOV"}

# ============================================================
# 字幕判定参数 (减少误杀的关键配置)
# ============================================================
# 有效文字最短长度：低于此长度的识别结果视为噪声(logo/标签)而非字幕
CFG_MIN_TEXT_LEN = 3
# 检测框最小宽度占比：宽度/图像宽度 < 此值的框视为非字幕
CFG_MIN_BOX_WIDTH_RATIO = 0.15
# 帧间一致性：至少这个比例的帧检测到有效文字才判定为有字幕
CFG_FRAME_CONSISTENCY_RATIO = 0.5
# 有效文字框的识别置信度下限
CFG_MIN_REC_CONFIDENCE = 0.6

# ============================================================
# ROI 预设模式 (比例坐标 x1,y1,x2,y2, 左上角为原点)
# 可通过 --mode 参数直接选用，也可用 --roi 自定义坐标
# ============================================================
ROI_PRESETS = {
    "full":          None,                   # 全图识别
    "subtitle":      (0.0, 0.75, 1.0, 1.0),  # 底部25% — 字幕区
    "subtitle_slim": (0.1, 0.85, 0.9, 1.0),  # 底部15%居中 — 窄字幕
    "watermark_top": (0.0, 0.0, 1.0, 0.15),  # 顶部15% — 顶部水印/台标
    "watermark_br":  (0.7, 0.85, 1.0, 1.0),  # 右下角 — 水印常见位置
    "watermark_bl":  (0.0, 0.85, 0.3, 1.0),  # 左下角 — 水印常见位置
    "center":        (0.1, 0.2, 0.9, 0.8),   # 中心区域60% — 画面主体
    "top_half":      (0.0, 0.0, 1.0, 0.5),   # 上半部分
    "bottom_half":   (0.0, 0.5, 1.0, 1.0),   # 下半部分
}


# ============================================================
# 核心函数
# ============================================================

def setup_paddle_env():
    """设置PaddlePaddle环境变量，避免oneDNN兼容性问题"""
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    import paddle
    paddle.set_flags({"FLAGS_use_mkldnn": False})


def to_paddle_device(device):
    """torch 风格设备名转 Paddle 风格：cuda / cuda:N → gpu / gpu:N；cpu、gpu:N 等原样返回。"""
    device = (device or "").strip()
    if device.startswith("cuda"):
        return "gpu" + device[len("cuda"):]
    return device


def create_ocr_engine(lang=CFG_LANG, ocr_version=CFG_OCR_VERSION, device=CFG_DEVICE):
    """创建PaddleOCR推理引擎

    Args:
        lang: 语言
        ocr_version: OCR版本
        device: 推理设备（cpu / cuda[:N] / gpu[:N]）

    Returns:
        PaddleOCR 实例
    """
    from paddleocr import PaddleOCR

    kwargs = dict(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=CFG_ENABLE_MKLDNN,
        text_det_box_thresh=CFG_DET_BOX_THRESH,
        text_rec_score_thresh=CFG_REC_SCORE_THRESH,
        text_det_limit_side_len=CFG_DET_LIMIT_SIDE_LEN,
    )
    # 本地有对应模型时显式传 模型名+目录：PaddleX 要求模型名与目录内 inference.yml 的 model_name 一致，
    # 且传了模型名/目录后 PaddleOCR 会忽略 lang/ocr_version；本地没有时沿用 lang/ocr_version，由 PaddleX 读缓存或下载
    det_name, rec_name = OCR_MODEL_NAMES.get((ocr_version, lang), (None, None))
    det_dir = os.path.join(MODEL_CACHE_DIR, det_name) if det_name else ""
    rec_dir = os.path.join(MODEL_CACHE_DIR, rec_name) if rec_name else ""
    if os.path.isdir(det_dir) and os.path.isdir(rec_dir):
        kwargs.update(
            text_detection_model_name=det_name, text_detection_model_dir=det_dir,
            text_recognition_model_name=rec_name, text_recognition_model_dir=rec_dir,
        )
    else:
        print(f"[OCR] {MODEL_CACHE_DIR} 下没有 {det_name}/{rec_name}，按 lang={lang} ocr_version={ocr_version} 由 PaddleX 解析模型")
        kwargs.update(lang=lang, ocr_version=ocr_version)
    # PaddleX 只认 cpu / gpu[:N] 等（cuda 会触发 assert），原来只透传 gpu*，cuda / cpu 都被丢掉、落到 PaddleOCR 默认设备
    device = to_paddle_device(device)
    if device:
        kwargs["device"] = device

    ocr = PaddleOCR(**kwargs)
    return ocr


def sample_frames_from_video(video_path: str, num_frames: int = CFG_NUM_FRAMES):
    """从视频中均匀采样帧

    Args:
        video_path: 视频文件路径
        num_frames: 采样帧数

    Returns:
        list of numpy arrays (RGB, H x W x 3)
    """
    import decord
    decord.bridge.set_bridge("native")

    try:
        vr = decord.VideoReader(video_path)
    except Exception as e:
        print(f"[WARN] 无法读取视频 {video_path}: {e}")
        return []

    total = len(vr)
    if total == 0:
        return []

    # 均匀采样
    if total <= num_frames:
        indices = list(range(total))
    else:
        indices = [int(i * total / num_frames) for i in range(num_frames)]

    frames = []
    for idx in indices:
        frame = vr[idx].asnumpy()  # RGB, H x W x 3
        frames.append(frame)

    return frames


def get_roi(mode=None, roi_str=None):
    """根据模式名或自定义坐标获取ROI

    优先级: roi_str > mode > None(全图)

    Args:
        mode: 预设模式名 (见 ROI_PRESETS)
        roi_str: 自定义坐标 "x1,y1,x2,y2"

    Returns:
        tuple (x1,y1,x2,y2) 或 None
    """
    if roi_str:
        return parse_roi(roi_str)
    if mode:
        if mode not in ROI_PRESETS:
            avail = ", ".join(ROI_PRESETS.keys())
            raise ValueError(f"未知模式 '{mode}'，可选: {avail}")
        return ROI_PRESETS[mode]
    return None


def parse_roi(roi_str):
    """解析ROI区域字符串

    支持两种格式:
      - 比例模式 (0~1): "0.0,0.8,1.0,1.0" 表示画面底部20%区域
      - 像素模式 (>1):  "0,800,1920,1080" 表示绝对像素坐标

    Args:
        roi_str: "x1,y1,x2,y2" 格式字符串

    Returns:
        tuple: (x1, y1, x2, y2)
    """
    parts = [float(x.strip()) for x in roi_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"ROI格式错误，需要4个值: x1,y1,x2,y2，实际得到 {len(parts)} 个")
    x1, y1, x2, y2 = parts
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"ROI坐标无效: 需要 x1<x2 且 y1<y2, 得到 ({x1},{y1},{x2},{y2})")
    return (x1, y1, x2, y2)


def crop_frame_roi(frame, roi):
    """根据ROI裁剪帧

    Args:
        frame: numpy array (H, W, 3)
        roi: (x1, y1, x2, y2) 比例(0~1)或像素坐标

    Returns:
        裁剪后的 numpy array
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi

    # 判断是比例还是像素 (所有值<=1视为比例)
    if all(v <= 1.0 for v in roi):
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    else:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # 裁剪边界保护
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    return frame[y1:y2, x1:x2]


def compute_text_area_ratio(polys, img_shape):
    """计算文字区域面积占图像面积的比例

    Args:
        polys: list of polygons, each shape (4, 2)
        img_shape: (H, W, C) or (H, W)

    Returns:
        float: 文字面积占总面积比例 [0, 1]
    """
    import numpy as np

    h, w = img_shape[0], img_shape[1]
    total_area = h * w
    if total_area == 0 or len(polys) == 0:
        return 0.0

    text_area = 0.0
    for poly in polys:
        poly = np.array(poly, dtype=np.float32)
        # Shoelace 公式计算多边形面积
        n = len(poly)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += poly[i][0] * poly[j][1]
            area -= poly[j][0] * poly[i][1]
        text_area += abs(area) / 2.0

    return min(text_area / total_area, 1.0)


def filter_subtitle_boxes(polys, texts, scores, img_shape):
    """过滤出"疑似字幕"的检测框，排除场景文字/噪声

    过滤逻辑:
      1. 识别置信度 >= CFG_MIN_REC_CONFIDENCE
      2. 文字长度 >= CFG_MIN_TEXT_LEN (排除logo、短标签)
      3. 检测框宽度 >= 图像宽度 * CFG_MIN_BOX_WIDTH_RATIO (排除窄小文字)

    Args:
        polys: list of (4,2) polygons
        texts: list of str
        scores: list of float
        img_shape: (H, W, C)

    Returns:
        (filtered_polys, filtered_texts, filtered_scores)
    """
    import numpy as np

    w = img_shape[1]
    min_box_w = w * CFG_MIN_BOX_WIDTH_RATIO

    f_polys, f_texts, f_scores = [], [], []
    for poly, text, score in zip(polys, texts, scores):
        # 置信度过滤
        if score < CFG_MIN_REC_CONFIDENCE:
            continue
        # 文字长度过滤 (去空格后)
        if len(text.strip()) < CFG_MIN_TEXT_LEN:
            continue
        # 检测框宽度过滤
        poly_arr = np.array(poly, dtype=np.float32)
        box_w = poly_arr[:, 0].max() - poly_arr[:, 0].min()
        if box_w < min_box_w:
            continue
        f_polys.append(poly)
        f_texts.append(text)
        f_scores.append(score)

    return f_polys, f_texts, f_scores


def infer_single_frame(ocr, frame, roi=None):
    """对单帧进行OCR推理

    Args:
        ocr: PaddleOCR 实例
        frame: numpy array (H, W, 3)
        roi: (x1, y1, x2, y2) 指定识别区域，None表示全图

    Returns:
        dict: {
            'text_area_ratio': float,       # 原始全部文字面积占比
            'text_count': int,              # 原始检测框数
            'texts': list[str],             # 原始识别文字
            'scores': list[float],          # 原始置信度
            'subtitle_area_ratio': float,   # 过滤后疑似字幕面积占比
            'subtitle_count': int,          # 过滤后疑似字幕框数
            'subtitle_texts': list[str],    # 过滤后疑似字幕文字
            'has_text': bool,               # 该帧是否有疑似字幕
        }
    """
    import numpy as np

    if roi is not None:
        frame = crop_frame_roi(frame, roi)

    result = ocr.predict(frame)
    empty = {
        "text_area_ratio": 0.0, "text_count": 0, "texts": [], "scores": [],
        "subtitle_area_ratio": 0.0, "subtitle_count": 0,
        "subtitle_texts": [], "has_text": False,
    }
    if not result:
        return empty

    r = result[0]
    polys = r.get("dt_polys", [])
    texts = r.get("rec_texts", [])
    scores = r.get("rec_scores", [])

    text_area_ratio = compute_text_area_ratio(polys, frame.shape)

    # 过滤出疑似字幕
    sub_polys, sub_texts, sub_scores = filter_subtitle_boxes(
        polys, texts, scores, frame.shape
    )
    subtitle_area_ratio = compute_text_area_ratio(sub_polys, frame.shape)

    return {
        "text_area_ratio": text_area_ratio,
        "text_count": len(polys),
        "texts": texts,
        "scores": scores,
        "subtitle_area_ratio": subtitle_area_ratio,
        "subtitle_count": len(sub_polys),
        "subtitle_texts": sub_texts,
        "has_text": len(sub_polys) > 0,
    }


def infer_video(ocr, video_path: str, num_frames: int = CFG_NUM_FRAMES, roi=None):
    """对视频进行OCR推理，并判定是否有字幕

    判定逻辑 (has_subtitle):
      1. 对每帧进行过滤: 去掉短文字/窄框/低置信度
      2. 计算帧间一致性: 有多少比例的帧检测到了"疑似字幕"
      3. 一致性 >= CFG_FRAME_CONSISTENCY_RATIO 才判定为有字幕

    Args:
        ocr: PaddleOCR 实例
        video_path: 视频路径
        num_frames: 采样帧数
        roi: (x1, y1, x2, y2) 指定识别区域，None表示全图

    Returns:
        dict 包含原始指标和字幕判定结果
    """
    frames = sample_frames_from_video(video_path, num_frames)
    empty = {
        "video_path": video_path, "num_frames": 0,
        "avg_text_area_ratio": 0.0, "max_text_area_ratio": 0.0,
        "avg_text_count": 0.0,
        "subtitle_area_ratio": 0.0, "frame_consistency": 0.0,
        "has_subtitle": False, "texts_sample": "",
    }
    if not frames:
        return empty

    all_ratios = []
    all_counts = []
    all_texts = []
    # 字幕相关
    sub_ratios = []
    frames_with_subtitle = 0
    sub_texts_all = []

    for frame in frames:
        res = infer_single_frame(ocr, frame, roi=roi)
        all_ratios.append(res["text_area_ratio"])
        all_counts.append(res["text_count"])
        all_texts.extend(res["texts"])
        # 字幕过滤后指标
        sub_ratios.append(res["subtitle_area_ratio"])
        if res["has_text"]:
            frames_with_subtitle += 1
        sub_texts_all.extend(res["subtitle_texts"])

    n = len(frames)
    # 帧间一致性 = 检测到字幕的帧数 / 总帧数
    frame_consistency = frames_with_subtitle / n
    # 过滤后的平均字幕面积
    avg_sub_ratio = sum(sub_ratios) / n
    # 判定: 一致性达标才算有字幕
    has_subtitle = frame_consistency >= CFG_FRAME_CONSISTENCY_RATIO

    unique_texts = list(dict.fromkeys(all_texts))[:10]
    unique_sub_texts = list(dict.fromkeys(sub_texts_all))[:10]

    return {
        "video_path": video_path,
        "num_frames": n,
        "avg_text_area_ratio": sum(all_ratios) / n,
        "max_text_area_ratio": max(all_ratios),
        "avg_text_count": sum(all_counts) / n,
        "subtitle_area_ratio": avg_sub_ratio,
        "frame_consistency": frame_consistency,
        "has_subtitle": has_subtitle,
        "texts_sample": " | ".join(unique_sub_texts if unique_sub_texts else unique_texts),
    }


def get_video_list(input_path: str):
    """获取视频文件列表

    Args:
        input_path: 视频文件或目录路径

    Returns:
        list of video file paths
    """
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        videos = []
        for f in sorted(p.iterdir()):
            if f.suffix in VIDEO_EXTENSIONS:
                videos.append(str(f))
        return videos
    else:
        print(f"[ERROR] 路径不存在: {input_path}")
        return []


# ============================================================
# 命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PaddleOCR 视频文字检测推理")
    parser.add_argument("--input", "-i", required=True,
                        help="输入视频文件或目录路径")
    parser.add_argument("--output", "-o", default=None,
                        help="输出CSV路径 (默认: 打印到终端)")
    parser.add_argument("--num_frames", "-n", type=int, default=CFG_NUM_FRAMES,
                        help=f"每个视频采样帧数 (默认: {CFG_NUM_FRAMES})")
    parser.add_argument("--ocr_version", default=CFG_OCR_VERSION,
                        help=f"OCR模型版本 (默认: {CFG_OCR_VERSION})")
    parser.add_argument("--lang", default=CFG_LANG,
                        help=f"语言 (默认: {CFG_LANG})")
    parser.add_argument("--device", default=CFG_DEVICE,
                        help=f"推理设备 (默认: {CFG_DEVICE})")
    parser.add_argument("--mode", "-m", default=None,
                        choices=list(ROI_PRESETS.keys()),
                        help="预设识别模式: "
                             "subtitle(字幕区), subtitle_slim(窄字幕), "
                             "watermark_top(顶部水印), watermark_br(右下水印), "
                             "watermark_bl(左下水印), center(中心), "
                             "top_half(上半), bottom_half(下半), full(全图)")
    parser.add_argument("--roi", default=None,
                        help="自定义识别区域: 'x1,y1,x2,y2' (比例0~1或像素坐标). "
                             "优先级高于 --mode. "
                             "例: '0,0.8,1,1' 底部20%%, '0,0,1,0.15' 顶部15%%")
    args = parser.parse_args()

    # 解析ROI (--roi 优先于 --mode)
    roi = get_roi(mode=args.mode, roi_str=args.roi)

    # 设置环境
    setup_paddle_env()

    # 获取视频列表
    videos = get_video_list(args.input)
    if not videos:
        print("[ERROR] 未找到视频文件")
        sys.exit(1)

    num_frames = args.num_frames
    print(f"[INFO] 找到 {len(videos)} 个视频文件")
    print(f"[INFO] 模型版本: {args.ocr_version}, 语言: {args.lang}, 采样帧数: {num_frames}")
    if roi:
        print(f"[INFO] 指定识别区域 ROI: {roi}")

    # 创建OCR引擎
    ocr = create_ocr_engine(lang=args.lang, ocr_version=args.ocr_version, device=args.device)
    print("[INFO] OCR引擎创建成功")

    # 推理
    results = []
    for i, vpath in enumerate(videos):
        print(f"[{i+1}/{len(videos)}] 处理: {os.path.basename(vpath)}")
        res = infer_video(ocr, vpath, num_frames, roi=roi)
        results.append(res)
        tag = "[字幕]" if res['has_subtitle'] else "[正常]"
        print(f"  -> {tag} consistency={res['frame_consistency']:.0%}, "
              f"sub_ratio={res['subtitle_area_ratio']:.4f}, "
              f"raw_ratio={res['avg_text_area_ratio']:.4f}")

    # 统计
    subtitle_count = sum(1 for r in results if r["has_subtitle"])
    print(f"\n[INFO] 判定结果: {subtitle_count}/{len(results)} 个视频有字幕")

    # 输出结果
    if args.output:
        fieldnames = ["video_path", "has_subtitle", "frame_consistency",
                      "subtitle_area_ratio", "avg_text_area_ratio",
                      "max_text_area_ratio", "avg_text_count",
                      "num_frames", "texts_sample"]
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"[INFO] 结果已保存到: {args.output}")
    else:
        # 打印到终端
        print("\n" + "=" * 100)
        print(f"{'视频':<38} {'判定':>6} {'一致性':>6} {'字幕ratio':>10} {'原始ratio':>10} {'文字样例'}")
        print("-" * 100)
        for r in results:
            name = os.path.basename(r["video_path"])[:36]
            tag = "字幕" if r["has_subtitle"] else "正常"
            print(f"{name:<38} {tag:>6} {r['frame_consistency']:>6.0%} "
                  f"{r['subtitle_area_ratio']:>10.4f} "
                  f"{r['avg_text_area_ratio']:>10.4f} "
                  f"{r['texts_sample'][:30]}")
        print("=" * 100)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()