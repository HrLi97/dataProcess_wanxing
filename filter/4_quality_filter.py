from __future__ import annotations

import argparse
import csv
import fcntl
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import yaml

# 让 from filter.xxx import ... 可用
_FILTER_ROOT = Path(__file__).resolve().parents[1]
if str(_FILTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FILTER_ROOT))

from filter.filter_utils import project_root, ckps_root, env_or, worker_syspath_setup  # noqa: E402
from filter.obs_helper import add_obs_args, ObsIO  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# third_part 根目录（包含三个子项目）
# 优先级：env DATA_PROCESS_ROOT > 项目根/third_part
CFG_third_part   = project_root() / "third_part"

# DOVER
CFG_DOVER_CFG    = CFG_third_part / "DOVER-master" / "dover.yml"
CFG_DOVER_WEIGHT = CFG_third_part / "DOVER-master" / "pretrained_weights" / "DOVER.pth"

# Aesthetic Predictor V2.5
CFG_AES_WEIGHT   = CFG_third_part / "aesthetic-predictor-v2-5" / "models" / "aesthetic_predictor_v2_5.pth"
# AES 编码器是 HuggingFace snapshot 目录；默认放在 third_part/google 下
# 跨 server 部署时可被 env SIGLIP_ENCODER_PATH 覆盖
CFG_AES_ENCODER  = str(env_or("SIGLIP_ENCODER_PATH", ckps_root() / "google" / "siglip-so400m-patch14-384" / "models--google--siglip-so400m-patch14-384" / "snapshots" / "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"))

# KVQ
CFG_KVQ_CFG      = CFG_third_part / "KVQ" / "configs" / "test.yaml"
CFG_KVQ_WEIGHT   = CFG_third_part / "KVQ" / "weights" / "KVQ.pth"

CFG_VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

# ── 模型开关（True = 启用，False = 跳过）────────────────────────────────
CFG_USE_DOVER    = True
CFG_USE_AES      = True
CFG_USE_KVQ      = True
CFG_USE_OCR      = True

# ── OCR (PaddleOCR) 路径配置 ─────────────────────────────────────────────
# PaddleOCR 项目根（默认 third_part/PaddleOCR，可被 env PADDLEOCR_ROOT 覆盖）
CFG_PADDLEOCR_ROOT  = env_or("PADDLEOCR_ROOT",  CFG_third_part / "PaddleOCR")
# PaddleOCR 模型缓存目录（PP-OCRv4 mobile det/rec 等）
CFG_PADDLEOCR_MODEL = env_or("PADDLEOCR_MODEL_DIR", Path("/mnt/workspace/lihaoran/ckps/paddleocr"))

# ══════════════════════════════════════════════════════════════════════════


# ─── 内部路径变量（由 CFG_* 初始化，可被命令行覆盖）─────────────────────
_third_part = CFG_third_part
_DOVER_ROOT = _third_part / "DOVER-master"
_AES_ROOT   = _third_part / "aesthetic-predictor-v2-5"
_KVQ_ROOT   = _third_part / "KVQ"


def _update_roots(third_part: Optional[str]) -> None:
    """若命令行指定了 --third-part，更新内部路径变量。"""
    if third_part is None:
        return
    global _third_part, _DOVER_ROOT, _AES_ROOT, _KVQ_ROOT
    _third_part = Path(third_part).resolve()
    _DOVER_ROOT = _third_part / "DOVER-master"
    _AES_ROOT   = _third_part / "aesthetic-predictor-v2-5"
    _KVQ_ROOT   = _third_part / "KVQ"


_DEFAULT_VIDEO_EXTS = CFG_VIDEO_EXTS
VIDEO_EXTS = _DEFAULT_VIDEO_EXTS   # 运行时由 args 覆盖


# ─── 工具函数 ────────────────────────────────────────────────────────────────

def _add_path(p: Path) -> None:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)


def _sample_video_frames(video_path: str, n_frames: int = 8) -> List[np.ndarray]:
    """从视频均匀采样 n_frames 帧，返回 RGB numpy 列表。"""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    indices = [int(round((i + 0.5) * total / n_frames)) for i in range(n_frames)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(idx, total - 1)))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


# ─── DOVER ──────────────────────────────────────────────────────────────────
_DOVER_MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
_DOVER_STD  = torch.FloatTensor([58.395, 57.12, 57.375])
_T_SHIFT, _T_SCALE = 0.1107,  0.07355
_A_SHIFT, _A_SCALE = 0.08285, 0.03774
_T_W,     _A_W     = 0.6104,  0.3896


def load_dover(
    device: str,
    cfg_path: Optional[str] = None,
    weight_path: Optional[str] = None,
) -> Optional[Dict]:

    _add_path(_DOVER_ROOT)
    from dover.models import DOVER  # noqa

    cfg = Path(cfg_path) if cfg_path else _DOVER_ROOT / "dover.yml"
    with open(cfg) as f:
        opt = yaml.safe_load(f)

    model = DOVER(**opt["model"]["args"]).to(device)

    pth = Path(weight_path) if weight_path else _DOVER_ROOT / opt["test_load_path"].lstrip("./")
    if not pth.exists():
        print(f"[DOVER] 权重不存在: {pth}，跳过")
        return None
    model.load_state_dict(torch.load(str(pth), map_location=device))
    model.eval()
    print(f"[DOVER] 加载成功  ({pth.name})")
    return {"model": model, "opt": opt, "device": device}



@torch.no_grad()
def dover_score(video_path: str, state: Dict) -> Dict[str, float]:
    """返回 {"aesthetic", "technical", "overall"}，范围 [0,1]。"""
    _add_path(_DOVER_ROOT)
    from dover.datasets import spatial_temporal_view_decomposition, UnifiedFrameSampler  # noqa

    opt, device = state["opt"], state["device"]
    stypes = opt["sample_types"]

    samplers = {}
    for stype, sopt in stypes.items():
        if "t_frag" not in sopt:
            samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
            )
        else:
            samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"],
                sopt["frame_interval"],
                sopt["num_clips"],
            )

    try:
        views, _ = spatial_temporal_view_decomposition(video_path, stypes, samplers)
    except Exception as e:
        print(f"  [DOVER] 读取失败: {e}")
        return {k: float("nan") for k in ("aesthetic", "technical", "overall")}

    video = {}
    for k, v in views.items():
        num_clips = stypes[k].get("num_clips", 1)
        v = ((v.permute(1, 2, 3, 0) - _DOVER_MEAN) / _DOVER_STD).permute(3, 0, 1, 2)
        c, t, h, w = v.shape
        video[k] = (
            v.reshape(c, num_clips, t // num_clips, h, w)
             .permute(1, 0, 2, 3, 4)
             .to(device)
        )

    try:
        raw = state["model"](video, reduce_scores=False)
        t_raw = float(np.mean(raw[1].cpu().numpy()))
        a_raw = float(np.mean(raw[0].cpu().numpy()))
    except Exception as e:
        print(f"  [DOVER] 推理失败: {e}")
        return {k: float("nan") for k in ("aesthetic", "technical", "overall")}

    t = (t_raw - _T_SHIFT) / _T_SCALE
    a = (a_raw + _A_SHIFT) / _A_SCALE
    x = t * _T_W + a * _A_W
    return {
        "aesthetic": float(1 / (1 + np.exp(-a))),
        "technical": float(1 / (1 + np.exp(-t))),
        "overall":   float(1 / (1 + np.exp(-x))),
    }


# ─── Aesthetic Predictor V2.5 ────────────────────────────────────────────────
def load_aes_v25(
    device: str,
    weight_path: Optional[str] = None,
    encoder_model: str = "google/siglip-so400m-patch14-384",
) -> Optional[Dict]:
    """
    加载 Aesthetic Predictor V2.5。

    Args:
        device:        推理设备
        weight_path:   打分头 .pth 路径（默认 aesthetic-predictor-v2-5/models/aesthetic_predictor_v2_5.pth）
        encoder_model: SigLIP 底座模型名（默认 google/siglip-so400m-patch14-384）
    """
    _add_path(_AES_ROOT / "src")
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip  # noqa

    pth = Path(weight_path) if weight_path else _AES_ROOT / "models" / "aesthetic_predictor_v2_5.pth"
    pth_arg = str(pth) if pth.exists() else None
    if pth_arg is None:
        print("[AesV25] 本地头权重不存在，将从网络下载...")

    model, processor = convert_v2_5_from_siglip(
        predictor_name_or_path=pth_arg,
        encoder_model_name=encoder_model,
        low_cpu_mem_usage=True,
    )
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = model.to(dtype).to(device)
    model.eval()
    print(f"[AesV25] 加载成功  (encoder={encoder_model.split('/')[-1]})")
    return {"model": model, "processor": processor, "device": device, "dtype": dtype}



@torch.no_grad()
def aes_v25_score(video_path: str, state: Dict, n_frames: int = 8) -> Dict[str, float]:
    """
    对视频均匀采样 n_frames 帧，分别打分后聚合。
    返回 {"aes_mean", "aes_min"}，范围约 [1,10]。
    """
    from PIL import Image  # noqa

    frames = _sample_video_frames(video_path, n_frames)
    if not frames:
        return {"aes_mean": float("nan"), "aes_min": float("nan")}

    model    = state["model"]
    proc     = state["processor"]
    device   = state["device"]
    dtype    = state["dtype"]

    scores = []
    for frame in frames:
        img = Image.fromarray(frame)
        pv = proc(images=img.convert("RGB"), return_tensors="pt").pixel_values
        pv = pv.to(dtype).to(device)
        score = model(pv).logits.squeeze().float().cpu().item()
        scores.append(score)

    return {
        "aes_mean": float(np.mean(scores)),
        "aes_min":  float(np.min(scores)),
    }


# ─── KVQ ────────────────────────────────────────────────────────────────────

def load_kvq(
    device: str,
    cfg_path: Optional[str] = None,
    weight_path: Optional[str] = None,
) -> Optional[Dict]:
    """
    加载 KVQ 模型。

    Args:
        device:      推理设备
        cfg_path:    KVQ 配置文件路径（默认 KVQ/configs/test.yaml）
        weight_path: KVQ.pth 路径（默认读 yaml 的 test_load_path）
    """
    try:
        _add_path(_KVQ_ROOT)
        import vqa.models as kvq_models  # noqa
        from vqa.datasets.fusion_datasets import FragmentSampleFrames  # noqa
        from collections import OrderedDict

        cfg = Path(cfg_path) if cfg_path else _KVQ_ROOT / "configs" / "test.yaml"
        with open(cfg) as f:
            config = yaml.safe_load(f)

        pth_rel = config.get("test_load_path", "weights/KVQ.pth")
        pth = Path(weight_path) if weight_path else _KVQ_ROOT / pth_rel.lstrip("./")
        if not pth.exists():
            print(f"[KVQ] 权重不存在: {pth}，跳过")
            return None

        data_cfg = config.get("data", {})
        split_key = next(iter(data_cfg))
        sample_types = data_cfg[split_key]["args"]["sample_types"]

        model = getattr(kvq_models, config["model"]["type"])(
            **config["model"]["args"]
        ).to(device)
        try:
            ckpt = torch.load(str(pth), map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(str(pth), map_location=device)

        sd = ckpt.get("state_dict", ckpt)
        clean_sd = OrderedDict(
            (k[7:] if k.startswith("module.") else k, v) for k, v in sd.items()
        )
        model.load_state_dict(clean_sd, strict=False)
        model.eval()

        samplers = {}
        for stype, sopt in sample_types.items():
            if "t_frag" not in sopt:
                samplers[stype] = FragmentSampleFrames(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                samplers[stype] = FragmentSampleFrames(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

        print(f"[KVQ]   加载成功  ({pth.name})")
        return {
            "model": model,
            "sample_types": sample_types,
            "samplers": samplers,
            "device": device,
        }
    except Exception as e:
        print(f"[KVQ] 加载失败: {e}")
        return None


_KVQ_MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
_KVQ_STD  = torch.FloatTensor([58.395, 57.12, 57.375])
_KVQ_SAMPLE_TYPES_ORDER = ["resize", "fragments", "crop", "arp_resize", "arp_fragments"]


@torch.no_grad()
def kvq_score(video_path: str, state: Dict) -> Dict[str, float]:
    """返回 {"kvq_score"}（raw score，越高越好）。"""
    from vqa.datasets.fusion_datasets import get_spatial_and_temporal_samples  # noqa

    sample_types = state["sample_types"]
    samplers     = state["samplers"]
    device       = state["device"]

    try:
        data, _ = get_spatial_and_temporal_samples(
            str(video_path), sample_types, samplers, is_train=False, augment=False
        )
        # 归一化
        data = {
            k: ((v.permute(1, 2, 3, 0) - _KVQ_MEAN) / _KVQ_STD).permute(3, 0, 1, 2)
            for k, v in data.items()
        }
    except Exception as e:
        print(f"  [KVQ] 预处理失败: {e}")
        return {"kvq_score": float("nan")}

    video = {}
    for key in _KVQ_SAMPLE_TYPES_ORDER:
        if key not in data:
            continue
        clip = data[key].to(device).unsqueeze(0)
        b, c, t, h, w = clip.shape
        num_clips = int(sample_types[key].get("num_clips", 1))
        video[key] = (
            clip.reshape(b, c, num_clips, t // num_clips, h, w)
                .permute(0, 2, 1, 3, 4, 5)
                .reshape(b * num_clips, c, t // num_clips, h, w)
        )

    try:
        scores, *_ = state["model"](video)
        return {"kvq_score": float(scores.detach().mean().cpu().item())}
    except Exception as e:
        print(f"  [KVQ] 推理失败: {e}")
        return {"kvq_score": float("nan")}


# ─── Pipeline 主逻辑 ─────────────────────────────────────────────────────────

QUALITY_FIELDS = [
    "dover_overall", "aes_mean", "kvq_score",
    "quality_reject_reason",
]


def process_segment(args, item, worker):
    """处理单个视频：质量评分。短路求值: ocr -> dover -> aes -> kvq,
    任一指标不达标立即终止, 后续模型不再运行; 失败原因写入 quality_reject_reason。

    新增 reject reason 取值:
      - ocr_subtitle_c{consistency}    帧一致性 >= 阈值, 判定有字幕
      - ocr_subtitle_a{area_ratio}     字幕面积比超过 --max_subtitle_area_ratio
      - ocr_error                      OCR 推理异常
      - ocr_engine_missing             启用了 OCR 但引擎未初始化
    """
    row = dict(item)
    for k in QUALITY_FIELDS:
        row.setdefault(k, "")

    video_path = row.get(args.video_path_column, "")
    if not video_path or not Path(video_path).is_file():
        row["quality_reject_reason"] = "file_not_found"
        return row

    print(f"  处理: {Path(video_path).name}")
    reasons: List[str] = []

    # ── 1) OCR（最便宜，放第一位；失败立即跳过后续三模型） ──
    ocr_enabled = (not args.no_ocr) and CFG_USE_OCR
    if ocr_enabled:
        if worker.ocr_engine is None:
            reasons.append("ocr_engine_missing")
            row["quality_reject_reason"] = ",".join(reasons)
            print(f"  不通过(OCR 引擎未就绪): {Path(video_path).name}")
            return row
        try:
            from filter.filter_ocr import infer_video as ocr_infer_video
            oc = ocr_infer_video(
                worker.ocr_engine, video_path,
                num_frames=args.ocr_frames, roi=worker.ocr_roi,
            )
            row["ocr_has_subtitle"]        = bool(oc["has_subtitle"])
            row["ocr_subtitle_area_ratio"] = round(oc["subtitle_area_ratio"], 6)
            row["ocr_frame_consistency"]   = round(oc["frame_consistency"], 6)
            row["ocr_avg_text_area_ratio"] = round(oc["avg_text_area_ratio"], 6)

            if oc["has_subtitle"] and oc["frame_consistency"] >= args.min_ocr_consistency:
                reasons.append(f"ocr_subtitle_c{oc['frame_consistency']:.2f}")
            elif (args.max_subtitle_area_ratio > 0
                  and oc["subtitle_area_ratio"] > args.max_subtitle_area_ratio):
                reasons.append(f"ocr_subtitle_a{oc['subtitle_area_ratio']:.4f}")
        except Exception as e:
            print(f"  [OCR] 推理失败: {e}")
            row["ocr_has_subtitle"] = "ERR"
            reasons.append("ocr_error")

    if reasons:
        row["quality_reject_reason"] = ",".join(reasons)
        print(f"  不通过(OCR): {Path(video_path).name} | {row['quality_reject_reason']}")
        return row

    # ── 2) DOVER ──
    if worker.dover_state:
        d = dover_score(video_path, worker.dover_state)
        row["dover_overall"] = round(d["overall"], 6)
        if args.min_dover_overall > 0 and row["dover_overall"] < args.min_dover_overall:
            reasons.append(f"dover_low_{row['dover_overall']}")
            row["quality_reject_reason"] = ",".join(reasons)
            print(f"  不通过(DOVER): {Path(video_path).name}")
            return row

    # ── 3) AesV25 ──
    if worker.aes_state:
        a = aes_v25_score(video_path, worker.aes_state, n_frames=args.aes_frames)
        row["aes_mean"] = round(a["aes_mean"], 6)
        if args.min_aes_mean > 0 and row["aes_mean"] < args.min_aes_mean:
            reasons.append(f"aes_low_{row['aes_mean']}")
            row["quality_reject_reason"] = ",".join(reasons)
            print(f"  不通过(AES): {Path(video_path).name}")
            return row

    # ── 4) KVQ ──
    if worker.kvq_state:
        k = kvq_score(video_path, worker.kvq_state)
        row["kvq_score"] = round(k["kvq_score"], 6)
        if args.min_kvq_score > 0 and row["kvq_score"] < args.min_kvq_score:
            reasons.append(f"kvq_low_{row['kvq_score']}")
            row["quality_reject_reason"] = ",".join(reasons)
            print(f"  不通过(KVQ): {Path(video_path).name}")
            return row

    if reasons:
        row["quality_reject_reason"] = ",".join(reasons)
    else:
        print(f"  通过: {Path(video_path).name}")
    return row


# ─── Worker ─────────────────────────────────────────────────────────────────
class Worker:
    def __init__(self):
        worker_syspath_setup()  # Ray worker 端把 data_process/ 加进 sys.path
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_csv = self.output_dir / "4_quality_all.csv"
        self.qual_csv = self.output_dir / "4_quality_qualified.csv"
        # 读取输入 CSV 的列名
        with open(args.input_csv, encoding="utf-8") as f:
            input_fields = csv.DictReader(f).fieldnames or []
        self.fieldnames = list(input_fields) + [c for c in QUALITY_FIELDS if c not in input_fields]
        for p in (self.all_csv, self.qual_csv):
            if not p.exists():
                with open(p, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
        # 断点续跑：加载已处理的 video_path（输入行里 video_path 字段是源视频路径）
        self.processed = set()
        if self.all_csv.exists():
            with open(self.all_csv, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    vp = r.get("video_path", "")
                    if vp:
                        self.processed.add(vp)
        print(f"Worker: 已处理 {len(self.processed)} 条。")
        # 路径初始化
        _update_roots(args.third_part)
        # 加载模型
        device = args.device
        print(f"加载质量评估模型 (device={device})")
        self.dover_state = None if args.no_dover else load_dover(
            device, args.dover_cfg, args.dover_weight
        )
        self.aes_state = None if args.no_aes else load_aes_v25(
            device, args.aes_weight, args.aes_encoder
        )
        self.kvq_state = None if args.no_kvq else load_kvq(
            device, args.kvq_cfg, args.kvq_weight
        )
        # OCR 引擎（懒加载：首次 __call__ 时才创建, 避免 Ray Actor 启动期 fork Paddle 崩溃）
        self.ocr_engine = None
        self.ocr_roi    = None
        if not args.no_ocr and CFG_USE_OCR:
            if not CFG_PADDLEOCR_ROOT.exists():
                print(f"[OCR] PaddleOCR 路径不存在: {CFG_PADDLEOCR_ROOT}, 跳过")
            else:
                os.environ.setdefault("PADDLEOCR_ROOT",       str(CFG_PADDLEOCR_ROOT))
                os.environ.setdefault("PADDLEOCR_MODEL_DIR",  str(CFG_PADDLEOCR_MODEL))
                os.environ.setdefault("OMP_NUM_THREADS",      "1")
                os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
                print(f"[OCR] 已就绪(懒加载)  root={CFG_PADDLEOCR_ROOT}  model={CFG_PADDLEOCR_MODEL}")
        # OBS
        self.obs_io = ObsIO(args)
        print(f"质量评估模型加载完成. OBS={self.obs_io.enabled}")

    def _ensure_ocr_engine(self):
        """首次需要 OCR 时再创建引擎。"""
        if self.ocr_engine is not None:
            return
        from filter.filter_ocr import (
            create_ocr_engine, setup_paddle_env, get_roi, PADDLEOCR_ROOT,
        )
        setup_paddle_env()
        self.ocr_engine = create_ocr_engine(
            lang=args.ocr_lang, ocr_version=args.ocr_version, device=args.ocr_device,
        )
        self.ocr_roi = get_roi(mode=args.ocr_mode, roi_str=args.ocr_roi)
        print(f"[OCR] 引擎就绪  version={args.ocr_version} lang={args.ocr_lang} "
              f"roi={self.ocr_roi} frames={args.ocr_frames}")

    def _append_csv(self, filepath, row):
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore").writerow(row)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)

    def __call__(self, item):
        video_path = item.get(args.video_path_column, "")
        if not video_path:
            row = dict(item)
            row.update({k: "" for k in QUALITY_FIELDS})
            return row
        if video_path in self.processed:
            row = dict(item)
            row.update({k: "" for k in QUALITY_FIELDS})
            return row
        # 懒加载 OCR 引擎
        if (not args.no_ocr) and CFG_USE_OCR and self.ocr_engine is None \
                and CFG_PADDLEOCR_ROOT.exists():
            try:
                self._ensure_ocr_engine()
            except Exception as e:
                print(f"  [OCR] 引擎初始化失败: {e}")

        local_video = None
        try:
            # ── OBS: 下载视频用于质量评分 ──
            if self.obs_io.enabled:
                local_video = self.obs_io.download(video_path)
                item = dict(item)
                item[args.video_path_column] = local_video
            else:
                local_video = video_path

            row = process_segment(args, item, self)

            # ── OBS: video_path 保持为 obs_key（不产出新视频，只打分） ──
            if self.obs_io.enabled and not row.get("quality_reject_reason"):
                row["video_path"] = video_path  # 恢复为 obs_key

            self._append_csv(self.all_csv, row)
            if not row.get("quality_reject_reason"):
                self._append_csv(self.qual_csv, row)
            self.processed.add(video_path)
            return row
        finally:
            # ── OBS: 清理下载的临时文件 ──
            if self.obs_io.enabled and local_video and local_video != video_path:
                self.obs_io.cleanup(local_video)


# ─── CLI ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="视频质量三模型集成评估 Pipeline")
parser.add_argument("--input_csv", default="/mnt/workspace/lihaoran/data_process/test/result/0618/motion_filter_raft/data_5no_pepole000_all/3_motion_filter_qualified.csv")
parser.add_argument("--output_dir", default="/mnt/workspace/lihaoran/data_process/test/result/0618/quality_filter/no_pepole")
parser.add_argument("--video_path_column", default="video_path")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

# 模型开关
parser.add_argument("--no-dover", action="store_true")
parser.add_argument("--no-aes", action="store_true")
parser.add_argument("--no-kvq", action="store_true")
parser.add_argument("--no-ocr", action="store_true")

# third_part 总目录
parser.add_argument("--third-part", default=str(CFG_third_part))

# DOVER
parser.add_argument("--dover-cfg", default=str(CFG_DOVER_CFG))
parser.add_argument("--dover-weight", default=str(CFG_DOVER_WEIGHT))

# AesV2.5
parser.add_argument("--aes-weight", default=str(CFG_AES_WEIGHT))
parser.add_argument("--aes-encoder", default=CFG_AES_ENCODER)
parser.add_argument("--aes-frames", type=int, default=16)


# KVQ
parser.add_argument("--kvq-cfg", default=str(CFG_KVQ_CFG))
parser.add_argument("--kvq-weight", default=str(CFG_KVQ_WEIGHT))

# OCR (PaddleOCR)
parser.add_argument("--ocr-version", default="PP-OCRv4", choices=["PP-OCRv4", "PP-OCRv5"])
parser.add_argument("--ocr-lang",    default="ch",choices=["ch", "en"])
parser.add_argument("--ocr-device",  default="cpu",)
parser.add_argument("--ocr-frames",  type=int, default=21)
parser.add_argument("--ocr-mode",    default="full",choices=["full","subtitle","subtitle_slim","watermark_top","watermark_br","watermark_bl","center","top_half","bottom_half"])
parser.add_argument("--ocr-roi",     default=None,help="自定义 ROI: 'x1,y1,x2,y2' (比例或像素), 覆盖 --ocr-mode")
parser.add_argument("--min_ocr_consistency",       type=float, default=0.5,help="字幕帧一致性下限, >= 则视为有字幕")
parser.add_argument("--max_subtitle_area_ratio",   type=float, default=0.0,help="过滤后字幕面积均值上限, >0 时启用; 0 表示不按面积过滤")

# 质量阈值（默认 0 表示不过滤）
parser.add_argument("--min_dover_overall", type=float, default=0.4)
parser.add_argument("--min_aes_mean", type=float, default=4.2)
parser.add_argument("--min_kvq_score", type=float, default=0.008)

# OBS
add_obs_args(parser)

# Ray
parser.add_argument("--ray", action="store_true", help="启用 Ray 分布式模式")
parser.add_argument("--ray_num_gpus", type=float, default=0.2)
parser.add_argument("--ray_num_cpus", type=float, default=2)
parser.add_argument("--concurrency", type=int, default=10)
args = parser.parse_args()


if __name__ == "__main__":

    # ── 去重：按 video_path_column 去重，避免重复处理 ──
    import pandas as pd
    df = pd.read_csv(args.input_csv, encoding="utf-8")
    before_cnt = len(df)
    df = df.drop_duplicates(subset=[args.video_path_column], keep="first")
    after_cnt = len(df)
    if before_cnt != after_cnt:
        print(f"去重: {before_cnt} → {after_cnt} (去除 {before_cnt - after_cnt} 条重复)")
    deduped_csv = args.input_csv.replace(".csv", "_deduped.csv")
    df.to_csv(deduped_csv, index=False)

    if not args.ray:
        # ── 本地模式 ──
        samples = list(csv.DictReader(open(deduped_csv, encoding="utf-8")))
        print(f"本地模式，共 {len(samples)} 条")
        worker = Worker()
        for i, item in enumerate(samples, 1):
            print(f"[{i}/{len(samples)}]")
            worker(item)
    else:
        # ── Ray 分布式模式 ──
        import ray
        ray.init(address="auto",
                 runtime_env={"env_vars":
                                {"PYTHONPATH": str(_FILTER_ROOT),
                                 "DATA_PROCESS_ROOT": str(_FILTER_ROOT)},
                            }
                )
        from ray.data import ActorPoolStrategy
        samples = ray.data.read_csv(deduped_csv).materialize()
        target_parallelism = min(args.concurrency, samples.count())
        if samples.num_blocks() < target_parallelism:
            samples = samples.repartition(target_parallelism)
        predictions = samples.map(
            Worker,
            num_gpus=args.ray_num_gpus,
            num_cpus=max(args.ray_num_cpus, 1),
            compute=ActorPoolStrategy(size=target_parallelism),
        )
        predictions.take_all()
        print("Ray 完成")
