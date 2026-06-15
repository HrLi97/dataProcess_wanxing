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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# third_part 根目录（包含三个子项目）
CFG_third_part   = Path("/mnt/workspace/lihaoran/data_process/third_part")

# DOVER
CFG_DOVER_CFG    = CFG_third_part / "DOVER-master" / "dover.yml"
CFG_DOVER_WEIGHT = CFG_third_part / "DOVER-master" / "pretrained_weights" / "DOVER.pth"

# Aesthetic Predictor V2.5
CFG_AES_WEIGHT   = CFG_third_part / "aesthetic-predictor-v2-5" / "models" / "aesthetic_predictor_v2_5.pth"
CFG_AES_ENCODER  = "/mnt/workspace/lihaoran/ckps/google/siglip-so400m-patch14-384/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3" 

# KVQ
CFG_KVQ_CFG      = CFG_third_part / "KVQ" / "configs" / "test.yaml"
CFG_KVQ_WEIGHT   = CFG_third_part / "KVQ" / "weights" / "KVQ.pth"

CFG_VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

# ── 模型开关（True = 启用，False = 跳过）────────────────────────────────
CFG_USE_DOVER    = True
CFG_USE_AES      = True
CFG_USE_KVQ      = True

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


def process_segment(args, item, dover_state, aes_state, kvq_state):
    """处理单个视频：质量评分。保留输入 CSV 全部列，追加质量评分结果。"""
    row = dict(item)
    row["dover_overall"] = ""
    row["aes_mean"] = ""
    row["kvq_score"] = ""
    row["quality_reject_reason"] = ""

    video_path = row.get(args.video_path_column, "")
    if not video_path or not Path(video_path).is_file():
        row["quality_reject_reason"] = "file_not_found"
        return row

    print(f"  处理: {Path(video_path).name}")

    # DOVER
    if dover_state:
        d = dover_score(video_path, dover_state)
        row["dover_overall"] = round(d["overall"], 6)

    # Aesthetic Predictor V2.5
    if aes_state:
        a = aes_v25_score(video_path, aes_state, n_frames=args.aes_frames)
        row["aes_mean"] = round(a["aes_mean"], 6)

    # KVQ
    if kvq_state:
        k = kvq_score(video_path, kvq_state)
        row["kvq_score"] = round(k["kvq_score"], 6)

    # 质量判定
    reasons = []
    if args.min_dover_overall > 0 and dover_state:
        if row["dover_overall"] and row["dover_overall"] < args.min_dover_overall:
            reasons.append(f"dover_low_{row['dover_overall']}")
    if args.min_aes_mean > 0 and aes_state:
        if row["aes_mean"] and row["aes_mean"] < args.min_aes_mean:
            reasons.append(f"aes_low_{row['aes_mean']}")
    if args.min_kvq_score > 0 and kvq_state:
        if row["kvq_score"] and row["kvq_score"] < args.min_kvq_score:
            reasons.append(f"kvq_low_{row['kvq_score']}")

    if reasons:
        row["quality_reject_reason"] = ",".join(reasons)
        print(f"  不通过: {Path(video_path).name} | {', '.join(reasons)}")
    else:
        print(f"  通过: {Path(video_path).name}")

    return row


# ─── Worker ─────────────────────────────────────────────────────────────────
class Worker:
    def __init__(self):
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
        print("质量评估模型加载完成")

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
        row = process_segment(args, item, self.dover_state, self.aes_state, self.kvq_state)
        self._append_csv(self.all_csv, row)
        if not row.get("quality_reject_reason"):
            self._append_csv(self.qual_csv, row)
        return row


# ─── CLI ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="视频质量三模型集成评估 Pipeline")
parser.add_argument("--input_csv", default="/mnt/workspace/lihaoran/data_process/test/result/0612/motion_filter_raft/data_5000_all_last/motion_filter_qualified.csv")
parser.add_argument("--output_dir", default="/mnt/workspace/lihaoran/data_process/test/result/0612/quality_filter/data_5000_all_last/")
parser.add_argument("--video_path_column", default="video_path")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

# 模型开关
parser.add_argument("--no-dover", action="store_true")
parser.add_argument("--no-aes", action="store_true")
parser.add_argument("--no-kvq", action="store_true")

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

# 质量阈值（默认 0 表示不过滤）
parser.add_argument("--min_dover_overall", type=float, default=0.3)
parser.add_argument("--min_aes_mean", type=float, default=4.2)
parser.add_argument("--min_kvq_score", type=float, default=0.008)

# Ray
parser.add_argument("--ray", action="store_true", help="启用 Ray 分布式模式")
parser.add_argument("--ray_num_gpus", type=float, default=0.5)
parser.add_argument("--ray_num_cpus", type=float, default=0)
parser.add_argument("--concurrency", type=int, default=4)
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
        ray.init(address="auto")
        samples = ray.data.read_csv(deduped_csv)
        predictions = samples.map(
            Worker,
            num_gpus=args.ray_num_gpus,
            num_cpus=args.ray_num_cpus,
            concurrency=args.concurrency,
        )
        predictions.take_all()
        print("Ray 完成")
