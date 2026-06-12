"""
视频质量评估集成 Pipeline
=========================
整合三个模型对视频进行全维度质量评分：

  1. DOVER      — 视频美学(aesthetic) + 技术(technical) + 综合(overall)，范围 [0,1]
  2. AesV2.5    — SigLIP 图像美学分，对视频均匀采帧后平均，范围约 [1,10]
  3. KVQ        — 视频质量感知分 (raw score，越高越好)

════════════════════════════════════════════════════════════════
  ★ 快速修改入口 ★
  ─ 所有路径 / 参数的默认值集中在本文件顶部的 CFG_* 区块 ─
    /xsky/yangxin/data_process_repo/filter/video_quality_pipeline.py
    第 74 行起 → "全局配置（统一修改入口）"
════════════════════════════════════════════════════════════════

─── 命令行参数（命令行优先级 > CFG_* 默认值）───────────────────────────────

  必选:
    -in  / --input   VIDEO_PATH_OR_DIR
          视频文件或目录（目录时递归搜索）

  输出:
    -out / --output  CSV
          结果 CSV 路径
          默认: /xsky/yangxin/data_process_repo/scores.csv

  设备:
    -d   / --device  cuda|cpu
          推理设备（默认 cuda，无 GPU 时自动回退 cpu）

  模型开关（默认三个全开）:
    --no-dover       跳过 DOVER
    --no-aes         跳过 Aesthetic Predictor V2.5
    --no-kvq         跳过 KVQ

  Third_part 总目录（覆盖三个子项目的根路径）:
    --third-part     DIR
          默认: /xsky/yangxin/data_process_repo/Third_part

  DOVER 路径:
    --dover-cfg      YML
          默认: /xsky/yangxin/data_process_repo/Third_part/DOVER-master/dover.yml
    --dover-weight   PTH
          默认: /xsky/yangxin/data_process_repo/Third_part/DOVER-master/pretrained_weights/DOVER.pth

  Aesthetic Predictor V2.5 参数:
    --aes-weight     PTH
          默认: /xsky/yangxin/data_process_repo/Third_part/aesthetic-predictor-v2-5/models/aesthetic_predictor_v2_5.pth
    --aes-encoder    MODEL
          默认: google/siglip-so400m-patch14-384
    --aes-frames     N
          对视频均匀采帧数（默认 8）

  KVQ 路径:
    --kvq-cfg        YML
          默认: /xsky/yangxin/data_process_repo/Third_part/KVQ/configs/test.yaml
    --kvq-weight     PTH
          默认: /xsky/yangxin/data_process_repo/Third_part/KVQ/weights/KVQ.pth

  视频格式:
    --exts           EXT...
          要处理的后缀，多个用空格分隔（默认 .avi .flv .mkv .mov .mp4 .webm）

─── 用法示例 ─────────────────────────────────────────────────────────────────

  # 批量推理目录（使用 CFG_* 默认值）
  python /xsky/yangxin/data_process_repo/filter/video_quality_pipeline.py \\
      -in  /xsky/yangnq/LM_data/zgtz_test \\
      -out /xsky/yangxin/data_process_repo/scores.csv

  # 单视频，只跑 DOVER
  python /xsky/yangxin/data_process_repo/filter/video_quality_pipeline.py \\
      -in /xsky/yangnq/LM_data/sample.mp4 \\
      --no-aes --no-kvq \\
      -out /xsky/yangxin/data_process_repo/scores.csv

  # 自定义权重 + 采帧数
  python /xsky/yangxin/data_process_repo/filter/video_quality_pipeline.py \\
      -in  /xsky/yangnq/LM_data/zgtz_test \\
      --dover-weight /xsky/yangxin/data_process_repo/Third_part/DOVER-master/pretrained_weights/DOVER.pth \\
      --kvq-weight   /xsky/yangxin/data_process_repo/Third_part/KVQ/weights/KVQ.pth \\
      --aes-frames   16 \\
      -out /xsky/yangxin/data_process_repo/scores.csv

─── CSV 列 ───────────────────────────────────────────────────────────────────
  path, dover_aesthetic, dover_technical, dover_overall, aes_mean, aes_min, kvq_score
"""

from __future__ import annotations

import argparse
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

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                         全局配置（统一修改入口）                        ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║  以下所有参数均可被同名命令行参数覆盖，命令行优先级更高                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── 路径配置 ──────────────────────────────────────────────────────────────
# Third_part 根目录（包含三个子项目）
CFG_THIRD_PART   = Path("/xsky/yangxin/data_process_repo/Third_part")

# DOVER
CFG_DOVER_CFG    = CFG_THIRD_PART / "DOVER-master" / "dover.yml"
CFG_DOVER_WEIGHT = CFG_THIRD_PART / "DOVER-master" / "pretrained_weights" / "DOVER.pth"

# Aesthetic Predictor V2.5
CFG_AES_WEIGHT   = CFG_THIRD_PART / "aesthetic-predictor-v2-5" / "models" / "aesthetic_predictor_v2_5.pth"
CFG_AES_ENCODER  = "google/siglip-so400m-patch14-384"   # SigLIP 底座模型（HF model id 或本地路径）

# KVQ
CFG_KVQ_CFG      = CFG_THIRD_PART / "KVQ" / "configs" / "test.yaml"
CFG_KVQ_WEIGHT   = CFG_THIRD_PART / "KVQ" / "weights" / "KVQ.pth"

# ── 推理参数 ──────────────────────────────────────────────────────────────
CFG_AES_FRAMES   = 8          # AesV2.5 对每个视频均匀采帧数
CFG_DEVICE       = "cuda"     # 推理设备：cuda / cpu（命令行未指定时使用）
CFG_OUTPUT       = "/xsky/yangxin/data_process_repo/scores.csv"   # 默认输出路径

# ── 视频格式 ──────────────────────────────────────────────────────────────
CFG_VIDEO_EXTS   = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

# ── 模型开关（True = 启用，False = 跳过）────────────────────────────────
CFG_USE_DOVER    = True
CFG_USE_AES      = True
CFG_USE_KVQ      = True

# ══════════════════════════════════════════════════════════════════════════


# ─── 内部路径变量（由 CFG_* 初始化，可被命令行覆盖）─────────────────────
_THIRD_PART = CFG_THIRD_PART
_DOVER_ROOT = _THIRD_PART / "DOVER-master"
_AES_ROOT   = _THIRD_PART / "aesthetic-predictor-v2-5"
_KVQ_ROOT   = _THIRD_PART / "KVQ"


def _update_roots(third_part: Optional[str]) -> None:
    """若命令行指定了 --third-part，更新内部路径变量。"""
    if third_part is None:
        return
    global _THIRD_PART, _DOVER_ROOT, _AES_ROOT, _KVQ_ROOT
    _THIRD_PART = Path(third_part).resolve()
    _DOVER_ROOT = _THIRD_PART / "DOVER-master"
    _AES_ROOT   = _THIRD_PART / "aesthetic-predictor-v2-5"
    _KVQ_ROOT   = _THIRD_PART / "KVQ"


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
    """
    加载 DOVER 模型。

    Args:
        device:      推理设备
        cfg_path:    dover.yml 路径（默认 DOVER-master/dover.yml）
        weight_path: DOVER.pth 路径（默认读 yml 的 test_load_path）
    """
    try:
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
    except Exception as e:
        print(f"[DOVER] 加载失败: {e}")
        return None


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
    try:
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
    except Exception as e:
        print(f"[AesV25] 加载失败: {e}")
        return None


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

def collect_videos(input_path: str) -> List[Path]:
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
        return [p]
    if p.is_dir():
        return sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTS)
    return []


def process_video(
    video_path: Path,
    dover_state:  Optional[Dict],
    aes_state:    Optional[Dict],
    kvq_state:    Optional[Dict],
    aes_n_frames: int = 8,
) -> Dict:
    path_str = str(video_path)
    result: Dict = {"path": path_str}

    print(f"  处理: {video_path.name}")

    # DOVER
    if dover_state:
        d = dover_score(path_str, dover_state)
        result.update({
            "dover_aesthetic": d["aesthetic"],
            "dover_technical": d["technical"],
            "dover_overall":   d["overall"],
        })
    else:
        result.update({"dover_aesthetic": float("nan"), "dover_technical": float("nan"), "dover_overall": float("nan")})

    # Aesthetic Predictor V2.5
    if aes_state:
        a = aes_v25_score(path_str, aes_state, n_frames=aes_n_frames)
        result.update({"aes_mean": a["aes_mean"], "aes_min": a["aes_min"]})
    else:
        result.update({"aes_mean": float("nan"), "aes_min": float("nan")})

    # KVQ
    if kvq_state:
        k = kvq_score(path_str, kvq_state)
        result["kvq_score"] = k["kvq_score"]
    else:
        result["kvq_score"] = float("nan")

    return result


def save_csv(results: List[Dict], output_path: str) -> None:
    columns = [
        "path",
        "dover_aesthetic", "dover_technical", "dover_overall",
        "aes_mean", "aes_min",
        "kvq_score",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")
        for r in results:
            row = []
            for col in columns:
                v = r.get(col, float("nan"))
                if isinstance(v, float):
                    row.append(f"{v:.6f}" if not (v != v) else "nan")  # nan check
                else:
                    row.append(str(v).replace(",", " "))
            f.write(",".join(row) + "\n")
    print(f"\nCSV 已保存到: {output_path}")


def print_summary(results: List[Dict]) -> None:
    if not results:
        return
    print(f"\n{'='*60}")
    print(f"推理完成，共 {len(results)} 个视频")
    print(f"{'='*60}")
    cols = ["dover_aesthetic", "dover_technical", "dover_overall", "aes_mean", "kvq_score"]
    for col in cols:
        vals = [r[col] for r in results if r.get(col) == r.get(col)]  # filter nan
        if vals:
            print(f"  {col:25s}: 均值={np.mean(vals):.4f}, 最小={np.min(vals):.4f}, 最大={np.max(vals):.4f}")


# ─── 入口 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="视频质量三模型集成评估 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── 必选 ──
    parser.add_argument("-in", "--input",  required=True, metavar="PATH",
                        help="视频文件或目录（目录时递归搜索）")

    # ── 输出 ──
    parser.add_argument("-out", "--output", default=CFG_OUTPUT,
                        metavar="CSV", help=f"输出 CSV 路径（默认 {CFG_OUTPUT}）")

    # ── 设备 ──
    parser.add_argument("-d", "--device",
                        default=CFG_DEVICE if torch.cuda.is_available() else "cpu",
                        help=f"推理设备 cuda|cpu（默认 {CFG_DEVICE}，无 GPU 时自动回退 cpu）")

    # ── 模型开关 ──
    tog = parser.add_argument_group("模型开关（默认全开）")
    tog.add_argument("--no-dover", action="store_true",
                     help=f"跳过 DOVER（默认启用={CFG_USE_DOVER}）")
    tog.add_argument("--no-aes",   action="store_true",
                     help=f"跳过 Aesthetic Predictor V2.5（默认启用={CFG_USE_AES}）")
    tog.add_argument("--no-kvq",   action="store_true",
                     help=f"跳过 KVQ（默认启用={CFG_USE_KVQ}）")

    # ── Third_part 总目录（优先级最高）──
    parser.add_argument("--third-part", default=None, metavar="DIR",
                        help=(f"Third_part 目录的绝对路径（默认 {CFG_THIRD_PART}）\n"
                               "指定后 DOVER-master / aesthetic-predictor-v2-5 / KVQ 均从此目录下寻找，\n"
                               "可被各模型具体的 --dover-weight 等参数覆盖"))

    # ── DOVER 参数 ──
    dg = parser.add_argument_group("DOVER 路径（可选，有默认值）")
    dg.add_argument("--dover-cfg",    default=None, metavar="YML",
                    help=f"DOVER 配置文件（默认 {CFG_DOVER_CFG}）")
    dg.add_argument("--dover-weight", default=None, metavar="PTH",
                    help=f"DOVER 权重文件（默认 {CFG_DOVER_WEIGHT}）")

    # ── AesV2.5 参数 ──
    ag = parser.add_argument_group("Aesthetic Predictor V2.5 参数")
    ag.add_argument("--aes-weight",  default=None, metavar="PTH",
                    help=f"打分头权重（默认 {CFG_AES_WEIGHT}）")
    ag.add_argument("--aes-encoder", default=CFG_AES_ENCODER,
                    metavar="MODEL",
                    help=f"SigLIP 底座模型名称（默认 {CFG_AES_ENCODER}）")
    ag.add_argument("--aes-frames",  type=int, default=CFG_AES_FRAMES, metavar="N",
                    help=f"对视频均匀采帧数（默认 {CFG_AES_FRAMES}）")

    # ── KVQ 参数 ──
    kg = parser.add_argument_group("KVQ 路径（可选，有默认值）")
    kg.add_argument("--kvq-cfg",    default=None, metavar="YML",
                    help=f"KVQ 配置文件（默认 {CFG_KVQ_CFG}）")
    kg.add_argument("--kvq-weight", default=None, metavar="PTH",
                    help=f"KVQ 权重文件（默认 {CFG_KVQ_WEIGHT}）")

    # ── 视频格式 ──
    parser.add_argument("--exts", nargs="+",
                        default=sorted(CFG_VIDEO_EXTS),
                        metavar="EXT",
                        help=f"要处理的视频后缀（默认 {' '.join(sorted(CFG_VIDEO_EXTS))}）")

    args = parser.parse_args()

    # — 路径初始化：--third-part 优先于模块级默认值 —
    _update_roots(args.third_part)

    # 更新全局视频扩展名集合
    global VIDEO_EXTS
    VIDEO_EXTS = {e if e.startswith(".") else f".{e}" for e in args.exts}

    print(f"{'='*60}")
    print(f"设备       : {args.device}")
    print(f"输入       : {args.input}")
    print(f"输出       : {args.output}")
    print(f"Third_part : {_THIRD_PART}")
    print(f"视频格式   : {' '.join(sorted(VIDEO_EXTS))}")
    print(f"{'='*60}")

    # 加载模型
    dover_state = None if args.no_dover else load_dover(
        args.device, args.dover_cfg, args.dover_weight
    )
    aes_state = None if args.no_aes else load_aes_v25(
        args.device, args.aes_weight, args.aes_encoder
    )
    kvq_state = None if args.no_kvq else load_kvq(
        args.device, args.kvq_cfg, args.kvq_weight
    )

    if not any([dover_state, aes_state, kvq_state]):
        print("[ERROR] 所有模型均加载失败，退出")
        sys.exit(1)

    print(f"{'='*60}")

    # 收集视频
    videos = collect_videos(args.input)
    if not videos:
        print(f"[ERROR] 未找到视频文件: {args.input}")
        sys.exit(1)
    print(f"找到 {len(videos)} 个视频\n")

    # 逐视频推理
    results = []
    for i, vp in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}]")
        r = process_video(vp, dover_state, aes_state, kvq_state, args.aes_frames)
        results.append(r)

    # 保存
    save_csv(results, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()
