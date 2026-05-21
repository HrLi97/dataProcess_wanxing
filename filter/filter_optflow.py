"""
基于光流的视频动态性检测脚本 (支持 Ray 分布式 + 本地模式)

方法:
  1. MEMFOF (推荐, GPU, ICCV 2025 Highlight, 多帧光流, Full HD 原生支持)
     - pip install git+https://github.com/msu-video-group/memfof
     - 或使用本项目 submodule: third_part/memfof
  2. torchvision RAFT (GPU, 经典稳定)
     - 基于 torchvision.models.optical_flow.raft_large / raft_small
  3. OpenCV Farneback (CPU 降级方案, 无需 GPU)

输出指标:
  - motion_score:  帧间光流 magnitude 的平均值 (越大=运动越剧烈)
  - motion_std:    帧间光流 magnitude 的标准差 (越大=运动变化越大)
  - dynamic_ratio: magnitude > threshold 的像素占比 (越大=动态区域越多)

Usage:
    # 单视频 (默认 MEMFOF)
    python motion_dynamics_optflow.py --video /path/to/video.mp4

    # 使用 RAFT
    python motion_dynamics_optflow.py --video /path/to/video.mp4 --method raft

    # CSV 批量
    python motion_dynamics_optflow.py --input_csv /path/to/input.csv --output_csv /path/to/result.csv

    # 使用 OpenCV 方案 (无需 GPU)
    python motion_dynamics_optflow.py --video /path/to/video.mp4 --method opencv

    # Ray 分布式
    python motion_dynamics_optflow.py --input_csv input.csv --output_csv out.csv --is_local False --concurrency 8

依赖:  pip install torch torchvision decord numpy opencv-python-headless
MEMFOF: pip install git+https://github.com/msu-video-group/memfof  (或用本项目 submodule)
"""

import os
import sys
import csv
import json
import time
import argparse
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def sample_frame_pairs(total_frames: int, fps: float, sample_fps: float = 2.0,
                       max_pairs: int = 60) -> List[Tuple[int, int]]:
    """按 sample_fps 采样相邻帧对, 最多 max_pairs 对."""
    if total_frames < 2:
        return []
    step = max(1, round(fps / sample_fps))
    indices = list(range(0, total_frames, step))
    if len(indices) < 2:
        indices = [0, total_frames - 1]
    pairs = [(indices[i], indices[i + 1]) for i in range(len(indices) - 1)]
    if len(pairs) > max_pairs:
        sel = np.linspace(0, len(pairs) - 1, max_pairs, dtype=int)
        pairs = [pairs[i] for i in sel]
    return pairs


def load_video_info(video_path: str) -> Tuple[int, float]:
    """只读取视频元信息 (总帧数, fps), 不加载帧数据."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    return len(vr), vr.get_avg_fps()


def load_frames_by_indices(video_path: str, indices: List[int]) -> np.ndarray:
    """只加载指定帧, 大幅减少内存占用."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    frames = vr.get_batch(indices).asnumpy()
    return frames


# ────────────────────── 方法 1: MEMFOF (ICCV 2025 Highlight) ──────────────────────
MEMFOF_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "third_part", "memfof")
)


class MEMFOFFlowEstimator:
    """MEMFOF: 多帧光流, Full HD 省显存"""

    DEFAULT_LOCAL_PATH = "/mnt/workspace/lihaoran/ckps/MEMFOF-Tartan-T-TSKH"
    DEFAULT_HF_NAME = "egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH"

    def __init__(self, model_path=None, device="cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("MEMFOF 需要 GPU, 当前 CUDA 不可用。请使用 --method opencv 或 --method raft")
        # 优先从 submodule 导入, 否则从 pip 安装的包导入
        if os.path.isdir(os.path.join(MEMFOF_ROOT, "memfof")):
            sys.path.insert(0, MEMFOF_ROOT)
        from memfof import MEMFOF
        self.device = torch.device(device)
        # 优先级: 指定路径 > 本地 ckps > HuggingFace 在线
        if model_path is None:
            if os.path.isdir(self.DEFAULT_LOCAL_PATH):
                model_path = self.DEFAULT_LOCAL_PATH
            else:
                model_path = self.DEFAULT_HF_NAME
        logger.info(f"MEMFOF 加载模型: {model_path} -> {self.device}")
        t0 = time.time()
        self.model = MEMFOF.from_pretrained(model_path).eval().to(self.device)
        logger.info(f"MEMFOF 模型加载完成, 耗时 {time.time()-t0:.1f}s")
        self._torch = torch

    def estimate_triplet(self, frame1: np.ndarray, frame2: np.ndarray, frame3: np.ndarray):
        """3 帧输入, 返回 (backward_mag, forward_mag), 各为 [H, W] float32."""
        import torch
        # MEMFOF 输入: [B=1, T=3, C=3, H, W] uint8
        triplet = np.stack([frame1, frame2, frame3], axis=0)  # [3, H, W, 3]
        tensor = torch.from_numpy(triplet).permute(0, 3, 1, 2).unsqueeze(0).to(self.device)  # [1,3,3,H,W]
        with torch.inference_mode():
            result = self.model(tensor)
            bw_flow, fw_flow = result["flow"][-1].unbind(dim=1)  # each [1,2,H,W]
        bw_np = bw_flow[0].cpu().numpy()
        fw_np = fw_flow[0].cpu().numpy()
        bw_mag = np.sqrt(bw_np[0] ** 2 + bw_np[1] ** 2)
        fw_mag = np.sqrt(fw_np[0] ** 2 + fw_np[1] ** 2)
        return bw_mag, fw_mag

    def estimate(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """双帧接口 (兼容), 用 frame1 作为前后文构造 3 帧, 返回 forward magnitude."""
        _, fw_mag = self.estimate_triplet(frame1, frame1, frame2)
        return fw_mag


# ────────────────────── 方法 2: torchvision RAFT ──────────────────────
class RAFTFlowEstimator:
    """使用 torchvision 内置 RAFT 模型计算光流."""

    def __init__(self, model_size="large", device="cuda"):
        import torch
        import torchvision.models.optical_flow as of

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if model_size == "large":
            self.model = of.raft_large(weights=of.Raft_Large_Weights.DEFAULT).to(self.device)
        else:
            self.model = of.raft_small(weights=of.Raft_Small_Weights.DEFAULT).to(self.device)
        self.model.eval()
        self.transforms = of.Raft_Large_Weights.DEFAULT.transforms() if model_size == "large" \
            else of.Raft_Small_Weights.DEFAULT.transforms()

    def estimate(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """返回光流 magnitude map [H, W], float32."""
        import torch
        f1 = torch.from_numpy(frame1).permute(2, 0, 1).float().unsqueeze(0)
        f2 = torch.from_numpy(frame2).permute(2, 0, 1).float().unsqueeze(0)
        f1, f2 = self.transforms(f1, f2)
        f1, f2 = f1.to(self.device), f2.to(self.device)
        with torch.inference_mode():
            flow = self.model(f1, f2)[-1]  # 取最后一次迭代
        flow_np = flow[0].cpu().numpy()  # [2, H, W]
        magnitude = np.sqrt(flow_np[0] ** 2 + flow_np[1] ** 2)
        return magnitude


# ────────────────────── 方法 3: OpenCV Farneback ──────────────────────
class FarnebackFlowEstimator:
    """OpenCV Farneback 密集光流, CPU 方案."""

    def estimate(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        import cv2
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return magnitude


EMPTY_METRICS = {"motion_score": 0.0, "motion_std": 0.0, "dynamic_ratio": 0.0, "max_motion": 0.0}


def _resize_frame(frame: np.ndarray, resize_short: int) -> np.ndarray:
    import cv2
    h, w = frame.shape[:2]
    if min(h, w) > resize_short:
        scale = resize_short / min(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


def compute_motion_metrics(video_path: str, estimator, sample_fps: float = 2.0,
                           max_pairs: int = 60, dynamic_threshold: float = 2.0,
                           resize_short: int = 480) -> Dict[str, float]:
    """
    计算视频的运动指标.
    Returns:
        motion_score:   平均光流 magnitude
        motion_std:     光流 magnitude 标准差
        dynamic_ratio:  动态像素占比
        max_motion:     最大帧间运动
    """
    t0 = time.time()

    # 1) 只读元信息
    total_frames, fps = load_video_info(video_path)
    logger.debug(f"  视频信息: {total_frames} frames, {fps:.1f} fps")
    if total_frames < 2:
        return EMPTY_METRICS.copy()

    pairs = sample_frame_pairs(total_frames, fps, sample_fps, max_pairs)
    if not pairs:
        return EMPTY_METRICS.copy()

    # 2) 只加载需要的帧 (而非全部帧)
    use_memfof = isinstance(estimator, MEMFOFFlowEstimator)
    needed_indices = sorted(set(i for pair in pairs for i in pair))
    logger.debug(f"  加载 {len(needed_indices)}/{total_frames} 帧...")
    t1 = time.time()
    frames = load_frames_by_indices(video_path, needed_indices)
    idx_map = {orig: pos for pos, orig in enumerate(needed_indices)}
    logger.debug(f"  帧加载完成, 耗时 {time.time()-t1:.1f}s")

    # 3) 光流计算
    magnitudes = []
    dynamic_ratios = []
    t2 = time.time()

    if use_memfof:
        # MEMFOF 三帧模式: 对每个采样点取 (prev, cur, next) 三帧
        for idx in range(1, len(needed_indices) - 1):
            f_prev = _resize_frame(frames[idx - 1], resize_short)
            f_cur  = _resize_frame(frames[idx],     resize_short)
            f_next = _resize_frame(frames[idx + 1], resize_short)
            bw_mag, fw_mag = estimator.estimate_triplet(f_prev, f_cur, f_next)
            avg_mag = (bw_mag + fw_mag) / 2.0
            magnitudes.append(avg_mag.mean())
            dynamic_ratios.append((avg_mag > dynamic_threshold).mean())
    else:
        # 双帧模式: RAFT / OpenCV
        for i1, i2 in pairs:
            f1 = _resize_frame(frames[idx_map[i1]], resize_short)
            f2 = _resize_frame(frames[idx_map[i2]], resize_short)
            mag = estimator.estimate(f1, f2)
            magnitudes.append(mag.mean())
            dynamic_ratios.append((mag > dynamic_threshold).mean())

    logger.debug(f"  光流计算完成, {len(magnitudes)} 对, 耗时 {time.time()-t2:.1f}s, 总耗时 {time.time()-t0:.1f}s")

    magnitudes = np.array(magnitudes)
    return {
        "motion_score": float(magnitudes.mean()),
        "motion_std": float(magnitudes.std()),
        "dynamic_ratio": float(np.mean(dynamic_ratios)),
        "max_motion": float(magnitudes.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="光流动态性检测")
    parser.add_argument("--video", type=str, default="", help="单个视频路径")
    parser.add_argument("--input_csv", type=str, default="", help="批量 CSV (需含 file_path 列)")
    parser.add_argument("--output_csv", type=str, default="", help="输出 CSV")
    parser.add_argument("--method", type=str, default="memfof", choices=["memfof", "raft", "raft_small", "opencv"],
                        help="光流方法: memfof (GPU, ICCV2025, 推荐), raft (GPU), raft_small (GPU, 快), opencv (CPU)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="MEMFOF 模型本地路径 (默认: /mnt/workspace/lihaoran/ckps/MEMFOF-Tartan-T-TSKH)")
    parser.add_argument("--gpu_id", type=int, default=0, help="指定 GPU ID")
    parser.add_argument("--sample_fps", type=float, default=2.0, help="光流采样帧率")
    parser.add_argument("--debug", action="store_true", help="开启 DEBUG 日志")
    parser.add_argument("--max_pairs", type=int, default=60, help="最大帧对数")
    parser.add_argument("--dynamic_threshold", type=float, default=2.0, help="动态像素阈值")
    parser.add_argument("--resize_short", type=int, default=480, help="短边 resize 加速")
    parser.add_argument("--is_local", type=str, default="True", help="True=本地; False=Ray 分布式")
    parser.add_argument("--concurrency", type=int, default=8, help="Ray 并发数")
    args = parser.parse_args()

    if not args.video and not args.input_csv:
        parser.error("请指定 --video 或 --input_csv")

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── 检查 GPU 可用性 ──
    gpu_available = False
    if args.method != "opencv":
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            torch.cuda.set_device(args.gpu_id)
            logger.info(f"使用 GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
        else:
            logger.warning("CUDA 不可用! 自动降级为 opencv 方法 (CPU)")
            logger.warning("若需 GPU, 请确认: 1) nvidia-smi 正常; 2) torch 与 CUDA 驱动版本匹配")
            logger.warning(f"当前 torch={torch.__version__}, 建议: pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu124")
            args.method = "opencv"

    # ── 构建 estimator ──
    t_start = time.time()
    if args.method == "memfof":
        estimator = MEMFOFFlowEstimator(model_path=args.model_path, device=f"cuda:{args.gpu_id}")
    elif args.method == "raft":
        estimator = RAFTFlowEstimator(model_size="large", device=f"cuda:{args.gpu_id}")
    elif args.method == "raft_small":
        estimator = RAFTFlowEstimator(model_size="small", device=f"cuda:{args.gpu_id}")
    else:
        estimator = FarnebackFlowEstimator()
    logger.info(f"Estimator ({args.method}) 初始化完成, 耗时 {time.time()-t_start:.1f}s")

    # ── 单视频模式 ──
    if args.video:
        metrics = compute_motion_metrics(
            args.video, estimator, args.sample_fps, args.max_pairs,
            args.dynamic_threshold, args.resize_short
        )
        logger.info(f"视频: {args.video}")
        logger.info(f"  motion_score  = {metrics['motion_score']:.4f}  (平均光流幅度)")
        logger.info(f"  motion_std    = {metrics['motion_std']:.4f}  (帧间变化)")
        logger.info(f"  dynamic_ratio = {metrics['dynamic_ratio']:.4f}  (动态像素占比)")
        logger.info(f"  max_motion    = {metrics['max_motion']:.4f}  (最大帧间运动)")
        return

    # ── CSV 批量模式 ──
    with open(args.input_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    is_local = args.is_local.lower() == "true"
    output_csv = args.output_csv or args.input_csv.replace(".csv", "_motion.csv")

    if is_local:
        results = []
        for i, row in enumerate(rows):
            vpath = row["file_path"]
            try:
                metrics = compute_motion_metrics(
                    vpath, estimator, args.sample_fps, args.max_pairs,
                    args.dynamic_threshold, args.resize_short
                )
                row.update({f"optflow_{k}": f"{v:.4f}" for k, v in metrics.items()})
                logger.info(f"[{i+1}/{len(rows)}] {os.path.basename(vpath)} → score={metrics['motion_score']:.4f}")
            except Exception as e:
                row.update({f"optflow_{k}": "" for k in ("motion_score", "motion_std", "dynamic_ratio", "max_motion")})
                logger.error(f"[{i+1}/{len(rows)}] {os.path.basename(vpath)} 失败: {e}")
            results.append(row)
    else:
        import ray

        ray.init(ignore_reinit_error=True)

        @ray.remote(num_gpus=0.5 if args.method != "opencv" else 0)
        class FlowWorker:
            def __init__(self, method):
                if method == "memfof":
                    self.estimator = MEMFOFFlowEstimator()
                elif method == "raft":
                    self.estimator = RAFTFlowEstimator(model_size="large")
                elif method == "raft_small":
                    self.estimator = RAFTFlowEstimator(model_size="small")
                else:
                    self.estimator = FarnebackFlowEstimator()

            def process(self, row, sample_fps, max_pairs, dynamic_threshold, resize_short):
                vpath = row["file_path"]
                try:
                    metrics = compute_motion_metrics(
                        vpath, self.estimator, sample_fps, max_pairs,
                        dynamic_threshold, resize_short
                    )
                    row.update({f"optflow_{k}": f"{v:.4f}" for k, v in metrics.items()})
                except Exception as e:
                    row.update({f"optflow_{k}": "" for k in ("motion_score", "motion_std", "dynamic_ratio", "max_motion")})
                return row

        workers = [FlowWorker.remote(args.method) for _ in range(args.concurrency)]
        futures = []
        for i, row in enumerate(rows):
            worker = workers[i % len(workers)]
            futures.append(worker.process.remote(row, args.sample_fps, args.max_pairs,
                                                  args.dynamic_threshold, args.resize_short))
        results = ray.get(futures)
        ray.shutdown()

    fieldnames = list(results[0].keys()) if results else []
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"结果写入: {output_csv}")


if __name__ == "__main__":
    main()
