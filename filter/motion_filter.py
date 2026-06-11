import gc
import shutil
from pathlib import Path
from contextlib import contextmanager

import cv2
import numpy as np


@contextmanager
def VideoCapture(video_path):
    cap = cv2.VideoCapture(str(video_path))
    try:
        yield cap
    finally:
        cap.release()
        del cap
        gc.collect()


NORMALIZE_WIDTH = 640
NORMALIZE_HEIGHT = 360


def compute_motion_score(video_path, sampling_fps=2):
    """
    基于光流法计算视频的运动分数。
    使用 Farneback 稠密光流，对采样帧计算平均运动幅度。
    帧会统一缩放到 640x360 以消除分辨率对光流幅度的影响。

    Args:
        video_path: 视频路径
        sampling_fps: 采样帧率（默认2fps）

    Returns:
        dict: {"motion_score": float, "frame_scores": list} 或 None（失败时）
    """
    video_motion_scores = []

    try:
        with VideoCapture(video_path) as cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                return None
            valid_fps = min(max(sampling_fps, 1), fps)
            frame_interval = int(fps / valid_fps)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < 2:
                return None

            frame_interval = min(frame_interval, total_frames - 1)

            prev_frame = None
            frame_count = -1
            while cap.isOpened():
                ret, frame = cap.read()
                frame_count += 1

                if not ret:
                    break

                if frame_count % frame_interval != 0:
                    continue

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.resize(gray_frame, (NORMALIZE_WIDTH, NORMALIZE_HEIGHT))
                if prev_frame is None:
                    prev_frame = gray_frame
                    continue

                flow = cv2.calcOpticalFlowFarneback(
                    prev_frame,
                    gray_frame,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                frame_motion_score = np.mean(mag)
                video_motion_scores.append(frame_motion_score)
                prev_frame = gray_frame

        if not video_motion_scores:
            return None

        return {
            "motion_score": round(float(np.mean(video_motion_scores)), 5),
            "frame_scores": video_motion_scores,
        }

    except Exception as e:
        print(f"计算运动分数失败 {video_path}: {e}")
        return None


def filter_by_motion(
    input_dir,
    output_dir=None,
    min_motion_score=2.0,
    max_motion_score=float("inf"),
    sampling_fps=2,
    extensions=(".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"),
    copy_mode=True,
):
    """
    基于画面动态性（光流运动分数）过滤视频。

    运动分数含义（经验参考值）：
        - < 1.0: 几乎静止（静态画面、幻灯片）
        - 1.0 ~ 3.0: 低动态（缓慢移动、talking head）
        - 3.0 ~ 8.0: 中等动态（正常视频内容）
        - 8.0 ~ 15.0: 高动态（运动、快速镜头移动）
        - > 15.0: 极高动态（可能是抖动、场景频繁切换）

    Args:
        input_dir: 输入视频目录
        output_dir: 输出目录，为 None 时只打印结果不复制/移动
        min_motion_score: 最小运动分数阈值（低于此值视为静态，过滤掉）
        max_motion_score: 最大运动分数阈值（高于此值可能是抖动/异常，过滤掉）
        sampling_fps: 光流计算的采样帧率
        extensions: 支持的视频扩展名
        copy_mode: True=复制, False=移动

    Returns:
        passed: 满足条件的视频列表
        failed: 不满足条件的视频列表
    """
    input_dir = Path(input_dir)
    passed = []
    failed = []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    video_files = [
        f for f in input_dir.rglob("*")
        if f.suffix.lower() in extensions and f.is_file()
    ]
    video_files.sort()

    print(f"找到 {len(video_files)} 个视频文件，开始动态性检测...")

    for i, video_path in enumerate(video_files, 1):
        result = compute_motion_score(video_path, sampling_fps=sampling_fps)

        if result is None:
            print(f"[{i}/{len(video_files)}] 跳过（无法计算）: {video_path.name}")
            failed.append({"path": str(video_path), "reason": "无法计算运动分数", "motion_score": None})
            continue

        score = result["motion_score"]

        if score < min_motion_score:
            reason = f"运动分数 {score:.3f} < {min_motion_score}（画面过于静态）"
            print(f"[{i}/{len(video_files)}] 不通过: {video_path.name} | {reason}")
            failed.append({"path": str(video_path), "reason": reason, "motion_score": score})
        elif score > max_motion_score:
            reason = f"运动分数 {score:.3f} > {max_motion_score}（画面动态过大/抖动）"
            print(f"[{i}/{len(video_files)}] 不通过: {video_path.name} | {reason}")
            failed.append({"path": str(video_path), "reason": reason, "motion_score": score})
        else:
            print(f"[{i}/{len(video_files)}] 通过: {video_path.name} | 运动分数: {score:.3f}")
            passed.append({"path": str(video_path), "motion_score": score})

            if output_dir:
                dest = output_dir / video_path.name
                if copy_mode:
                    shutil.copy2(video_path, dest)
                else:
                    shutil.move(str(video_path), str(dest))

    print(f"\n动态性过滤完成: 通过 {len(passed)}, 不通过 {len(failed)}")
    return passed, failed


if __name__ == "__main__":
    input_directory = "/xsky/yangnq/LM_data/zgtz_test/"
    # input_directory = "/xsky/yangnq/yangfile/data_deal_video_audio/videos"
    passed, failed = filter_by_motion(
        input_dir=input_directory,
        output_dir=None,              # 设为目录路径则复制/移动满足条件的视频
        min_motion_score=3,         # 最小运动分数（过滤静态视频）
        max_motion_score=100.0,        # 最大运动分数（过滤剧烈抖动/异常视频）
        sampling_fps=4,               # 采样帧率
        copy_mode=True,               # True=复制, False=移动
    )
