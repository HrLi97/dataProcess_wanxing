import os
import json
import subprocess
import shutil
from pathlib import Path


def get_video_info(video_path):
    """使用 ffprobe 获取视频信息"""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return None


def parse_video_info(info):
    """从 ffprobe 输出中解析码率、分辨率和时长"""
    if not info:
        return None

    # 获取视频流
    video_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        return None

    # 分辨率
    width = video_stream.get("width", 0)
    height = video_stream.get("height", 0)

    # 时长（秒）
    duration = float(info.get("format", {}).get("duration", 0))
    if duration == 0:
        duration = float(video_stream.get("duration", 0))

    # 码率（bps）
    bitrate = int(info.get("format", {}).get("bit_rate", 0))
    if bitrate == 0:
        bitrate = int(video_stream.get("bit_rate", 0))

    return {
        "width": width,
        "height": height,
        "duration": duration,
        "bitrate": bitrate,
    }


def parse_audio_info(info):
    """从 ffprobe 输出中解析音频码率、采样率和时长"""
    if not info:
        return None

    # 获取音频流
    audio_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "audio":
            audio_stream = stream
            break

    if not audio_stream:
        return None

    # 采样率（Hz）
    try:
        sample_rate = int(audio_stream.get("sample_rate", 0))
    except (TypeError, ValueError):
        sample_rate = 0

    # 时长（秒）
    try:
        duration = float(info.get("format", {}).get("duration", 0))
    except (TypeError, ValueError):
        duration = 0.0
    if duration == 0:
        try:
            duration = float(audio_stream.get("duration", 0))
        except (TypeError, ValueError):
            duration = 0.0

    # 码率（bps）
    try:
        bitrate = int(info.get("format", {}).get("bit_rate", 0))
    except (TypeError, ValueError):
        bitrate = 0
    if bitrate == 0:
        try:
            bitrate = int(audio_stream.get("bit_rate", 0))
        except (TypeError, ValueError):
            bitrate = 0

    return {
        "sample_rate": sample_rate,
        "duration": duration,
        "bitrate": bitrate,
    }


# 分辨率预设（总像素阈值）
# RESOLUTION_PRESETS = {
#     "360p": 512 * 512,
#     "480p": 640 * 640,
#     "720p": 960 * 960,
#     "1080p": 1440 * 1440,
# }
RESOLUTION_PRESETS = {
    "360p": 480 * 480,
    "480p": 600 * 600,
    "720p": 900 * 900,
    "1080p": 1376 * 1376,
}   # 考虑到要使用线上的数据，线上数据大多为模型生成的不是标准的对应分辨率。以更宽松的像素阈值替代传统分辨率标准，适应不同宽高比的视频


def filter_videos(
    input_dir,
    output_dir=None,
    min_resolution="720p",
    min_duration=0,
    max_duration=float("inf"),
    min_bitrate=0,
    max_bitrate=float("inf"),
    extensions=(".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"),
    copy_mode=False,
):
    """
    过滤视频文件，按总像素、码率和时长筛选。

    Args:
        input_dir: 输入视频目录
        output_dir: 输出目录（满足条件的视频复制/移动到此目录），为 None 时只打印结果
        min_resolution: 最小分辨率等级，可选 "360p"/"480p"/"720p"/"1080p"，
                        也可直接传入整数表示最小总像素值
        min_duration: 最小时长（秒）
        max_duration: 最大时长（秒）
        min_bitrate: 最小码率（bps）
        max_bitrate: 最大码率（bps）
        extensions: 支持的视频扩展名
        copy_mode: True 为复制，False 为移动（仅在 output_dir 不为 None 时生效）

    Returns:
        passed: 满足条件的视频列表
        failed: 不满足条件的视频列表
    """
    # 解析最小像素阈值
    if isinstance(min_resolution, str):
        min_pixels = RESOLUTION_PRESETS[min_resolution]
    else:
        min_pixels = int(min_resolution)
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

    print(f"找到 {len(video_files)} 个视频文件，开始过滤...")

    for i, video_path in enumerate(video_files, 1):
        info = get_video_info(video_path)
        parsed = parse_video_info(info)

        if not parsed:
            print(f"[{i}/{len(video_files)}] 跳过（无法解析）: {video_path.name}")
            failed.append({"path": str(video_path), "reason": "无法解析"})
            continue

        w, h = parsed["width"], parsed["height"]
        dur = parsed["duration"]
        br = parsed["bitrate"]

        # 检查是否满足条件
        total_pixels = w * h
        reasons = []
        if total_pixels < min_pixels:
            reasons.append(f"总像素 {total_pixels} < {min_pixels} ({min_resolution})")
        if dur < min_duration or dur > max_duration:
            reasons.append(f"时长 {dur:.1f}s 不在 [{min_duration}, {max_duration}]")
        if br < min_bitrate or br > max_bitrate:
            reasons.append(f"码率 {br} 不在 [{min_bitrate}, {max_bitrate}]")

        if reasons:
            print(f"[{i}/{len(video_files)}] 不通过: {video_path.name} | {'; '.join(reasons)}")
            failed.append({"path": str(video_path), "reason": "; ".join(reasons)})
        else:
            print(f"[{i}/{len(video_files)}] 通过: {video_path.name} | {w}x{h}, {dur:.1f}s, {br//1000}kbps")
            passed.append({"path": str(video_path), "info": parsed})

            if output_dir:
                dest = output_dir / video_path.name
                if copy_mode:
                    shutil.copy2(video_path, dest)
                else:
                    shutil.move(str(video_path), str(dest))

    print(f"\n过滤完成: 通过 {len(passed)}, 不通过 {len(failed)}")
    return passed, failed


def filter_audios(
    input_dir,
    output_dir=None,
    min_sample_rate=32000,
    min_duration=0,
    max_duration=float("inf"),
    min_bitrate=96000,
    max_bitrate=float("inf"),
    extensions=(
        ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
        ".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm",
    ),
    copy_mode=False,
):
    """
    过滤媒体文件中的音轨，按码率、采样率和时长筛选。

    Args:
        input_dir: 输入目录（可包含音频文件或视频文件）
        output_dir: 输出目录（满足条件的媒体文件复制/移动到此目录），为 None 时只打印结果
        min_sample_rate: 最小采样率（Hz）
        min_duration: 最小时长（秒）
        max_duration: 最大时长（秒）
        min_bitrate: 最小码率（bps）
        max_bitrate: 最大码率（bps）
        extensions: 支持的媒体扩展名（默认含音频和视频）
        copy_mode: True 为复制，False 为移动（仅在 output_dir 不为 None 时生效）

    Returns:
        passed: 满足条件的文件列表（音频文件或包含合格音轨的视频文件）
        failed: 不满足条件的文件列表
    """
    input_dir = Path(input_dir)
    passed = []
    failed = []

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    media_files = [
        f for f in input_dir.rglob("*")
        if f.suffix.lower() in extensions and f.is_file()
    ]

    print(f"找到 {len(media_files)} 个媒体文件，开始按音轨过滤...")

    for i, media_path in enumerate(media_files, 1):
        info = get_video_info(media_path)
        parsed = parse_audio_info(info)

        if not parsed:
            print(f"[{i}/{len(media_files)}] 跳过（无可用音轨/无法解析）: {media_path.name}")
            failed.append({"path": str(media_path), "reason": "无可用音轨/无法解析"})
            continue

        sr = parsed["sample_rate"]
        dur = parsed["duration"]
        br = parsed["bitrate"]

        # 检查是否满足条件
        reasons = []
        if sr < min_sample_rate:
            reasons.append(f"采样率 {sr} < {min_sample_rate}")
        if dur < min_duration or dur > max_duration:
            reasons.append(f"时长 {dur:.1f}s 不在 [{min_duration}, {max_duration}]")
        if br < min_bitrate or br > max_bitrate:
            reasons.append(f"码率 {br} 不在 [{min_bitrate}, {max_bitrate}]")

        if reasons:
            print(f"[{i}/{len(media_files)}] 不通过: {media_path.name} | {'; '.join(reasons)}")
            failed.append({"path": str(media_path), "reason": "; ".join(reasons)})
        else:
            print(f"[{i}/{len(media_files)}] 通过: {media_path.name} | {sr}Hz, {dur:.1f}s, {br//1000}kbps")
            passed.append({"path": str(media_path), "info": parsed})

            if output_dir:
                dest = output_dir / media_path.name
                if copy_mode:
                    shutil.copy2(media_path, dest)
                else:
                    shutil.move(str(media_path), str(dest))

    print(f"\n过滤完成: 通过 {len(passed)}, 不通过 {len(failed)}")
    return passed, failed


if __name__ == "__main__":
    # 示例用法
    input_video_directory = "./videos"
    input_audio_directory = "./audios"

    passed, failed = filter_videos(
        input_dir=input_video_directory,
        output_dir=None,          # 设为目录路径则复制/移动满足条件的视频
        min_resolution="720p",    # 最小分辨率等级（按总像素过滤）
        min_duration=3,           # 最小时长 3 秒
        max_duration=30,         # 最大时长 15 秒
        min_bitrate=1_000_000,      # 最小码率 1Mbps
        max_bitrate=50_000_000,   # 最大码率 50Mbps
        copy_mode=True,           # True=复制, False=移动
    )

    passed, failed = filter_audios(
        input_dir=input_audio_directory,
        output_dir=None,          # 设为目录路径则复制/移动满足条件的音频
        min_sample_rate=32000,    # 最小采样率 32kHz
        min_duration=3,           # 最小时长 3 秒
        max_duration=600,          # 最大时长 600 秒
        min_bitrate=96_000,       # 最小码率 96kbps
        max_bitrate=512_000,      # 最大码率 512kbps
        copy_mode=True,           # True=复制, False=移动
    )
