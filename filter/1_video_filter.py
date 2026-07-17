#!/usr/bin/env python3
"""
视频质量过滤 + 标准化转码(CSV 输入/输出)

功能：
  1. 分辨率过滤(总像素阈值)
  2. 码率过滤
  3. 时长过滤
  4. FPS 标准化：非 25fps → 自动转码为 25fps
  5. 音频采样率标准化：非 48kHz → 自动转码为 48kHz

输入：CSV 文件(需含 video_path 列)
输出：CSV 结果 + qualified_videos/(含转码后的合格视频)
"""
import argparse
import csv
import fcntl
import json
import re
import shutil
import subprocess
from pathlib import Path
import sys

_FILTER_ROOT = Path(__file__).resolve().parents[1]
if str(_FILTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FILTER_ROOT))

from filter.filter_utils import unique_output_name, worker_syspath_setup
from filter.obs_helper import add_obs_args, ObsIO

# ─── ffprobe ─────────────────────────────────────────────────────────────────

def get_media_info(video_path):
    """使用 ffprobe 获取媒体信息(JSON 格式)。返回 (info, error_reason)。"""
    cmd = [
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            err = (result.stderr or "").strip().replace("\n", " | ")
            return None, f"ffprobe_rc={result.returncode} stderr={err[:300]}"

        stdout = (result.stdout or "").strip()
        if not stdout:
            return None, "ffprobe_empty_stdout"

        try:
            return json.loads(stdout), ""
        except json.JSONDecodeError as e:
            return None, f"ffprobe_json_decode_error={e} stdout_head={stdout[:200]}"
    except subprocess.TimeoutExpired:
        return None, "ffprobe_timeout_30s"
    except FileNotFoundError:
        return None, "ffprobe_not_found"
    except Exception as e:
        return None, f"ffprobe_exception={type(e).__name__}:{e}"


def parse_media_info(info):
    """从 ffprobe JSON 中同时解析视频 + 音频参数。返回 (parsed, error_reason)。"""
    if not info:
        return None, "empty_ffprobe_info"

    result = {}

    # ── 视频流 ──
    video_stream = None
    for s in info.get("streams", []):
        if s.get("codec_type") == "video":
            video_stream = s
            break
    if not video_stream:
        return None, "no_video_stream"

    result["width"] = video_stream.get("width", 0)
    result["height"] = video_stream.get("height", 0)

    # FPS(from r_frame_rate, e.g. "30000/1001")
    r_fr = video_stream.get("r_frame_rate", "0/1")
    try:
        num, den = r_fr.split("/")
        result["fps"] = round(int(num) / max(int(den), 1), 2)
    except Exception:
        result["fps"] = 0.0

    # Duration(秒)
    dur = info.get("format", {}).get("duration", 0)
    result["duration"] = float(dur) if dur else 0.0
    if result["duration"] == 0:
        dur2 = video_stream.get("duration", 0)
        result["duration"] = float(dur2) if dur2 else 0.0

    # Bitrate(bps)
    br = info.get("format", {}).get("bit_rate", 0)
    result["bitrate"] = int(br) if br else 0
    if result["bitrate"] == 0:
        br2 = video_stream.get("bit_rate", 0)
        result["bitrate"] = int(br2) if br2 else 0

    # 总帧数
    nb_frames = video_stream.get("nb_frames")
    if nb_frames and str(nb_frames).isdigit():
        result["total_frames"] = int(nb_frames)
    else:
        result["total_frames"] = int(round(result["fps"] * result["duration"])) if result["fps"] > 0 else 0

    # ── 音频流 ──
    result["audio_sample_rate"] = 0
    result["audio_bitrate"] = 0
    for s in info.get("streams", []):
        if s.get("codec_type") == "audio":
            try:
                result["audio_sample_rate"] = int(s.get("sample_rate", 0))
            except (TypeError, ValueError):
                pass
            try:
                result["audio_bitrate"] = int(s.get("bit_rate", 0) or 0)
            except (TypeError, ValueError):
                pass
            break

    return result, ""


# ─── 分辨率预设(总像素阈值) ────────────────────────────────────────────────
# 以更宽松的像素阈值替代传统分辨率标准，适应不同宽高比的视频
RESOLUTION_PRESETS = {
    "360p": 480 * 480,
    "480p": 600 * 600,
    "720p": 900 * 900,
    "1080p": 1376 * 1376,
}


# ─── 验证 MP4 文件有效性 ──────────────────────────────────────────────────────
def _is_valid_mp4(path):
    """用 ffprobe 快速验证文件是否可读（moov atom 存在且至少有一个视频流）。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        return r.returncode == 0 and len(r.stdout.strip()) > 0
    except Exception:
        return False

# ─── ffmpeg 转码 ─────────────────────────────────────────────────────────────
def transcode_video(input_path, output_path,
                    need_fps=False, need_audio=False,
                    target_fps=25, target_audio_sr=48000):
    """
    用 ffmpeg 转码。仅重编码需要变更的流，其余 stream copy。

    Returns:
        bool: 是否成功
    """
    cmd = ["ffmpeg", "-y", "-i", str(input_path)]

    # Video
    if need_fps:
        cmd += ["-r", str(target_fps),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "15",
                "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]

    # Audio
    if need_audio:
        cmd += ["-ar", str(target_audio_sr), "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-c:a", "copy"]

    cmd += ["-movflags", "+faststart", "-loglevel", "warning", str(output_path)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            print(f"  [transcode] ffmpeg error: {r.stderr[:300]}")
            # 删除不完整文件
            Path(output_path).unlink(missing_ok=True)
            return False
        return Path(output_path).exists()
    except subprocess.TimeoutExpired:
        print("  [transcode] timeout")
        # 删除不完整文件
        Path(output_path).unlink(missing_ok=True)
        return False


def get_audio_max_volume_db(media_path, timeout_sec=60):
    """返回 (max_volume_db, err_reason)。"""
    cmd = [
        "ffmpeg", "-hide_banner", "-i", str(media_path),
        "-af", "volumedetect", "-f", "null", "-",
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return None, "audio_probe_timeout"
    except FileNotFoundError:
        return None, "ffmpeg_not_found"
    except Exception as e:
        return None, f"audio_probe_exception={type(e).__name__}:{e}"

    output = (r.stdout or "") + "\n" + (r.stderr or "")
    m = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", output)
    if not m:
        return None, "audio_probe_parse_failed"
    try:
        return float(m.group(1)), ""
    except Exception:
        return None, "audio_probe_parse_failed"


# ─── CSV 字段 ────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "video_path", "_source_key",
    "video_filter_reject_reason",
    "resolution", "source_fps", "source_audio_sample_rate", "source_audio_max_volume_db", "source_bitrate",
    "duration", "was_transcoded",
]


# ─── 单段处理 ────────────────────────────────────────────────────────────────
def process_segment(args, video_path, qualified_dir):
    """处理单个视频：质量检查 + 必要时转码。返回 CSV row dict。"""
    row = {
        "video_path": "",
        "video_filter_reject_reason": "",
        "resolution": "",
        "source_fps": "", "source_audio_sample_rate": "",
        "source_audio_max_volume_db": "",
        "source_bitrate": "", "duration": "",
        "was_transcoded": False,
    }

    seg_path = Path(video_path)
    if not seg_path.is_file():
        row["video_filter_reject_reason"] = "file_not_found"
        return row

    # ── 解析媒体信息 ──
    info, probe_err = get_media_info(video_path)
    parsed, parse_err = parse_media_info(info)
    if not parsed:
        row["video_filter_reject_reason"] = "parse_failed"
        try:
            size_bytes = seg_path.stat().st_size
        except Exception:
            size_bytes = -1
        print(
            f"  [parse_failed] path={seg_path} size={size_bytes} "
            f"probe_err={probe_err or '-'} parse_err={parse_err or '-'}"
        )
        return row

    row.update({
        "resolution": f"{parsed['width']}x{parsed['height']}",
        "source_fps": parsed["fps"],
        "source_audio_sample_rate": parsed["audio_sample_rate"],
        "source_bitrate": parsed["bitrate"],
        "duration": parsed["total_frames"],
    })

    # ── 质量检查 ──
    reasons = []

    # 分辨率
    if isinstance(args.min_resolution, str):
        min_pixels = RESOLUTION_PRESETS.get(args.min_resolution, 0)
    else:
        min_pixels = int(args.min_resolution)
    total_px = parsed["width"] * parsed["height"]
    if min_pixels > 0 and total_px < min_pixels:
        reasons.append(f"low_resolution_{parsed['width']}x{parsed['height']}")

    # 无音频是否过滤由开关控制（默认不过滤）
    if parsed["audio_sample_rate"] == 0:
        if args.filter_no_audio:
            reasons.append("no_audio")
    elif args.enable_audio_max_volume_filter:
        audio_max_db, audio_probe_err = get_audio_max_volume_db(seg_path)
        if audio_max_db is None:
            reasons.append(audio_probe_err or "audio_probe_failed")
        else:
            row["source_audio_max_volume_db"] = round(audio_max_db, 2)
            if audio_max_db < args.min_audio_max_volume_db:
                reasons.append(f"audio_too_quiet_maxdb_{audio_max_db:.2f}")

    # 时长(帧)
    total_frames = parsed["total_frames"]
    if total_frames < args.min_duration:
        reasons.append(f"too_short_{total_frames}frames")
    if args.max_duration > 0 and total_frames > args.max_duration:
        reasons.append(f"too_long_{total_frames}frames")

    if reasons:
        row["video_filter_reject_reason"] = ",".join(reasons)
        print(f"  不通过: {seg_path.name} | {', '.join(reasons)}")
        return row

    # ── 转码检查 ──
    need_fps = abs(parsed["fps"] - args.target_fps) > 0.5
    need_audio = parsed["audio_sample_rate"] != args.target_audio_sr

    if need_fps or need_audio:
        out_name = unique_output_name(video_path, suffix=".mp4")
        out_path = Path(qualified_dir) / out_name

        # 已存在则校验有效性（ffprobe 能否打开），损坏文件直接删除重新转码
        if out_path.exists():
            if _is_valid_mp4(str(out_path)):
                print(f"  跳过(已存在): {out_name}")
                row["video_path"] = str(out_path)
                row["was_transcoded"] = True
                return row
            else:
                print(f"  删除损坏文件: {out_name}")
                out_path.unlink(missing_ok=True)

        tc_info = []
        if need_fps:
            tc_info.append(f"fps:{parsed['fps']}->{args.target_fps}")
        if need_audio:
            tc_info.append(f"sr:{parsed['audio_sample_rate']}->{args.target_audio_sr}")
        print(f"  转码: {seg_path.name} ({', '.join(tc_info)})")

        ok = transcode_video(
            video_path, str(out_path),
            need_fps=need_fps, need_audio=need_audio,
            target_fps=args.target_fps, target_audio_sr=args.target_audio_sr,
        )
        if not ok:
            row["video_filter_reject_reason"] = "transcode_failed"
            return row

        row["video_path"] = str(out_path)
        row["was_transcoded"] = True
    else:
        # 无需转码，symlink / copy / 原路径
        out_name = unique_output_name(video_path, suffix=".mp4")
        out_path = Path(qualified_dir) / out_name

        # 已存在则跳过
        if out_path.exists():
            print(f"  跳过(已存在): {out_name}")
            row["video_path"] = str(out_path)
            return row

        if args.copy_mode == "symlink":
            out_path.symlink_to(seg_path.resolve())
        elif args.copy_mode == "copy":
            shutil.copy2(str(seg_path), str(out_path))
        else:
            out_path = seg_path
        row["video_path"] = str(out_path)

    print(f"  通过: {seg_path.name} | {parsed['width']}x{parsed['height']}, "
          f"fps={parsed['fps']}, sr={parsed['audio_sample_rate']}, "
          f"tc={row['was_transcoded']}")
    return row


class Worker:
    def __init__(self):
        worker_syspath_setup()  # Ray worker 端把 data_process/ 加进 sys.path
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.qualified_dir = self.output_dir / "qualified_videos"
        self.qualified_dir.mkdir(parents=True, exist_ok=True)
        self.all_csv = self.output_dir / "video_filter_all.csv"
        self.qual_csv = self.output_dir / "video_filter_qualified.csv"
        for p in (self.all_csv, self.qual_csv):
            if not p.exists():
                with open(p, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()
        # 断点续跑：加载已处理的 video_path
        self.processed = set()
        if self.all_csv.exists():
            with open(self.all_csv, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    sk = r.get("_source_key") or r.get("video_path", "")
                    if sk:
                        self.processed.add(sk)
        # OBS
        self.obs_io = ObsIO(args)
        print(f"Worker initialized. OBS={self.obs_io.enabled}. 已处理 {len(self.processed)} 条。")

    def _append_csv(self, filepath, row):
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore").writerow(row)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)

    def __call__(self, item):
        video_path = item.get(args.video_path_column, "")
        if not video_path:
            return {k: "" for k in FIELDNAMES}
        if video_path in self.processed:
            return {k: "" for k in FIELDNAMES}

        local_video = None
        try:
            # ── OBS: 下载到本地 ──
            if self.obs_io.enabled:
                local_video = self.obs_io.download(video_path)
                if local_video is None:
                    row = {k: "" for k in FIELDNAMES}
                    row["video_filter_reject_reason"] = "obs_download_failed"
                    row["_source_key"] = video_path
                    self._append_csv(self.all_csv, row)
                    self.processed.add(video_path)
                    return row
            else:
                local_video = video_path

            row = process_segment(args, local_video, str(self.qualified_dir))

            # ── OBS: 上传合格视频，CSV 写 obs_key 而非本地路径 ──
            if self.obs_io.enabled and not row.get("video_filter_reject_reason"):
                local_result = row.get("video_path", "")
                if local_result and Path(local_result).is_file():
                    obs_key = self.obs_io.make_obs_key(
                        "qualified_videos", Path(local_result).name
                    )
                    self.obs_io.upload(local_result, obs_key)
                    row["video_path"] = obs_key
                    # 删除本地输出
                    self.obs_io.cleanup(local_result)

            row["_source_key"] = video_path
            self._append_csv(self.all_csv, row)
            if not row.get("video_filter_reject_reason"):
                self._append_csv(self.qual_csv, row)
            self.processed.add(video_path)
            return row
        finally:
            # ── OBS: 清理下载的临时文件 ──
            if self.obs_io.enabled and local_video and local_video != video_path:
                self.obs_io.cleanup(local_video)


# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="视频质量过滤 + FPS/音频采样率标准化转码")
parser.add_argument("--input_csv", default="/home/ma-user/work/lihaoran/all_data/testcsv/0622/OpenS2V-5M.csv")
parser.add_argument("--output_dir", default="/home/ma-user/work/lihaoran/data_process/test/result/0624/video_filter/")
parser.add_argument("--video_path_column", default="video_path")
parser.add_argument("--min_resolution", default="1080p")
parser.add_argument("--min_duration", type=int, default=121)
parser.add_argument("--max_duration", type=int, default=0)
parser.add_argument("--target_fps", type=int, default=25)
parser.add_argument("--target_audio_sr", type=int, default=48000)
parser.add_argument("--filter_no_audio", action="store_true",
                    help="开启后过滤无音频视频(默认关闭)")
parser.add_argument("--enable_audio_max_volume_filter", action="store_true",
                    help="开启后才按 --min_audio_max_volume_db 过滤，默认关闭")
parser.add_argument("--min_audio_max_volume_db", type=float, default=-50.0,
                    help="最小可听峰值音量(dB)。低于阈值判为近静音")
parser.add_argument("--copy_mode", choices=["symlink", "copy", "none"], default="symlink")

# OBS
add_obs_args(parser)

# Ray
parser.add_argument("--ray", action="store_true", help="启用 Ray 分布式模式")
parser.add_argument("--ray_num_gpus", type=float, default=0, help="GPU 资源 (无 GPU 时设 0)")
parser.add_argument("--ray_num_npus", type=float, default=0, help="NPU 自定义资源 (video_filter 不需要 NPU)")
parser.add_argument("--ray_num_cpus", type=float, default=1)
parser.add_argument("--concurrency", type=int, default=48)
args = parser.parse_args()


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(args.input_csv, encoding="utf-8", encoding_errors="replace")

    if not args.ray:
        # ── 本地模式 ──
        samples = list(csv.DictReader(open(args.input_csv, encoding="utf-8")))
        total = len(samples)
        print(f"读取 {total} 条记录 from {args.input_csv}")
        worker = Worker()
        for i, item in enumerate(samples, 1):
            print(f"[{i}/{total}] {item.get(args.video_path_column, '')}")
            worker(item)
        print("完成")

    else:
        # ── Ray 分布式模式 ──
        import ray
        ray.init(address="auto")
        try:
            import ray.autoscaler._private.util  # noqa: F401
        except Exception:
            pass
        from ray.data import ActorPoolStrategy
        samples = ray.data.from_pandas(df).materialize()
        target_parallelism = min(args.concurrency, samples.count())
        if samples.num_blocks() < target_parallelism:
            samples = samples.repartition(target_parallelism)
        map_kwargs = dict(
            num_cpus=max(args.ray_num_cpus, 1),
            compute=ActorPoolStrategy(size=target_parallelism),
        )
        if args.ray_num_gpus > 0:
            map_kwargs["num_gpus"] = args.ray_num_gpus
        if args.ray_num_npus > 0:
            map_kwargs["resources"] = {"NPU": args.ray_num_npus}
        predictions = samples.map(Worker, **map_kwargs)
        predictions.take_all()
        print("Ray 完成")