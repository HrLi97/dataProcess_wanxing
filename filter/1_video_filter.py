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
from collections import Counter
from pathlib import Path
import sys

_FILTER_ROOT = Path(__file__).resolve().parents[1]
if str(_FILTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FILTER_ROOT))

from filter.filter_utils import unique_output_name, worker_syspath_setup, init_csv
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

    # 旋转角（手机竖拍常见）：ffmpeg 解码时会自动转正，cropdetect 给出的是转正后的坐标
    result["rotation"] = 0
    for sd in video_stream.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                result["rotation"] = int(float(sd["rotation"]))
            except (TypeError, ValueError):
                pass
    if not result["rotation"]:
        try:
            result["rotation"] = int(float((video_stream.get("tags") or {}).get("rotate", 0)))
        except (TypeError, ValueError):
            pass

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
    "none": 0,
    "360p": 480 * 480,
    "480p": 600 * 600,
    "720p": 900 * 900,
    "1080p": 1376 * 1376,
}


def resolve_min_pixels(value) -> int:
    """--min_resolution: 预设名（见 RESOLUTION_PRESETS）或纯数字像素数；未知值直接报错，不再静默关闭过滤。"""
    text = str(value).strip().lower()
    if text.isdigit():
        return int(text)
    if text not in RESOLUTION_PRESETS:
        raise ValueError(f"未知的 --min_resolution {value!r}，可选 {sorted(RESOLUTION_PRESETS)} 或像素数")
    return RESOLUTION_PRESETS[text]


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


# ─── 黑边检测 ────────────────────────────────────────────────────────────────
_CROPDETECT_RE = re.compile(r"\bt:\s*(-?[\d.]+).*?crop=(-?\d+):(-?\d+):(-?\d+):(-?\d+)")


def _cropdetect_at(video_path, t_sec, timeout_sec=60):
    """从 t_sec 起用 cropdetect 找内容框，返回 (最后评估帧的时间戳, (w, h, x, y))；失败返回 None。

    先只解码关键帧（快，且几帧之间相隔较远、画面更多样）；t_sec 之后关键帧不够时（短视频、关键帧间隔长）
    改为解码连续帧。cropdetect 默认跳过前 2 帧，reset=0 表示取评估过的各帧内容的并集。
    """
    for decode_opts, n_frames in ((["-skip_frame", "nokey"], 5), ([], 8)):
        # -copyts：时间戳保持源视频的绝对时间（不从 seek 点归零），用于去重；-vsync 0：不为凑恒定帧率而复制帧
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats", "-copyts",
            *decode_opts, "-ss", f"{t_sec:.3f}", "-i", str(video_path),
            "-map", "0:v:0", "-vf", "cropdetect=limit=24:round=2:reset=0",
            "-vsync", "0", "-frames:v", str(n_frames), "-an", "-f", "null", "-",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
        except Exception:
            return None
        found = _CROPDETECT_RE.findall(r.stderr or "")
        if found:
            t, *box = found[-1]
            return round(float(t), 3), tuple(int(v) for v in box)
    return None


def _stable_bar(values):
    """一条边的黑边宽度：在重复出现的值（按 4 像素分桶，至少 20% 的采样点）里取最大的。

    字幕、台标压在黑边上只会让黑边变窄；暗场景会让黑边变宽，但每个场景暗得不一样，很少重复。
    所以重复出现的最宽值就是真实黑边。没有重复值时取中位数。
    """
    need = max(2, -(-len(values) // 5))
    bins = Counter(v // 4 for v in values)
    stable = [b for b, c in bins.items() if c >= need]
    if stable:
        top_bin = max(stable)
        return max(v for v in values if v // 4 == top_bin)
    return sorted(values)[len(values) // 2]


def detect_letterbox(video_path, duration_sec, frame_w, frame_h, samples=12, min_bar_ratio=0.01):
    """估计整片的黑边，返回裁切框 (w, h, x, y)；没有明显黑边或检测失败时返回 None。

    在 5%~95% 时长内均匀取 samples 个点，每个点得到上下左右四条黑边的宽度，每条边各自用 _stable_bar 汇总。
    黑边里的字幕会随黑边一起裁掉。frame_w / frame_h 是转正后的显示尺寸。
    不足 min_bar_ratio 的边（压缩噪声、一两个像素的脏边）不裁。
    """
    if samples <= 0 or duration_sec <= 0 or frame_w <= 0 or frame_h <= 0:
        return None
    bars = {}  # 最后评估帧的时间戳 -> (左, 上, 右, 下)；短视频里相邻采样点可能落在同一组关键帧上，按时间戳去重
    for i in range(samples):
        t = duration_sec * (0.05 + 0.9 * i / max(samples - 1, 1))
        res = _cropdetect_at(video_path, t)
        if not res:
            continue
        frame_t, (w, h, x, y) = res
        # 全黑帧（片头片尾、淡入淡出）会给出负数或很小的框，丢掉
        if x < 0 or y < 0 or w < frame_w * 0.3 or h < frame_h * 0.3:
            continue
        bars[frame_t] = (x, y, max(frame_w - x - w, 0), max(frame_h - y - h, 0))
    if len(bars) < 3:
        return None

    left, top, right, bottom = (_stable_bar([b[k] for b in bars.values()]) for k in range(4))
    min_x, min_y = frame_w * min_bar_ratio, frame_h * min_bar_ratio
    left, right = [v if v >= min_x else 0 for v in (left, right)]
    top, bottom = [v if v >= min_y else 0 for v in (top, bottom)]
    if not (left or right or top or bottom):
        return None
    w = frame_w - left - right
    h = frame_h - top - bottom
    w, h = w - w % 2, h - h % 2  # yuv420p 要求偶数宽高
    if w < frame_w * 0.3 or h < frame_h * 0.3:
        return None
    return w, h, left, top


# ─── CSV 字段 ────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "video_path", "_source_key",
    "video_filter_reject_reason",
    "resolution", "source_fps", "source_audio_sample_rate", "source_audio_max_volume_db", "source_bitrate",
    "duration", "was_transcoded",
    "crop",  # 黑边裁切框 w:h:x:y（Step 2 切片时裁），空 = 不裁
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
        "crop": "",
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

    # 分辨率：总像素（--min_resolution）和短边（--min_short_side）两道门槛，任一不满足即拒。
    # 短边门槛不会误杀 1920x800 这类宽银幕（总像素门槛 1080p=1376² 会）。
    min_pixels = resolve_min_pixels(args.min_resolution)
    total_px = parsed["width"] * parsed["height"]
    if min_pixels > 0 and total_px < min_pixels:
        reasons.append(f"low_resolution_{parsed['width']}x{parsed['height']}")
    elif args.min_short_side > 0 and min(parsed["width"], parsed["height"]) < args.min_short_side:
        reasons.append(f"low_short_side_{parsed['width']}x{parsed['height']}")

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

    # 黑边：只对通过上面检查的片子检测（每片几秒）；去掉黑边后的内容再按短边门槛判一次
    if not reasons and args.letterbox_samples > 0:
        disp_w, disp_h = parsed["width"], parsed["height"]
        if abs(parsed.get("rotation", 0)) % 180 == 90:
            disp_w, disp_h = disp_h, disp_w
        crop = detect_letterbox(video_path, parsed["duration"], disp_w, disp_h,
                                samples=args.letterbox_samples)
        if crop:
            cw, ch, cx, cy = crop
            row["crop"] = f"{cw}:{ch}:{cx}:{cy}"
            if args.min_short_side > 0 and min(cw, ch) < args.min_short_side:
                reasons.append(f"low_content_short_side_{cw}x{ch}")

    if reasons:
        row["video_filter_reject_reason"] = ",".join(reasons)
        print(f"  不通过: {seg_path.name} | {', '.join(reasons)}")
        return row

    # ── 转码检查 ──
    need_fps = abs(parsed["fps"] - args.target_fps) > 0.5
    # target_audio_sr <= 0：保留源采样率（不为改音频重写整部片，统一到训练采样率放在切片时做）；
    # 无音轨（采样率 0）不触发转码
    need_audio = (args.target_audio_sr > 0 and parsed["audio_sample_rate"] > 0
                  and parsed["audio_sample_rate"] != args.target_audio_sr)

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
          f"tc={row['was_transcoded']}" + (f", crop={row['crop']}" if row["crop"] else ""))
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
        # 以文件实际表头追加（旧 CSV 缺 crop 等新列时原地补列）
        self.csv_fields = {p: init_csv(p, FIELDNAMES) for p in (self.all_csv, self.qual_csv)}
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
            csv.DictWriter(f, fieldnames=self.csv_fields[filepath], extrasaction="ignore").writerow(row)
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
# 默认值按 MiniMax-H3 训练数据：24 fps、768p、片段 124 帧（17n+5）、音频 32 kHz 立体声（切片时统一）
parser.add_argument("--min_resolution", default="none",
                    help="总像素门槛：none/360p/480p/720p/1080p 或像素数")
parser.add_argument("--min_short_side", type=int, default=768,
                    help="短边门槛（像素），0 关闭；H3 训练分辨率为 768p")
parser.add_argument("--min_duration", type=int, default=124, help="最短帧数（源帧率下）")
parser.add_argument("--max_duration", type=int, default=0)
parser.add_argument("--target_fps", type=int, default=24,
                    help="目标帧率（偏差 >0.5 才转码）；H3 为 24")
parser.add_argument("--target_audio_sr", type=int, default=0,
                    help="目标音频采样率，0=保留源采样率（默认；切片时再统一到 32 kHz）")
parser.add_argument("--filter_no_audio", action="store_true",
                    help="开启后过滤无音频视频(默认关闭)")
parser.add_argument("--enable_audio_max_volume_filter", action="store_true",
                    help="开启后才按 --min_audio_max_volume_db 过滤，默认关闭")
parser.add_argument("--min_audio_max_volume_db", type=float, default=-50.0,
                    help="最小可听峰值音量(dB)。低于阈值判为近静音")
parser.add_argument("--copy_mode", choices=["symlink", "copy", "none"], default="symlink")
parser.add_argument("--letterbox_samples", type=int, default=12,
                    help="黑边检测的采样点数，0=不检测；检测到的裁切框写入 crop 列，Step 2 切片时裁掉")

# OBS
add_obs_args(parser)

# Ray
parser.add_argument("--ray", action="store_true", help="启用 Ray 分布式模式")
parser.add_argument("--ray_num_gpus", type=float, default=0, help="GPU 资源 (无 GPU 时设 0)")
parser.add_argument("--ray_num_npus", type=float, default=0, help="NPU 自定义资源 (video_filter 不需要 NPU)")
parser.add_argument("--ray_num_cpus", type=float, default=1)
parser.add_argument("--concurrency", type=int, default=48)
args = parser.parse_args()
resolve_min_pixels(args.min_resolution)  # 非法值在启动时报错，而不是在每个 worker 里


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