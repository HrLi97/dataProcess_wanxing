"""
基于 OmniShotCut 的视频场景切分脚本 (支持 Ray 分布式 + 本地模式)
Usage:
    # 本地单机模式
    python filter/filter_cutShot.py --input_csv /path/to/input.csv --output_dir /path/to/output
    # Ray 分布式模式
    python filter/filter_cutShot.py --input_csv /path/to/input.csv --output_dir /path/to/output --ray
CSV 要求:  必须包含 file_path 列，指向视频文件路径
输出: output_dir/scenes.csv（每个 clean_shot 一行）
"""

import os
import sys
import csv
import fcntl
import argparse
import logging
import re
import subprocess
from pathlib import Path

_FILTER_ROOT = str(Path(__file__).resolve().parents[1]) 
if _FILTER_ROOT not in sys.path:
    sys.path.insert(0, _FILTER_ROOT)

from filter.filter_utils import unique_output_name, safe_name, init_csv, load_processed_inputs, INPUT_PATH_FIELD
from filter.obs_helper import add_obs_args, ObsIO

# ── 将 OmniShotCut 项目根目录加入 sys.path ──
OMNISHOTCUT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "third_part", "OmniShotCut")
)
sys.path.insert(0, OMNISHOTCUT_ROOT)
UTILS_INFERENCE_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "third_part", "utils_inference", "src")
)
sys.path.insert(0, UTILS_INFERENCE_SRC)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


CUTSHOT_FIELDS = [
    "video_path", "duration_frames","transition_type",
    "is_clean_shot", "cutshot_reject_reason",
    "num_shots", "cut_frames",  # 多镜头片段：镜头数、片段内切点（相对片段第 0 帧，分号分隔；单镜头为空）
    "source_path", "clip_start_sec", "clip_end_sec",  # 原始视频（Step 1 的 _source_key）和片段在其中的起止秒，用来对回原片标注
]


def safe_name(text):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_")


def parse_crop(value):
    """Step 1 写的 crop 列（w:h:x:y）→ (w, h, x, y)；空值、NaN（pandas/Ray 读空列）或格式不对时返回 None（不裁）。"""
    if not isinstance(value, str) or value.count(":") != 3:
        return None
    try:
        w, h, x, y = (int(v) for v in value.split(":"))
    except ValueError:
        return None
    return (w, h, x, y) if w > 0 and h > 0 and x >= 0 and y >= 0 else None


def write_segment_video(video_path, scene, output_dir, fps, crop=None):
    """用 ffmpeg 切出片段并重编码（libx264 CRF18，8-bit yuv420p）。

    视频用 -frames:v 保证正好 frame_count 帧（H3 要求 17n+5 帧）；只取第一条视频轨和第一条音轨（可无），
    音频按 --audio_sr / --audio_channels 统一（默认 32 kHz 立体声，H3 音频 VAE 的格式）。
    crop=(w, h, x, y) 时同时裁掉黑边（Step 1 检测），不多一次编码。
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # 目录名加源视频 hash 短码，避免不同源同名时在 segments/ 下目录冲突
    src_tag = unique_output_name(video_path, suffix="").rsplit("__", 1)[-1]
    segment_dir = os.path.join(output_dir, "segments", f"{safe_name(base_name)}__{src_tag}")
    os.makedirs(segment_dir, exist_ok=True)

    t_type = safe_name(scene["transition_type"])
    out_name = (
        f"{scene['start_frame']:06d}-{scene['end_frame']:06d}_"
        f"{t_type}.mp4"
    )
    out_path = os.path.join(segment_dir, out_name)

    start_frame = int(scene["start_frame"])
    end_frame = int(scene["end_frame"])
    frame_count = end_frame - start_frame
    if frame_count <= 0:
        return {"segment_video_path": "", "status": False}

    actual_fps = fps if fps and fps > 0 else 25.0
    start_sec = start_frame / actual_fps
    # 多给半帧时长，由 -frames:v 精确截到 frame_count 帧（只用 -t 时可能少 1 帧）
    duration_sec = (frame_count + 0.5) / actual_fps

    audio_opts = ["-c:a", "aac", "-b:a", "192k"]
    if args.audio_sr > 0:
        audio_opts += ["-ar", str(args.audio_sr)]
    if args.audio_channels > 0:
        audio_opts += ["-ac", str(args.audio_channels)]
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.4f}",
        "-i", video_path,
        "-t", f"{duration_sec:.4f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        "-frames:v", str(frame_count),
        *(["-vf", "crop={}:{}:{}:{}".format(*crop)] if crop else []),
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        *audio_opts,
        "-loglevel", "error",
        out_path,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning(f"ffmpeg 切割失败: {video_path} [{start_frame}-{end_frame}], err={e}")
        return {"segment_video_path": "", "status": False}

    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        return {"segment_video_path": out_path, "status": True}
    return {"segment_video_path": "", "status": False}


TARGET_FRAME_COUNT = 124  # 由 --clip_frames 覆盖；H3 默认 124 帧 = 17*7+5（5.17 s @ 24 fps）
TRIM_FRAMES = 3  # 由 --trim_frames 覆盖：每个镜头两端各裁掉的帧数，避开切点附近的闪帧和过渡帧
MIN_SHOT_FRAMES = 12  # 由 --min_shot_frames 覆盖：多镜头片段里每个镜头（含被片段截断的首尾两段）至少几帧
MAX_SHOTS = 4  # 由 --max_shots 覆盖：多镜头片段最多几个镜头（H3 团队 prompt 校验的上限也是 4）


def _split_fragment(fragment_start, fragment_end):
    fragment_start += TRIM_FRAMES
    fragment_end -= TRIM_FRAMES
    dur = fragment_end - fragment_start
    if dur < TARGET_FRAME_COUNT:
        return []

    results = []
    pos = fragment_start
    while pos + TARGET_FRAME_COUNT <= fragment_end:
        results.append({
            "start_frame": pos,
            "end_frame": pos + TARGET_FRAME_COUNT,
        })
        pos += TARGET_FRAME_COUNT

    return results


def _first_problem_jump(bounds):
    """片段内镜头边界 [起点, 切点…, 终点] 有问题时，返回下一次开窗的起点；没问题返回 None。"""
    n = len(bounds) - 1  # 镜头数
    for k in range(n):
        if bounds[k + 1] - bounds[k] < MIN_SHOT_FRAMES:
            cut = bounds[k + 1] if k + 1 < n else bounds[k]  # 末段过短时，问题出在它前面那个切点
            return cut + TRIM_FRAMES
    if n > MAX_SHOTS:
        return bounds[1] + TRIM_FRAMES
    return None


def _multi_shot_windows(ranges, intra_labels, inter_labels, inter_names, general_idx):
    """多镜头模式：把硬切相接的 general 段连成一段（run），在 run 里按 TARGET_FRAME_COUNT 连续开窗，窗口可以跨硬切。

    转场段（叠化、划像等）和 Transition / Transition_Source / Padding 边界会打断 run。窗口两端离 run 边界至少
    TRIM_FRAMES 帧；窗口里每个镜头至少 MIN_SHOT_FRAMES 帧（所以切点也离窗口两端至少这么远），镜头数不超过
    MAX_SHOTS；不满足就把窗口起点挪到出问题的切点之后再试。
    """
    runs, cur = [], None
    for (s, e), intra, inter in zip(ranges, intra_labels, inter_labels):
        s, e = int(s), int(e)
        name = inter_names.get(int(inter), str(inter))
        if int(intra) != general_idx or name in ("Transition", "Transition_Source", "Padding"):
            if cur:
                runs.append(cur)
            cur = None
            continue
        if cur and s == cur["end"]:  # 和上一段硬切相接
            cur["cuts"].append(s)
            cur["end"] = e
        else:
            if cur:
                runs.append(cur)
            cur = {"start": s, "end": e, "cuts": [], "type": name}
    if cur:
        runs.append(cur)

    scenes = []
    for run in runs:
        pos, stop = run["start"] + TRIM_FRAMES, run["end"] - TRIM_FRAMES
        while pos + TARGET_FRAME_COUNT <= stop:
            end = pos + TARGET_FRAME_COUNT
            inner = [c for c in run["cuts"] if pos < c < end]
            jump = _first_problem_jump([pos] + inner + [end])
            if jump is not None:
                pos = jump
                continue
            scenes.append({
                "start_frame": pos,
                "end_frame": end,
                "duration_frames": TARGET_FRAME_COUNT,
                "transition_type": f"{len(inner) + 1}shot",
                "is_clean_shot": True,
                "num_shots": len(inner) + 1,
                "cut_frames": [c - pos for c in inner],
            })
            pos = end
    return scenes


# ────────────────────── 核心推理函数 ──────────────────────
def run_scene_detection(video_path, model, model_args, num_context_frames,
                        mode="clean_shot"):
    import numpy as np
    from test_code.inference import single_video_inference
    from config.label_correspondence import (
        unique_intra_label_mapping,
        intra_int2string,
        inter_int2string,
    )

    pred_ranges, pred_intra_labels, pred_inter_labels, video_np_full, fps = \
        single_video_inference(video_path, model, model_args, num_context_frames)

    print(f"[DEBUG] video: {os.path.basename(video_path)}, fps={fps}, total_frames={len(video_np_full)}")
    print(f"[DEBUG] pred_ranges ({len(pred_ranges)}): {pred_ranges}")
    print(f"[DEBUG] pred_intra_labels: {pred_intra_labels}")
    print(f"[DEBUG] pred_inter_labels: {pred_inter_labels}")

    if mode == "multi_shot":
        scenes = _multi_shot_windows(pred_ranges, pred_intra_labels, pred_inter_labels, inter_int2string,
                                     unique_intra_label_mapping["general"])
        print(f"[DEBUG] multi_shot windows: {len(scenes)}")
        for sc in scenes:
            print(f"  [{sc['start_frame']:>5d} - {sc['end_frame']:>5d}] shots={sc['num_shots']} cuts={sc['cut_frames']}")
        return scenes, fps

    if mode == "clean_shot":
        general_idx = unique_intra_label_mapping["general"]
        keep = [i for i, lab in enumerate(pred_intra_labels) if lab == general_idx]
        pred_ranges = [pred_ranges[i] for i in keep]
        pred_intra_labels = [pred_intra_labels[i] for i in keep]
        pred_inter_labels = [pred_inter_labels[i] for i in keep]

    TRANSITION_INTER_ID = 3
    TRANSITION_SOURCE_INTER_ID = 2

    # 单 segment 视频（未检测到切点）= 单连续镜头，直接视为 clean shot
    single_segment = (len(pred_ranges) == 1)

    raw_scenes = []
    for i, (rng, intra, inter) in enumerate(zip(pred_ranges, pred_intra_labels, pred_inter_labels)):
        start_f, end_f = int(rng[0]), int(rng[1])
        is_transition = (int(inter) == TRANSITION_INTER_ID)
        is_transition_source = (int(inter) == TRANSITION_SOURCE_INTER_ID)
        inter_label = inter_int2string.get(int(inter), str(inter))
        # transition_type: 单segment视频 → "none", 否则用 inter_label
        if single_segment:
            t_type = "none"
            clean = True
        else:
            t_type = inter_label
            clean = not is_transition and not is_transition_source and inter_label != "Padding"
        raw_scenes.append({
            "start_frame": start_f,
            "end_frame": end_f,
            "duration_frames": end_f - start_f,
            "transition_type": t_type,
            "is_clean_shot": clean,
        })

    scenes = []
    for sc in raw_scenes:
        if not sc["is_clean_shot"]:
            scenes.append(sc)
            continue

        # 所有 clean shot 统一先裁两端 TRIM_FRAMES，再切成固定 TARGET_FRAME_COUNT 帧
        dur = sc["duration_frames"]
        sub_fragments = _split_fragment(sc["start_frame"], sc["end_frame"])
        if not sub_fragments:
            print(f"  [SKIP] [{sc['start_frame']}-{sc['end_frame']}] dur={dur} < {TARGET_FRAME_COUNT}, dropped")
            continue

        for frag in sub_fragments:
            scenes.append({
                "start_frame": frag["start_frame"],
                "end_frame": frag["end_frame"],
                "duration_frames": frag["end_frame"] - frag["start_frame"],
                "transition_type": sc["transition_type"],
                "is_clean_shot": True,
            })

    clean_shots = [s for s in scenes if s["is_clean_shot"]]
    print(f"[DEBUG] all segments: {len(scenes)}, clean shots: {len(clean_shots)}, rejected: {len(scenes) - len(clean_shots)}")
    for s in scenes:
        tag = "" if s["is_clean_shot"] else " *** REJECTED ***"
        print(f"  [{s['start_frame']:>5d} - {s['end_frame']:>5d}] type={s['transition_type']}{tag}")

    return scenes, fps


def cut_scenes_to_video(video_path, scenes, fps, output_dir, crop=None):
    rows = []
    for sc in scenes:
        write_info = {
            "segment_video_path": "",
            "status": False,
        }
        if sc.get("is_clean_shot"):
            try:
                write_info = write_segment_video(video_path, sc, output_dir, fps, crop=crop)
            except Exception as e:
                write_info = {
                    "segment_video_path": "",
                    "status": False,
                }
                logger.warning(f"segment 写出失败: {video_path}, [{sc['start_frame']}-{sc['end_frame']}], err={e}")
        row = dict(sc)
        row.update(write_info)
        rows.append(row)
    return rows


def process_single_video(video_path, model, model_args, opt, input_item):
    """处理单个视频，返回多行结果（每个 segment 一行）。"""
    if not os.path.isfile(video_path):
        return [{
            "video_path": "",
            "duration_frames": "",
            "transition_type": "", "is_clean_shot": False,
            "cutshot_reject_reason": "file_not_found",
        }]

    try:
        scenes, fps = run_scene_detection(
            video_path, model, model_args,
            opt.num_context_frames, opt.mode,
        )
    except Exception as e:
        logger.warning(f"场景检测失败: {video_path}, err={e}")
        return [{
            "video_path": "",
            "duration_frames": "",
            "transition_type": "", "is_clean_shot": False,
            "cutshot_reject_reason": "detection_failed",
        }]

    crop = parse_crop((input_item or {}).get("crop"))
    segment_rows = cut_scenes_to_video(video_path, scenes, fps, opt.output_dir, crop=crop)

    out_rows = []
    for seg_row in segment_rows:
        row = {
            "video_path": seg_row.get("segment_video_path", ""),
            "duration_frames": seg_row.get("duration_frames", ""),
            "transition_type": seg_row.get("transition_type", ""),
            "is_clean_shot": seg_row.get("is_clean_shot", False),
            "num_shots": seg_row.get("num_shots", 1),
            "cut_frames": ";".join(str(c) for c in seg_row.get("cut_frames", [])),
            "source_path": (input_item or {}).get("_source_key") or video_path,
            "clip_start_sec": round(int(seg_row["start_frame"]) / fps, 4) if fps else "",
            "clip_end_sec": round(int(seg_row["end_frame"]) / fps, 4) if fps else "",
        }
        if not seg_row.get("is_clean_shot"):
            row["cutshot_reject_reason"] = "not_clean_shot"
        elif not seg_row.get("segment_video_path"):
            row["cutshot_reject_reason"] = "write_failed"
        else:
            row["cutshot_reject_reason"] = ""
        out_rows.append(row)

    logger.info(f"✓ {video_path} → {len(scenes)} segments, {sum(1 for r in out_rows if not r['cutshot_reject_reason'])} qualified")
    return out_rows


# ────────────────────── Worker ──────────────────────────────
class Worker:
    def __init__(self):
        import sys
        import importlib
        from pathlib import Path
        try:
            _project_root = str(Path(__file__).resolve().parents[1])  # data_process/
        except NameError:
            _project_root = str(Path.cwd())
        if _project_root not in sys.path:
            sys.path.insert(0, _project_root)
        if "filter" not in sys.modules or sys.modules.get("filter") is None:
            try:
                importlib.import_module("filter")
            except Exception:
                pass

        os.chdir(OMNISHOTCUT_ROOT)
        sys.path.insert(0, OMNISHOTCUT_ROOT)
        # torch>=2.6 起 torch.load 默认 weights_only=True；OmniShotCut ckpt 的 state_dict["args"] 是 argparse.Namespace，
        # 不加白名单的话 third_part/OmniShotCut/test_code/inference.py 里的 torch.load 会抛 UnpicklingError
        import torch
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([argparse.Namespace])
        from test_code.inference import load_model
        self.model, self.model_args = load_model(args.checkpoint_path)
        # 初始化输出
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.all_csv = os.path.join(self.output_dir, "2_cutshot_all.csv")
        self.qual_csv = os.path.join(self.output_dir, "2_cutshot_qualified.csv")
        self.fieldnames = list(CUTSHOT_FIELDS)
        # 以文件实际表头追加（旧 CSV 缺列时原地补列）；all CSV 额外记录输入路径作断点续跑 key
        self.csv_fields = {
            self.all_csv: init_csv(self.all_csv, self.fieldnames + [INPUT_PATH_FIELD]),
            self.qual_csv: init_csv(self.qual_csv, self.fieldnames),
        }
        # 断点续跑：按输入路径判断（旧 CSV 无 input_video_path 列时回退到原 video_path 逻辑）
        self.processed = load_processed_inputs(self.all_csv)
        # OBS
        self.obs_io = ObsIO(args)
        logger.info(f"Worker initialized: OmniShotCut model loaded. OBS={self.obs_io.enabled}. 已处理 {len(self.processed)} 条。")

    def _append_csv(self, filepath, row):
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            csv.DictWriter(f, fieldnames=self.csv_fields[filepath], extrasaction="ignore").writerow(row)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)

    def __call__(self, item):
        vid_path = item.get(args.video_path_column, "")
        if not vid_path:
            row = {k: "" for k in CUTSHOT_FIELDS}
            row["cutshot_reject_reason"] = "empty_path"
            self._append_csv(self.all_csv, row)
            return row
        if vid_path in self.processed:
            return {k: "" for k in CUTSHOT_FIELDS}

        local_video = None
        try:
            # ── OBS: 下载源视频 ──
            if self.obs_io.enabled:
                local_video = self.obs_io.download(vid_path)
            else:
                local_video = vid_path

            rows = process_single_video(local_video, self.model, self.model_args, args, item)
            if not rows:
                # 一个 segment 都没切出（如 clean shot 全部短于 TARGET_FRAME_COUNT 被丢弃）也记一行，续跑时不再重算
                rows = [{"video_path": "", "duration_frames": "", "transition_type": "",
                         "is_clean_shot": False, "cutshot_reject_reason": "no_segments"}]

            # ── OBS: 上传切割出的合格 segment，CSV 写 obs_key ──
            if self.obs_io.enabled:
                for row in rows:
                    local_seg = row.get("video_path", "")
                    if local_seg and os.path.isfile(local_seg) and not row.get("cutshot_reject_reason"):
                        obs_key = self.obs_io.make_obs_key(
                            "segments", os.path.basename(local_seg)
                        )
                        self.obs_io.upload(local_seg, obs_key)
                        row["video_path"] = obs_key
                        self.obs_io.cleanup(local_seg)

            for row in rows:
                # all CSV 每行（含拒绝行）带本步输入路径；返回给 Ray 的 row 不加列，保持各行 schema 一致
                self._append_csv(self.all_csv, {**row, INPUT_PATH_FIELD: vid_path})
                if not row.get("cutshot_reject_reason"):
                    self._append_csv(self.qual_csv, row)
            self.processed.add(vid_path)
            return rows[0] if rows else {k: "" for k in CUTSHOT_FIELDS}
        finally:
            # ── OBS: 清理下载的源视频 + 本地 segment 目录 ──
            if self.obs_io.enabled and local_video and local_video != vid_path:
                self.obs_io.cleanup(local_video)
                # 清理本地 segments 目录（已上传）
                seg_dir = os.path.join(args.output_dir, "segments")
                if os.path.isdir(seg_dir):
                    import shutil
                    shutil.rmtree(seg_dir, ignore_errors=True)


# transition_type 取值:
#   "none"              = 单segment视频，无切点
#   "Hard_Cut"          = 硬切 (clean shot)
#   "New_Start"         = 新起始 (clean shot)
#   "Sudden_Jump"       = 突然跳变 (clean shot)
#   "Transition"        = 慢转场 (rejected)
#   "Transition_Source" = 转场源镜头尾部 (rejected)
#   "Padding"           = 无效 padding (rejected)
parser = argparse.ArgumentParser(description="OmniShotCut 场景切分 (Ray / Local)")
parser.add_argument("--input_csv", type=str, default="/mnt/workspace/lihaoran/data_process/test/result/0618/video_filter/no_pepole/video_filter_qualified.csv", help="输入 CSV")
parser.add_argument("--output_dir", type=str, default="/mnt/workspace/lihaoran/data_process/test/result/0618/filter_shotcut/no_pepole/", help="输出根目录")
parser.add_argument("--video_path_column", type=str, default="video_path", help="输入 CSV 中视频路径列名")
parser.add_argument("--checkpoint_path", type=str, default=os.path.join(OMNISHOTCUT_ROOT, "checkpoints", "OmniShotCut_ckpt.pth"))
parser.add_argument("--mode", type=str, default="multi_shot", choices=["default", "clean_shot", "multi_shot"],
                    help="clean_shot：片段只取单个镜头内部；multi_shot：片段可以跨硬切（H3 支持多镜头描述），切点写进 cut_frames")
parser.add_argument("--min_shot_frames", type=int, default=12, help="multi_shot：片段里每个镜头至少几帧")
parser.add_argument("--max_shots", type=int, default=4, help="multi_shot：一个片段最多几个镜头")
parser.add_argument("--num_context_frames", type=int, default=5, help="上下文帧数")
parser.add_argument("--clip_frames", type=int, default=124,
                    help="每个片段的帧数；MiniMax-H3 要求 17n+5（如 73、124、141、192）")
parser.add_argument("--trim_frames", type=int, default=3, help="每个镜头两端各裁掉的帧数")
parser.add_argument("--audio_sr", type=int, default=32000, help="片段音频采样率，0=保留源采样率；H3 为 32000")
parser.add_argument("--audio_channels", type=int, default=2, help="片段音频声道数，0=保留源声道；H3 为立体声")

# OBS
add_obs_args(parser)

# Ray
parser.add_argument("--ray", action="store_true", help="启用 Ray 分布式模式")
parser.add_argument("--ray_num_gpus", type=float, default=0.5)
parser.add_argument("--ray_num_cpus", type=float, default=0)
parser.add_argument("--concurrency", type=int, default=4, help="Ray 并发数")
args = parser.parse_args()
TARGET_FRAME_COUNT = args.clip_frames
TRIM_FRAMES = args.trim_frames
MIN_SHOT_FRAMES = args.min_shot_frames
MAX_SHOTS = args.max_shots
if (TARGET_FRAME_COUNT - 5) % 17 != 0:
    print(f"[WARN] --clip_frames {TARGET_FRAME_COUNT} 不是 17n+5，MiniMax-H3 的视频 VAE 不能直接编码这个帧数")

# ────────────────────────── main ──────────────────────────
if __name__ == "__main__":

    # ── 去重：按 video_path_column 去重，避免重复处理 ──
    import pandas as pd
    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    before_cnt = len(df)
    df = df.drop_duplicates(subset=[args.video_path_column], keep="first")
    after_cnt = len(df)
    if before_cnt != after_cnt:
        logger.info(f"去重: {before_cnt} → {after_cnt} (去除 {before_cnt - after_cnt} 条重复)")
    deduped_csv = args.input_csv.replace(".csv", "_deduped.csv")
    df.to_csv(deduped_csv, index=False)

    if not args.ray:
        # ── 本地模式 ──
        samples = list(csv.DictReader(open(deduped_csv, "r", encoding="utf-8-sig")))
        logger.info(f"本地模式: 共 {len(samples)} 条视频")
        worker = Worker()
        for i, item in enumerate(samples, 1):
            logger.info(f"[{i}/{len(samples)}]")
            worker(item)
    else:
        # ── Ray 分布式模式 ──
        import ray
        ray.init(address="auto",
                 runtime_env={"env_vars": 
                                {"PYTHONPATH": f"{_FILTER_ROOT}:{OMNISHOTCUT_ROOT}",
                                "DATA_PROCESS_ROOT": _FILTER_ROOT},
                                "working_dir": OMNISHOTCUT_ROOT
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

    logger.info(f"完成: {args.output_dir}")
