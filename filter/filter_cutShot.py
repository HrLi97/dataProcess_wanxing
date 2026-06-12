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
import argparse
import logging
import re
import subprocess

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


class CsvWriter:
    """批量写入 CSV，全部收集后一次性写出避免频繁IO。"""

    def __init__(self, path, fieldnames):
        self.path = path
        self.fieldnames = fieldnames
        self._rows = []

    def write_row(self, row):
        self._rows.append(row)

    def flush(self):
        if not self._rows:
            return
        import csv
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)
        self._rows.clear()


# 标签说明：
# intra_label_id：0=General, 1=Dissolve, 2=Wipes, 3=Push, 4=Slide, 5=Zoom, 6=Fade, 7=Doorway, 8=Padding
# inter_label_id：0=New_Start, 1=Hard_Cut, 2=Transition_Source, 3=Transition, 4=Sudden_Jump, 5=Padding
# segment_role：shot=普通镜头, transition_source=慢转场前源镜头尾部, transition=慢转场过程帧, padding=无效 padding
# 常用筛选：干净镜头用 is_clean_shot；慢转场相关片段用 is_transition 或 is_transition_source。

SEGMENT_IO_BACKEND = "utils_inference.VideoInfoFfmpeg.load_data_by_index + utils_inference.VideoStreamWriterFfmpeg"
parser = argparse.ArgumentParser(description="OmniShotCut 场景切分 (Ray / Local)")
parser.add_argument("--input_csv", type=str, default="/mnt/workspace/lihaoran/data_process/test/0521/YouTube_talk.csv", help="输入 CSV，需含 file_path 列")
parser.add_argument("--output_dir", type=str, default="/mnt/workspace/lihaoran/data_process/test/result/0603/filter_cutShot/1", help="输出根目录")
parser.add_argument("--checkpoint_path", type=str, default=os.path.join(OMNISHOTCUT_ROOT, "checkpoints", "OmniShotCut_ckpt.pth"))
parser.add_argument("--mode", type=str, default="clean_shot", choices=["default", "clean_shot"], help="clean_shot 只保留 general 硬切镜头")
parser.add_argument("--num_context_frames", type=int, default=5, help="上下文帧数")
parser.add_argument("--cut_video", action="store_true", default=True, help="是否切分子视频")
parser.add_argument("--min_scene_frames", type=int, default=81, help="最短场景帧数，低于此阈值的场景会被丢弃")
parser.add_argument("--ray", action="store_true", help="启用 Ray 分布式模式")
parser.add_argument("--concurrency", type=int, default=16, help="Ray 并发数")
args = parser.parse_args()


def safe_name(text):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_")


def get_video_meta(video_path):
    from utils_inference.video.video_info_ffmpeg import VideoInfoFfmpeg

    video_info = VideoInfoFfmpeg(video_path)
    return {
        "fps": float(video_info.fps) if video_info.fps and video_info.fps > 0 else 0.0,
        "frame_count": int(video_info.num_frame),
        "width": int(video_info.width),
        "height": int(video_info.height),
        "channel": int(video_info.channel),
        "pix_fmt": video_info.pix_fmt,
        "color_space": video_info.color_space,
        "color_transfer": video_info.color_transfer,
        "color_primaries": video_info.color_primaries,
        "duration": video_info.duration,
        "bit_rate": video_info.bit_rate,
    }


def segment_role(inter_label):
    if inter_label == "Transition":
        return "transition"
    if inter_label == "Transition_Source":
        return "transition_source"
    if inter_label == "Padding":
        return "padding"
    return "shot"


def _get_full_audio(video_path, audio_cache_dir):
    audio_wav = os.path.join(audio_cache_dir, f"{os.path.basename(video_path)}.full.wav")
    if os.path.exists(audio_wav):
        return audio_wav
    os.makedirs(audio_cache_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        audio_wav,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return audio_wav


def _extract_audio_segment(video_path, start_sec, end_sec, audio_cache_dir):
    full_audio = _get_full_audio(video_path, audio_cache_dir)
    seg_audio = full_audio.replace(".full.wav", f".s{start_sec:.3f}e{end_sec:.3f}.wav")
    if os.path.exists(seg_audio):
        return seg_audio
    duration = end_sec - start_sec
    cmd = [
        "ffmpeg", "-y",
        "-i", full_audio,
        "-ss", str(start_sec),
        "-t", str(duration),
        "-acodec", "pcm_s16le",
        seg_audio,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return seg_audio


def write_segment_video_stream(video_path, scene, output_dir, fps):
    from utils_inference.video.video_info_ffmpeg import VideoInfoFfmpeg
    from utils_inference.video.video_stream_writer_ffmpeg import VideoStreamWriterFfmpeg
    from utils_inference.video.video_writer import TensorSaveVideo

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    segment_dir = os.path.join(output_dir, "segments", safe_name(base_name))
    os.makedirs(segment_dir, exist_ok=True)

    role = scene["segment_role"]
    intra_name = safe_name(scene["intra_label"])
    inter_name = safe_name(scene["inter_label"])
    out_name = (
        f"{scene['scene_idx']:04d}_"
        f"{scene['start_frame']:06d}-{scene['end_frame']:06d}_"
        f"{role}_{intra_name}_{inter_name}.mp4"
    )
    out_path = os.path.join(segment_dir, out_name)

    video_reader = VideoInfoFfmpeg(video_path)
    if video_reader.width <= 0 or video_reader.height <= 0:
        raise RuntimeError(f"无法读取视频尺寸: {video_path}")

    start_frame = int(scene["start_frame"])
    end_frame = min(int(scene["end_frame"]), int(video_reader.num_frame))
    frame_count = max(0, end_frame - start_frame)
    if frame_count <= 0:
        return {
            "segment_video_path": "",
            "status": False,
        }

    start_sec = start_frame / video_reader.fps if video_reader.fps and video_reader.fps > 0 else start_frame / fps
    end_sec = end_frame / video_reader.fps if video_reader.fps and video_reader.fps > 0 else end_frame / fps

    video_only_path = out_path + ".video_only.mp4"
    process = VideoStreamWriterFfmpeg.create_writer(
        save_path=video_only_path,
        width=video_reader.width,
        height=video_reader.height,
        fps=video_reader.fps if video_reader.fps and video_reader.fps > 0 else fps,
        vcodec="libx264",
        encode_fmt="yuv420p",
        input_fmt="gray" if video_reader.channel == 1 else video_reader.rgb_fmt,
        crf=18,
        loglevel="error",
    )
    writer = VideoStreamWriterFfmpeg(save_path=video_only_path, writer=process)
    written_frames = 0
    try:
        chunk_size = 120
        for chunk_start in range(start_frame, end_frame, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end_frame)
            video_data = video_reader.load_data_by_index(
                indexs=list(range(chunk_start, chunk_end)),
                device="cpu",
                format="NHWC",
            )
            if video_data is None:
                continue
            TensorSaveVideo.videodata_to_stream(video_data, writer=writer)
            written_frames += int(video_data.num_frame)
    finally:
        writer.Close()
        video_reader.close()

    if written_frames > 0:
        audio_cache_dir = os.path.join(output_dir, "_audio_cache")
        audio_segment = _extract_audio_segment(video_path, start_sec, end_sec, audio_cache_dir)
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", video_only_path, "-i", audio_segment,
                 "-c:v", "copy", "-c:a", "aac", "-shortest",
                 "-loglevel", "error", out_path],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
            )
        except subprocess.CalledProcessError:
            import shutil
            shutil.copy2(video_only_path, out_path)
        finally:
            os.remove(video_only_path)
    else:
        out_path = ""

    return {
        "segment_video_path": out_path if written_frames > 0 else "",
        "status": written_frames > 0,
    }


# ────────────────────── 核心推理函数 ──────────────────────
def run_scene_detection(video_path, model, model_args, num_context_frames, mode="clean_shot", min_scene_frames=24):
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

    # clean_shot 模式：只保留 general 硬切类型的纯净 shot
    if mode == "clean_shot":
        general_idx = unique_intra_label_mapping["general"]
        keep = [i for i, lab in enumerate(pred_intra_labels) if lab == general_idx]
        pred_ranges = [pred_ranges[i] for i in keep]
        pred_intra_labels = [pred_intra_labels[i] for i in keep]
        pred_inter_labels = [pred_inter_labels[i] for i in keep]

    TRANSITION_INTER_ID = 3          # "Transition" 转场过程
    TRANSITION_SOURCE_INTER_ID = 2   # "Transition_Source" 转场前源

    scenes = []
    for i, (rng, intra, inter) in enumerate(zip(pred_ranges, pred_intra_labels, pred_inter_labels)):
        start_f, end_f = int(rng[0]), int(rng[1])
        if end_f - start_f < min_scene_frames:
            continue

        is_transition = (int(inter) == TRANSITION_INTER_ID)
        is_transition_source = (int(inter) == TRANSITION_SOURCE_INTER_ID)
        intra_label = intra_int2string.get(int(intra), str(intra))
        inter_label = inter_int2string.get(int(inter), str(inter))
        duration_frames = end_f - start_f

        scenes.append({
            "scene_idx": len(scenes),
            "start_frame": start_f,
            "end_frame": end_f,
            "duration_frames": duration_frames,
            "intra_label": intra_label,
            "inter_label": inter_label,
            "intra_label_id": int(intra),
            "inter_label_id": int(inter),
            "segment_role": segment_role(inter_label),
            "is_transition": is_transition,
            "is_transition_source": is_transition_source,
            "is_clean_shot": not is_transition and not is_transition_source and inter_label != "Padding",
        })

    clean_shots = [s for s in scenes if s["is_clean_shot"]]

    print(f"[DEBUG] all segments: {len(scenes)}, clean shots: {len(clean_shots)}, transitions: {len(scenes) - len(clean_shots)}")
    for s in scenes:
        tag = " *** TRANSITION ***" if s["is_transition"] else ""
        print(f"  [{s['start_frame']:>5d} - {s['end_frame']:>5d}] intra_id={s['intra_label_id']} inter_id={s['inter_label_id']}{tag}")

    return scenes, fps


def cut_scenes_to_video(video_path, scenes, fps, output_dir):
    rows = []
    for sc in scenes:
        write_info = {
            "segment_video_path": "",
            "status": False,
        }
        if sc.get("is_clean_shot"):
            try:
                write_info = write_segment_video_stream(video_path, sc, output_dir, fps)
            except Exception as e:
                write_info = {
                    "segment_video_path": "",
                    "status": False,
                    "write_error": str(e),
                }
                logger.warning(f"segment 写出失败: {video_path}, scene={sc['scene_idx']}, err={e}")
        row = dict(sc)
        row.update(write_info)
        rows.append(row)
    return rows


def process_single_video(video_path, model, model_args, opt):
    scenes, fps = run_scene_detection(
        video_path, model, model_args,
        opt.num_context_frames, opt.mode, opt.min_scene_frames,
    )
    segment_rows = cut_scenes_to_video(video_path, scenes, fps, opt.output_dir)

    out_rows = []
    for row in segment_rows:
        out_rows.append({
            "video_path": video_path,
            "fps": fps,
            "scene_idx": row["scene_idx"],
            "start_frame": row["start_frame"],
            "end_frame": row["end_frame"],
            "segment_video_path": row.get("segment_video_path", ""),
            "status": row.get("status", False),
            "is_clean_shot": row.get("is_clean_shot", False),
            "intra_label": row.get("intra_label", ""),
            "inter_label": row.get("inter_label", ""),
            "segment_role": row.get("segment_role", ""),
        })

    logger.info(f"✓ {video_path} → {len(scenes)} scenes, {sum(1 for r in out_rows if r['is_clean_shot'])} clean_shots")
    return out_rows


# ────────────────────── Worker (for Ray) ──────────────────
class Worker:
    def __init__(self):
        os.chdir(OMNISHOTCUT_ROOT)
        sys.path.insert(0, OMNISHOTCUT_ROOT)
        from test_code.inference import load_model
        self.model, self.model_args = load_model(args.checkpoint_path)
        logger.info("Worker initialized: OmniShotCut model loaded.")

    def __call__(self, item):
        vid_path = item["file_path"] if isinstance(item["file_path"], str) else item["file_path"][0]
        return process_single_video(vid_path, self.model, self.model_args, args)


# ────────────────────────── main ──────────────────────────
if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)
    scenes_csv = os.path.join(args.output_dir, "scenes.csv")
    fieldnames = ["video_path", "fps", "scene_idx", "start_frame", "end_frame",
                  "segment_video_path", "status", "is_clean_shot",
                  "intra_label", "inter_label", "segment_role"]
    csv_writer = CsvWriter(scenes_csv, fieldnames)

    if args.ray:
        import ray
        ray.init(address="auto",
                 runtime_env={"env_vars": 
                                        {
                                            "PYTHONPATH": OMNISHOTCUT_ROOT
                                        },
                                "working_dir": OMNISHOTCUT_ROOT
                            }
                )
        samples = ray.data.read_csv(args.input_csv)
        predictions = samples.map(Worker, num_gpus=1, concurrency=args.concurrency)
        rows = predictions.take_all()
        for r in rows:
            
            for row in r:
                csv_writer.write_row(row)
    else:
        os.chdir(OMNISHOTCUT_ROOT)
        from test_code.inference import load_model
        model, model_args = load_model(args.checkpoint_path)
        samples = list(csv.DictReader(open(args.input_csv, "r", encoding="utf-8-sig")))
        logger.info(f"本地模式: 共 {len(samples)} 条视频")
        for item in samples:
            for row in process_single_video(item["file_path"], model, model_args, args):
                csv_writer.write_row(row)

    csv_writer.flush()
    logger.info(f"完成: {scenes_csv}")
