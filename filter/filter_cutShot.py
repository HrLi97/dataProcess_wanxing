"""
基于 OmniShotCut 的视频场景切分脚本 (支持 Ray 分布式 + 本地模式)
Usage:
    # 本地单机模式
    python scene_split_omnishotcut.py --input_csv /path/to/input.csv --output_dir /path/to/output --is_local True
    # Ray 分布式模式
    python scene_split_omnishotcut.py --input_csv /path/to/input.csv --output_dir /path/to/output --is_local False
CSV 要求:  必须包含 file_path 列，指向视频文件路径
输出: 每条视频在 output_dir 下生成 <video_name>_scenes.json 同时可选切分子视频
"""

import os
import sys
import csv
import json
import argparse
import logging
import re

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


# 标签说明：
# intra_label_id：0=General, 1=Dissolve, 2=Wipes, 3=Push, 4=Slide, 5=Zoom, 6=Fade, 7=Doorway, 8=Padding
# inter_label_id：0=New_Start, 1=Hard_Cut, 2=Transition_Source, 3=Transition, 4=Sudden_Jump, 5=Padding
# segment_role：shot=普通镜头, transition_source=慢转场前源镜头尾部, transition=慢转场过程帧, padding=无效 padding
# 常用筛选：干净镜头用 is_clean_shot；慢转场相关片段用 is_transition 或 is_transition_source。

SEGMENT_IO_BACKEND = "utils_inference.VideoInfoFfmpeg.load_data_by_index + utils_inference.VideoStreamWriterFfmpeg"
parser = argparse.ArgumentParser(description="OmniShotCut 场景切分 (Ray / Local)")
parser.add_argument("--input_csv", type=str, default="/mnt/workspace/lihaoran/data_process/test/0521/YouTube_talk.csv", help="输入 CSV，需含 file_path 列")
parser.add_argument("--output_dir", type=str, default="/mnt/workspace/lihaoran/data_process/test/0521/YouTube_talk/result/", help="输出根目录")
parser.add_argument("--checkpoint_path", type=str,default=os.path.join(OMNISHOTCUT_ROOT, "checkpoints", "OmniShotCut_ckpt.pth"))
parser.add_argument("--mode", type=str, default="default", choices=["default", "clean_shot"],help="clean_shot 只保留 general 硬切镜头; default 保留全部标签")
parser.add_argument("--num_context_frames", type=int, default=5, help="上下文帧数")
parser.add_argument("--cut_video", action="store_true", default=True, help="是否切分子视频")
parser.add_argument("--no_cut_video", action="store_false", dest="cut_video", help="不切分子视频")
parser.add_argument("--min_scene_frames", type=int, default=1, help="最短场景帧数，低于此阈值的场景会被丢弃")
parser.add_argument("--is_local", type=str, default="True", help="True=本地; False=Ray分布式")
parser.add_argument("--concurrency", type=int, default=16, help="Ray 并发数")
parser.add_argument("--ray_log_dir", type=str, default="", help="Ray 日志目录")
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
            "write_success": False,
        }

    process = VideoStreamWriterFfmpeg.create_writer(
        save_path=out_path,
        width=video_reader.width,
        height=video_reader.height,
        fps=video_reader.fps if video_reader.fps and video_reader.fps > 0 else fps,
        vcodec="libx264",
        encode_fmt="yuv420p",
        input_fmt="gray" if video_reader.channel == 1 else video_reader.rgb_fmt,
        crf=18,
        loglevel="error",
    )
    writer = VideoStreamWriterFfmpeg(save_path=out_path, writer=process)
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

    return {
        "segment_video_path": out_path if written_frames > 0 else "",
        "write_success": written_frames > 0,
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

    # ── 构建完整的 scenes 列表（包含转场信息）──
    # inter_label 含义:
    #   0=New_Start, 1=Hard_Cut, 2=Transition_Source, 3=Transition, 4=Sudden_Jump
    # 策略：
    #   - inter=0,1,4 的片段是纯净 shot（可直接使用）
    #   - inter=2 是转场前源镜头尾部（可选保留）
    #   - inter=3 是转场过程帧（应排除或标记）
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
            "intra_label_id": int(intra),
            "inter_label_id": int(inter),
            "intra_label": intra_label,
            "inter_label": inter_label,
            "segment_role": segment_role(inter_label),
            "is_transition": is_transition,
            "is_transition_source": is_transition_source,
            "is_clean_shot": not is_transition and not is_transition_source and inter_label != "Padding",
        })

    # ── 提取纯净 shot（排除转场帧）──
    clean_shots = [s for s in scenes if s["is_clean_shot"]]

    print(f"[DEBUG] all segments: {len(scenes)}, clean shots: {len(clean_shots)}, transitions: {len(scenes) - len(clean_shots)}")
    for s in scenes:
        tag = " *** TRANSITION ***" if s["is_transition"] else ""
        print(f"  [{s['start_frame']:>5d} - {s['end_frame']:>5d}] intra={s['intra_label']:<10s} inter={s['inter_label']:<20s}{tag}")

    return scenes, fps


def cut_scenes_to_video(video_path, scenes, fps, output_dir):
    rows = []
    for sc in scenes:
        try:
            write_info = write_segment_video_stream(video_path, sc, output_dir, fps)
        except Exception as e:
            write_info = {
                "segment_video_path": "",
                "write_success": False,
                "write_error": str(e),
            }
            logger.warning(f"segment 写出失败: {video_path}, scene={sc['scene_idx']}, err={e}")
        row = dict(sc)
        row.update(write_info)
        rows.append(row)
    return rows


def process_single_video(video_path, model, model_args, opt):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    json_path = os.path.join(opt.output_dir, f"{base_name}_scenes.json")

    try:
        scenes, fps = run_scene_detection(
            video_path, model, model_args,
            opt.num_context_frames, opt.mode, opt.min_scene_frames,
        )
        source_meta = get_video_meta(video_path)
        if opt.cut_video and scenes:
            segment_rows = cut_scenes_to_video(video_path, scenes, fps, opt.output_dir)
        else:
            segment_rows = []
            for scene in scenes:
                row = dict(scene)
                row.update({
                    "segment_video_path": "",
                    "write_success": False,
                })
                segment_rows.append(row)

        clean_shot_rows = [s for s in segment_rows if s.get("is_clean_shot")]
        transition_rows = [s for s in segment_rows if s.get("is_transition")]
        transition_source_rows = [s for s in segment_rows if s.get("is_transition_source")]

        result = {
            "video_path": video_path,
            "video_name": os.path.basename(video_path),
            "detected_fps": fps,
            "source": source_meta,
            "run_config": {
                "mode": opt.mode,
                "num_context_frames": opt.num_context_frames,
                "min_scene_frames": opt.min_scene_frames,
                "cut_video": opt.cut_video,
                "segment_io_backend": SEGMENT_IO_BACKEND,
            },
            "num_scenes": len(scenes),
            "num_clean_shots": len(clean_shot_rows),
            "num_transitions": len(transition_rows),
            "num_transition_sources": len(transition_source_rows),
            "clean_shot_scene_indices": [s["scene_idx"] for s in clean_shot_rows],
            "scenes": segment_rows,
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ {video_path} → {len(scenes)} scenes")
        return {
            "video_path": video_path,
            "video_name": os.path.basename(video_path),
            "scenes_json_path": json_path,
            "num_scenes": len(scenes),
            "num_clean_shots": len(clean_shot_rows),
            "num_transitions": len(transition_rows),
            "num_transition_sources": len(transition_source_rows),
            "clean_shot_scene_indices": [s["scene_idx"] for s in clean_shot_rows],
            "clean_shot_segment_video_paths": [
                s["segment_video_path"] for s in clean_shot_rows
                if s.get("write_success") and s.get("segment_video_path")
            ],
            "status": 1,
        }

    except Exception as e:
        logger.error(f"✗ {video_path}: {e}")
        return {
            "video_path": video_path,
            "video_name": os.path.basename(video_path),
            "scenes_json_path": json_path,
            "num_scenes": 0,
            "num_clean_shots": 0,
            "num_transitions": 0,
            "num_transition_sources": 0,
            "clean_shot_scene_indices": [],
            "clean_shot_segment_video_paths": [],
            "status": 0,
            "error": str(e),
        }


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
        result = process_single_video(vid_path, self.model, self.model_args, args)
        item["scenes_json_path"] = [result["scenes_json_path"]]
        item["num_scenes"] = [result["num_scenes"]]
        item["num_clean_shots"] = [result["num_clean_shots"]]
        item["num_transitions"] = [result["num_transitions"]]
        item["num_transition_sources"] = [result["num_transition_sources"]]
        item["clean_shot_scene_indices"] = [json.dumps(result["clean_shot_scene_indices"], ensure_ascii=False)]
        item["clean_shot_segment_video_paths"] = [json.dumps(result["clean_shot_segment_video_paths"], ensure_ascii=False)]
        item["status"] = [result["status"]]
        if result.get("error"):
            item["error"] = [result["error"]]
        return item


# ────────────────────────── main ──────────────────────────
if __name__ == "__main__":
    os.makedirs(args.output_dir, exist_ok=True)

    is_local = args.is_local.lower() in ("true", "1", "yes")

    if is_local:
        os.chdir(OMNISHOTCUT_ROOT)
        from test_code.inference import load_model
        model, model_args = load_model(args.checkpoint_path)

        samples = list(csv.DictReader(open(args.input_csv, "r", encoding="utf-8-sig")))
        logger.info(f"本地模式: 共 {len(samples)} 条视频")

        results = []
        for item in samples:
            vid_path = item["file_path"]
            res = process_single_video(vid_path, model, model_args, args)
            results.append(res)

        summary_path = os.path.join(args.output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"完成! 汇总写入 {summary_path}")

    else:
        import ray

        ray.init(
            address="auto",
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": OMNISHOTCUT_ROOT,
                },
                "working_dir": OMNISHOTCUT_ROOT,
            },
        )

        log_dir = args.ray_log_dir or os.path.join(args.output_dir, "ray_log")
        os.makedirs(log_dir, exist_ok=True)

        samples = ray.data.read_csv(args.input_csv)
        predictions = samples.map(
            Worker,
            num_gpus=1,
            concurrency=args.concurrency,
        )
        predictions.write_csv(log_dir)
        logger.info(f"Ray 完成! 日志写入 {log_dir}")
