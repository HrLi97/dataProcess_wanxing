"""共享工具函数，所有 Runner 均可引用。"""
import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


# ─── 上下文管理器 ──────────────────────────────────────────────────────────────

@contextlib.contextmanager
def pushd(path):
    old = os.getcwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(old)


@contextlib.contextmanager
def prepend_sys_path(path):
    old = list(sys.path)
    path = str(path)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path = old


@contextlib.contextmanager
def block_torch_hub_downloads():
    import torch
    old_load = torch.hub.load_state_dict_from_url

    def offline_load_state_dict_from_url(*args, **kwargs):
        return {}

    torch.hub.load_state_dict_from_url = offline_load_state_dict_from_url
    try:
        yield
    finally:
        torch.hub.load_state_dict_from_url = old_load


def clear_modules(*prefixes):
    """清除指定前缀的 sys.modules 条目，用于在切换模块版本前做清理。"""
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            del sys.modules[name]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_name(text):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text)).strip("_") or "item"


def link_or_copy(src, dst, mode):
    ensure_dir(Path(dst).parent)
    if Path(dst).exists() or Path(dst).is_symlink():
        Path(dst).unlink()
    if mode == "none":
        return ""
    if mode == "copy":
        shutil.copy2(src, dst)
        return str(dst)
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return str(dst)


# ─── BBox 工具 ────────────────────────────────────────────────────────────────
def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def video_meta(video_path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"fps": 0.0, "frame_count": 0, "width": 0, "height": 0, "duration": 0.0}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": round(frame_count / fps, 3) if fps > 0 else 0.0,
    }


def has_audio_stream(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(video_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    return proc.returncode == 0 and bool(proc.stdout.strip())


def scene_times(scene, scenes_json):
    fps = float(scenes_json.get("detected_fps") or scenes_json.get("source", {}).get("fps") or 0.0)
    if fps <= 0:
        return None, None
    start_frame = int(scene.get("start_frame", 0))
    end_frame = int(scene.get("end_frame", start_frame))
    start_sec = max(0.0, start_frame / fps)
    duration_sec = max(0.001, (end_frame - start_frame) / fps)
    return start_sec, duration_sec


# ─── 音频处理 ─────────────────────────────────────────────────────────────────
def extract_audio_16k(video_path):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "ignore"))
    if not proc.stdout:
        import numpy as np
        return np.zeros((0,), dtype=np.float32)
    import numpy as np
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


# ─── 音视频片段截取 ──────────────────────────────────────────────────────────
def build_av_segment(args, video_item, scene, output_dir):
    original_segment = scene.get("segment_video_path", "")
    if not args.use_source_audio_segments:
        return original_segment, {"status": "disabled", "path": original_segment}

    source_video = (
        video_item.get("summary_row", {}).get("video_path")
        or video_item.get("scenes_json", {}).get("video_path")
    )
    if not source_video or not Path(source_video).is_file():
        return original_segment, {"status": "source_missing", "path": original_segment, "source_video": source_video}
    if not has_audio_stream(source_video):
        return original_segment, {"status": "source_no_audio", "path": original_segment, "source_video": source_video}

    start_sec, duration_sec = scene_times(scene, video_item.get("scenes_json", {}))
    if start_sec is None:
        return original_segment, {"status": "bad_fps", "path": original_segment, "source_video": source_video}

    video_name = video_item.get("summary_row", {}).get("video_name") or Path(source_video).name
    out_dir = Path(output_dir) / safe_name(Path(video_name).stem)
    ensure_dir(out_dir)
    scene_idx = int(scene.get("scene_idx", 0))
    out_path = out_dir / f"{scene_idx:04d}_{int(scene.get('start_frame', 0)):06d}-{int(scene.get('end_frame', 0)):06d}_av.mp4"

    if out_path.is_file() and has_audio_stream(out_path):
        return str(out_path), {"status": "cached", "path": str(out_path), "source_video": source_video,
                                "start_sec": start_sec, "duration_sec": duration_sec}

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_sec:.6f}", "-i", str(source_video),
        "-t", f"{duration_sec:.6f}",
        "-map", "0:v:0", "-map", "0:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-ar", "16000", "-ac", "1", "-shortest",
        str(out_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0 or not out_path.is_file():
        return original_segment, {"status": "ffmpeg_error", "path": original_segment,
                                   "source_video": source_video, "stderr": proc.stderr[-2000:]}
    return str(out_path), {"status": "created", "path": str(out_path),
                            "source_video": source_video, "start_sec": start_sec, "duration_sec": duration_sec}


# ─── 决策函数 ─────────────────────────────────────────────────────────────────
def attach_speaker_scores(tracks, talknce_result):
    by_id = {x.get("track_id"): x for x in talknce_result.get("tracks", [])}
    out = []
    for tr in tracks:
        row = dict(tr)
        row["active_speaker"] = by_id.get(tr["track_id"], {
            "status": talknce_result.get("status", "not_configured"),
            "is_active_speaker": False,
        })
        out.append(row)
    return out


def qualify_segment(args, face_result, talknce_result, sync_result, lrasd_result=None):
    """Determine whether a segment passes the lipsync quality bar.

    Args:
        args:             argparse namespace with all threshold flags
        face_result:      InsightFaceTracker output dict
        talknce_result:   TalkNCE active-speaker detection result
        sync_result:      Synchformer/LRASD AV-sync result
        lrasd_result:     Optional LR-ASD active-speaker result; if provided,
                          used to set per-track speaking status when talknce is
                          absent or disabled
    """
    reasons, attention = [], []
    primary_id = face_result.get("primary_track_id")

    if face_result.get("num_face_tracks", 0) == 0:
        reasons.append("no_face")
        attention.append("no_face")

    active_ids = []
    if talknce_result.get("status") == "ok":
        active_ids = [x["track_id"] for x in talknce_result.get("tracks", []) if x.get("is_active_speaker")]
        if primary_id is not None and primary_id not in active_ids:
            reasons.append("target_not_speaking")
            attention.append("target_not_speaking")
        other_ids = [x for x in active_ids if x != primary_id]
        if other_ids:
            attention.append("non_target_speaker")
            if not args.allow_non_target_speaker:
                reasons.append("non_target_speaker")
    elif lrasd_result is not None and lrasd_result.get("status") == "ok":
        # Fall back to LR-ASD speaking results when TalkNCE is unavailable
        active_ids = [x["track_id"] for x in lrasd_result.get("tracks", []) if x.get("is_speaking")]
        if primary_id is not None and primary_id not in active_ids:
            reasons.append("target_not_speaking")
            attention.append("target_not_speaking")
        other_ids = [x for x in active_ids if x != primary_id]
        if other_ids:
            attention.append("non_target_speaker")
            if not args.allow_non_target_speaker:
                reasons.append("non_target_speaker")
    elif args.require_talknce:
        reasons.append(f"talknce_{talknce_result.get('status', 'ok')}")

    if sync_result.get("status") == "ok":
        if not sync_result.get("is_synced"):
            reasons.append("av_not_synced")
    elif args.require_synchformer:
        reasons.append(f"synchformer_{sync_result.get('status', 'ok')}")

    return {
        "is_lipsync_qualified": len(reasons) == 0,
        "reject_reasons": reasons,
        "requires_attention": bool(attention),
        "attention_reasons": sorted(set(attention)),
        "target_track_id": primary_id,
        "active_speaker_track_ids": active_ids,
        "non_target_speaker_track_ids": [x for x in active_ids if x != primary_id],
    }


def ns(value):
    if isinstance(value, dict):
        return argparse.Namespace(**{k: ns(v) for k, v in value.items()})
    if isinstance(value, list):
        return [ns(x) for x in value]
    return value


class JsonlWriter:
    """追加写入 JSONL，每行一个 JSON 对象，不要求所有数据同时在内存中。"""

    def __init__(self, path):
        self.path = Path(path)
        self._buf = []

    def write_json(self, obj):
        self._buf.append(json.dumps(obj, ensure_ascii=False))

    def flush(self):
        if not self._buf:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(self._buf) + "\n")
        self._buf.clear()
