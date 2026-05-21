#!/usr/bin/env python3
import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TALKNCE_ROOT = PROJECT_ROOT / "third_part" / "TalkNCE"
SYNCHFORMER_ROOT = PROJECT_ROOT / "third_part" / "Synchformer"
DEFAULT_INSIGHTFACE_ROOT = Path("/mnt/workspace/lihaoran/ckps/insightface")


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
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
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


def build_av_segment(args, video_item, scene, output_dir):
    original_segment = scene.get("segment_video_path", "")
    if not args.use_source_audio_segments:
        return original_segment, {"status": "disabled", "path": original_segment}
    source_video = video_item.get("summary_row", {}).get("video_path") or video_item.get("scenes_json", {}).get("video_path")
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
        return str(out_path), {"status": "cached", "path": str(out_path), "source_video": source_video, "start_sec": start_sec, "duration_sec": duration_sec}
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.6f}",
        "-i",
        str(source_video),
        "-t",
        f"{duration_sec:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-shortest",
        str(out_path),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if proc.returncode != 0 or not out_path.is_file():
        return original_segment, {"status": "ffmpeg_error", "path": original_segment, "source_video": source_video, "stderr": proc.stderr[-2000:]}
    return str(out_path), {"status": "created", "path": str(out_path), "source_video": source_video, "start_sec": start_sec, "duration_sec": duration_sec}


def crop_face(frame, bbox, size=112):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(x)) for x in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(frame[y1:y2, x1:x2], (size, size))


def extract_audio_16k(video_path):
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "ignore"))
    if not proc.stdout:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0


class InsightFaceTracker:
    def __init__(self, device="cuda", model_name="buffalo_l", det_size=640, image_size=112, root="", allow_download=False):
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        if device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        root_path = Path(root).expanduser() if root else Path.home() / ".insightface"
        model_dir = root_path / "models" / model_name
        if not model_dir.is_dir() and not allow_download:
            raise RuntimeError(
                "insightface model nonono..."
            )
        self.image_size = image_size
        self.face_align = face_align
        self.app = FaceAnalysis(name=model_name, root=str(root_path), providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def detect(self, frame):
        detections = []
        for face in self.app.get(frame):
            bbox = np.asarray(face.bbox, dtype=np.float32).tolist()
            kps = getattr(face, "kps", None)
            aligned = None
            if kps is not None:
                try:
                    aligned = self.face_align.norm_crop(frame, landmark=np.asarray(kps, dtype=np.float32), image_size=self.image_size)
                except Exception:
                    aligned = None
            if aligned is None:
                aligned = crop_face(frame, bbox, self.image_size)
            detections.append({
                "bbox": bbox,
                "score": float(getattr(face, "det_score", 0.0)),
                "crop": aligned,
            })
        detections.sort(key=lambda x: bbox_area(x["bbox"]), reverse=True)
        return detections

    def extract_tracks(self, video_path, iou_threshold=0.35, max_missing=5, min_track_frames=12):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        tracks, active_ids, next_id = [], [], 0
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections = self.detect(frame)
            assigned_tracks, assigned_dets = set(), set()
            for det_idx, det in enumerate(detections):
                best_id, best_iou = None, 0.0
                for track_id in active_ids:
                    if track_id in assigned_tracks:
                        continue
                    cur_iou = bbox_iou(det["bbox"], tracks[track_id]["last_bbox"])
                    if cur_iou > best_iou:
                        best_id, best_iou = track_id, cur_iou
                if best_id is not None and best_iou >= iou_threshold:
                    tr = tracks[best_id]
                    tr["frames"].append(frame_idx)
                    tr["bboxes"].append(det["bbox"])
                    tr["scores"].append(det["score"])
                    tr["crops"].append(det["crop"])
                    tr["last_bbox"] = det["bbox"]
                    tr["missing"] = 0
                    assigned_tracks.add(best_id)
                    assigned_dets.add(det_idx)
            for det_idx, det in enumerate(detections):
                if det_idx in assigned_dets:
                    continue
                tracks.append({
                    "track_id": next_id,
                    "frames": [frame_idx],
                    "bboxes": [det["bbox"]],
                    "scores": [det["score"]],
                    "crops": [det["crop"]],
                    "last_bbox": det["bbox"],
                    "missing": 0,
                })
                active_ids.append(next_id)
                next_id += 1
            live = []
            for track_id in active_ids:
                if track_id not in assigned_tracks:
                    tracks[track_id]["missing"] += 1
                if tracks[track_id]["missing"] <= max_missing:
                    live.append(track_id)
            active_ids = live
            frame_idx += 1
        cap.release()
        out_tracks = []
        for tr in tracks:
            if len(tr["frames"]) < min_track_frames:
                continue
            areas = [bbox_area(b) for b in tr["bboxes"]]
            start = int(min(tr["frames"]))
            end = int(max(tr["frames"]))
            out_tracks.append({
                "track_id": int(tr["track_id"]),
                "start_frame": start,
                "end_frame": end,
                "duration_frames": int(end - start + 1),
                "num_detected_frames": int(len(tr["frames"])),
                "avg_face_area": float(np.mean(areas)) if areas else 0.0,
                "avg_det_score": float(np.mean(tr["scores"])) if tr["scores"] else 0.0,
                "frames": tr["frames"],
                "bboxes": tr["bboxes"],
                "crops": tr["crops"],
            })
        primary_track_id = None
        if out_tracks:
            primary = max(out_tracks, key=lambda x: x["avg_face_area"] * x["num_detected_frames"])
            primary_track_id = int(primary["track_id"])
        return {
            "video_meta": {
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": frame_idx,
                "duration": round(frame_idx / fps, 3) if fps > 0 else 0.0,
            },
            "num_face_tracks": len(out_tracks),
            "primary_track_id": primary_track_id,
            "tracks": out_tracks,
        }


def dense_gray_crops(track, max_frames):
    frame_to_crop = {int(f): crop for f, crop in zip(track["frames"], track["crops"])}
    start, end = int(track["start_frame"]), int(track["end_frame"])
    crops, last = [], None
    for frame_idx in range(start, end + 1):
        crop = frame_to_crop.get(frame_idx)
        if crop is None:
            crop = last if last is not None else np.zeros((112, 112, 3), dtype=np.uint8)
        last = crop
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        if max_frames > 0 and len(crops) >= max_frames:
            break
    return np.asarray(crops, dtype=np.uint8)


def compact_tracks(face_result):
    compact = []
    for tr in face_result["tracks"]:
        compact.append({
            "track_id": tr["track_id"],
            "start_frame": tr["start_frame"],
            "end_frame": tr["end_frame"],
            "duration_frames": tr["duration_frames"],
            "num_detected_frames": tr["num_detected_frames"],
            "avg_face_area": round(tr["avg_face_area"], 3),
            "avg_det_score": round(tr["avg_det_score"], 5),
        })
    return compact


class StaticScoreWorker:
    def __init__(self, result):
        self.result = result

    def score(self, *args, **kwargs):
        return dict(self.result)

    def close(self):
        pass


def ns(value):
    if isinstance(value, dict):
        return argparse.Namespace(**{k: ns(v) for k, v in value.items()})
    if isinstance(value, list):
        return [ns(x) for x in value]
    return value


class TalkNCERunner:
    def __init__(self, args):
        import torch
        import torch.nn.functional as F
        import yaml

        self.args = args
        self.torch = torch
        self.F = F
        self.device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
        root = Path(args.talknce_root).resolve()
        ckpt = Path(args.talknce_ckpt)
        if not ckpt.is_file():
            raise FileNotFoundError(f"TalkNCE checkpoint not found: {ckpt}")
        with prepend_sys_path(root), pushd(root):
            from loss_multi import lossAV
            from model.loconet_encoder import locoencoder
            from torchvggish import vggish_input
            cfg = ns(yaml.safe_load(open(args.talknce_config_path, "r", encoding="utf-8")))
            download_ctx = contextlib.nullcontext() if args.talknce_allow_weight_download else block_torch_hub_downloads()
            with download_ctx:
                self.model = locoencoder(cfg).to(self.device).eval()
            self.classifier = lossAV(cfg).to(self.device).eval()
            self.vggish_input = vggish_input
        loaded = torch.load(str(ckpt), map_location="cpu")
        if isinstance(loaded, dict) and "model" in loaded:
            loaded = loaded["model"]
        model_state, cls_state = {}, {}
        for key, val in loaded.items():
            key = key.replace("model.module.", "").replace("module.", "")
            if key.startswith("model."):
                model_state[key[6:]] = val
            elif key.startswith("lossAV."):
                cls_state[key[7:]] = val
            elif key in self.model.state_dict():
                model_state[key] = val
            elif key in self.classifier.state_dict() or key.startswith("FC."):
                cls_state[key] = val
        if not model_state or not cls_state:
            raise RuntimeError(f"bad TalkNCE checkpoint: model_keys={len(model_state)}, classifier_keys={len(cls_state)}")
        self.model.load_state_dict(model_state, strict=False)
        self.classifier.load_state_dict(cls_state, strict=False)

    def score_arrays(self, audio, fps, track_ids, start_frames, crops_list, min_frames):
        results = []
        for idx, track_id in enumerate(track_ids):
            crops = crops_list[idx].astype(np.float32)
            num_frames = int(crops.shape[0])
            if num_frames < min_frames:
                results.append({"track_id": int(track_id), "status": "too_short", "num_frames": num_frames, "is_active_speaker": False})
                continue
            start_sample = max(0, int(round(start_frames[idx] / fps * 16000)))
            end_sample = min(len(audio), int(round((start_frames[idx] + num_frames) / fps * 16000)))
            audio_slice = audio[start_sample:end_sample]
            if audio_slice.size == 0:
                results.append({"track_id": int(track_id), "status": "no_audio", "num_frames": num_frames, "is_active_speaker": False})
                continue
            audio_feat = self.vggish_input.waveform_to_examples(audio_slice, 16000, num_frames, fps, return_tensor=False)
            audio_feat = self.torch.FloatTensor(audio_feat)[None, None, :, :].to(self.device)
            visual_feat = self.torch.FloatTensor(crops)[None, :, :, :].to(self.device)
            with self.torch.no_grad():
                audio_embed = self.model.forward_audio_frontend(audio_feat)
                visual_embed = self.model.forward_visual_frontend(visual_feat)
                audio_embed, visual_embed = self.model.forward_cross_attention(audio_embed, visual_embed)
                outs = self.model.forward_audio_visual_backend(audio_embed, visual_embed, b=1, s=1)
                probs = self.F.softmax(self.classifier.FC(outs.reshape(-1, 256)), dim=-1)[:, 1]
                mean_score = float(probs.mean().cpu().item())
                max_score = float(probs.max().cpu().item())
                active_ratio = float((probs >= self.args.active_speaker_threshold).float().mean().cpu().item())
            results.append({
                "track_id": int(track_id),
                "status": "ok",
                "num_frames": num_frames,
                "active_speaker_score": mean_score,
                "active_speaker_max_score": max_score,
                "active_speaker_ratio": active_ratio,
                "is_active_speaker": bool(mean_score >= self.args.active_speaker_threshold or active_ratio >= self.args.active_speaker_min_ratio),
            })
        return {"status": "ok", "tracks": results}

    def score(self, face_result, segment_path):
        if not face_result["tracks"]:
            return {"status": "no_face", "tracks": []}
        audio = extract_audio_16k(segment_path)
        fps = float(face_result["video_meta"]["fps"])
        track_ids = [int(t["track_id"]) for t in face_result["tracks"]]
        start_frames = [int(t["start_frame"]) for t in face_result["tracks"]]
        crops_list = [dense_gray_crops(t, self.args.max_talknce_frames) for t in face_result["tracks"]]
        return self.score_arrays(audio, fps, track_ids, start_frames, crops_list, self.args.min_talknce_frames)


class TalkNCEWorker:
    def __init__(self, args):
        self.args = args
        self.runner = None
        self.static_status = None
        if args.disable_talknce:
            self.static_status = "disabled"
            return
        if not args.talknce_ckpt:
            self.static_status = "not_configured"
            return
        runner_args = argparse.Namespace(**vars(args))
        runner_args.device = args.device if args.talknce_device == "auto" else args.talknce_device
        self.runner = TalkNCERunner(runner_args)

    def score(self, face_result, segment_path, work_dir=None):
        if self.runner is None:
            return {"status": self.static_status, "tracks": []}
        if not face_result["tracks"]:
            return {"status": "no_face", "tracks": []}
        try:
            result = self.runner.score(face_result, segment_path)
        except Exception as e:
            return {"status": "error", "error": str(e), "tracks": []}
        if "tracks" not in result:
            result["tracks"] = []
        return result

    def close(self):
        pass


class SynchformerRunner:
    def __init__(self, args):
        import torch
        import torchvision
        from omegaconf import OmegaConf

        self.args = args
        self.torch = torch
        self.torchvision = torchvision
        self.root = Path(args.synchformer_root).resolve()
        self.vfps = 25
        self.afps = 16000
        self.in_size = 256
        self.cache_dir = Path(args.output_dir) / "synchformer_inputs"
        ensure_dir(self.cache_dir)
        cfg_path = "/mnt/workspace/lihaoran/data_process/third_part/Synchformer/logs/sync_models/23-12-23T18-33-57/cfg-23-12-23T18-33-57.yaml"
        ckpt_path = "/mnt/workspace/lihaoran/data_process/third_part/Synchformer/logs/sync_models/23-12-23T18-33-57/23-12-23T18-33-57.1.pt"
        with prepend_sys_path(self.root), pushd(self.root):
            from dataset.dataset_utils import get_video_and_audio
            from dataset.transforms import make_class_grid
            from scripts.train_utils import get_model, get_transforms, prepare_inputs
            from example import patch_config
            cfg = patch_config(OmegaConf.load(cfg_path))
            self.device = torch.device(args.device)
            _, self.model = get_model(cfg, self.device)
            ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            self.transform = get_transforms(cfg, ["test"])["test"]
            self.prepare_inputs = prepare_inputs
            self.get_video_and_audio = get_video_and_audio
            self.cfg = cfg
            max_off_sec = cfg.data.max_off_sec
            num_cls = cfg.model.params.transformer.params.off_head_cfg.params.out_features
            self.grid = make_class_grid(-max_off_sec, max_off_sec, num_cls)

    def prepare_video(self, segment_path):
        segment_path = str(segment_path)
        v, _, info = self.torchvision.io.read_video(segment_path, pts_unit="sec")
        _, h, w, _ = v.shape
        video_fps = float(info.get("video_fps") or 0.0)
        audio_fps = int(info.get("audio_fps") or 0)
        if abs(video_fps - self.vfps) < 0.01 and audio_fps == self.afps and min(h, w) == self.in_size:
            return segment_path, {"status": "already_compatible", "path": segment_path}
        out_path = self.cache_dir / f"{safe_name(Path(segment_path).stem)}_{self.vfps}fps_{self.in_size}side_{self.afps}hz.mp4"
        if out_path.is_file() and has_audio_stream(out_path):
            return str(out_path), {"status": "cached", "path": str(out_path)}
        vf = f"fps={self.vfps},scale=iw*{self.in_size}/min(iw\\,ih):ih*{self.in_size}/min(iw\\,ih),crop=trunc(iw/2)*2:trunc(ih/2)*2"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            segment_path,
            "-vf",
            vf,
            "-ar",
            str(self.afps),
            "-ac",
            "1",
            "-c:a",
            "aac",
            str(out_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode != 0 or not out_path.is_file():
            raise RuntimeError(proc.stderr[-2000:])
        return str(out_path), {"status": "created", "path": str(out_path)}

    def score(self, segment_path):
        with pushd(self.root):
            prepared_path, prep = self.prepare_video(segment_path)
            rgb, audio, meta = self.get_video_and_audio(prepared_path, get_meta=True)
            item = dict(
                video=rgb,
                audio=audio,
                meta=meta,
                path=prepared_path,
                split="test",
                targets={"v_start_i_sec": 0.0, "offset_sec": 0.0},
            )
            item = self.transform(item)
            batch = self.torch.utils.data.default_collate([item])
            aud, vid, targets = self.prepare_inputs(batch, self.device)
            device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
            with self.torch.set_grad_enabled(False):
                with self.torch.autocast(device_type, enabled=bool(device_type == "cuda" and self.cfg.training.use_half_precision)):
                    _, logits = self.model(vid, aud)
            off_logits = logits[0]
            off_probs = self.torch.softmax(off_logits, dim=-1)
            k = min(off_probs.shape[-1], 5)
            _, topk_preds = self.torch.topk(off_logits, k)
            top_offsets = []
            for pred in topk_preds:
                pred_i = int(pred.item())
                top_offsets.append({"probability": float(off_probs[pred_i].cpu().item()), "offset_sec": float(self.grid[pred_i])})
            best = top_offsets[0]
            return {
                "status": "ok",
                "predicted_offset_sec": best["offset_sec"],
                "sync_confidence": best["probability"],
                "top_offsets": top_offsets,
                "synchformer_input": prep,
            }


class SynchformerWorker:
    def __init__(self, args):
        self.args = args
        self.runner = None
        self.static_status = None
        if args.disable_synchformer:
            self.static_status = "disabled"
            return
        if not args.synchformer_exp_name:
            self.static_status = "not_configured"
            return
        self.runner = SynchformerRunner(args)

    def score(self, segment_path):
        if self.runner is None:
            return {"status": self.static_status, "is_synced": False}
        meta = video_meta(segment_path)
        if meta["duration"] < self.args.min_synchformer_duration:
            return {"status": "too_short", "duration": meta["duration"], "is_synced": False}
        try:
            result = self.runner.score(segment_path)
        except Exception as e:
            return {"status": "error", "error": str(e), "is_synced": False}
        if result.get("status") == "ok":
            result["is_synced"] = bool(abs(result["predicted_offset_sec"]) <= self.args.max_sync_offset_sec and result["sync_confidence"] >= self.args.min_sync_confidence)
            result["synchronizability_proxy_score"] = result["sync_confidence"]
            result["is_synchronizable"] = bool(result["sync_confidence"] >= self.args.min_sync_confidence)
            result["synchronizability_note"] = "uses Synchformer offset-class probability as a proxy; separate synchronizability checkpoint is not invoked"
        else:
            result["is_synced"] = False
        return result

    def close(self):
        pass


def attach_speaker_scores(tracks, talknce_result):
    by_id = {x.get("track_id"): x for x in talknce_result.get("tracks", [])}
    out = []
    for tr in tracks:
        row = dict(tr)
        row["active_speaker"] = by_id.get(tr["track_id"], {"status": talknce_result.get("status", "not_configured"), "is_active_speaker": False})
        out.append(row)
    return out


def qualify_segment(args, face_result, talknce_result, sync_result):
    reasons = []
    attention = []
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
    elif args.require_talknce:
        reasons.append(f"talknce_{talknce_result.get('status', 'not_ok')}")
    if sync_result.get("status") == "ok":
        if not sync_result.get("is_synced"):
            reasons.append("av_not_synced")
    elif args.require_synchformer:
        reasons.append(f"synchformer_{sync_result.get('status', 'not_ok')}")
    qualified = len(reasons) == 0
    return {
        "is_lipsync_qualified": qualified,
        "reject_reasons": reasons,
        "requires_attention": bool(attention),
        "attention_reasons": sorted(set(attention)),
        "target_track_id": primary_id,
        "active_speaker_track_ids": active_ids,
        "non_target_speaker_track_ids": [x for x in active_ids if x != primary_id],
    }


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


def process_segment(args, video_item, scene, tracker, talknce_worker, synchformer_worker, video_out_dir, qualified_dir):
    segment_path = scene.get("segment_video_path", "")
    print(f"  [{time.strftime('%H:%M:%S')}] prepare av segment", flush=True)
    av_segment_path, av_segment_info = build_av_segment(args, video_item, scene, Path(video_out_dir).parent / "av_segments")
    processing_video_path = av_segment_path or segment_path
    result = {
        "scene_idx": scene.get("scene_idx"),
        "segment_video_path": segment_path,
        "processing_video_path": processing_video_path,
        "av_segment": av_segment_info,
        "start_frame": scene.get("start_frame"),
        "end_frame": scene.get("end_frame"),
        "duration_frames": scene.get("duration_frames"),
        "source_scenes_json_path": video_item.get("scenes_json_path"),
    }
    if not processing_video_path or not Path(processing_video_path).is_file():
        result.update({"is_lipsync_qualified": False, "reject_reasons": ["missing_segment_video"], "requires_attention": True, "attention_reasons": ["missing_segment_video"]})
        return result
    print(f"  [{time.strftime('%H:%M:%S')}] insightface tracks", flush=True)
    face_result = tracker.extract_tracks(processing_video_path, args.face_iou_threshold, args.face_max_missing, args.min_face_track_frames)
    print(f"  [{time.strftime('%H:%M:%S')}] talknce start tracks={face_result['num_face_tracks']}", flush=True)
    talknce_result = talknce_worker.score(face_result, processing_video_path)
    print(f"  [{time.strftime('%H:%M:%S')}] talknce done status={talknce_result.get('status')}", flush=True)
    print(f"  [{time.strftime('%H:%M:%S')}] synchformer start", flush=True)
    sync_result = synchformer_worker.score(processing_video_path)
    print(f"  [{time.strftime('%H:%M:%S')}] synchformer done status={sync_result.get('status')}", flush=True)
    face_compact = {
        "video_meta": face_result["video_meta"],
        "num_face_tracks": face_result["num_face_tracks"],
        "primary_track_id": face_result["primary_track_id"],
        "tracks": attach_speaker_scores(compact_tracks(face_result), talknce_result),
    }
    decision = qualify_segment(args, face_result, talknce_result, sync_result)
    result.update({"face": face_compact, "talknce": talknce_result, "synchformer": sync_result, **decision})
    if result["is_lipsync_qualified"]:
        out_name = f"{safe_name(Path(processing_video_path).stem)}.mp4"
        result["qualified_video_path"] = link_or_copy(processing_video_path, Path(qualified_dir) / out_name, args.qualified_copy_mode)
    return result


def build_candidate_item(scenes_json_path, summary_row=None):
    scenes_json_path = Path(scenes_json_path)
    scenes_json = read_json(scenes_json_path)
    row = dict(summary_row or {})
    row.setdefault("video_path", scenes_json.get("video_path"))
    row.setdefault("video_name", scenes_json.get("video_name") or Path(str(row.get("video_path", scenes_json_path.stem))).name)
    row.setdefault("scenes_json_path", str(scenes_json_path))
    row.setdefault("clean_shot_scene_indices", scenes_json.get("clean_shot_scene_indices", []))
    scenes = []
    idx_set = set(row.get("clean_shot_scene_indices", []))
    for scene in scenes_json.get("scenes", []):
        if not scene.get("is_clean_shot"):
            continue
        if not scene.get("write_success"):
            continue
        if idx_set and scene.get("scene_idx") not in idx_set:
            continue
        scenes.append(scene)
    return {"summary_row": row, "scenes_json": scenes_json, "scenes_json_path": str(scenes_json_path), "candidate_scenes": scenes}


def collect_candidate_scenes(input_path):
    path = Path(input_path)
    items = []
    if path.is_file():
        summary = read_json(path)
        if isinstance(summary, dict):
            summary = summary.get("videos", summary.get("items", []))
        for row in summary:
            scenes_json_path = row.get("scenes_json_path")
            if not scenes_json_path:
                continue
            scenes_json_path = Path(scenes_json_path)
            if not scenes_json_path.is_absolute():
                scenes_json_path = path.parent / scenes_json_path
            if scenes_json_path.is_file():
                items.append(build_candidate_item(scenes_json_path, row))
        return items
    scenes_dir = path if path.is_dir() else path.parent
    scene_paths = sorted(scenes_dir.glob("*_scenes.json"))
    if not scene_paths:
        raise FileNotFoundError(f"找不到 summary.json，也没有在目录中发现 *_scenes.json: {scenes_dir}")
    for scenes_json_path in scene_paths:
        items.append(build_candidate_item(scenes_json_path))
    return items


def create_lipsync_workers(args):
    tracker = InsightFaceTracker(
        args.device,
        args.insightface_model,
        args.insightface_det_size,
        112,
        args.insightface_root,
        args.allow_insightface_download,
    )
    print(f"[{time.strftime('%H:%M:%S')}] starting TalkNCE worker (one-time load)", flush=True)
    try:
        talknce_worker = TalkNCEWorker(args)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] TalkNCE worker failed: {e}", flush=True)
        talknce_worker = StaticScoreWorker({"status": "error", "error": str(e), "tracks": []})
    print(f"[{time.strftime('%H:%M:%S')}] starting Synchformer worker (one-time load)", flush=True)
    try:
        synchformer_worker = SynchformerWorker(args)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] Synchformer worker failed: {e}", flush=True)
        synchformer_worker = StaticScoreWorker({"status": "error", "error": str(e), "is_synced": False})
    return tracker, talknce_worker, synchformer_worker


def close_lipsync_workers(talknce_worker, synchformer_worker):
    if talknce_worker is not None:
        talknce_worker.close()
    if synchformer_worker is not None:
        synchformer_worker.close()


def process_video_item(args, item, tracker, talknce_worker, synchformer_worker, qualified_dir):
    src_row = item["summary_row"]
    video_name = src_row.get("video_name") or Path(src_row.get("video_path", "video")).name
    video_stem = safe_name(Path(video_name).stem)
    video_out_dir = Path(args.output_dir) / "details"
    ensure_dir(video_out_dir)
    results = []
    attention_rows = []
    for scene in item["candidate_scenes"]:
        print(f"[{time.strftime('%H:%M:%S')}] lipsync {video_name} scene={scene.get('scene_idx')}", flush=True)
        try:
            row = process_segment(args, item, scene, tracker, talknce_worker, synchformer_worker, video_out_dir, qualified_dir)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] lipsync error {video_name} scene={scene.get('scene_idx')}: {e}", flush=True)
            row = {
                "scene_idx": scene.get("scene_idx"),
                "segment_video_path": scene.get("segment_video_path", ""),
                "processing_video_path": scene.get("segment_video_path", ""),
                "start_frame": scene.get("start_frame"),
                "end_frame": scene.get("end_frame"),
                "duration_frames": scene.get("duration_frames"),
                "source_scenes_json_path": item.get("scenes_json_path"),
                "is_lipsync_qualified": False,
                "reject_reasons": ["processing_error"],
                "requires_attention": True,
                "attention_reasons": ["processing_error"],
                "error": str(e),
                "error_type": type(e).__name__,
            }
        results.append(row)
        if row.get("requires_attention"):
            attention_rows.append({"video_name": video_name, "scene_idx": row.get("scene_idx"), "segment_video_path": row.get("segment_video_path"), "attention_reasons": row.get("attention_reasons", []), "reject_reasons": row.get("reject_reasons", []), "error": row.get("error", "")})
    valid = [x for x in results if x.get("is_lipsync_qualified")]
    video_json_path = video_out_dir / f"{video_stem}_lipsync.json"
    write_json(video_json_path, {
        "video_path": src_row.get("video_path"),
        "video_name": video_name,
        "source_scenes_json_path": item["scenes_json_path"],
        "num_candidate_scenes": len(item["candidate_scenes"]),
        "num_lipsync_qualified": len(valid),
        "num_attention_segments": sum(1 for x in results if x.get("requires_attention")),
        "segments": results,
    })
    summary_row = {
        "video_path": src_row.get("video_path"),
        "video_name": video_name,
        "source_scenes_json_path": item["scenes_json_path"],
        "lipsync_json_path": str(video_json_path),
        "num_candidate_scenes": len(item["candidate_scenes"]),
        "num_lipsync_qualified": len(valid),
        "qualified_scene_indices": [x["scene_idx"] for x in valid],
        "qualified_video_paths": [x.get("qualified_video_path") for x in valid if x.get("qualified_video_path")],
        "num_attention_segments": sum(1 for x in results if x.get("requires_attention")),
    }
    return summary_row, attention_rows


def build_failed_video_result(args, item, error):
    src_row = item.get("summary_row", {})
    video_name = src_row.get("video_name") or Path(src_row.get("video_path", "video")).name
    detail_path = Path(args.output_dir) / "details" / f"{safe_name(Path(video_name).stem)}_lipsync.json"
    summary_row = {
        "video_path": src_row.get("video_path"),
        "video_name": video_name,
        "source_scenes_json_path": item.get("scenes_json_path"),
        "lipsync_json_path": str(detail_path),
        "num_candidate_scenes": len(item.get("candidate_scenes", [])),
        "num_lipsync_qualified": 0,
        "qualified_scene_indices": [],
        "qualified_video_paths": [],
        "num_attention_segments": len(item.get("candidate_scenes", [])),
        "status": 0,
        "error": str(error),
        "error_type": type(error).__name__,
    }
    attention_rows = []
    for scene in item.get("candidate_scenes", []):
        attention_rows.append({
            "video_name": video_name,
            "scene_idx": scene.get("scene_idx"),
            "segment_video_path": scene.get("segment_video_path", ""),
            "attention_reasons": ["video_processing_error"],
            "reject_reasons": ["video_processing_error"],
            "error": str(error),
        })
    try:
        write_json(detail_path, {
            "video_path": src_row.get("video_path"),
            "video_name": video_name,
            "source_scenes_json_path": item.get("scenes_json_path"),
            "num_candidate_scenes": len(item.get("candidate_scenes", [])),
            "num_lipsync_qualified": 0,
            "num_attention_segments": len(attention_rows),
            "status": 0,
            "error": str(error),
            "segments": [],
        })
    except Exception:
        pass
    return summary_row, attention_rows


def run_pipeline(args):
    ensure_dir(args.output_dir)
    qualified_dir = Path(args.output_dir) / "qualified_videos"
    ensure_dir(qualified_dir)
    items = collect_candidate_scenes(args.summary_json)
    tracker, talknce_worker, synchformer_worker = create_lipsync_workers(args)
    all_summary, attention_rows = [], []
    try:
        for item in items:
            summary_row, video_attention_rows = process_video_item(args, item, tracker, talknce_worker, synchformer_worker, qualified_dir)
            all_summary.append(summary_row)
            attention_rows.extend(video_attention_rows)
    finally:
        close_lipsync_workers(talknce_worker, synchformer_worker)
    write_json(Path(args.output_dir) / "lipsync_summary.json", all_summary)
    write_json(Path(args.output_dir) / "attention_segments.json", attention_rows)
    print(f"完成: {Path(args.output_dir) / 'lipsync_summary.json'}")


def make_ray_lipsync_worker(ray_args):
    class RayLipSyncWorker:
        def __init__(self):
            import atexit
            self.args = argparse.Namespace(**vars(ray_args))
            self.closed = False
            self.talknce_worker = None
            self.synchformer_worker = None
            ensure_dir(self.args.output_dir)
            self.qualified_dir = Path(self.args.output_dir) / "qualified_videos"
            ensure_dir(self.qualified_dir)
            self.tracker, self.talknce_worker, self.synchformer_worker = create_lipsync_workers(self.args)
            atexit.register(self.close)

        def __call__(self, item):
            item_index = item.get("_item_index")
            try:
                summary_row, attention_rows = process_video_item(
                    self.args,
                    item,
                    self.tracker,
                    self.talknce_worker,
                    self.synchformer_worker,
                    self.qualified_dir,
                )
                status = 1
                error = ""
            except Exception as e:
                summary_row, attention_rows = build_failed_video_result(self.args, item, e)
                status = 0
                error = str(e)
            return {
                "item_index": item_index,
                "video_name": summary_row.get("video_name"),
                "lipsync_json_path": summary_row.get("lipsync_json_path"),
                "status": status,
                "error": error,
                "summary_row": summary_row,
                "attention_rows": attention_rows,
            }

        def close(self):
            if self.closed:
                return
            self.closed = True
            close_lipsync_workers(self.talknce_worker, self.synchformer_worker)

        def __del__(self):
            self.close()

    return RayLipSyncWorker


def run_ray_pipeline(args):
    ensure_dir(args.output_dir)
    qualified_dir = Path(args.output_dir) / "qualified_videos"
    ensure_dir(qualified_dir)
    items = collect_candidate_scenes(args.summary_json)
    if not items:
        write_json(Path(args.output_dir) / "lipsync_summary.json", [])
        write_json(Path(args.output_dir) / "attention_segments.json", [])
        print(f"完成: {Path(args.output_dir) / 'lipsync_summary.json'}")
        return

    import ray

    ray_init_kwargs = {
        "runtime_env": {
            "env_vars": {
                "PYTHONPATH": str(PROJECT_ROOT),
            }
        },
    }
    if args.ray_working_dir:
        ray_init_kwargs["runtime_env"]["working_dir"] = args.ray_working_dir
    if args.ray_address:
        ray_init_kwargs["address"] = args.ray_address
    ray.init(**ray_init_kwargs)

    ray_items = []
    for idx, item in enumerate(items):
        row = dict(item)
        row["_item_index"] = idx
        ray_items.append(row)

    samples = ray.data.from_items(ray_items)
    map_kwargs = {
        "concurrency": args.concurrency,
        "num_gpus": args.ray_num_gpus,
    }
    if args.ray_num_cpus > 0:
        map_kwargs["num_cpus"] = args.ray_num_cpus
    predictions = samples.map(make_ray_lipsync_worker(args), **map_kwargs)
    worker_rows = predictions.take_all()
    worker_rows.sort(key=lambda x: x.get("item_index", 0))

    all_summary = [x.get("summary_row", {}) for x in worker_rows]
    attention_rows = []
    for row in worker_rows:
        attention_rows.extend(row.get("attention_rows") or [])

    write_json(Path(args.output_dir) / "lipsync_summary.json", all_summary)
    write_json(Path(args.output_dir) / "attention_segments.json", attention_rows)
    log_dir = Path(args.ray_log_dir) if args.ray_log_dir else Path(args.output_dir) / "ray_log"
    ensure_dir(log_dir)
    write_json(log_dir / "worker_results.json", worker_rows)
    print(f"Ray 完成: {Path(args.output_dir) / 'lipsync_summary.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description="OmniShotCut 后处理：InsightFace + TalkNCE + Synchformer 嘴型同步筛选")
    parser.add_argument("--summary_json", default="/mnt/workspace/lihaoran/data_process/test/0521/YouTube_talk/result/summary.json")
    parser.add_argument("--output_dir", default="/mnt/workspace/lihaoran/data_process/test/0521/YouTube_talk/result/lipsync")
    parser.add_argument("--is_local", type=str, default="True", help="True=本地串行; False=Ray 分布式")
    parser.add_argument("--concurrency", type=int, default=1, help="Ray actor 并发数")
    parser.add_argument("--ray_address", default="auto", help="Ray 集群地址；传空字符串时启动/连接本地 Ray")
    parser.add_argument("--ray_log_dir", default="", help="Ray worker 结果日志目录")
    parser.add_argument("--ray_num_gpus", type=float, default=1.0, help="每个 Ray lipsync worker 申请的 GPU 数")
    parser.add_argument("--ray_num_cpus", type=float, default=0.0, help="每个 Ray lipsync worker 申请的 CPU 数；0 表示不显式设置")
    parser.add_argument("--ray_working_dir", default="", help="Ray runtime_env working_dir；默认不打包工作目录")
    parser.add_argument("--use_source_audio_segments", action="store_true", default=True)
    parser.add_argument("--no_use_source_audio_segments", action="store_false", dest="use_source_audio_segments")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--insightface_model", default="buffalo_l")
    parser.add_argument("--insightface_root", default=str(DEFAULT_INSIGHTFACE_ROOT))
    parser.add_argument("--allow_insightface_download", action="store_true", default=False)
    parser.add_argument("--insightface_det_size", type=int, default=640)
    parser.add_argument("--face_iou_threshold", type=float, default=0.35)
    parser.add_argument("--face_max_missing", type=int, default=5)
    parser.add_argument("--min_face_track_frames", type=int, default=12)
    parser.add_argument("--talknce_root", default=str(TALKNCE_ROOT))
    parser.add_argument("--talknce_config_path", default=str(TALKNCE_ROOT / "configs" / "test.yaml"))
    parser.add_argument("--talknce_ckpt", default="/mnt/workspace/lihaoran/ckps/lipsync/talknce_ava_pretrained.model")
    parser.add_argument("--talknce_device", default="cpu")
    parser.add_argument("--disable_talknce", action="store_true", default=False)
    parser.add_argument("--talknce_allow_weight_download", action="store_true", default=False)
    parser.add_argument("--active_speaker_threshold", type=float, default=0.5)
    parser.add_argument("--active_speaker_min_ratio", type=float, default=0.2)
    parser.add_argument("--min_talknce_frames", type=int, default=16)
    parser.add_argument("--max_talknce_frames", type=int, default=200)
    parser.add_argument("--require_talknce", action="store_true", default=False)
    parser.add_argument("--allow_non_target_speaker", action="store_true", default=False)
    parser.add_argument("--synchformer_root", default=str(SYNCHFORMER_ROOT))
    parser.add_argument("--synchformer_exp_name", default="23-12-23T18-33-57")
    parser.add_argument("--disable_synchformer", action="store_true", default=False)
    parser.add_argument("--min_synchformer_duration", type=float, default=5.0)
    parser.add_argument("--max_sync_offset_sec", type=float, default=0.2)
    parser.add_argument("--min_sync_confidence", type=float, default=0.25)
    parser.add_argument("--require_synchformer", action="store_true", default=True)
    parser.add_argument("--no_require_synchformer", action="store_false", dest="require_synchformer")
    parser.add_argument("--qualified_copy_mode", choices=["symlink", "copy", "none"], default="symlink")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    is_local = args.is_local.lower() in ("true", "1", "yes")
    if is_local:
        run_pipeline(args)
    else:
        run_ray_pipeline(args)
