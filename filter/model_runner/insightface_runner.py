"""InsightFace 人脸检测与跟踪 Runner。"""
import cv2
import numpy as np
from pathlib import Path

from filter.filter_utils import bbox_area, bbox_iou


def crop_face(frame, bbox, size=112):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(x)) for x in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size, size, 3), dtype=np.uint8)
    return cv2.resize(frame[y1:y2, x1:x2], (size, size))


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


def dense_gray_crops_range(track, start_frame, num_frames):
    frame_to_crop = {int(f): crop for f, crop in zip(track["frames"], track["crops"])}
    track_start, track_end = int(track["start_frame"]), int(track["end_frame"])
    crops, masks, last = [], [], None
    zero = np.zeros((112, 112), dtype=np.uint8)
    for offset in range(num_frames):
        frame_idx = start_frame + offset
        inside = track_start <= frame_idx <= track_end
        crop = frame_to_crop.get(frame_idx)
        if crop is not None:
            last = crop
        elif inside and last is not None:
            crop = last
        if crop is None:
            gray = zero
        elif len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop
        if gray.shape[:2] != (112, 112):
            gray = cv2.resize(gray, (112, 112))
        crops.append(gray)
        masks.append(bool(inside))
    return np.asarray(crops, dtype=np.uint8), np.asarray(masks, dtype=bool)


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


class InsightFaceTracker:
    def __init__(self, device="cuda", model_name="buffalo_l", det_size=640, image_size=112, root="", allow_download=False):
        from insightface.app import FaceAnalysis
        from insightface.utils import face_align

        det_size = max(128, det_size)
        det_size = ((det_size + 31) // 32) * 32

        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        if device.startswith("cuda"):
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0

        root_path = Path(root).expanduser() if root else Path.home() / ".insightface"
        model_dir = root_path / "models" / model_name
        if not model_dir.is_dir() and not allow_download:
            raise RuntimeError("insightface model not found...")

        self.image_size = image_size
        self.face_align = face_align
        self.app = FaceAnalysis(name=model_name, root=str(root_path), providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))

    def detect(self, frame):
        detections = []
        try:
            raw_faces = self.app.get(frame)
        except ValueError:
            raw_faces = []
        for face in raw_faces:
            bbox = np.asarray(face.bbox, dtype=np.float32).tolist()
            kps = getattr(face, "kps", None)
            aligned = None
            if kps is not None:
                try:
                    aligned = self.face_align.norm_crop(
                        frame, landmark=np.asarray(kps, dtype=np.float32), image_size=self.image_size
                    )
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
