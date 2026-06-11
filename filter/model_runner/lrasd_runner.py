"""LR-ASD Active Speaker Detection Runner.

LR-ASD (Lightweight and Robust Network for Active Speaker Detection, IJCV 2025)
performs audio-visual active speaker detection: given a face track and corresponding
audio, it outputs a per-frame score indicating whether that person is speaking.

Integration notes:
  - InsightFace provides face detection + tracking (unchanged).
  - LR-ASD consumes pre-cropped face clips + audio.
  - In multi-speaker mode, audio is aligned per-speaker: each speaker gets
    their own audio window (from PyAnnote) passed via speaker_window arg.
"""
import glob
import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import wavfile

from filter.filter_utils import (
    clear_modules,
    ensure_dir,
    prepend_sys_path,
    pushd,
    safe_name,
)


# ──────────────────────────────────────────────────────────────────────────────
# Audio utilities
# ──────────────────────────────────────────────────────────────────────────────

def extract_audio_slice(video_path, start_sec, end_sec, output_wav, sr=16000):
    """Extract a mono audio slice from a video between start_sec and end_sec."""
    if start_sec >= end_sec:
        return False
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}", "-i", str(video_path),
        "-t", f"{end_sec - start_sec:.3f}",
        "-vn", "-ac", "1", "-ar", str(sr),
        "-acodec", "pcm_s16le",
        str(output_wav),
        "-loglevel", "panic",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and output_wav.exists() and output_wav.stat().st_size > 0


def pad_audio_to_duration(wav_path, duration_sec, sr=16000):
    """Zero-pad an audio file to at least duration_sec using ffmpeg apad."""
    target_samples = int(duration_sec * sr)
    _, audio = wavfile.read(wav_path)
    if len(audio) >= target_samples:
        return
    cmd = [
        "ffmpeg", "-y",
        "-i", str(wav_path),
        "-af", f"apad=whole_dur={duration_sec}",
        "-ar", str(sr), "-ac", "1", "-acodec", "pcm_s16le",
        str(wav_path.with_suffix(".padded.wav")),
        "-loglevel", "panic",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and wav_path.with_suffix(".padded.wav").exists():
        wav_path.with_suffix(".padded.wav").replace(wav_path)


# ──────────────────────────────────────────────────────────────────────────────
# Face crop utilities
# ──────────────────────────────────────────────────────────────────────────────

def _medfilt(x, kernel_size=13):
    """1-D median filter."""
    if kernel_size <= 1:
        return x
    pad = kernel_size // 2
    x = np.concatenate([[x[0]] * pad, x, [x[-1]] * pad])
    from scipy.ndimage import median_filter
    return median_filter(x, size=kernel_size)


def write_face_crop_video(track, output_avi, fps, crop_scale=0.40, source_video=None):
    """Write a face crop video for a single track.

    Mirrors Columbia_test.py crop_video() logic but:
    - Reads frames from source_video (avoids per-frame file I/O)
    - Crops to 224x224 (LR-ASD input size)
    - Smooths bbox with median filter

    Args:
        track:          face track dict from InsightFaceTracker
        output_avi:     output .avi path
        fps:            video fps
        crop_scale:     bbox padding scale
        source_video:   path to source video (if None, uses cached crops in track)

    Returns:
        True on success, False on failure.
    """
    crops = track.get("crops", [])
    frames = track.get("frames", [])

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vout = cv2.VideoWriter(str(output_avi), fourcc, fps, (224, 224))
    if not vout.isOpened():
        return False

    if source_video and os.path.exists(source_video):
        # Read from source video (smoother, uses median-filtered bbox)
        frame_to_idx = {int(f): i for i, f in enumerate(frames)}

        dets = {"x": [], "y": [], "s": []}
        for bbox in track.get("bboxes", []):
            dets["s"].append(max(bbox[3] - bbox[1], bbox[2] - bbox[0]) / 2)
            dets["y"].append((bbox[1] + bbox[3]) / 2)
            dets["x"].append((bbox[0] + bbox[2]) / 2)

        dets["s"] = _medfilt(np.array(dets["s"]), kernel_size=13)
        dets["x"] = _medfilt(np.array(dets["x"]), kernel_size=13)
        dets["y"] = _medfilt(np.array(dets["y"]), kernel_size=13)

        cap = cv2.VideoCapture(str(source_video))
        last_frame_idx = -1
        last_face = None

        for fidx in range(int(track["start_frame"]), int(track["end_frame"]) + 1):
            if fidx != last_frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                _, image = cap.read()
                last_frame_idx = fidx
                last_face = image

            image = last_face
            if image is None:
                image = np.zeros((512, 512, 3), dtype=np.uint8)

            # Map fidx to track array index
            arr_idx = fidx - int(track["start_frame"])
            if arr_idx < 0 or arr_idx >= len(dets["s"]):
                gray = np.zeros((224, 224), dtype=np.uint8)
                vout.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                continue

            cs = crop_scale
            bs = dets["s"][arr_idx]
            bsi = int(bs * (1 + 2 * cs))

            ih, iw = image.shape[:2]
            padded = np.pad(
                image, ((bsi, bsi), (bsi, bsi), (0, 0)),
                mode="constant", constant_values=110,
            )
            my = dets["y"][arr_idx] + bsi
            mx = dets["x"][arr_idx] + bsi
            face = padded[
                int(my - bs): int(my + bs * (1 + 2 * cs)),
                int(mx - bs * (1 + cs)): int(mx + bs * (1 + cs)),
            ]
            gray = cv2.resize(face, (224, 224))
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            vout.write(gray)

        cap.release()
        vout.release()
        return os.path.exists(str(output_avi))

    elif crops and frames:
        # Fall back to pre-cached crops from InsightFaceTracker
        frame_to_crop = {int(f): c for f, c in zip(frames, crops)}
        last_crop = None
        for fidx in range(int(track["start_frame"]), int(track["end_frame"]) + 1):
            crop = frame_to_crop.get(fidx)
            if crop is not None:
                last_crop = crop
            crop = last_crop
            if crop is None:
                gray = np.zeros((224, 224), dtype=np.uint8)
            else:
                if len(crop.shape) == 3:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                else:
                    gray = crop
                gray = cv2.resize(gray, (224, 224))
            vout.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        vout.release()
        return os.path.exists(str(output_avi))

    else:
        vout.release()
        return False


def crop_face_to_window(track, window_start_sec, window_end_sec, fps, output_avi):
    """Crop face to a specific speaker time window.

    Computes the overlap between the track's frame range and the speaker window,
    then writes a face crop video containing only frames within the window.

    Args:
        track:          face track dict
        window_start_sec: speaker window start (seconds)
        window_end_sec:  speaker window end (seconds)
        fps:            video fps
        output_avi:     output .avi path

    Returns:
        True on success, False on failure.
    """
    # Convert speaker window to frame indices
    win_start_f = window_start_sec * fps
    win_end_f = window_end_sec * fps

    # Find overlap with track
    tr_start = float(track["start_frame"])
    tr_end = float(track["end_frame"])

    overlap_start = max(tr_start, win_start_f)
    overlap_end = min(tr_end, win_end_f)

    if overlap_end <= overlap_start:
        return False

    # Build a sub-track dict for the overlapping frames
    crops = track.get("crops", [])
    frames = track.get("frames", [])
    bboxes = track.get("bboxes", [])
    scores = track.get("scores", [])

    frame_to_idx = {int(f): i for i, f in enumerate(frames)}

    sub_crops = []
    sub_frames = []
    sub_bboxes = []
    sub_scores = []

    for fidx in range(int(overlap_start), int(overlap_end) + 1):
        if fidx not in frame_to_idx:
            continue
        idx = frame_to_idx[fidx]
        sub_crops.append(crops[idx] if idx < len(crops) else None)
        sub_frames.append(fidx)
        if idx < len(bboxes):
            sub_bboxes.append(bboxes[idx])
        if idx < len(scores):
            sub_scores.append(scores[idx])

    if not sub_frames:
        return False

    sub_track = {
        "track_id": track["track_id"],
        "start_frame": int(overlap_start),
        "end_frame": int(overlap_end),
        "crops": sub_crops,
        "frames": sub_frames,
        "bboxes": sub_bboxes,
        "scores": sub_scores,
    }

    return write_face_crop_video(sub_track, output_avi, fps)


# ──────────────────────────────────────────────────────────────────────────────
# LR-ASD inference
# ──────────────────────────────────────────────────────────────────────────────

def _run_lrasd_inference(avi_path, wav_path, model, device, duration_set=None):
    """Run LR-ASD inference on a face crop video + aligned audio.

    Mirrors Columbia_test.py evaluate_network() logic with multi-duration averaging.

    Args:
        avi_path:   path to face crop video (224x224, 25fps)
        wav_path:   path to aligned audio WAV (16kHz mono)
        model:      ASD torch.nn.Module instance
        device:     torch device
        duration_set: list of chunk durations for averaging

    Returns:
        np.ndarray of per-frame scores, or None on failure.
    """
    import python_speech_features

    if duration_set is None:
        duration_set = [1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6]

    if not os.path.exists(wav_path):
        return None
    _, audio = wavfile.read(wav_path)
    audio_feature = python_speech_features.mfcc(
        audio, 16000, numcep=13, winlen=0.025, winstep=0.010,
    )  # (N, 13)

    cap = cv2.VideoCapture(avi_path)
    video_feature = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (224, 224))
        gray = gray[56:168, 56:168]  # center-crop to 112x112
        video_feature.append(gray)
    cap.release()
    video_feature = np.array(video_feature)

    if video_feature.shape[0] == 0:
        return None

    length = min(
        (audio_feature.shape[0] - audio_feature.shape[0] % 4) / 100,
        video_feature.shape[0] / 25,
    )
    # Hard minimum: LR-ASD needs at least 1s of aligned audio+video to produce
    # meaningful scores. Shorter windows produce empty tensors and crash.
    MIN_DURATION = 1.0
    if length < MIN_DURATION:
        return None

    # Filter out durations longer than the available length to avoid OOB slices
    valid_durations = [d for d in duration_set if d <= length]
    if not valid_durations:
        return None

    audio_feature = audio_feature[: int(round(length * 100)), :]
    video_feature = video_feature[: int(round(length * 25)), :, :]

    all_scores = []
    for duration in valid_durations:
        batch_size = math.ceil(length / duration)
        scores = []
        with torch.no_grad():
            for i in range(batch_size):
                a_start = i * duration * 100
                a_end = (i + 1) * duration * 100
                v_start = i * duration * 25
                v_end = (i + 1) * duration * 25

                inputA = torch.FloatTensor(audio_feature[a_start:a_end, :]).unsqueeze(0).to(device)
                inputV = torch.FloatTensor(video_feature[v_start:v_end, :, :]).unsqueeze(0).to(device)

                embedA = model.model.forward_audio_frontend(inputA)
                embedV = model.model.forward_visual_frontend(inputV)
                out = model.model.forward_audio_visual_backend(embedA, embedV)
                score = model.lossAV.forward(out, labels=None)
                scores.extend(score)

        all_scores.append(scores)

    result = np.round(np.mean(np.array(all_scores), axis=0), 1).astype(float)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# LRASDRunner
# ──────────────────────────────────────────────────────────────────────────────

class LRASDRunner:
    """Loaded LR-ASD model with per-speaker audio alignment support."""

    def __init__(self, args):
        self.args = args
        self.root = Path(args.lrasd_root).resolve()
        self.model_path = Path(args.lrasd_ckpt).expanduser()
        self.cache_dir = Path(args.output_dir) / "lrasd_crops"
        ensure_dir(self.cache_dir)

        clear_modules(
            "model", "model.Model", "model.Encoder", "model.Classifier",
            "model.faceDetector", "ASD", "loss", "dataLoader",
            "utils", "scenedetect",
        )
        with prepend_sys_path(self.root), pushd(self.root):
            from ASD import ASD

        self.device = torch.device(args.device)
        self.model = ASD()
        self.model.loadParameters(str(self.model_path))
        self.model.eval()
        self.model.to(self.device)

        print(f"[LR-ASD] Model loaded from {self.model_path} on {self.device}")

    def score(self, video_path, face_result, track_id=None, speaker_window=None):
        """Run ASD with optional per-speaker audio alignment.

        In multi-speaker mode, pass speaker_window=(start_sec, end_sec) to
        extract the speaker's audio slice and crop face to that window.

        In single-speaker / no-window mode, uses the full segment audio and
        all face tracks (original behavior).

        Args:
            video_path:       path to video segment
            face_result:      dict from InsightFaceTracker.extract_tracks()
            track_id:         optional, evaluate only this track
            speaker_window:    optional tuple (start_sec, end_sec). If provided,
                              audio is extracted for this window and face is
                              cropped to overlap with it. If None, uses full
                              track range with full segment audio.

        Returns:
            dict with keys:
              status: "ok" | "no_tracks" | "no_overlap" | "error"
              tracks: list of per-track results
              primary_track_id: track with highest mean score
              primary_mean_score: mean score of primary track
        """
        tracks = face_result.get("tracks", [])
        if not tracks:
            return {"status": "no_tracks", "tracks": [], "primary_track_id": None}

        fps = float(face_result.get("video_meta", {}).get("fps") or 25.0)
        base_stem = safe_name(Path(video_path).stem)
        crop_dir = self.cache_dir / base_stem
        ensure_dir(crop_dir)

        # Filter to specific track if requested
        if track_id is not None:
            tracks = [t for t in tracks if t["track_id"] == track_id]
            if not tracks:
                return {"status": "track_not_found", "tracks": [], "primary_track_id": None}

        scored_tracks = []
        for tr in tracks:
            tid = tr["track_id"]

            # ── Determine audio + video paths for this track+speaker window ──
            if speaker_window is not None:
                spk_start, spk_end = speaker_window
                # Cache key includes speaker window to avoid collision
                cache_key = f"{tid:05d}_{int(spk_start*1000):06d}_{int(spk_end*1000):06d}"
            else:
                spk_start = tr["start_frame"] / fps
                spk_end = (tr["end_frame"] + 1) / fps
                cache_key = f"{tid:05d}_full"

            avi_path = crop_dir / f"crop_{cache_key}.avi"
            wav_path = crop_dir / f"crop_{cache_key}.wav"

            # ── Build aligned face video + audio (with cache) ──────────────────
            if not (avi_path.exists() and wav_path.exists()):
                # Step 1: crop face to window
                if speaker_window is not None:
                    ok = crop_face_to_window(tr, spk_start, spk_end, fps, avi_path)
                else:
                    ok = write_face_crop_video(tr, avi_path, fps, source_video=video_path)
                if not ok or not avi_path.exists():
                    continue

                # Step 2: extract aligned audio slice
                ok = extract_audio_slice(video_path, spk_start, spk_end, wav_path)
                if not ok:
                    # Fall back: use segment-level audio extraction
                    tr_start = tr["start_frame"] / fps
                    tr_end = (tr["end_frame"] + 1) / fps
                    ok = extract_audio_slice(video_path, tr_start, tr_end, wav_path)
                if not ok or not wav_path.exists():
                    continue

                # Step 3: ensure audio is long enough (LR-ASD expects ≥ some duration)
                min_dur = 2.0  # seconds; pad if needed
                pad_audio_to_duration(wav_path, min_dur)

            if not (avi_path.exists() and wav_path.exists()):
                continue

            # ── Run LR-ASD inference ──────────────────────────────────────────
            try:
                scores = _run_lrasd_inference(
                    str(avi_path), str(wav_path), self.model, self.device
                )
            except Exception as e:
                print(f"[LR-ASD] Inference error track {tid}: {e}")
                continue

            if scores is None or len(scores) == 0:
                continue

            mean_score = float(np.mean(scores))
            threshold = getattr(self.args, "lrasd_score_threshold", 0.0)
            scored_tracks.append({
                "track_id": int(tid),
                "scores": scores.tolist(),
                "mean_score": mean_score,
                "is_speaking": mean_score > threshold,
                "num_frames": len(scores),
                "speaker_window": (spk_start, spk_end) if speaker_window else None,
            })

        if not scored_tracks:
            return {"status": "no_valid_tracks", "tracks": [], "primary_track_id": None}

        primary = max(scored_tracks, key=lambda x: x["mean_score"])
        return {
            "status": "ok",
            "tracks": scored_tracks,
            "primary_track_id": primary["track_id"],
            "primary_mean_score": primary["mean_score"],
            "primary_is_speaking": primary["is_speaking"],
        }


# ──────────────────────────────────────────────────────────────────────────────
# LRASDWorker
# ──────────────────────────────────────────────────────────────────────────────

class LRASDWorker:
    """High-level wrapper with per-speaker audio alignment API."""

    def __init__(self, args):
        self.args = args
        self.runner = None
        self.static_status = None

        if getattr(args, "disable_lrasd", False):
            self.static_status = "disabled"
            return

        ckpt = getattr(args, "lrasd_ckpt", "")
        root = getattr(args, "lrasd_root", "")
        if not ckpt or not root:
            self.static_status = "not_configured"
            return

        if not Path(ckpt).is_file():
            self.static_status = "model_missing"
            return

        try:
            self.runner = LRASDRunner(args)
        except Exception as e:
            self.static_status = f"load_error: {e}"
            self.runner = None

    def score(self, video_path, face_result, track_id=None, speaker_window=None):
        """Score active speaker with optional per-speaker audio alignment.

        Args:
            video_path:      path to video segment
            face_result:     dict from InsightFaceTracker.extract_tracks()
            track_id:        optional specific track to score
            speaker_window:   optional (start_sec, end_sec). When provided,
                             audio is extracted for this window and face is
                             cropped to the overlap with the speaker window.

        Returns:
            dict with status, tracks, primary_track_id, primary_is_speaking.
        """
        if self.runner is None:
            return {
                "status": self.static_status or "not_initialized",
                "tracks": [],
                "primary_track_id": None,
                "primary_is_speaking": False,
            }

        result = self.runner.score(video_path, face_result, track_id, speaker_window)
        if result.get("status") == "ok":
            result["primary_is_speaking"] = any(
                t["is_speaking"] for t in result.get("tracks", [])
            )
        else:
            result["primary_is_speaking"] = False

        return result

    def close(self):
        pass
