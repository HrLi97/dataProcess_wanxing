"""SyncNet Lip-Sync Evaluation Runner.

Ported from filter_two_in_one.py's SyncNetInstance usage.
SyncNet evaluates temporal audio-visual synchronization by computing
offset and confidence between lip movements and speech audio.

Decision logic (from filter_two_in_one.py):
  - offset in [min_offset, max_offset] (default [-6, 6])
  - confidence >= min_confidence (default 1.0)

Replaces LR-ASD as the primary lip-sync gate in the pipeline.
"""
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import signal
from scipy.io import wavfile

from filter.filter_utils import ensure_dir, safe_name


# ──────────────────────────────────────────────────────────────────────────────
# SyncNet Model Definition (from syncnet_python-master/SyncNetModel.py)
# ──────────────────────────────────────────────────────────────────────────────

class _SyncNetModel(torch.nn.Module):
    def __init__(self, num_layers_in_fc_layers=1024):
        super().__init__()

        self.netcnnaud = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(1, 1), stride=(1, 1)),

            torch.nn.Conv2d(64, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 2)),

            torch.nn.Conv2d(192, 384, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

            torch.nn.Conv2d(256, 512, kernel_size=(5, 4), padding=(0, 0)),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
        )

        self.netfcaud = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netfclip = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, num_layers_in_fc_layers),
        )

        self.netcnnlip = torch.nn.Sequential(
            torch.nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=0),
            torch.nn.BatchNorm3d(96),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            torch.nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),

            torch.nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

            torch.nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU(inplace=True),
        )

    def forward_aud(self, x):
        mid = self.netcnnaud(x)
        mid = mid.view((mid.size()[0], -1))
        out = self.netfcaud(mid)
        return out

    def forward_lip(self, x):
        mid = self.netcnnlip(x)
        mid = mid.view((mid.size()[0], -1))
        out = self.netfclip(mid)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Core SyncNet evaluation functions
# ──────────────────────────────────────────────────────────────────────────────

def _calc_pdist(feat1, feat2, vshift=10):
    """Compute pairwise distances with temporal shift."""
    win_size = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
    dists = []
    for i in range(len(feat1)):
        dists.append(
            torch.nn.functional.pairwise_distance(
                feat1[[i], :].repeat(win_size, 1), feat2p[i:i + win_size, :]
            )
        )
    return dists


def _syncnet_evaluate(model, video_frames, audio_data, vshift=10, batch_size=20):
    """Evaluate synchronization between video frames and audio.

    Args:
        model:        _SyncNetModel instance (on CUDA)
        video_frames: numpy array of face crops (N, H, W, C) - 224x224 RGB
        audio_data:   numpy array of audio samples (16kHz mono int16/float)
        vshift:       temporal shift window
        batch_size:   batch size for feature extraction

    Returns:
        (offset, confidence, dists_npy) or (0, 0, None) if too short
    """
    import python_speech_features

    model.eval()

    # Process video frames: (N, H, W, C) -> (1, N, C, H, W) for Conv3D
    im = np.stack(video_frames, axis=3)  # (H, W, C, N)
    im = np.expand_dims(im, axis=0)      # (1, H, W, C, N)
    im = np.transpose(im, (0, 3, 4, 1, 2))  # (1, C, N, H, W)
    imtv = torch.from_numpy(im.astype(np.float32))

    # Process audio: MFCC features
    if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
        audio_int16 = (audio_data * 32768).astype(np.int16)
    else:
        audio_int16 = audio_data
    mfcc = zip(*python_speech_features.mfcc(audio_int16, 16000))
    mfcc = np.stack([np.array(i) for i in mfcc])
    cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
    cct = torch.from_numpy(cc.astype(np.float32))

    # Check lengths
    min_length = min(len(video_frames), math.floor(len(audio_int16) / 640))
    if min_length <= 5:
        return 0, 0.0, None

    lastframe = min_length - 5
    im_feat = []
    cc_feat = []

    with torch.no_grad():
        for i in range(0, lastframe, batch_size):
            # Video features
            im_batch = [imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, min(lastframe, i + batch_size))]
            im_in = torch.cat(im_batch, 0)
            im_out = model.forward_lip(im_in.cuda())
            im_feat.append(im_out.data.cpu())

            # Audio features
            cc_batch = [cct[:, :, :, vframe * 4:vframe * 4 + 20] for vframe in range(i, min(lastframe, i + batch_size))]
            cc_in = torch.cat(cc_batch, 0)
            cc_out = model.forward_aud(cc_in.cuda())
            cc_feat.append(cc_out.data.cpu())

    im_feat = torch.cat(im_feat, 0)
    cc_feat = torch.cat(cc_feat, 0)

    # Compute offset
    dists = _calc_pdist(im_feat, cc_feat, vshift=vshift)
    mdist = torch.mean(torch.stack(dists, 1), 1)
    minval, minidx = torch.min(mdist, 0)

    offset = int((vshift - minidx).item())
    conf = float((torch.median(mdist) - minval).item())

    dists_npy = np.array([dist.numpy() for dist in dists])
    return offset, conf, dists_npy


# ──────────────────────────────────────────────────────────────────────────────
# Face crop utility (matches filter_two_in_one.py crop_video_fan_based_time)
# ──────────────────────────────────────────────────────────────────────────────

def _crop_face_for_syncnet(track, video_path, frame_indices=None, crop_scale=0.40):
    """Crop face regions from video for SyncNet evaluation.

    Args:
        track:         face track dict from InsightFaceTracker
        video_path:    path to source video
        frame_indices: optional list of frame indices to crop (subset)
        crop_scale:    bbox padding scale (default 0.40)

    Returns:
        numpy array of face crops (N, 224, 224, 3) RGB
    """
    frames_list = track.get("frames", [])
    bboxes = track.get("bboxes", [])

    if not frames_list or not bboxes:
        return np.array([])

    frame_to_bbox = {int(f): b for f, b in zip(frames_list, bboxes)}

    if frame_indices is not None:
        target_frames = [f for f in frame_indices if f in frame_to_bbox]
    else:
        target_frames = sorted(frame_to_bbox.keys())

    if not target_frames:
        return np.array([])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.array([])

    cropped = []
    last_frame_idx = -1

    for fidx in target_frames:
        if fidx != last_frame_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue
            last_frame_idx = fidx
            # Convert BGR to RGB for SyncNet
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame_rgb  # reuse

        bbox = frame_to_bbox[fidx]
        x1, y1, x2, y2 = bbox

        cs = crop_scale
        bs = max(y2 - y1, x2 - x1) / 2
        bsi = int(bs * (1 + 2 * cs))

        padded = np.pad(
            frame_rgb, ((bsi, bsi), (bsi, bsi), (0, 0)),
            mode="constant", constant_values=110,
        )
        my = (y1 + y2) / 2 + bsi
        mx = (x1 + x2) / 2 + bsi
        face = padded[
            int(my - bs): int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)): int(mx + bs * (1 + cs)),
        ]

        if face.size == 0:
            continue
        try:
            face_resized = cv2.resize(face, (224, 224))
        except Exception:
            continue
        cropped.append(face_resized)

    cap.release()
    return np.array(cropped) if cropped else np.array([])


def _extract_audio_by_frames(video_path, frame_indices, fps=25, sr=16000):
    """Extract audio samples corresponding to frame indices.

    Matches filter_two_in_one.py's extract_audio_by_track logic.
    """
    # First extract full audio from video
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(sr),
            "-acodec", "pcm_s16le",
            tmp_path,
            "-loglevel", "quiet",
        ]
        r = subprocess.run(cmd, capture_output=True, check=False)
        if r.returncode != 0 or not os.path.exists(tmp_path):
            return np.zeros(0, dtype=np.int16)
        _, audio_data = wavfile.read(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Extract snippets per frame
    snippets = []
    for fidx in frame_indices:
        t0 = fidx / fps
        t1 = (fidx + 1) / fps
        s0 = int(round(t0 * sr))
        s1 = int(round(t1 * sr))
        if s1 <= len(audio_data):
            snippets.append(audio_data[s0:s1])
    if snippets:
        return np.concatenate(snippets)
    return np.zeros(0, dtype=np.int16)


# ──────────────────────────────────────────────────────────────────────────────
# SyncNetRunner
# ──────────────────────────────────────────────────────────────────────────────

class SyncNetRunner:
    """Loaded SyncNet model for lip-sync evaluation."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(getattr(args, "device", "cuda:0"))
        self.vshift = getattr(args, "syncnet_vshift", 10)
        self.batch_size = getattr(args, "syncnet_batch_size", 20)
        self.min_confidence = getattr(args, "syncnet_min_confidence", 1.0)
        self.min_offset = getattr(args, "syncnet_min_offset", -6)
        self.max_offset = getattr(args, "syncnet_max_offset", 6)
        self.crop_scale = getattr(args, "syncnet_crop_scale", 0.40)

        ckpt_path = getattr(args, "syncnet_ckpt", "")
        self.model = _SyncNetModel()
        loaded_state = torch.load(ckpt_path, map_location="cpu")
        self_state = self.model.state_dict()
        for name, param in loaded_state.items():
            if name in self_state:
                self_state[name].copy_(param)
        self.model.cuda()
        self.model.eval()
        print(f"[SyncNet] Model loaded from {ckpt_path}")

    def score(self, video_path, face_result, track_id=None, speaker_window=None):
        """Evaluate lip-sync for face tracks using SyncNet.

        Args:
            video_path:      path to video segment
            face_result:     dict from InsightFaceTracker.extract_tracks()
            track_id:        optional specific track to score
            speaker_window:  optional (start_sec, end_sec) for per-speaker evaluation

        Returns:
            dict with keys:
              status: "ok" | "no_tracks" | "too_short" | "error"
              tracks: list of per-track results
              primary_track_id: track with highest confidence
              primary_mean_score: confidence of primary track
              primary_is_speaking: bool
        """
        tracks = face_result.get("tracks", [])
        if not tracks:
            return {"status": "no_tracks", "tracks": [], "primary_track_id": None, "primary_is_speaking": False}

        fps = float(face_result.get("video_meta", {}).get("fps") or 25.0)

        if track_id is not None:
            tracks = [t for t in tracks if t["track_id"] == track_id]
            if not tracks:
                return {"status": "track_not_found", "tracks": [], "primary_track_id": None, "primary_is_speaking": False}

        scored_tracks = []
        for tr in tracks:
            tid = tr["track_id"]
            frames_list = tr.get("frames", [])

            # Determine frame range based on speaker window
            if speaker_window is not None:
                spk_start, spk_end = speaker_window
                start_frame = int(round(spk_start * fps))
                end_frame = int(round(spk_end * fps))
                frame_indices = [f for f in frames_list if start_frame <= f <= end_frame]
            else:
                frame_indices = list(frames_list)

            if len(frame_indices) < 6:
                continue

            # Crop face regions
            video_frames = _crop_face_for_syncnet(
                tr, video_path, frame_indices=frame_indices, crop_scale=self.crop_scale
            )
            if len(video_frames) < 6:
                continue

            # Extract corresponding audio
            audio_data = _extract_audio_by_frames(video_path, frame_indices, fps=fps)
            if len(audio_data) == 0:
                continue

            # Run SyncNet evaluation
            offset, conf, dists = _syncnet_evaluate(
                self.model, video_frames, audio_data,
                vshift=self.vshift, batch_size=self.batch_size,
            )

            # Decision: offset in range AND confidence above threshold
            is_synced = (
                self.min_offset <= offset <= self.max_offset
                and conf >= self.min_confidence
            )

            scored_tracks.append({
                "track_id": int(tid),
                "offset": int(offset),
                "confidence": round(float(conf), 4),
                "is_speaking": is_synced,
                "mean_score": round(float(conf), 4),
                "speaker_window": (spk_start, spk_end) if speaker_window else None,
            })

        if not scored_tracks:
            return {"status": "no_valid_tracks", "tracks": [], "primary_track_id": None, "primary_is_speaking": False}

        primary = max(scored_tracks, key=lambda x: x["confidence"])
        return {
            "status": "ok",
            "tracks": scored_tracks,
            "primary_track_id": primary["track_id"],
            "primary_mean_score": primary["confidence"],
            "primary_is_speaking": primary["is_speaking"],
        }

    def close(self):
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# SyncNetWorker (high-level wrapper, replaces LRASDWorker)
# ──────────────────────────────────────────────────────────────────────────────

class SyncNetWorker:
    """High-level wrapper matching the LRASDWorker interface."""

    def __init__(self, args):
        self.args = args
        self.runner = None
        self.static_status = None

        if getattr(args, "disable_syncnet", False):
            self.static_status = "disabled"
            return

        ckpt = getattr(args, "syncnet_ckpt", "")
        if not ckpt or not Path(ckpt).is_file():
            self.static_status = "not_configured"
            return

        self.runner = SyncNetRunner(args)

    def score(self, video_path, face_result, track_id=None, speaker_window=None):
        """Score lip-sync using SyncNet.

        Interface matches LRASDWorker.score() for drop-in replacement.
        """
        if self.runner is None:
            return {
                "status": self.static_status or "not_initialized",
                "tracks": [],
                "primary_track_id": None,
                "primary_is_speaking": False,
            }
        return self.runner.score(video_path, face_result, track_id, speaker_window)

    def close(self):
        if self.runner is not None:
            self.runner.close()
