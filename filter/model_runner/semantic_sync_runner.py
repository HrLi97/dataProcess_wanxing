"""Semantic Audio-Visual Sync Runner using ImageBind.

ImageBind (Meta, 2023) embeds multiple modalities (audio, video, text, etc.)
into a shared latent space. By computing cosine similarity between audio and
video embeddings, we can detect whether the speech content semantically matches
the visual content.

Use cases:
  - Reject segments where audio is dubbed/mismatched (e.g., background music
    playing over a silent person, or a voiceover describing unrelated content)
  - Combined with LR-ASD (temporal lip-sync) provides both:
    * Temporal alignment: lips move in sync with audio
    * Semantic alignment: audio content matches visual scene

Integration:
  - Runs after audio enhancement (uses cleaned audio for better embedding)
  - Outputs a similarity score [0, 1] per segment
  - score > threshold → semantically synchronized
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from filter.filter_utils import ensure_dir, safe_name


# ──────────────────────────────────────────────────────────────────────────────
# ImageBind loader
# ──────────────────────────────────────────────────────────────────────────────

IMAGEBIND_ROOT = Path(__file__).resolve().parents[2] / "third_part" / "ImageBind"


def _ensure_imagebind_importable():
    """Add ImageBind to sys.path if needed."""
    root = str(IMAGEBIND_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _load_imagebind_model(device):
    """Load pretrained ImageBind model."""
    _ensure_imagebind_importable()
    from imagebind.models import imagebind_model
    from imagebind.models.imagebind_model import ModalityType

    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    return model, ModalityType


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_audio_wav(video_path, output_wav, sr=16000):
    """Extract mono 16kHz WAV from video."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sr),
        "-acodec", "pcm_s16le",
        str(output_wav),
        "-loglevel", "quiet",
    ]
    r = subprocess.run(cmd, capture_output=True, check=False)
    return r.returncode == 0 and Path(output_wav).exists()


def _extract_keyframes(video_path, output_dir, max_frames=5):
    """Extract evenly-spaced keyframes from video as images."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Sample evenly spaced frames
    indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
    paths = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame_path = Path(output_dir) / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        paths.append(str(frame_path))

    cap.release()
    return paths


# ──────────────────────────────────────────────────────────────────────────────
# SemanticSyncRunner
# ──────────────────────────────────────────────────────────────────────────────

class SemanticSyncRunner:
    """Compute audio-visual semantic similarity using ImageBind."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(getattr(args, "device", "cuda:0"))
        self.model = None
        self.ModalityType = None
        self.threshold = getattr(args, "semantic_sync_threshold", 0.15)
        self.max_frames = getattr(args, "semantic_sync_max_frames", 5)

        try:
            self.model, self.ModalityType = _load_imagebind_model(self.device)
            print(f"[SemanticSync] ImageBind loaded on {self.device}")
        except Exception as e:
            print(f"[SemanticSync] Failed to load ImageBind: {e}")
            self.model = None

    @property
    def is_available(self):
        return self.model is not None

    def score(self, video_path, enhanced_audio_path=None):
        """Compute semantic sync score between video and audio.

        Args:
            video_path:          path to video segment
            enhanced_audio_path: optional path to enhanced audio WAV.
                                 If None, audio is extracted from video directly.

        Returns:
            dict with keys:
              status: "ok" | "no_model" | "error"
              semantic_score: float cosine similarity [0, 1]
              is_semantically_synced: bool (score > threshold)
        """
        if self.model is None:
            return {
                "status": "no_model",
                "semantic_score": None,
                "is_semantically_synced": False,
            }

        _ensure_imagebind_importable()
        from imagebind import data as imagebind_data

        tmp_dir = tempfile.mkdtemp(prefix="semantic_sync_")
        try:
            # Prepare audio
            if enhanced_audio_path and Path(enhanced_audio_path).is_file():
                audio_path = enhanced_audio_path
            else:
                audio_path = os.path.join(tmp_dir, "audio.wav")
                ok = _extract_audio_wav(video_path, audio_path)
                if not ok:
                    return {
                        "status": "no_audio",
                        "semantic_score": None,
                        "is_semantically_synced": False,
                    }

            # Prepare video frames
            frame_dir = os.path.join(tmp_dir, "frames")
            ensure_dir(frame_dir)
            frame_paths = _extract_keyframes(video_path, frame_dir, self.max_frames)
            if not frame_paths:
                return {
                    "status": "no_frames",
                    "semantic_score": None,
                    "is_semantically_synced": False,
                }

            # Compute embeddings
            with torch.no_grad():
                # Audio embedding
                audio_input = imagebind_data.load_and_transform_audio_data(
                    [audio_path], self.device
                )
                # Vision embedding (use frames as images)
                vision_input = imagebind_data.load_and_transform_vision_data(
                    frame_paths, self.device
                )

                inputs = {
                    self.ModalityType.AUDIO: audio_input,
                    self.ModalityType.VISION: vision_input,
                }

                embeddings = self.model(inputs)
                audio_emb = embeddings[self.ModalityType.AUDIO]  # (1, D)
                vision_emb = embeddings[self.ModalityType.VISION]  # (N, D)

                # Normalize
                audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
                vision_emb = vision_emb / vision_emb.norm(dim=-1, keepdim=True)

                # Compute cosine similarity (audio vs mean of vision frames)
                vision_mean = vision_emb.mean(dim=0, keepdim=True)
                vision_mean = vision_mean / vision_mean.norm(dim=-1, keepdim=True)
                similarity = torch.nn.functional.cosine_similarity(
                    audio_emb, vision_mean
                ).item()

            # Clamp to [0, 1] for readability
            similarity = max(0.0, similarity)

            return {
                "status": "ok",
                "semantic_score": round(similarity, 4),
                "is_semantically_synced": similarity > self.threshold,
            }

        except Exception as e:
            print(f"[SemanticSync] Error: {e}")
            return {
                "status": "error",
                "semantic_score": None,
                "is_semantically_synced": False,
            }
        finally:
            # Cleanup temp files
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def close(self):
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# SemanticSyncWorker (high-level wrapper)
# ──────────────────────────────────────────────────────────────────────────────

class SemanticSyncWorker:
    """High-level wrapper matching the Worker interface pattern."""

    def __init__(self, args):
        self.args = args
        self.runner = None
        self.static_status = None

        if getattr(args, "disable_semantic_sync", False):
            self.static_status = "disabled"
            return

        # Check if ImageBind is available
        if not IMAGEBIND_ROOT.is_dir():
            self.static_status = "imagebind_not_found"
            return

        try:
            self.runner = SemanticSyncRunner(args)
            if not self.runner.is_available:
                self.static_status = "load_failed"
                self.runner = None
        except Exception as e:
            self.static_status = f"load_error: {e}"
            self.runner = None

    def score(self, video_path, enhanced_audio_path=None):
        """Score semantic audio-visual sync.

        Args:
            video_path:          path to video segment
            enhanced_audio_path: optional enhanced audio path (from AudioEnhanceWorker)

        Returns:
            dict with status, semantic_score, is_semantically_synced
        """
        if self.runner is None:
            return {
                "status": self.static_status or "not_initialized",
                "semantic_score": None,
                "is_semantically_synced": False,
            }
        return self.runner.score(video_path, enhanced_audio_path)

    def close(self):
        if self.runner is not None:
            self.runner.close()
