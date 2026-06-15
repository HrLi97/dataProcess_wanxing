"""PyAnnote speaker diarization runner."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def _get_pyannote_src_root():
    """Locate pyannote-audio source tree relative to this file."""
    root = Path(__file__).resolve().parents[2]
    src = root / "third_part" / "pyannote-audio" / "src"
    if src.is_dir():
        return str(src)
    return None


def check_pyannote_available():
    """Check if pyannote.audio is importable (handles dynamic src injection).

    Returns:
        tuple(bool, str): (is_available, reason_if_not)
    """
    try:
        import pyannote.audio
        import pyannote.core
        return True, ""
    except ImportError as e:
        src_root = _get_pyannote_src_root()
        if src_root and src_root not in sys.path:
            sys.path.insert(0, src_root)
        try:
            import pyannote.audio
            import pyannote.core
            return True, ""
        except ImportError as e2:
            return False, f"{e2}"


class PyannoteDiarizationRunner:
    def __init__(self, args):
        self.args = args
        self.pipeline = None
        self._available, self._check_reason = check_pyannote_available()
        if not self._available:
            return
        self._ensure_imported()
        import torch
        from pyannote.audio import Pipeline
        self.pipeline = Pipeline.from_pretrained(str(args.pyannote_model_path), token=args.pyannote_hf_token)
        device_str = getattr(args, "device", "cpu")
        self.pipeline.to(torch.device(device_str))

    def _ensure_imported(self):
        """Guarantee pyannote.audio is importable in this process."""
        src_root = _get_pyannote_src_root()
        if src_root and src_root not in sys.path:
            sys.path.insert(0, src_root)

    def diarize(self, video_path):
        """Run diarization on a video file. Returns dict with speaker segments."""
        if self.pipeline is None:
            status = "not_available"
            if not self._available:
                status = "not_installed"
            return {"status": status, "reason": self._check_reason, "speakers": []}

        self._ensure_imported()
        audio_wav = self._extract_audio(video_path)
        if audio_wav is None:
            return {"status": "no_audio", "reason": "ffmpeg audio extraction failed", "speakers": []}

        diarization = self.pipeline(audio_wav)

        speakers = []
        diar = diarization.speaker_diarization
        for turn, _, speaker in diar.itertracks(yield_label=True):
            speakers.append({
                "speaker_id": speaker,
                "start_sec": round(turn.start, 3),
                "end_sec": round(turn.end, 3),
            })

        return {"status": "ok", "speakers": speakers}

    def _extract_audio(self, video_path):
        """Extract audio from video to waveform dict (torchcodec-free).
        
        Returns a dict {'waveform': tensor(C, T), 'sample_rate': int} as expected by
        pyannote.audio Pipeline, or None on failure.
        """
        wav_path = Path(tempfile.mktemp(suffix=".wav"))
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le",
            str(wav_path).replace(".wav", ".raw"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        raw_path = Path(str(wav_path).replace(".wav", ".raw"))
        if proc.returncode != 0 or not raw_path.is_file() or raw_path.stat().st_size == 0:
            return None
        import numpy as np
        import torch
        pcm = np.fromfile(str(raw_path), dtype=np.int16)
        waveform = torch.from_numpy(pcm.astype(np.float32) / 32768.0).unsqueeze(0)
        raw_path.unlink(missing_ok=True)
        return {"waveform": waveform, "sample_rate": 16000}


class PyannoteDiarizationWorker:
    """Lightweight wrapper matching the Worker interface pattern used in the project."""

    def __init__(self, args):
        self.args = args
        self.runner = None
        self.static_status = None
        self.static_reason = ""

        if getattr(args, "disable_pyannote", False):
            self.static_status = "disabled"
            return
        available, reason = check_pyannote_available()
        if not available:
            self.static_status = "not_installed"
            self.static_reason = reason
            return
        if not getattr(args, "pyannote_model_path", None):
            self.static_status = "not_configured"
            return

        runner_args = argparse.Namespace(**vars(args))
        self.runner = PyannoteDiarizationRunner(runner_args)

    def diarize(self, video_path):
        if self.runner is None:
            return {"status": self.static_status, "reason": self.static_reason, "speakers": []}
        return self.runner.diarize(video_path)

    def score(self, video_path):
        """Alias for compatibility."""
        return self.diarize(video_path)

    def close(self):
        pass
