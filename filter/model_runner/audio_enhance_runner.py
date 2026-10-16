"""Audio Enhancement Runner: Voice Separation + Speech Enhancement.

Ported from filter_two_in_one.py's first_step logic:
  1. MelBandRoformer for music/vocals source separation
  2. ClearVoice (MossFormer2_SE_48K) for speech enhancement

This produces a clean vocals WAV that improves downstream:
  - PyAnnote speaker diarization accuracy
  - LR-ASD / SyncNet lip-sync scoring
  - ImageBind semantic similarity computation
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from filter.filter_utils import ensure_dir, safe_name, ckps_root, env_or, prepend_sys_path, pushd


# ──────────────────────────────────────────────────────────────────────────────
# Audio separation (MelBandRoformer)
# ──────────────────────────────────────────────────────────────────────────────

def _load_msst_module(mss_root, rel_path):
    """按文件路径加载 Music-Source-Separation-Training 里的模块（utils/settings.py、utils/model_utils.py）。

    固定的 MSST 提交里没有 utils_separation 模块。MSST 的 utils/ 与 TalkNCE 的 utils/（还有项目根 utils/）
    都是没有 __init__.py 的顶层 namespace 包，且都有 model_utils.py；同一进程里 TalkNCE 已经
    import 了 utils.distributed，再 `from utils.xxx import ...` 会串包。这里用私有模块名加载，不碰 sys.modules["utils"]。
    """
    name = f"_msst_{Path(rel_path).stem}"
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, str(Path(mss_root) / rel_path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return mod


def _extract_audio_from_video(video_path, sr=44100):
    """Extract audio from video as numpy array at given sample rate."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sr), "-ac", "2",
        "-f", "wav", "-loglevel", "quiet",
        tmp_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    import librosa
    mix, _ = librosa.load(tmp_path, sr=sr, mono=False)
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=0)
    return mix



def _run_separation(model, config, mix, device, demix, use_tta=False):
    """Run MelBandRoformer source separation. demix: MSST utils/model_utils.py 里的 demix。"""
    mix_orig = mix.copy()
    if config.inference.get("normalize", False):
        mono = mix.mean(0)
        mean, std = mono.mean(), mono.std() + 1e-8
        mix = (mix - mean) / std
    else:
        mean, std = None, None

    track_proc_list = [mix.copy()]
    if use_tta:
        track_proc_list.extend([mix[::-1].copy(), -1.0 * mix.copy()])

    full_result = []
    for proc_mix in track_proc_list:
        waveforms = demix(
            config, model, proc_mix, device,
            pbar=False, model_type="mel_band_roformer",
        )
        full_result.append(waveforms)

    waveforms = full_result[0]
    for i in range(1, len(full_result)):
        d = full_result[i]
        for el in d:
            if i == 2:
                waveforms[el] += -1.0 * d[el]
            elif i == 1:
                waveforms[el] += d[el][::-1].copy()
            else:
                waveforms[el] += d[el]
    for el in waveforms:
        waveforms[el] /= len(full_result)

    return {
        "waveforms": waveforms,
        "sr": 44100,
        "mix_orig": mix_orig,
        "normalize_params": (mean, std) if mean is not None else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# AudioEnhanceRunner
# ──────────────────────────────────────────────────────────────────────────────

class AudioEnhanceRunner:
    """Loads MelBandRoformer + ClearVoice for audio separation and enhancement."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device(getattr(args, "device", "cuda:0"))
        self.model = None
        self.config = None
        self.demix = None
        self.clearvoice = None

        sep_config_path = getattr(args, "audio_sep_config", "")
        sep_ckpt_path = getattr(args, "audio_sep_ckpt", "")
        mss_root = getattr(args, "audio_sep_root", "")

        if not sep_config_path or not sep_ckpt_path:
            print("[AudioEnhance] Separation model not configured, skipping init.")
            return

        # Load MelBandRoformer（get_model_from_config 在 MSST utils/settings.py，demix 在 utils/model_utils.py）
        settings = _load_msst_module(mss_root, "utils/settings.py")
        self.demix = _load_msst_module(mss_root, "utils/model_utils.py").demix
        # get_model_from_config 内部 `from models.bs_roformer import ...`，构建模型期间需要 MSST 根在 sys.path 最前
        with prepend_sys_path(mss_root):
            model, config = settings.get_model_from_config("mel_band_roformer", sep_config_path)
        state_dict = torch.load(sep_ckpt_path, map_location=self.device)
        model.load_state_dict(state_dict)
        self.model = model.to(self.device).eval()
        self.config = config
        print(f"[AudioEnhance] MelBandRoformer loaded from {sep_ckpt_path}")

        # Load ClearVoice
        try:
            from clearvoice.clearvoice import ClearVoice  # 源码仓库布局（原写法）
        except ImportError:
            from clearvoice import ClearVoice  # pip 包 clearvoice 0.1.x：ClearVoice 定义在包 __init__ 里
        # ClearVoice 按 yaml 里的相对路径 checkpoints/MossFormer2_SE_48K（相对当前工作目录）找权重；找不到会
        # 尝试联网下载，失败只打 warning、模型保持随机初始化。这里先确认权重存在，再切到 CLEARVOICE_DIR 下构建。
        clearvoice_dir = env_or("CLEARVOICE_DIR", ckps_root() / "clearvoice")
        best_ckpt = clearvoice_dir / "checkpoints" / "MossFormer2_SE_48K" / "last_best_checkpoint"
        if not best_ckpt.is_file():
            raise FileNotFoundError(f"[AudioEnhance] ClearVoice 权重不存在: {best_ckpt}（检查 CLEARVOICE_DIR）")
        with pushd(clearvoice_dir):
            self.clearvoice = ClearVoice(
                task="speech_enhancement",
                model_names=["MossFormer2_SE_48K"],
            )
        print(f"[AudioEnhance] ClearVoice (MossFormer2_SE_48K) loaded from {clearvoice_dir}.")


    @property
    def is_available(self):
        return self.model is not None

    def enhance(self, video_path, output_dir=None):
        """Run separation + enhancement on a video.

        Returns:
            dict with keys:
              status: "ok" | "no_model" | "no_audio" | "error"
              enhanced_audio_path: path to enhanced mono WAV (16kHz)
              vocals_path: path to separated vocals WAV (44.1kHz stereo)
        """
        if self.model is None:
            return {"status": "no_model", "enhanced_audio_path": None, "vocals_path": None}

        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="audio_enhance_"))
        else:
            output_dir = Path(output_dir)
        ensure_dir(output_dir)

        # Step 1: Extract audio
        mix = _extract_audio_from_video(video_path, sr=44100)
        if mix is None:
            return {"status": "no_audio", "enhanced_audio_path": None, "vocals_path": None}

        # Step 2: Source separation
        sep_result = _run_separation(
            self.model, self.config, mix, self.device, self.demix,
            use_tta=getattr(self.args, "audio_sep_tta", False),
        )

        # Step 3: Save vocals
        import soundfile as sf
        stem = safe_name(Path(video_path).stem)
        vocals_wav = sep_result["waveforms"].get("vocals")
        if vocals_wav is None:
            return {"status": "error", "enhanced_audio_path": None, "vocals_path": None}

        estimates = vocals_wav.T
        if sep_result["normalize_params"]:
            mean, std = sep_result["normalize_params"]
            estimates = estimates * std + mean

        vocals_path = output_dir / f"{stem}_vocals.wav"
        sf.write(str(vocals_path), estimates, sep_result["sr"], subtype="FLOAT")

        # Step 4: Speech enhancement (ClearVoice)
        enhanced_path = vocals_path
        if self.clearvoice is not None:
            enhanced_out = output_dir / f"{stem}_enhanced.wav"
            output_wav = self.clearvoice(input_path=str(vocals_path), online_write=False)
            self.clearvoice.write(output_wav, output_path=str(enhanced_out))
            enhanced_path = enhanced_out

        # Step 5: Convert to 16kHz mono for downstream
        final_16k_path = output_dir / f"{stem}_enhanced_16k.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(enhanced_path),
            "-ar", "16000", "-ac", "1",
            "-acodec", "pcm_s16le",
            str(final_16k_path),
            "-loglevel", "quiet",
        ]
        subprocess.run(cmd, capture_output=True, check=False)
        if not final_16k_path.exists():
            final_16k_path = enhanced_path

        return {
            "status": "ok",
            "enhanced_audio_path": str(final_16k_path),
            "vocals_path": str(vocals_path),
        }

    def close(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.clearvoice is not None:
            del self.clearvoice
            self.clearvoice = None
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# AudioEnhanceWorker (high-level wrapper)
# ──────────────────────────────────────────────────────────────────────────────

class AudioEnhanceWorker:
    """High-level wrapper matching the Worker interface pattern."""

    def __init__(self, args):
        self.args = args
        self.runner = None
        self.static_status = None

        if getattr(args, "disable_audio_enhance", False):
            self.static_status = "disabled"
            return

        sep_ckpt = getattr(args, "audio_sep_ckpt", "")
        if not sep_ckpt or not Path(sep_ckpt).is_file():
            self.static_status = "not_configured"
            return

        self.runner = AudioEnhanceRunner(args)
        if not self.runner.is_available:
            self.static_status = "load_failed"
            self.runner = None

    def enhance(self, video_path, output_dir=None):
        """Enhance audio for a video segment.

        Returns:
            dict with status, enhanced_audio_path, vocals_path
        """
        if self.runner is None:
            return {
                "status": self.static_status or "not_initialized",
                "enhanced_audio_path": None,
                "vocals_path": None,
            }
        return self.runner.enhance(video_path, output_dir)

    def close(self):
        if self.runner is not None:
            self.runner.close()
