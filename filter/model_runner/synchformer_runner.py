"""Synchformer 音视频同步检测 Runner。"""
import signal
import subprocess
from pathlib import Path

from filter.filter_utils import (
    pushd,
    prepend_sys_path,
    clear_modules,
    safe_name,
    ensure_dir,
    has_audio_stream,
    video_meta,
)


# ── Signal-based timeout for Synchformer inference (Linux only) ─────────────────
class _SigTimeout(Exception):
    pass


def _sigalarm_handler(signum, frame):
    raise _SigTimeout()


def _timeout_call(fn, args, timeout_sec):
    """Run fn(*args) with SIGALRM timeout. Returns (ok, result_or_error_str)."""
    old_handler = signal.signal(signal.SIGALRM, _sigalarm_handler)
    signal.alarm(int(timeout_sec))
    try:
        result = fn(*args)
        signal.alarm(0)
        return True, result
    except _SigTimeout:
        return False, f"timeout after {timeout_sec}s"
    except Exception as e:
        signal.alarm(0)
        return False, f"{type(e).__name__}: {e}"
    finally:
        signal.signal(signal.SIGALRM, old_handler)


class SynchformerRunner:
    """加载 Synchformer 模型，预测音视频偏移量，判断是否同步。"""

    VFPS = 25        # 目标视频帧率
    AFPS = 16000     # 目标音频采样率
    IN_SIZE = 256    # 目标帧分辨率

    def __init__(self, args):
        import torch
        import torchvision
        from omegaconf import OmegaConf
        import av
        if not hasattr(av, "AVError"):
            av.AVError = av.error.InvalidDataError

        self.args = args
        self.torch = torch
        self.torchvision = torchvision
        self.root = Path(args.synchformer_root).resolve()
        self.cache_dir = Path(args.output_dir) / "synchformer_inputs"
        ensure_dir(self.cache_dir)

        # ── av-based readers: avoids torchvision.io.read_video hang with PyAV 16.x ──
        import av

        def read_video_av(path, start_sec=0.0, end_sec=None):
            container = av.open(str(path))
            video_stream = container.streams.video[0]
            video_stream.thread_type = "AUTO"
            fps = float(video_stream.average_rate)
            target_fps = args.synchformer_vfps if hasattr(args, 'synchformer_vfps') else 25.0
            start_pts = int(start_sec * float(video_stream.time_base))
            end_ts = None if end_sec is None else end_sec
            frames = []
            for frame in container.decode(video=0):
                pts_sec = float(frame.pts * video_stream.time_base)
                if pts_sec < start_sec:
                    continue
                if end_ts is not None and pts_sec >= end_ts:
                    break
                img = frame.to_ndarray(format="rgb24")
                if abs(fps - target_fps) > 0.1:
                    # rescale to target fps by dropping/duplicating frames
                    frames.append(torch.from_numpy(img))
                else:
                    frames.append(torch.from_numpy(img))
            container.close()
            if not frames:
                return torch.empty(0, 3, 256, 256, dtype=torch.uint8)
            tensor = torch.stack(frames)  # (T, H, W, 3)
            tensor = tensor.permute(0, 3, 1, 2)  # (T, 3, H, W)
            # Zero-pad video to 5 s (125 frames @ 25fps) for Synchformer
            min_frames = 5 * 25
            if tensor.shape[0] < min_frames:
                pad = torch.zeros(min_frames - tensor.shape[0], *tensor.shape[1:], dtype=tensor.dtype)
                tensor = torch.cat([tensor, pad], dim=0)
            return tensor

        def read_audio_av(path, start_sec=0.0, end_sec=None):
            container = av.open(str(path))
            audio_stream = container.streams.audio[0] if container.streams.audio else None
            if audio_stream is None:
                dur = end_sec - start_sec if end_sec else 5.0
                n = int(dur * 16000)
                return torch.zeros(n, dtype=torch.float32)
            audio_stream.thread_type = "AUTO"
            samples = []
            for frame in container.decode(audio=0):
                pts_sec = float(frame.pts * audio_stream.time_base)
                if end_sec is not None and pts_sec >= end_sec:
                    break
                nd = frame.to_ndarray()
                # nd is (channels, samples) e.g. (1, 1024); squeeze to (samples,)
                if nd.ndim == 2:
                    nd = nd[0]
                samples.append(torch.from_numpy(nd))
            container.close()
            if not samples:
                return torch.zeros(int((end_sec - start_sec) * 16000), dtype=torch.float32)
            audio = torch.cat(samples, dim=0)
            # Zero-pad to at least 5 s (80000 samples) for Synchformer
            min_len = 5 * 16000
            if audio.shape[0] < min_len:
                audio = torch.nn.functional.pad(audio, (0, min_len - audio.shape[0]))
            return audio

        def get_video_and_audio_av(path, get_meta=False, start_sec=0, end_sec=None):
            target_fps = args.synchformer_vfps if hasattr(args, 'synchformer_vfps') else 25.0
            rgb = read_video_av(path, start_sec, end_sec)
            audio = read_audio_av(path, start_sec, end_sec)
            if get_meta:
                meta = {
                    "video_fps": target_fps,
                    "audio_fps": 16000.0,
                }
                return rgb, audio, meta
            return rgb, audio

        self._orig_get_va = None

        # 加载模型配置和权重
        model_dir = self.root / "logs" / "sync_models" / args.synchformer_exp_name
        cfg_path = model_dir / f"cfg-{args.synchformer_exp_name}.yaml"
        ckpt_path = model_dir / f"{args.synchformer_exp_name}.pt"
        fallback_ckpt_path = model_dir / f"{args.synchformer_exp_name}.1.pt"

        if (not ckpt_path.is_file() or ckpt_path.stat().st_size == 0) and fallback_ckpt_path.is_file():
            ckpt_path = fallback_ckpt_path
        if not cfg_path.is_file() or not ckpt_path.is_file():
            raise FileNotFoundError(f"Synchformer weights missing: cfg={cfg_path}, ckpt={ckpt_path}")
        if ckpt_path.stat().st_size == 0:
            raise RuntimeError(f"Synchformer checkpoint is empty: {ckpt_path}")

        clear_modules("model", "dataset", "scripts", "utils")
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
            self.get_video_and_audio = get_video_and_audio_av
            self.cfg = cfg

            max_off_sec = cfg.data.max_off_sec
            num_cls = cfg.model.params.transformer.params.off_head_cfg.params.out_features
            self.grid = make_class_grid(-max_off_sec, max_off_sec, num_cls)

    def prepare_video(self, segment_path):
        """将视频标准化为 Synchformer 所需格式：25fps、256px、16kHz。"""
        segment_path = str(segment_path)

        # 用 ffprobe 获取视频宽高
        probe = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "csv=p=0", segment_path],
            capture_output=True, text=True, check=True,
        )
        w, h, fps_str = probe.stdout.strip().split(",")
        video_fps = eval(fps_str)  # e.g. "30/1" -> 30.0

        already_ok = (
            abs(video_fps - self.VFPS) < 0.01
            and min(int(w), int(h)) == self.IN_SIZE
        )
        audio_probe = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "a",
             "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", segment_path],
            capture_output=True, text=True,
        )
        has_audio = "audio" in audio_probe.stdout

        if already_ok and has_audio:
            return segment_path, {"status": "already_compatible", "path": segment_path}

        out_path = (
            self.cache_dir
            / f"{safe_name(Path(segment_path).stem)}_{self.VFPS}fps_{self.IN_SIZE}side_{self.AFPS}hz.mp4"
        )
        if out_path.is_file() and has_audio_stream(out_path):
            return str(out_path), {"status": "cached", "path": str(out_path)}

        vf = (
            f"fps={self.VFPS},scale=iw*{self.IN_SIZE}/min(iw\\,ih):ih*{self.IN_SIZE}/min(iw\\,ih),"
            "crop=trunc(iw/2)*2:trunc(ih/2)*2"
        )

        if has_audio:
            audio_input = []
            audio_maps = ["-map", "0:a"]
        else:
            audio_input = ["-f", "lavfi", "-i", f"anullsrc=channel_layout=mono:sample_rate={self.AFPS}"]
            audio_maps = ["-map", "1:a"]

        # 获取视频时长，用于强制音频 Pad 到同样长度（不用 -shortest，避免音频不够长）
        duration_probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "format=duration", "-of", "csv=p=0", segment_path],
            capture_output=True, text=True, check=True,
        )
        video_dur = float(duration_probe.stdout.strip())

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", segment_path, *audio_input,
            "-vf", vf,
            "-map", "0:v", *audio_maps,
            "-ar", str(self.AFPS), "-ac", "1", "-c:a", "aac",
            "-t", f"{video_dur:.3f}",
            "-movflags", "+faststart",
            str(out_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False, timeout=60)
        if proc.returncode != 0 or not out_path.is_file():
            raise RuntimeError(proc.stderr[-2000:])
        return str(out_path), {"status": "created", "path": str(out_path)}

    def score(self, segment_path):
        """执行同步检测推理。先用 prepare_video 标准化，再用 signal alarm 做超时保护。"""
        prepared_path, prep = self.prepare_video(segment_path)

        def _inference():
            rgb, audio, raw_meta = self.get_video_and_audio(prepared_path, get_meta=True)
            meta = {
                "video": {"fps": [float(raw_meta.get("video_fps", self.VFPS))]},
                "audio": {"framerate": [float(raw_meta.get("audio_fps", self.AFPS))]},
            }
            item = dict(
                video=rgb, audio=audio, meta=meta, path=prepared_path, split="test",
                targets={"v_start_i_sec": 0.0, "offset_sec": 0.0},
            )
            item = self.transform(item)
            batch = self.torch.utils.data.default_collate([item])
            aud, vid, targets = self.prepare_inputs(batch, self.device)
            device_type = "cuda" if str(self.device).startswith("cuda") else "cpu"
            with self.torch.set_grad_enabled(False):
                with self.torch.autocast(
                    device_type,
                    enabled=bool(device_type == "cuda" and self.cfg.training.use_half_precision),
                ):
                    _, logits = self.model(vid, aud)
            off_logits = logits[0]
            off_probs = self.torch.softmax(off_logits, dim=-1)
            k = min(off_probs.shape[-1], 5)
            _, topk_preds = self.torch.topk(off_logits, k)
            top_offsets = [
                {
                    "probability": float(off_probs[pred_i].cpu().item()),
                    "offset_sec": float(self.grid[pred_i]),
                }
                for pred in topk_preds
                for pred_i in [int(pred.item())]
            ]
            best = top_offsets[0]
            return {
                "status": "ok",
                "predicted_offset_sec": best["offset_sec"],
                "sync_confidence": best["probability"],
                "top_offsets": top_offsets,
                "synchformer_input": prep,
            }

        ok, val = _timeout_call(_inference, (), timeout_sec=120)
        if not ok:
            raise RuntimeError(f"Synchformer inference failed: {val}")
        return val


class SynchformerWorker:
    """Synchformer 的封装，带时长检查和错误处理。"""

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

        result = self.runner.score(segment_path)

        if result.get("status") == "ok":
            result["is_synced"] = bool(
                abs(result["predicted_offset_sec"]) <= self.args.max_sync_offset_sec
                and result["sync_confidence"] >= self.args.min_sync_confidence
            )
            result["synchronizability_proxy_score"] = result["sync_confidence"]
            result["is_synchronizable"] = bool(result["sync_confidence"] >= self.args.min_sync_confidence)
            result["synchronizability_note"] = (
                "uses Synchformer offset-class probability as a proxy; "
                "separate synchronizability checkpoint is not invoked"
            )
        else:
            result["is_synced"] = False
        return result

    def close(self):
        pass
