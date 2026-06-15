"""TalkNCE 说话人判别 Runner。"""
import argparse
import contextlib
import cv2
import numpy as np
from pathlib import Path

from filter.filter_utils import (
    prepend_sys_path,
    pushd,
    block_torch_hub_downloads,
    ns,
    extract_audio_16k,
)


def dense_gray_crops_range_for_talknce(track, start_frame, num_frames):
    """从 face track 中提取指定帧范围的人脸灰度图序列，供 TalkNCE 使用。"""
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


class TalkNCERunner:
    """加载 TalkNCE 模型，执行音频-视觉联合推理，输出说话人概率。"""

    def __init__(self, args):
        import torch
        import torch.nn.functional as F
        import yaml

        self.args = args
        self.torch = torch
        self.F = F
        self.device = torch.device(
            args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
        )

        root = Path(args.talknce_root).resolve()
        ckpt = Path(args.talknce_ckpt)
        if not ckpt.is_file():
            raise FileNotFoundError(f"TalkNCE checkpoint not found: {ckpt}")

        with prepend_sys_path(root), pushd(root):
            from loss_multi import lossAV
            from model.loconet_encoder import locoencoder
            from torchvggish import vggish_input

            cfg = ns(yaml.safe_load(open(args.talknce_config_path, "r", encoding="utf-8")))
            download_ctx =  block_torch_hub_downloads()
            with download_ctx:
                self.model = locoencoder(cfg).to(self.device).eval()
                self.model.audioEncoder.to(self.device)
                self.model.audioEncoder.device = self.device
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
            raise RuntimeError(
                f"bad TalkNCE checkpoint: model_keys={len(model_state)}, classifier_keys={len(cls_state)}"
            )
        self.model.load_state_dict(model_state, strict=False)
        self.classifier.load_state_dict(cls_state, strict=False)

    def score(self, face_result, segment_path):
        """对给定片段执行说话人判别。"""
        if not face_result["tracks"]:
            return {"status": "no_face", "tracks": []}

        audio = extract_audio_16k(segment_path)
        fps = float(face_result["video_meta"]["fps"])
        tracks = list(face_result["tracks"])
        required_speakers = int(getattr(self.model.cfg.MODEL, "NUM_SPEAKERS", len(tracks)))
        tracks = tracks[:required_speakers]
        if len(tracks) < required_speakers:
            tracks = tracks + [None] * (required_speakers - len(tracks))

        valid_tracks = [t for t in tracks if t is not None]
        start_frame = min(int(t["start_frame"]) for t in valid_tracks)
        end_frame = max(int(t["end_frame"]) for t in valid_tracks)
        num_frames = end_frame - start_frame + 1

        if self.args.max_talknce_frames > 0:
            num_frames = min(num_frames, self.args.max_talknce_frames)
        if num_frames < self.args.min_talknce_frames:
            return {
                "status": "ok",
                "tracks": [
                    {"track_id": int(t["track_id"]), "status": "too_short", "num_frames": num_frames, "is_active_speaker": False}
                    for t in valid_tracks
                ],
            }

        # 构建视觉输入: num_speakers x num_frames x 112 x 112 (灰度)
        visual_rows, masks, track_ids = [], [], []
        for track in tracks:
            if track is None:
                visual_rows.append(np.zeros((num_frames, 112, 112), dtype=np.uint8))
                masks.append(np.zeros((num_frames,), dtype=bool))
                track_ids.append(None)
            else:
                crops, mask = dense_gray_crops_range_for_talknce(track, start_frame, num_frames)
                visual_rows.append(crops)
                masks.append(mask)
                track_ids.append(int(track["track_id"]))

        # 音频切片
        start_sample = max(0, int(round(start_frame / fps * 16000)))
        end_sample = min(len(audio), int(round((start_frame + num_frames) / fps * 16000)))
        audio_slice = audio[start_sample:end_sample]
        if audio_slice.size == 0:
            return {
                "status": "ok",
                "tracks": [
                    {"track_id": int(t["track_id"]), "status": "no_audio", "num_frames": num_frames, "is_active_speaker": False}
                    for t in valid_tracks
                ],
            }

        audio_feat_np = self.vggish_input.waveform_to_examples(audio_slice, 16000, num_frames, fps, return_tensor=False)
        audio_feat = self.torch.FloatTensor(audio_feat_np)[None, None, :, :].to(self.device)
        visual_feat = self.torch.FloatTensor(np.asarray(visual_rows, dtype=np.float32)).to(self.device)

        # 前向传播
        with self.torch.no_grad():
            audio_embed = self.model.forward_audio_frontend(audio_feat)
            visual_embed = self.model.forward_visual_frontend(visual_feat)
            audio_embed = audio_embed.repeat(required_speakers, 1, 1)
            audio_embed, visual_embed = self.model.forward_cross_attention(audio_embed, visual_embed)
            outs = self.model.forward_audio_visual_backend(audio_embed, visual_embed, b=1, s=required_speakers)
            probs = self.F.softmax(self.classifier.FC(outs.reshape(-1, 256)), dim=-1)[:, 1].view(required_speakers, num_frames)

        # 解析结果
        results = []
        for idx, track_id in enumerate(track_ids):
            if track_id is None:
                continue
            mask = self.torch.BoolTensor(masks[idx]).to(self.device)
            valid_probs = probs[idx][mask]
            if valid_probs.numel() == 0:
                results.append({
                    "track_id": int(track_id), "status": "too_short",
                    "num_frames": int(mask.sum().cpu().item()), "is_active_speaker": False,
                })
                continue
            mean_score = float(valid_probs.mean().cpu().item())
            max_score = float(valid_probs.max().cpu().item())
            active_ratio = float((valid_probs >= self.args.active_speaker_threshold).float().mean().cpu().item())
            results.append({
                "track_id": int(track_id),
                "status": "ok",
                "num_frames": int(mask.sum().cpu().item()),
                "active_speaker_score": mean_score,
                "active_speaker_max_score": max_score,
                "active_speaker_ratio": active_ratio,
                "is_active_speaker": bool(
                    mean_score >= self.args.active_speaker_threshold
                    or active_ratio >= self.args.active_speaker_min_ratio
                ),
            })
        return {"status": "ok", "tracks": results}


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
        runner_args.device = args.device
        self.runner = TalkNCERunner(runner_args)

    def score(self, face_result, segment_path, work_dir=None):
        if self.runner is None:
            return {"status": self.static_status, "tracks": []}
        if not face_result["tracks"]:
            return {"status": "no_face", "tracks": []}
        result = self.runner.score(face_result, segment_path)
        if "tracks" not in result:
            result["tracks"] = []
        return result

    def close(self):
        pass
