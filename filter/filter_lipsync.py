#!/usr/bin/env python3
"""
OmniShotCut 后处理：SyncNet 音画同步筛选

流程：InsightFace 人脸跟踪 → SyncNet 音画同步检测 → 综合决策

SyncNet evaluates temporal audio-visual synchronization by computing
offset and confidence between lip movements and speech audio.
Decision: offset in [-6,6] AND confidence >= threshold → speaking.

支持多说话人模式（PyAnnote）：
  - 使用 PyAnnote 进行说话人分割（diarization）
  - 将每个人脸轨迹匹配到对应的说话人时间段
  - 对每个说话人分别运行 TalkNCE 和 SyncNet
  - 任一说话人满足条件即视为合格片段
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch
torch.backends.cuda.matmul.allow_tf32
torch.backends.cudnn.allow_tf32 = True

_FILTER_ROOT = Path(__file__).resolve().parent.parent
if str(_FILTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FILTER_ROOT))

from filter.filter_utils import (
    ensure_dir,
    safe_name,
    link_or_copy,
    attach_speaker_scores,
    qualify_segment,
)


def collect_candidate_scenes(scenes_csv_path):
    """从 scenes.csv 读取，按 video_name 分组，返回 items 列表。"""
    grouped = {}
    with open(scenes_csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            video_key = row.get("video_name") or row.get("video_path", "")
            if video_key not in grouped:
                grouped[video_key] = {
                    "summary_row": {
                        "video_path": row.get("video_path", ""),
                        "video_name": row.get("video_name", ""),
                    },
                    "candidate_scenes": [],
                }
            grouped[video_key]["candidate_scenes"].append({
                "scene_idx": int(row["scene_idx"]) if row.get("scene_idx") else 0,
                "start_frame": int(row["start_frame"]) if row.get("start_frame") else 0,
                "end_frame": int(row["end_frame"]) if row.get("end_frame") else 0,
                "duration_frames": int(row["end_frame"]) - int(row["start_frame"])
                if row.get("start_frame") and row.get("end_frame") else 0,
                "segment_video_path": row.get("segment_video_path", ""),
                "fps": float(row["fps"]) if row.get("fps") else 0.0,
            })
    return list(grouped.values())


from filter.model_runner import (
    InsightFaceTracker,
    TalkNCEWorker,
    SyncNetWorker,
    AudioEnhanceWorker,
    SemanticSyncWorker,
)
from filter.model_runner.pyannote_diarization import PyannoteDiarizationWorker
from filter.model_runner.face_speaker_match import match_faces_to_speakers


# ─── Single-segment processing ─────────────────────────────────────────────────
def process_segment(
    args,
    video_item,
    scene,
    tracker,
    pyannote_worker,
    talknce_worker,
    syncnet_worker,
    qualified_dir,
    audio_enhance_worker=None,
    semantic_sync_worker=None,
):
    segment_path = scene.get("segment_video_path", "")

    if not segment_path or not Path(segment_path).is_file():
        return _make_row(
            video_item, scene, segment_path,
            qualified_dir, args,
            is_qualified=False,
            reject_reason="missing_segment",
        )

    processing_video_path = segment_path

    # ── Step 0: Audio Enhancement (separation + denoising) ───────────────────
    enhanced_audio_path = None
    if audio_enhance_worker is not None:
        enhance_output_dir = Path(args.output_dir) / "enhanced_audio"
        enhance_result = audio_enhance_worker.enhance(
            processing_video_path, output_dir=enhance_output_dir
        )
        if enhance_result.get("status") == "ok":
            enhanced_audio_path = enhance_result["enhanced_audio_path"]
            print(f"  [AudioEnhance] Enhanced audio: {enhanced_audio_path}")
        else:
            print(f"  [AudioEnhance] status={enhance_result.get('status')}")

    # ── Step 1: PyAnnote Diarization (uses enhanced audio if available) ───────
    diarization = {"status": "disabled", "speakers": []}
    # If enhanced audio is available, use it for better diarization
    diarize_input = enhanced_audio_path if enhanced_audio_path else processing_video_path
    diarization = pyannote_worker.diarize(diarize_input)

    # ── Step 2: InsightFace Tracking ─────────────────────────────────────────
    face_result = tracker.extract_tracks(
        processing_video_path,
        args.face_iou_threshold,
        args.face_max_missing,
        args.min_face_track_frames,
    )

    # ── Step 3: Match faces to speakers ──────────────────────────────────────
    use_multi_speaker = (
        not args.disable_pyannote
        and diarization.get("status") == "ok"
        and len(diarization.get("speakers", [])) > 0
    )


    num_speakers = 0
    if use_multi_speaker:
        fps = float(face_result["video_meta"]["fps"])
        speaker_matches = match_faces_to_speakers(
            face_result["tracks"], diarization["speakers"], fps,
        )
        num_speakers = speaker_matches["num_unique_speakers"]

        min_speakers = getattr(args, "min_speakers", 0)
        if num_speakers < min_speakers:
            use_multi_speaker = False

    # ── SyncNet evaluation ────────────────────────────────────────────────────
    syncnet_result = {"status": "skipped", "tracks": [], "primary_track_id": None}
    if not use_multi_speaker:
        syncnet_result = syncnet_worker.score(processing_video_path, face_result)

    talknce_result = {"status": "disabled", "tracks": []}
    if not args.disable_talknce:
        talknce_result = talknce_worker.score(face_result, processing_video_path)

    # 使用多说话人双层循环
    if use_multi_speaker:
        speaker_segments = diarization["speakers"]
        face_tracks = face_result["tracks"]

        score_matrix: dict[int, list] = {tr["track_id"]: [] for tr in face_tracks}

        for tr in face_tracks:
            tr_id = tr["track_id"]
            for seg in speaker_segments:
                spk_id = seg["speaker_id"]
                spk_start = seg["start_sec"]
                spk_end = seg["end_sec"]

                sync_res = syncnet_worker.score(
                    processing_video_path, face_result,
                    track_id=tr_id,
                    speaker_window=(spk_start, spk_end),
                )

                sync_mean = None
                if sync_res.get("status") == "ok":
                    for t in sync_res.get("tracks", []):
                        sync_mean = t.get("mean_score")
                        break

                score_matrix[tr_id].append({
                    "spk_id": spk_id,
                    "spk_start": spk_start,
                    "spk_end": spk_end,
                    "sync_mean": sync_mean,
                    "sync_status": sync_res.get("status", "unknown"),
                })

        # Per-track: pick the best speaker window by SyncNet score
        track_best = {}
        for tr in face_tracks:
            tr_id = tr["track_id"]
            entries = [e for e in score_matrix[tr_id] if e["sync_status"] == "ok" and e["sync_mean"] is not None]
            if not entries:
                track_best[tr_id] = {"spk_id": None, "spk_start": None, "spk_end": None,
                                     "sync_mean": None, "sync_status": "no_valid_window"}
            else:
                best = max(entries, key=lambda e: e["sync_mean"])
                track_best[tr_id] = best


        # Select primary face track (highest SyncNet confidence)
        valid = [(tr_id, info) for tr_id, info in track_best.items()
                 if info["sync_mean"] is not None]
        if valid:
            primary_tr_id, primary_info = max(valid, key=lambda x: x[1]["sync_mean"])
        else:
            primary_tr_id, primary_info = None, None

        primary_tr = next((t for t in face_tracks if t["track_id"] == primary_tr_id), None)

        # ── Semantic sync evaluation (multi-speaker path) ─────────────────────
        semantic_score = None
        semantic_status = "skipped"
        sem_result = {"is_semantically_synced": True}
        if semantic_sync_worker is not None:
            sem_result = semantic_sync_worker.score(
                processing_video_path, enhanced_audio_path
            )
            semantic_score = sem_result.get("semantic_score")
            semantic_status = sem_result.get("status", "unknown")
            print(f"  [SemanticSync] score={semantic_score}, status={semantic_status}")

        threshold = getattr(args, "syncnet_min_confidence", 1.0)
        if primary_info and primary_info["sync_mean"] >= threshold:
            # Check semantic sync gate
            reject_reasons = []
            if (
                semantic_status == "ok"
                and semantic_score is not None
                and not sem_result.get("is_semantically_synced", True)
            ):
                reject_reasons.append("semantic_not_synced")

            is_qual = len(reject_reasons) == 0

            qualified_video_path = ""
            if is_qual:
                out_name = f"{safe_name(Path(processing_video_path).stem)}.mp4"
                qualified_video_path = link_or_copy(
                    processing_video_path,
                    Path(qualified_dir) / out_name,
                    args.qualified_copy_mode,
                )

            talknce_spoke = False
            if talknce_result.get("status") == "ok":
                for t in talknce_result.get("tracks", []):
                    if t["track_id"] == primary_tr_id and t.get("is_active_speaker"):
                        talknce_spoke = True
                        break

            return _make_row(
                video_item, scene, segment_path,
                qualified_dir, args,
                qualified_video_path=qualified_video_path,
                is_qualified=is_qual,
                reject_reason=",".join(reject_reasons),
            )

        else:
            # No face track passed SyncNet threshold (or track_best is empty)
            best_mean = primary_info["sync_mean"] if primary_info else None
            reject = ["target_not_speaking"]
            if best_mean is None and primary_info is not None:
                reject.append(primary_info["sync_status"])
            elif primary_info is None:
                reject.append("no_valid_track")

            return _make_row(
                video_item, scene, segment_path,
                qualified_dir, args,
                qualified_video_path="",
                is_qualified=False,
                reject_reason=",".join(reject),
            )

    # ── Single-speaker evaluation path ───────────────────────────────────────
    from filter.model_runner.insightface_runner import compact_tracks

    face_compact = {
        "video_meta": face_result["video_meta"],
        "num_face_tracks": face_result["num_face_tracks"],
        "primary_track_id": face_result["primary_track_id"],
        "tracks": attach_speaker_scores(compact_tracks(face_result), talknce_result),
    }

    primary_id = face_result.get("primary_track_id")
    talknce_spoke = False
    if talknce_result.get("status") == "ok":
        for t in talknce_result.get("tracks", []):
            if t["track_id"] == primary_id and t.get("is_active_speaker"):
                talknce_spoke = True
                break

    syncnet_spoke = syncnet_result.get("primary_is_speaking", False)
    syncnet_mean = syncnet_result.get("primary_mean_score")
    syncnet_status = syncnet_result.get("status", "unknown")
    primary_track = next(
        (t for t in face_compact["tracks"] if t["track_id"] == primary_id), None
    )

    # ── Semantic sync evaluation (single-speaker path) ────────────────────────
    semantic_score = None
    semantic_status = "skipped"
    sem_result = {"is_semantically_synced": True}
    if semantic_sync_worker is not None:
        sem_result = semantic_sync_worker.score(
            processing_video_path, enhanced_audio_path
        )
        semantic_score = sem_result.get("semantic_score")
        semantic_status = sem_result.get("status", "unknown")
        print(f"  [SemanticSync] score={semantic_score}, status={semantic_status}")

    # SyncNet is the primary gate: require both face + speaking + semantic sync
    reasons = []
    if face_result.get("num_face_tracks", 0) == 0:
        reasons.append("no_face")
    if syncnet_status == "ok" and not syncnet_spoke:
        reasons.append("target_not_speaking")
    if (
        semantic_status == "ok"
        and semantic_score is not None
        and not sem_result.get("is_semantically_synced", True)
    ):
        reasons.append("semantic_not_synced")

    is_qualified = len(reasons) == 0

    row = _make_row(
        video_item, scene, segment_path,
        qualified_dir, args,
        qualified_video_path="",
        is_qualified=is_qualified,
        reject_reason=",".join(reasons),
    )

    if is_qualified:
        out_name = f"{safe_name(Path(processing_video_path).stem)}.mp4"
        row["qualified_video_path"] = link_or_copy(
            processing_video_path,
            Path(qualified_dir) / out_name,
            args.qualified_copy_mode,
        )

    return row


def _make_row(
    video_item, scene, segment_path,
    qualified_dir, args,
    qualified_video_path="",
    is_qualified=False,
    reject_reason="",
):
    """Construct a result row dict with only essential fields."""
    src_row = video_item.get("summary_row", {})
    row = {
        "video_name": src_row.get("video_name")
        or Path(src_row.get("video_path", "video")).name,
        "scene_idx": scene.get("scene_idx", 0),
        "segment_video_path": segment_path,
        "qualified_video_path": qualified_video_path,
        "is_lipsync_qualified": is_qualified,
        "reject_reason": reject_reason,
    }
    return row


# ─── Single-video processing ───────────────────────────────────────────────────

def process_video_item(
    args, item, tracker, pyannote_worker, talknce_worker, syncnet_worker, qualified_dir,
    audio_enhance_worker=None, semantic_sync_worker=None,
):
    src_row = item["summary_row"]
    video_name = src_row.get("video_name") or Path(src_row.get("video_path", "video")).name

    rows = []
    for scene in item["candidate_scenes"]:
        print(f"[{time.strftime('%H:%M:%S')}] syncnet {video_name} scene={scene.get('scene_idx')}", flush=True)
        row = process_segment(
            args, item, scene,
            tracker, pyannote_worker, talknce_worker, syncnet_worker, qualified_dir,
            audio_enhance_worker=audio_enhance_worker,
            semantic_sync_worker=semantic_sync_worker,
        )
        rows.append(row)
    return rows


# ─── Worker lifecycle ──────────────────────────────────────────────────────────

def create_lipsync_workers(args):
    tracker = InsightFaceTracker(
        args.device,
        args.insightface_model,
        args.insightface_det_size,
        112,
        args.insightface_root,
        False,
    )
    pyannote_worker = PyannoteDiarizationWorker(args)
    talknce_worker = TalkNCEWorker(args)
    syncnet_worker = SyncNetWorker(args)
    audio_enhance_worker = AudioEnhanceWorker(args)
    semantic_sync_worker = SemanticSyncWorker(args)
    return tracker, pyannote_worker, talknce_worker, syncnet_worker, audio_enhance_worker, semantic_sync_worker


def close_lipsync_workers(
    pyannote_worker, talknce_worker, syncnet_worker,
    audio_enhance_worker=None, semantic_sync_worker=None,
):
    if pyannote_worker is not None:
        pyannote_worker.close()
    if talknce_worker is not None:
        talknce_worker.close()
    if syncnet_worker is not None:
        syncnet_worker.close()
    if audio_enhance_worker is not None:
        audio_enhance_worker.close()
    if semantic_sync_worker is not None:
        semantic_sync_worker.close()


# ─── CSV field names ───────────────────────────────────────────────────────────

LIPSYNC_FIELDNAMES = [
    "video_name", "scene_idx", "segment_video_path", "qualified_video_path",
    "is_lipsync_qualified", "reject_reason",
]


# ─── Sequential pipeline ───────────────────────────────────────────────────────

def run_pipeline(args):
    ensure_dir(args.output_dir)
    qualified_dir = Path(args.output_dir) / "qualified_videos"
    ensure_dir(qualified_dir)

    all_csv = Path(args.output_dir) / "lipsync_all.csv"
    qual_csv = Path(args.output_dir) / "lipsync_qualified.csv"

    items = collect_candidate_scenes(args.scenes_csv)
    (
        tracker, pyannote_worker, talknce_worker, syncnet_worker,
        audio_enhance_worker, semantic_sync_worker,
    ) = create_lipsync_workers(args)

    try:
        with open(all_csv, "w", newline="", encoding="utf-8") as all_f, \
             open(qual_csv, "w", newline="", encoding="utf-8") as qual_f:
            all_writer = csv.DictWriter(all_f, fieldnames=LIPSYNC_FIELDNAMES, extrasaction='ignore')
            all_writer.writeheader()
            qual_writer = csv.DictWriter(qual_f, fieldnames=LIPSYNC_FIELDNAMES, extrasaction='ignore')
            qual_writer.writeheader()
            all_f.flush(); qual_f.flush()

            for item in items:
                for row in process_video_item(
                    args, item, tracker, pyannote_worker,
                    talknce_worker, syncnet_worker, qualified_dir,
                    audio_enhance_worker=audio_enhance_worker,
                    semantic_sync_worker=semantic_sync_worker,
                ):
                    all_writer.writerow(row)
                    all_f.flush()
                    if row.get("is_lipsync_qualified"):
                        qual_writer.writerow(row)
                        qual_f.flush()
    finally:
        close_lipsync_workers(
            pyannote_worker, talknce_worker, syncnet_worker,
            audio_enhance_worker, semantic_sync_worker,
        )

    print(f"完成: {all_csv} + {qual_csv}")


# ─── Ray distributed pipeline ───────────────────────────────────────────────────

def make_ray_lipsync_worker(ray_args):
    class RayLipSyncWorker:
        def __init__(self):
            import atexit
            self.args = argparse.Namespace(**vars(ray_args))
            self.closed = False
            ensure_dir(self.args.output_dir)
            self.qualified_dir = Path(self.args.output_dir) / "qualified_videos"
            ensure_dir(self.qualified_dir)
            (
                self.tracker, self.pyannote_worker,
                self.talknce_worker, self.syncnet_worker,
                self.audio_enhance_worker, self.semantic_sync_worker,
            ) = create_lipsync_workers(self.args)
            atexit.register(self.close)

        def __call__(self, item):
            item_index = item.get("_item_index")
            rows = process_video_item(
                self.args, item,
                self.tracker, self.pyannote_worker,
                self.talknce_worker, self.syncnet_worker,
                self.qualified_dir,
                audio_enhance_worker=self.audio_enhance_worker,
                semantic_sync_worker=self.semantic_sync_worker,
            )
            return {"item_index": item_index, "qualified_rows": rows}

        def close(self):
            if not self.closed:
                self.closed = True
                close_lipsync_workers(
                    self.pyannote_worker,
                    self.talknce_worker,
                    self.syncnet_worker,
                    self.audio_enhance_worker,
                    self.semantic_sync_worker,
                )

        def __del__(self):
            self.close()

    return RayLipSyncWorker


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TALKNCE_ROOT = PROJECT_ROOT / "third_part" / "TalkNCE"
SYNCNET_CKPT_DEFAULT = "/mnt/workspace/lihaoran/data_process/data-process-old/111/a6000/ckps/syncnet_v2.model"
DEFAULT_INSIGHTFACE_ROOT = Path("/mnt/workspace/lihaoran/ckps/insightface")


def run_ray_pipeline(args):
    ensure_dir(args.output_dir)
    qualified_dir = Path(args.output_dir) / "qualified_videos"
    ensure_dir(qualified_dir)

    all_csv = Path(args.output_dir) / "lipsync_all.csv"
    qual_csv = Path(args.output_dir) / "lipsync_qualified.csv"

    items = collect_candidate_scenes(args.scenes_csv)
    if not items:
        print(f"完成: {all_csv} + {qual_csv}")
        return

    import ray
    ray_init_kwargs = {
        "runtime_env": {"env_vars": {"PYTHONPATH": str(PROJECT_ROOT)}}
    }
    if args.ray_address:
        ray_init_kwargs["address"] = args.ray_address
    ray.init(**ray_init_kwargs)

    ray_items = [{**item, "_item_index": idx} for idx, item in enumerate(items)]
    samples = ray.data.from_items(ray_items)
    map_kwargs = {"concurrency": args.concurrency, "num_gpus": args.ray_num_gpus}
    if args.ray_num_cpus > 0:
        map_kwargs["num_cpus"] = args.ray_num_cpus

    predictions = samples.map(make_ray_lipsync_worker(args), **map_kwargs)
    worker_rows = sorted(predictions.take_all(), key=lambda x: x.get("item_index", 0))

    with open(all_csv, "w", newline="", encoding="utf-8") as all_f, \
         open(qual_csv, "w", newline="", encoding="utf-8") as qual_f:
        all_writer = csv.DictWriter(all_f, fieldnames=LIPSYNC_FIELDNAMES)
        all_writer.writeheader()
        qual_writer = csv.DictWriter(qual_f, fieldnames=LIPSYNC_FIELDNAMES)
        qual_writer.writeheader()

        for x in worker_rows:
            for row in x.get("qualified_rows", []):
                all_writer.writerow(row)
                if row.get("is_lipsync_qualified"):
                    qual_writer.writerow(row)

    print(f"Ray 完成: {all_csv} + {qual_csv}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="OmniShotCut 后处理：InsightFace + SyncNet 唇形同步筛选"
    )
    # Paths
    parser.add_argument(
        "--scenes_csv",
        default="/mnt/workspace/lihaoran/data_process/test/result/0603/filter_cutShot/1/scenes.csv",
        help="cutShot 输出的 scenes.csv 路径",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/workspace/lihaoran/data_process/test/result/0611/filter_lipsync/1",
    )
    parser.add_argument("--device", default="cuda:0")

    # InsightFace
    parser.add_argument("--insightface_model", default="buffalo_l")
    parser.add_argument(
        "--insightface_root", default=str(DEFAULT_INSIGHTFACE_ROOT)
    )
    parser.add_argument(
        "--insightface_det_size", type=int, default=640,
        help="InsightFace detection resolution (must be divisible by 32, min 128)",
    )
    parser.add_argument("--face_iou_threshold", type=float, default=0.5)
    parser.add_argument("--face_max_missing", type=int, default=5)
    parser.add_argument("--min_face_track_frames", type=int, default=20)

    # TalkNCE
    parser.add_argument("--talknce_root", default=str(TALKNCE_ROOT))
    parser.add_argument(
        "--talknce_config_path", default=str(TALKNCE_ROOT / "configs" / "test.yaml")
    )
    parser.add_argument(
        "--talknce_ckpt",
        default="/mnt/workspace/lihaoran/ckps/lipsync/talknce_ava_pretrained.model",
    )
    parser.add_argument("--active_speaker_threshold", type=float, default=0.6)
    parser.add_argument("--active_speaker_min_ratio", type=float, default=0.2)
    parser.add_argument("--min_talknce_frames", type=int, default=35)
    parser.add_argument("--max_talknce_frames", type=int, default=200)
    parser.add_argument("--disable_talknce", action="store_true")

    # SyncNet (lip-sync evaluation)
    parser.add_argument(
        "--syncnet_ckpt",
        default="/mnt/workspace/lihaoran/data_process/data-process-old/111/a6000/ckps/syncnet_v2.model",
        help="SyncNet model checkpoint 路径",
    )
    parser.add_argument(
        "--syncnet_vshift", type=int, default=10,
        help="SyncNet 时序偏移窗口大小",
    )
    parser.add_argument(
        "--syncnet_batch_size", type=int, default=20,
        help="SyncNet 特征提取 batch size",
    )
    parser.add_argument(
        "--syncnet_min_confidence", type=float, default=1.0,
        help="SyncNet 最低置信度阈值 (conf >= threshold = speaking)",
    )
    parser.add_argument(
        "--syncnet_min_offset", type=int, default=-6,
        help="SyncNet 允许的最小音视频偏移",
    )
    parser.add_argument(
        "--syncnet_max_offset", type=int, default=6,
        help="SyncNet 允许的最大音视频偏移",
    )
    parser.add_argument(
        "--syncnet_crop_scale", type=float, default=0.40,
        help="SyncNet 人脸裁剪 padding 缩放",
    )
    parser.add_argument(
        "--disable_syncnet", action="store_true",
        help="禁用 SyncNet（使用 TalkNCE only）",
    )

    # PyAnnote (multi-speaker)
    parser.add_argument(
        "--disable_pyannote", action="store_true",
        help="禁用 PyAnnote 说话人分割，强制使用单说话人模式",
    )
    parser.add_argument(
        "--pyannote_model_path",
        default="/mnt/workspace/lihaoran/ckps/pyannote/config.yaml",
        help="PyAnnote Pipeline config.yaml 路径（注意：是 config.yaml，不是 pipeline.yaml）",
    )
    import os as _os
    parser.add_argument(
        "--pyannote_hf_token", default=_os.environ.get("HF_TOKEN", None),
        help="HuggingFace token，用于加载 PyAnnote Pipeline",
    )
    parser.add_argument(
        "--min_speakers", type=int, default=0,
        help="PyAnnote 检测到的最少说话人数才启用多说话人模式",
    )

    # Audio Enhancement (MelBandRoformer + ClearVoice)
    parser.add_argument(
        "--disable_audio_enhance", action="store_true",
        help="禁用音频增强（声源分离+语音增强）",
    )
    parser.add_argument(
        "--audio_sep_root",
        default="/mnt/workspace/lihaoran/ckps/Music-Source-Separation-Training",
        help="Music-Source-Separation-Training 代码根目录",
    )
    parser.add_argument(
        "--audio_sep_config",
        default="/mnt/workspace/lihaoran/ckps/Music-Source-Separation-Training/ckpts/config_vocals_mel_band_roformer_kj.yaml",
        help="MelBandRoformer config yaml 路径",
    )
    parser.add_argument(
        "--audio_sep_ckpt",
        default="/mnt/workspace/lihaoran/ckps/Music-Source-Separation-Training/ckpts/MelBandRoformer.ckpt",
        help="MelBandRoformer checkpoint 路径",
    )
    parser.add_argument(
        "--audio_sep_tta", action="store_true",
        help="启用 TTA (Test-Time Augmentation) 提高分离质量",
    )

    # Semantic Sync (ImageBind)
    parser.add_argument(
        "--disable_semantic_sync", action="store_true",
        help="禁用 ImageBind 语义同步检测",
    )
    parser.add_argument(
        "--semantic_sync_threshold", type=float, default=0.15,
        help="ImageBind 语义相似度阈值（cosine similarity > threshold = 语义同步）",
    )
    parser.add_argument(
        "--semantic_sync_max_frames", type=int, default=5,
        help="ImageBind 取样视频帧数（用于计算视觉 embedding）",
    )

    # Decision
    parser.add_argument(
        "--allow_non_target_speaker", action="store_true",
        help="允许目标说话人存在时仍有其他说话人出现（仅单说话人模式）",
    )

    # Output
    parser.add_argument(
        "--qualified_copy_mode",
        choices=["symlink", "copy", "none"],
        default="symlink",
    )
    parser.add_argument("--use_source_audio_segments", action="store_true", default=True)

    # Ray
    parser.add_argument("--ray", action="store_true", help="启用 Ray 分布式模式")
    parser.add_argument("--ray_address", default="auto")
    parser.add_argument("--ray_num_gpus", type=float, default=1.0)
    parser.add_argument("--ray_num_cpus", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.ray:
        run_ray_pipeline(args)
    else:
        run_pipeline(args)
