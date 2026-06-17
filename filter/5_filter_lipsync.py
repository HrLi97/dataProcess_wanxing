#!/usr/bin/env python3
import argparse
import csv
import fcntl
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
    unique_output_name,
    link_or_copy,
    attach_speaker_scores,
    qualify_segment,
    project_root,
    ckps_root,
    env_or,
    worker_syspath_setup,
)


LIPSYNC_FIELDS = [
    "lipsync_reject_reason",
    "semantic_score",
]


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
    item,
    tracker,
    pyannote_worker,
    talknce_worker,
    syncnet_worker,
    qualified_dir,
    audio_enhance_worker=None,
    semantic_sync_worker=None,
):
    """处理单个视频片段的唇形同步检测。保留输入 CSV 全部列，追加 lipsync 结果。"""
    row = dict(item)
    segment_path = row.get(args.video_path_column, "")
    # 不再保留 source_video_path：当前处理视频的路径就是 segment_path。
    row["video_path"] = ""
    row["lipsync_reject_reason"] = ""
    row["semantic_score"] = ""
    row["is_semantically_synced"] = ""
    row["semantic_status"] = ""

    if not segment_path or not Path(segment_path).is_file():
        row["lipsync_reject_reason"] = "missing_segment"
        return row

    processing_video_path = segment_path

    # ── Step 0: Audio Enhancement (separation + denoising) ───────────────────
    enhanced_audio_path = None
    if audio_enhance_worker is not None:
        enhance_output_dir = Path(args.output_dir) / "enhanced_audio"
        
        enhance_result = audio_enhance_worker.enhance(
            processing_video_path, output_dir=enhance_output_dir
        )
        enhanced_audio_path = enhance_result["enhanced_audio_path"]
        print(f"  [AudioEnhance] Enhanced audio: {enhanced_audio_path}")

    # ── Step 1: PyAnnote Diarization (uses enhanced audio if available) ───────
    diarization = {"status": "disabled", "speakers": []}
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

        # 对每个 face track，在每个 speaker 时间窗口内裁剪人脸 + 提取音频，调用 SyncNet 得到 confidence 分数
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
                sync_offset = None
                sync_is_speaking = False
                if sync_res.get("status") == "ok":
                    for t in sync_res.get("tracks", []):
                        sync_mean = t.get("mean_score")
                        sync_offset = t.get("offset")
                        sync_is_speaking = t.get("is_speaking", False)
                        break

                score_matrix[tr_id].append({
                    "spk_id": spk_id,
                    "spk_start": spk_start,
                    "spk_end": spk_end,
                    "sync_mean": sync_mean,
                    "sync_offset": sync_offset,
                    "sync_is_speaking": sync_is_speaking,
                    "sync_status": sync_res.get("status", "unknown"),
                })

        # 对每个 face track，优先选 is_speaking=True 且 confidence 最高的 window；
        # 若无 is_speaking=True 的，标记为未通过
        track_best = {}
        for tr in face_tracks:
            tr_id = tr["track_id"]
            entries = [e for e in score_matrix[tr_id] if e["sync_status"] == "ok" and e["sync_mean"] is not None]
            if not entries:
                track_best[tr_id] = {"spk_id": None, "spk_start": None, "spk_end": None,
                                     "sync_mean": None, "sync_offset": None,
                                     "sync_is_speaking": False, "sync_status": "no_valid_window"}
            else:
                # 优先选 is_speaking=True
                speaking_entries = [e for e in entries if e["sync_is_speaking"]]
                if speaking_entries:
                    best = max(speaking_entries, key=lambda e: e["sync_mean"])
                else:
                    best = max(entries, key=lambda e: e["sync_mean"])
                track_best[tr_id] = best
  
                                     
        # 从所有 track 的 best 中选 primary：优先 is_speaking=True，再按 confidence
        speaking = [(tr_id, info) for tr_id, info in track_best.items()
                    if info.get("sync_is_speaking")]
        non_speaking = [(tr_id, info) for tr_id, info in track_best.items()
                        if info["sync_mean"] is not None and not info.get("sync_is_speaking")]
        if speaking:
            primary_tr_id, primary_info = max(speaking, key=lambda x: x[1]["sync_mean"])
        elif non_speaking:
            primary_tr_id, primary_info = max(non_speaking, key=lambda x: x[1]["sync_mean"])
        else:
            primary_tr_id, primary_info = None, None

        primary_tr = next((t for t in face_tracks if t["track_id"] == primary_tr_id), None)

        # ── 语义同步（必经流程） ─────────────────────────────────────────────
        semantic_score = None
        semantic_status = "skipped"
        sem_result = {"is_semantically_synced": False}
        if semantic_sync_worker is not None:
            sem_result = semantic_sync_worker.score(
                processing_video_path, enhanced_audio_path
            )
            semantic_score = sem_result.get("semantic_score")
            semantic_status = sem_result.get("status", "unknown")
            print(f"  [SemanticSync] score={semantic_score}, status={semantic_status}")

        semantic_passed = sem_result.get("is_semantically_synced", False)

        row["semantic_score"] = "" if semantic_score is None else semantic_score

        # ── 判定逻辑 ──────────────────────────────────────────────────────────
        # 有 speaking face 存在 → 必须选它，同时语义同步也必须通过
        # 没有 speaking face → 退化到语义同步兜底判断
        has_speaking = primary_info and primary_info.get("sync_is_speaking")

        reject_reasons = []
        if has_speaking:
            # 有说话人：语义同步必须通过
            if not semantic_passed:
                reject_reasons.append("semantic_not_synced")
        else:
            # 无说话人：语义同步兜底
            reject_reasons.append("target_not_speaking")
            if not semantic_passed:
                reject_reasons.append("semantic_not_synced")

        if reject_reasons:
            row["lipsync_reject_reason"] = ",".join(reject_reasons)
        else:
            out_name = unique_output_name(processing_video_path, suffix=".mp4")
            row["video_path"] = link_or_copy(
                processing_video_path,
                Path(qualified_dir) / out_name,
                args.qualified_copy_mode,
            )
        return row

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

    # ── 语义同步（必经流程） ────────────────────────────────────────────────
    semantic_score = None
    semantic_status = "skipped"
    sem_result = {"is_semantically_synced": False}
    if semantic_sync_worker is not None:
        sem_result = semantic_sync_worker.score(
            processing_video_path, enhanced_audio_path
        )
        semantic_score = sem_result.get("semantic_score")
        semantic_status = sem_result.get("status", "unknown")
        print(f"  [SemanticSync] score={semantic_score}, status={semantic_status}")

    semantic_passed = sem_result.get("is_semantically_synced", False)

    row["semantic_score"] = "" if semantic_score is None else semantic_score

    # 判定：有说话人 → 必须选 + 语义必须通过；无说话人 → 语义兜底
    reasons = []
    has_face = face_result.get("num_face_tracks", 0) > 0

    if has_face and syncnet_spoke:
        # 有说话人：语义同步必须通过
        if not semantic_passed:
            reasons.append("semantic_not_synced")
    elif has_face and not syncnet_spoke:
        # 有脸但没在说话：语义兜底
        reasons.append("target_not_speaking")
        if not semantic_passed:
            reasons.append("semantic_not_synced")
    else:
        # 无脸：仅语义同步兜底
        if not semantic_passed:
            reasons.append("semantic_not_synced")

    if reasons:
        row["lipsync_reject_reason"] = ",".join(reasons)
    else:
        out_name = unique_output_name(processing_video_path, suffix=".mp4")
        row["video_path"] = link_or_copy(
            processing_video_path,
            Path(qualified_dir) / out_name,
            args.qualified_copy_mode,
        )
    return row


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TALKNCE_ROOT = PROJECT_ROOT / "third_part" / "TalkNCE"
# 优先级：env INSIGHTFACE_ROOT > ckps_root()/insightface
DEFAULT_INSIGHTFACE_ROOT = env_or("INSIGHTFACE_ROOT", ckps_root() / "insightface")


# ─── Worker ──────────────────────────────────────────────────────────────────────
class Worker:
    def __init__(self):
        worker_syspath_setup()  # Ray worker 端把 data_process/ 加进 sys.path
        self.qualified_dir = Path(args.output_dir) / "qualified_videos"
        ensure_dir(args.output_dir)
        ensure_dir(self.qualified_dir)
        # 加载模型
        self.tracker = InsightFaceTracker(
            args.device, args.insightface_model, args.insightface_det_size,
            112, args.insightface_root, False,
        )
        self.pyannote_worker = PyannoteDiarizationWorker(args)
        self.talknce_worker = TalkNCEWorker(args)
        self.syncnet_worker = SyncNetWorker(args)
        self.audio_enhance_worker = AudioEnhanceWorker(args)
        self.semantic_sync_worker = SemanticSyncWorker(args)
        # 初始化 CSV
        self.all_csv = str(Path(args.output_dir) / "5_lipsync_all.csv")
        self.qual_csv = str(Path(args.output_dir) / "5_lipsync_qualified.csv")
        with open(args.input_csv, encoding="utf-8-sig") as f:
            input_fields = csv.DictReader(f).fieldnames or []
        # 不再额外加 source_video_path；video_path 表示当前处理视频的最终落点
        extra_fields = ["video_path"] + LIPSYNC_FIELDS
        self.fieldnames = list(input_fields) + [c for c in extra_fields if c not in input_fields]
        for p in (self.all_csv, self.qual_csv):
            if not Path(p).exists():
                with open(p, "w", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()
        # 断点续跑：加载 all_csv 中已处理的 video_path（输出路径即当前处理视频）
        self.processed = set()
        if Path(self.all_csv).exists():
            with open(self.all_csv, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    sv = r.get("video_path", "")
                    if sv:
                        self.processed.add(sv)
        print(f"Worker initialized: lipsync models loaded. 已处理 {len(self.processed)} 条。")

    def _append_csv(self, filepath, row):
        with open(filepath, "a", newline="", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore").writerow(row)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)

    def __call__(self, item):
        vid_path = item.get(args.video_path_column, "")
        # 断点续跑：CSV 已记录 或 输出目录已存在对应文件 → 跳过
        out_file = self.qualified_dir / unique_output_name(vid_path, suffix=".mp4")
        if vid_path in self.processed or out_file.exists():
            print(f"[{time.strftime('%H:%M:%S')}] 跳过已处理: {vid_path}", flush=True)
            skip_row = dict(item)
            skip_row.setdefault("video_path", "")
            skip_row["lipsync_reject_reason"] = "already_processed"
            return skip_row
        print(f"[{time.strftime('%H:%M:%S')}] lipsync: {vid_path}", flush=True)
        row = process_segment(
            args, item,
            self.tracker, self.pyannote_worker,
            self.talknce_worker, self.syncnet_worker,
            str(self.qualified_dir),
            audio_enhance_worker=self.audio_enhance_worker,
            semantic_sync_worker=self.semantic_sync_worker,
        )
        self._append_csv(self.all_csv, row)
        if not row.get("lipsync_reject_reason"):
            self._append_csv(self.qual_csv, row)
        self.processed.add(vid_path)
        return row


# ─── CLI ─────
parser = argparse.ArgumentParser(
    description="唇形同步筛选：InsightFace + SyncNet + PyAnnote + SemanticSync"
)
# Paths
parser.add_argument("--input_csv", default=str(project_root() / "test" / "result" / "0612" / "motion_filter_raft" / "data_5000_all_last" / "motion_filter_qualified.csv"),
                    help="输入 CSV（通常为 cutshot_qualified.csv）")
parser.add_argument("--output_dir", default=str(project_root() / "test" / "result" / "0612" / "filter_lipsync" / "data_5000_all_last" / ""))
parser.add_argument("--video_path_column", default="video_path", help="输入 CSV 中视频路径列名")
parser.add_argument("--device", default="cuda:0")

# InsightFace
parser.add_argument("--insightface_model", default="buffalo_l")
parser.add_argument("--insightface_root", default=str(DEFAULT_INSIGHTFACE_ROOT))
parser.add_argument("--insightface_det_size", type=int, default=640,help="InsightFace detection resolution")
parser.add_argument("--face_iou_threshold", type=float, default=0.5)
parser.add_argument("--face_max_missing", type=int, default=5)
parser.add_argument("--min_face_track_frames", type=int, default=20)

# TalkNCE
parser.add_argument("--talknce_root", default=str(TALKNCE_ROOT))
parser.add_argument("--talknce_config_path", default=str(TALKNCE_ROOT / "configs" / "test.yaml"))
parser.add_argument("--talknce_ckpt", default=str(env_or("TALKNCE_CKPT", ckps_root() / "lipsync" / "talknce_ava_pretrained.model")))
parser.add_argument("--active_speaker_threshold", type=float, default=0.6)
parser.add_argument("--active_speaker_min_ratio", type=float, default=0.2)
parser.add_argument("--min_talknce_frames", type=int, default=35)
parser.add_argument("--max_talknce_frames", type=int, default=200)
parser.add_argument("--disable_talknce", action="store_true")

# SyncNet
parser.add_argument("--syncnet_ckpt", default=str(env_or("SYNCNET_CKPT", project_root() / "third_part" / "syncnet" / "syncnet_v2.model")))
parser.add_argument("--syncnet_vshift", type=int, default=10)
parser.add_argument("--syncnet_batch_size", type=int, default=20)
parser.add_argument("--syncnet_min_confidence", type=float, default=1.0)
parser.add_argument("--syncnet_min_offset", type=int, default=-6)
parser.add_argument("--syncnet_max_offset", type=int, default=6)
parser.add_argument("--syncnet_crop_scale", type=float, default=0.40)
parser.add_argument("--disable_syncnet", action="store_true")

# PyAnnote (multi-speaker)
parser.add_argument("--disable_pyannote", action="store_true")
parser.add_argument("--pyannote_model_path", default=str(env_or("PYANNOTE_CFG", ckps_root() / "pyannote" / "config.yaml")))
import os as _os
parser.add_argument("--pyannote_hf_token", default=_os.environ.get("HF_TOKEN", None))
parser.add_argument("--min_speakers", type=int, default=0)

# Audio Enhancement
parser.add_argument("--disable_audio_enhance", action="store_true")
parser.add_argument("--audio_sep_root", default=str(env_or("AUDIO_SEP_ROOT", project_root() / "third_part" / "Music-Source-Separation-Training")))
parser.add_argument("--audio_sep_config",
                    default=str(env_or("AUDIO_SEP_CONFIG",
                                       project_root() / "third_part" / "Music-Source-Separation-Training" / "ckpts" / "config_vocals_mel_band_roformer_kj.yaml")))
parser.add_argument("--audio_sep_ckpt",
                    default=str(env_or("AUDIO_SEP_CKPT",
                                       project_root() / "third_part" / "Music-Source-Separation-Training" / "ckpts" / "MelBandRoformer.ckpt")))
parser.add_argument("--audio_sep_tta", action="store_true")

# Semantic Sync (ImageBind)
parser.add_argument("--disable_semantic_sync", action="store_true")
parser.add_argument("--semantic_sync_threshold", type=float, default=0.25)
parser.add_argument("--semantic_sync_max_frames", type=int, default=15)

# Decision
parser.add_argument("--allow_non_target_speaker", action="store_true")

# Output
parser.add_argument("--qualified_copy_mode", choices=["symlink", "copy", "none"], default="symlink")
parser.add_argument("--use_source_audio_segments", action="store_true", default=True)

# Ray
parser.add_argument("--ray", action="store_true", help="启用 Ray 分布式模式")
parser.add_argument("--ray_address", default="auto")
parser.add_argument("--ray_num_gpus", type=float, default=1.0)
parser.add_argument("--ray_num_cpus", type=float, default=0.0)
parser.add_argument("--concurrency", type=int, default=1)

args = parser.parse_args()


# ─── main ───────────────────────────────────────────────────────────────────────
def _cleanup_rejected_audio(output_dir, qual_csv_path):
    """删除未通过 lipsync 的音频文件，保留 qualified 对应的音频。"""
    enhanced_dir = Path(output_dir) / "enhanced_audio"
    if not enhanced_dir.exists():
        return
    # 收集合格视频的 stem
    qualified_stems = set()
    if Path(qual_csv_path).exists():
        with open(qual_csv_path, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                vp = r.get("video_path", "")
                if vp:
                    qualified_stems.add(Path(vp).stem)
    # 遍历 enhanced_audio 目录，删除不在 qualified 中的文件
    removed = 0
    for audio_file in enhanced_dir.iterdir():
        if audio_file.is_file():
            # enhanced audio 文件名通常是 video_stem + 后缀
            stem = audio_file.stem
            # 去除可能的 _vocals / _enhanced / _enhanced_16k 后缀来匹配原始视频
            base_stem = stem.replace("_enhanced_16k", "").replace("_enhanced", "").replace("_vocals", "")
            if base_stem not in qualified_stems and stem not in qualified_stems:
                audio_file.unlink()
                removed += 1
    print(f"清理不合格音频: 删除 {removed} 个文件, 保留 {len(qualified_stems)} 个合格音频")


if __name__ == "__main__":
    # ── 去重：按 video_path_column 去重，避免重复处理 ──
    import pandas as pd
    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    before_cnt = len(df)
    df = df.drop_duplicates(subset=[args.video_path_column], keep="first")
    after_cnt = len(df)
    if before_cnt != after_cnt:
        print(f"去重: {before_cnt} → {after_cnt} (去除 {before_cnt - after_cnt} 条重复)")
    deduped_csv = args.input_csv.replace(".csv", "_deduped.csv")
    df.to_csv(deduped_csv, index=False)

    if not args.ray:
        # ── 本地模式 ──
        samples = list(csv.DictReader(open(deduped_csv, "r", encoding="utf-8-sig")))
        print(f"本地模式: 共 {len(samples)} 条片段")
        worker = Worker()
        for i, item in enumerate(samples, 1):
            print(f"[{i}/{len(samples)}]")
            worker(item)
    else:
        # ── Ray 分布式模式 ──
        import ray
        ray_init_kwargs = {
            "runtime_env": {"env_vars": {"PYTHONPATH": str(PROJECT_ROOT)}}
        }
        if args.ray_address:
            ray_init_kwargs["address"] = args.ray_address
        ray.init(**ray_init_kwargs)
        from ray.data import ActorPoolStrategy
        samples = ray.data.read_csv(deduped_csv).materialize()
        target_parallelism = min(args.concurrency, samples.count())
        if samples.num_blocks() < target_parallelism:
            samples = samples.repartition(target_parallelism)
        predictions = samples.map(
            Worker,
            num_gpus=args.ray_num_gpus,
            num_cpus=max(args.ray_num_cpus, 1),
            compute=ActorPoolStrategy(size=target_parallelism),
        )
        predictions.take_all()

    # ── 清理不合格的 enhanced audio 文件 ──
    qual_csv_path = str(Path(args.output_dir) / "5_lipsync_qualified.csv")
    _cleanup_rejected_audio(args.output_dir, qual_csv_path)

    print(f"完成: {args.output_dir}")
