#!/usr/bin/env python3
"""
Debug visualization for SyncNet lip-sync pipeline.

For each segment, generates an annotated video showing:
  - Face bounding boxes with track IDs
  - Per-track SyncNet offset & confidence
  - Diarization speaker windows (colored timeline bar)
  - Qualification decision overlay

Outputs:
  <output_dir>/debug_vis/qualified/   — segments that passed
  <output_dir>/debug_vis/unqualified/ — segments that failed
"""
import argparse
import csv
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

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filter.filter_utils import ensure_dir, safe_name
from filter.model_runner import InsightFaceTracker, SyncNetWorker, TalkNCEWorker
from filter.model_runner.pyannote_diarization import PyannoteDiarizationWorker
from filter.model_runner.face_speaker_match import match_faces_to_speakers
from filter.model_runner.syncnet_runner import (
    _crop_face_for_syncnet,
    _extract_audio_by_frames,
    _syncnet_evaluate,
)

# ── colours for tracks / speakers ────────────────────────────────────────────
TRACK_COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue (BGR)
    (0, 0, 255),    # red
    (255, 255, 0),  # cyan
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (128, 255, 0),
    (0, 128, 255),
]

SPK_COLORS = [
    (255, 200, 100),
    (100, 200, 255),
    (200, 100, 255),
    (100, 255, 200),
]


def color_for(idx, palette):
    return palette[idx % len(palette)]


# ── draw helpers ─────────────────────────────────────────────────────────────

def draw_bbox(frame, bbox, color, label, thickness=2):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_info_panel(frame, lines, origin=(10, 30), font_scale=0.6, color=(255, 255, 255)):
    """Draw multi-line text on top-left with dark background."""
    y = origin[1]
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(frame, (origin[0] - 2, y - th - 4), (origin[0] + tw + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(frame, line, (origin[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
        y += th + 12


def draw_timeline_bar(frame, current_sec, total_sec, speakers, h, w):
    """Draw a horizontal timeline bar at the bottom showing speaker windows."""
    bar_h = 30
    bar_y = h - bar_h - 10
    bar_x0, bar_x1 = 10, w - 10
    bar_w = bar_x1 - bar_x0

    # Background
    cv2.rectangle(frame, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), (40, 40, 40), -1)

    # Speaker segments
    spk_ids = sorted(set(s["speaker_id"] for s in speakers))
    spk_color_map = {sid: color_for(i, SPK_COLORS) for i, sid in enumerate(spk_ids)}

    for seg in speakers:
        if total_sec <= 0:
            continue
        sx = bar_x0 + int(seg["start_sec"] / total_sec * bar_w)
        ex = bar_x0 + int(seg["end_sec"] / total_sec * bar_w)
        c = spk_color_map[seg["speaker_id"]]
        cv2.rectangle(frame, (sx, bar_y + 2), (ex, bar_y + bar_h - 2), c, -1)
        cv2.putText(frame, seg["speaker_id"][-2:], (sx + 2, bar_y + bar_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    # Current position marker
    if total_sec > 0:
        cx = bar_x0 + int(current_sec / total_sec * bar_w)
        cv2.line(frame, (cx, bar_y), (cx, bar_y + bar_h), (0, 0, 255), 2)

    # Border
    cv2.rectangle(frame, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), (200, 200, 200), 1)


def draw_verdict_banner(frame, text, is_qualified, h, w):
    """Draw a banner at top-right with PASS/FAIL."""
    color = (0, 200, 0) if is_qualified else (0, 0, 220)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    bx = w - tw - 20
    cv2.rectangle(frame, (bx - 5, 5), (w - 5, th + 20), color, -1)
    cv2.putText(frame, text, (bx, th + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


# ── per-segment debug processing ──────────────────────────────────────────────

def debug_segment(
    args, segment_path, tracker, syncnet_worker, pyannote_worker,
    output_dir_qualified, output_dir_unqualified,
):
    """Process one segment and produce an annotated debug video."""
    if not segment_path or not Path(segment_path).is_file():
        print(f"  [SKIP] Segment not found: {segment_path}")
        return None

    video_path = str(segment_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps if fps > 0 else 0
    cap.release()

    # ── Step 1: Diarization ───────────────────────────────────────────────
    diarization = pyannote_worker.diarize(video_path)
    speakers = diarization.get("speakers", []) if diarization.get("status") == "ok" else []

    # ── Step 2: Face tracking ─────────────────────────────────────────────
    face_result = tracker.extract_tracks(
        video_path,
        args.face_iou_threshold,
        args.face_max_missing,
        args.min_face_track_frames,
    )
    tracks = face_result.get("tracks", [])
    print(f"  Tracks: {len(tracks)}, Speakers: {len(speakers)}")

    # ── Step 3: SyncNet scoring per track (global or per-speaker) ─────────
    track_scores = {}  # track_id -> {offset, confidence, is_speaking, speaker_window}

    use_multi = (
        not args.disable_pyannote
        and diarization.get("status") == "ok"
        and len(speakers) > 0
    )

    for tr in tracks:
        tid = tr["track_id"]
        if use_multi:
            best_conf = -1
            best_result = None
            for seg in speakers:
                res = syncnet_worker.score(
                    video_path, face_result,
                    track_id=tid,
                    speaker_window=(seg["start_sec"], seg["end_sec"]),
                )
                for t in res.get("tracks", []):
                    if t["confidence"] > best_conf:
                        best_conf = t["confidence"]
                        best_result = {**t, "spk_id": seg["speaker_id"]}
            if best_result:
                track_scores[tid] = best_result
        else:
            res = syncnet_worker.score(video_path, face_result, track_id=tid)
            for t in res.get("tracks", []):
                track_scores[tid] = t
                break

    # ── Determine qualification ───────────────────────────────────────────
    qualified_tracks = {tid: s for tid, s in track_scores.items() if s.get("is_speaking")}
    is_qualified = len(qualified_tracks) > 0

    # ── Build per-frame bbox lookup ───────────────────────────────────────
    frame_bboxes = {}  # frame_idx -> list of (bbox, track_id)
    for tr in tracks:
        tid = tr["track_id"]
        frames_list = tr.get("frames", [])
        bboxes = tr.get("bboxes", [])
        for fidx, bbox in zip(frames_list, bboxes):
            frame_bboxes.setdefault(int(fidx), []).append((bbox, tid))

    # ── Render annotated video ────────────────────────────────────────────
    out_dir = output_dir_qualified if is_qualified else output_dir_unqualified
    stem = safe_name(Path(video_path).stem)
    out_path = str(Path(out_dir) / f"{stem}_debug.mp4")

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_sec = fidx / fps

        # Draw face bboxes
        if fidx in frame_bboxes:
            for bbox, tid in frame_bboxes[fidx]:
                c = color_for(tid, TRACK_COLORS)
                score_info = track_scores.get(tid)
                if score_info:
                    label = f"T{tid} off={score_info['offset']} conf={score_info['confidence']:.2f}"
                    if score_info["is_speaking"]:
                        label += " SYNC"
                    else:
                        label += " NO-SYNC"
                else:
                    label = f"T{tid} (no score)"
                draw_bbox(frame, bbox, c, label)

        # Info panel
        info_lines = [
            f"Frame {fidx}/{total_frames}  t={current_sec:.2f}s",
            f"Tracks: {len(tracks)}  Speakers: {len(set(s['speaker_id'] for s in speakers))}",
        ]
        for tid, sc in track_scores.items():
            spk = sc.get("spk_id", "?")
            info_lines.append(
                f"  T{tid}: off={sc['offset']} conf={sc['confidence']:.2f} "
                f"{'PASS' if sc['is_speaking'] else 'FAIL'} spk={spk}"
            )
        draw_info_panel(frame, info_lines)

        # Timeline bar
        if speakers:
            draw_timeline_bar(frame, current_sec, total_sec, speakers, h, w)

        # Verdict banner
        verdict = "QUALIFIED" if is_qualified else "UNQUALIFIED"
        draw_verdict_banner(frame, verdict, is_qualified, h, w)

        writer.write(frame)
        fidx += 1

    cap.release()
    writer.release()

    # Re-encode with ffmpeg for browser compatibility (H.264)
    final_path = out_path.replace("_debug.mp4", "_debug_h264.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", out_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-an",  # no audio for debug viz
        final_path,
        "-loglevel", "quiet",
    ]
    subprocess.run(cmd, check=False)
    if Path(final_path).is_file():
        os.remove(out_path)
        out_path = final_path

    status = "QUALIFIED" if is_qualified else "UNQUALIFIED"
    print(f"  [{status}] -> {out_path}")

    return {
        "segment": video_path,
        "output": out_path,
        "is_qualified": is_qualified,
        "track_scores": track_scores,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    from filter.filter_lipsync import collect_candidate_scenes, parse_args as lipsync_parse_args

    # Reuse the same CLI args as the main pipeline
    parser = argparse.ArgumentParser(
        description="Debug visualization for SyncNet lip-sync pipeline"
    )
    # We'll manually add the args we need and inherit the rest
    args = lipsync_parse_args()

    # Override output dir to add debug subdirectory
    debug_dir = Path(args.output_dir) / "debug_vis"
    dir_qual = debug_dir / "qualified"
    dir_unqual = debug_dir / "unqualified"
    ensure_dir(dir_qual)
    ensure_dir(dir_unqual)

    print(f"=== Debug Visualization ===")
    print(f"Output: {debug_dir}")
    print(f"Scenes CSV: {args.scenes_csv}")

    # Load workers
    tracker = InsightFaceTracker(
        args.device,
        args.insightface_model,
        args.insightface_det_size,
        112,
        args.insightface_root,
        False,
    )
    syncnet_worker = SyncNetWorker(args)
    pyannote_worker = PyannoteDiarizationWorker(args)

    items = collect_candidate_scenes(args.scenes_csv)
    results = []

    for item in items:
        video_name = (
            item["summary_row"].get("video_name")
            or Path(item["summary_row"].get("video_path", "video")).name
        )
        for scene in item["candidate_scenes"]:
            scene_idx = scene.get("scene_idx", 0)
            segment_path = scene.get("segment_video_path", "")
            print(f"\n[{time.strftime('%H:%M:%S')}] {video_name} scene={scene_idx}")
            print(f"  Source: {segment_path}")

            result = debug_segment(
                args, segment_path,
                tracker, syncnet_worker, pyannote_worker,
                str(dir_qual), str(dir_unqual),
            )
            if result:
                results.append(result)

    # Write summary
    summary_path = debug_dir / "debug_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "segment", "output", "is_qualified",
            "best_track_id", "best_offset", "best_confidence", "best_is_speaking",
        ])
        writer.writeheader()
        for r in results:
            best = max(r["track_scores"].values(), key=lambda x: x["confidence"]) if r["track_scores"] else {}
            writer.writerow({
                "segment": r["segment"],
                "output": r["output"],
                "is_qualified": r["is_qualified"],
                "best_track_id": best.get("track_id", ""),
                "best_offset": best.get("offset", ""),
                "best_confidence": best.get("confidence", ""),
                "best_is_speaking": best.get("is_speaking", ""),
            })

    n_qual = sum(1 for r in results if r["is_qualified"])
    n_unqual = sum(1 for r in results if not r["is_qualified"])
    print(f"\n=== Done ===")
    print(f"Qualified: {n_qual}  Unqualified: {n_unqual}")
    print(f"Qualified videos:   {dir_qual}")
    print(f"Unqualified videos: {dir_unqual}")
    print(f"Summary CSV:        {summary_path}")

    # Cleanup
    syncnet_worker.close()
    pyannote_worker.close()


if __name__ == "__main__":
    main()
