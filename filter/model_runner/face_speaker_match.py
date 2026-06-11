"""Face-to-speaker matching based on temporal overlap.

Given InsightFace tracks and PyAnnote speaker segments, computes which
face track belongs to which speaker (or to no speaker) by IoU of their
time intervals.
"""


def match_faces_to_speakers(face_tracks, speaker_segments, fps):
    
    if not face_tracks:
        return {
            "matches": [
                {
                    "speaker_id": seg["speaker_id"],
                    "start_sec": seg["start_sec"],
                    "end_sec": seg["end_sec"],
                    "matched_track_id": None,
                    "matched_track": None,
                    "iou": 0.0,
                    "coverage": 0.0,
                }
                for seg in speaker_segments
            ],
            "unique_speaker_ids": {seg["speaker_id"] for seg in speaker_segments},
            "num_unique_speakers": len({seg["speaker_id"] for seg in speaker_segments}),
        }

    best_matches = []

    for seg in speaker_segments:
        spk_start_sec = seg["start_sec"]
        spk_end_sec = seg["end_sec"]
        spk_start_f = spk_start_sec * fps
        spk_end_f = spk_end_sec * fps
        spk_duration_f = spk_end_f - spk_start_f

        best_track = None
        best_iou = 0.0
        best_coverage = 0.0

        for tr in face_tracks:
            tr_start = float(tr["start_frame"])
            tr_end = float(tr["end_frame"])

            inter_start = max(tr_start, spk_start_f)
            inter_end = min(tr_end, spk_end_f)
            inter_len = max(0.0, inter_end - inter_start)

            union_len = max(tr_end, spk_end_f) - min(tr_start, spk_start_f)
            iou = inter_len / union_len if union_len > 0 else 0.0

            coverage = inter_len / spk_duration_f if spk_duration_f > 0 else 0.0

            score = iou + 0.5 * coverage
            if score > best_iou + 0.5 * best_coverage:
                best_iou = iou
                best_coverage = coverage
                best_track = tr

        best_matches.append({
            "speaker_id": seg["speaker_id"],
            "start_sec": spk_start_sec,
            "end_sec": spk_end_sec,
            "matched_track_id": int(best_track["track_id"]) if best_track else None,
            "matched_track": best_track,
            "iou": round(best_iou, 4),
            "coverage": round(best_coverage, 4),
        })

    return {
        "matches": best_matches,
        "unique_speaker_ids": {seg["speaker_id"] for seg in speaker_segments},
        "num_unique_speakers": len({seg["speaker_id"] for seg in speaker_segments}),
    }
