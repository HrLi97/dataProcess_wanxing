"""
Step 7: 导出 I2V（MiniMax-H3 的 I2VA，首帧图生视频）训练数据。

对 Step 6 合格的片段：
  1. 导出第 0 帧为 PNG（无损，原分辨率），作为生成时的参考图 <Picture 1>；
  2. 检查首帧：太暗（片头黑场、淡入）、太糊、正在淡入的首帧不适合当参考图；
  3. 写 7_i2v_all.csv / 7_i2v_qualified.csv，以及训练清单 i2v_manifest.jsonl（每行一个样本：
     片段、首帧、描述、帧率帧数、镜头切点、来源和各步分数）。
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

_FILTER_ROOT = Path(__file__).resolve().parents[1]
if str(_FILTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FILTER_ROOT))

from filter.filter_utils import init_csv, load_processed_inputs, unique_output_name, INPUT_PATH_FIELD  # noqa: E402

I2V_FIELDS = ["first_frame_path", "ff_luma", "ff_sharpness", "ff_fade_ratio", "i2v_reject_reason"]
SCORE_FIELDS = ["motion_score", "dover_overall", "aes_mean", "kvq_score", "num_shots", "cut_frames",
                "caption_words", "caption_speakers", "caption_warnings"]


def first_frame_stats(path, compare_frame=6, max_side=512):
    """返回 (第 0 帧 BGR 原图, 平均亮度, 清晰度, 淡入比)；读不到返回 None。

    亮度：灰度均值（0–255）。清晰度：长边缩到 max_side 后拉普拉斯方差（越大越清楚）。
    淡入比：第 0 帧亮度 / 第 compare_frame 帧亮度（明显小于 1 说明正在从黑场淡入）。
    """
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) <= compare_frame:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if not frames:
        return None

    def gray(f):
        h, w = f.shape[:2]
        s = max_side / max(h, w)
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else g

    g0 = gray(frames[0])
    luma = float(g0.mean())
    sharp = float(cv2.Laplacian(g0, cv2.CV_64F).var())
    later = float(gray(frames[-1]).mean())
    fade = luma / later if later > 1 else 1.0
    return frames[0], luma, sharp, fade


def process(item, frame_dir):
    row = dict(item)
    row.update({k: "" for k in I2V_FIELDS})
    clip = item.get(args.video_path_column, "")
    stats = first_frame_stats(clip) if clip and Path(clip).is_file() else None
    if stats is None:
        row["i2v_reject_reason"] = "read_failed"
        return row
    frame, luma, sharp, fade = stats
    png = frame_dir / unique_output_name(clip, suffix=".png")
    if not png.exists():
        cv2.imwrite(str(png), frame)
    row.update(first_frame_path=str(png), ff_luma=round(luma, 2), ff_sharpness=round(sharp, 1),
               ff_fade_ratio=round(fade, 3))
    reasons = []
    if luma < args.min_luma:
        reasons.append(f"first_frame_dark_{luma:.0f}")
    if sharp < args.min_sharpness:
        reasons.append(f"first_frame_blurry_{sharp:.0f}")
    if fade < args.max_fade_ratio and luma < args.fade_luma:
        reasons.append(f"first_frame_fading_{fade:.2f}")
    row["i2v_reject_reason"] = ",".join(reasons)
    return row


def manifest_record(row):
    rec = {
        "id": Path(row[args.video_path_column]).stem,
        "task": "i2va",
        "video": row[args.video_path_column],
        "first_frame": row["first_frame_path"],
        "caption": row.get("caption", ""),
        "fps": args.fps,
        "num_frames": int(float(row.get("duration_frames") or 0)) or None,
        "source_path": row.get("source_path", ""),
        "clip_start_sec": row.get("clip_start_sec", ""),
        "clip_end_sec": row.get("clip_end_sec", ""),
        "scores": {k: row.get(k, "") for k in SCORE_FIELDS if row.get(k, "") != ""},
        "first_frame_stats": {k: row[k] for k in ("ff_luma", "ff_sharpness", "ff_fade_ratio")},
    }
    return rec


parser = argparse.ArgumentParser(description="导出 MiniMax-H3 I2VA 训练数据：首帧 PNG、首帧检查、训练清单")
parser.add_argument("--input_csv", required=True, help="Step 6 的 6_caption_qualified.csv（i2va 模式打的标）")
parser.add_argument("--output_dir", required=True)
parser.add_argument("--video_path_column", default="video_path")
parser.add_argument("--fps", type=float, default=24.0)
parser.add_argument("--min_luma", type=float, default=8, help="首帧平均亮度下限（0–255），只去掉黑场；电影夜景 11–15 也算正常")
parser.add_argument("--min_sharpness", type=float, default=15, help="首帧拉普拉斯方差下限，去掉严重糊的")
parser.add_argument("--max_fade_ratio", type=float, default=0.7, help="首帧亮度 / 0.25 秒后亮度低于此值、且首帧偏暗时视为淡入")
parser.add_argument("--fade_luma", type=float, default=60, help="淡入判断只对亮度低于此值的首帧生效")
args = parser.parse_args()

if __name__ == "__main__":
    out = Path(args.output_dir)
    frame_dir = out / "first_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.input_csv, encoding="utf-8")))
    fields = list(rows[0].keys()) if rows else []
    fields += [c for c in I2V_FIELDS if c not in fields]
    all_csv, qual_csv = out / "7_i2v_all.csv", out / "7_i2v_qualified.csv"
    cols = {all_csv: init_csv(all_csv, fields + [INPUT_PATH_FIELD]), qual_csv: init_csv(qual_csv, fields)}
    done = load_processed_inputs(all_csv)
    n_ok = n_bad = 0
    with open(all_csv, "a", newline="", encoding="utf-8") as fa, open(qual_csv, "a", newline="", encoding="utf-8") as fq, \
            open(out / "i2v_manifest.jsonl", "a", encoding="utf-8") as fm:
        wa = csv.DictWriter(fa, fieldnames=cols[all_csv], extrasaction="ignore")
        wq = csv.DictWriter(fq, fieldnames=cols[qual_csv], extrasaction="ignore")
        for item in rows:
            clip = item.get(args.video_path_column, "")
            if not clip or clip in done:
                continue
            row = process(item, frame_dir)
            wa.writerow({**row, INPUT_PATH_FIELD: clip})
            if row["i2v_reject_reason"]:
                n_bad += 1
                print(f"  不通过: {Path(clip).name} | {row['i2v_reject_reason']}")
            else:
                n_ok += 1
                wq.writerow(row)
                fm.write(json.dumps(manifest_record(row), ensure_ascii=False) + "\n")
    print(f"完成: 通过 {n_ok}，不通过 {n_bad}；清单 {out / 'i2v_manifest.jsonl'}")
