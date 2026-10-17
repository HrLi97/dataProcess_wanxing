"""
把 pipeline 各步的 CSV 导入 FiftyOne，开网页浏览、筛选、打标签；定时重新同步，处理进度实时可见。

    # 在 fiftyone 环境里运行（conda/envs/fiftyone）
    python script/fiftyone_sync.py --root /mnt/shared/lihaoran/Data/CineDance_01_h3/pilot500 --name cinedance_i2v_pilot500

数据集：
  <name>          每个片段一个样本（Step 2 切出的 mp4），字段：各步分数、走到哪一步、被哪一步拒掉及原因、
                  镜头数和切点、描述、首帧 PNG 路径和首帧检查值、来源视频和语言风格
  <name>_sources  每个源视频一个样本：Step 1 结果、切出几个片段、最后合格几个
目录约定：<root>/shard*/{1_video_filter,2_cutshot,3_motion_filter,4_quality_filter}/*_all.csv，
          <root>/6_caption/6_caption_all.csv，<root>/7_i2v/7_i2v_all.csv（不分 shard，可选）
"""
import argparse
import csv
import glob
import json
import time
from pathlib import Path

STEPS = [  # (步骤名, all CSV 相对路径模式, 拒绝原因列)
    ("motion", "shard*/3_motion_filter/3_motion_filter_all.csv", "motion_reject_reason"),
    ("quality", "shard*/4_quality_filter/4_quality_all.csv", "quality_reject_reason"),
    ("caption", "6_caption/6_caption_all.csv", "caption_reject_reason"),
    ("i2v", "7_i2v/7_i2v_all.csv", "i2v_reject_reason"),
]
FLOATS = {"motion_score", "dover_overall", "aes_mean", "kvq_score", "caption_words", "caption_speakers",
          "ff_luma", "ff_sharpness", "ff_fade_ratio", "clip_start_sec", "clip_end_sec", "num_shots"}
KEEP = ["motion_score", "dover_overall", "aes_mean", "kvq_score", "caption", "caption_asr", "caption_words",
        "caption_speakers", "caption_warnings", "first_frame_path", "ff_luma", "ff_sharpness", "ff_fade_ratio"]


def read(pattern, root):
    rows = []
    for f in sorted(glob.glob(str(Path(root) / pattern))):
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r["_shard"] = Path(f).parts[-3] if "shard" in f else ""
                rows.append(r)
    return rows


def value(k, v):
    if v in ("", None):
        return None
    if k in FLOATS:
        try:
            x = float(v)
            return None if x != x else x
        except ValueError:
            return None
    return v


def collect(root):
    """返回 (片段记录 {片段路径: dict}, 源视频记录 {源路径: dict})。"""
    meta = {}
    sel = Path(root) / "selection.jsonl"
    if sel.exists():
        for line in open(sel, encoding="utf-8"):
            r = json.loads(line)
            meta[r["id"]] = r

    sources, step1_out = {}, {}
    for r in read("shard*/1_video_filter/video_filter_all.csv", root):
        src = r.get("_source_key") or r.get("input_video_path")
        if not src:
            continue
        m = meta.get(Path(src).stem, {})
        sources[src] = {"filepath": src, "shard": r["_shard"], "lang": m.get("lang"),
                        "style": (m.get("style") or "")[:120], "duration": m.get("dur"),
                        "step1_reject": r.get("video_filter_reject_reason") or None,
                        "resolution": r.get("resolution"), "crop": r.get("crop") or None, "clips": 0, "ready": 0}
        if r.get("video_path"):
            step1_out[r["video_path"]] = src

    clips, alias = {}, {}
    for r in read("shard*/2_cutshot/2_cutshot_all.csv", root):
        seg = r.get("video_path")
        src = step1_out.get(r.get("input_video_path"), r.get("source_path") or r.get("input_video_path"))
        if src in sources and seg:
            sources[src]["clips"] += 1
        if not seg:
            continue
        clips[seg] = {"filepath": seg, "shard": r["_shard"], "source": src, "stage": "cut", "status": "passed",
                      "reject_step": None, "reject_reason": None,
                      "num_shots": value("num_shots", r.get("num_shots")), "cut_frames": r.get("cut_frames") or None,
                      "clip_start_sec": value("clip_start_sec", r.get("clip_start_sec")),
                      "clip_end_sec": value("clip_end_sec", r.get("clip_end_sec")),
                      "lang": (sources.get(src) or {}).get("lang"), "style": (sources.get(src) or {}).get("style")}
        alias[seg] = seg

    for step, pattern, reason_col in STEPS:
        for r in read(pattern, root):
            seg = alias.get(r.get("input_video_path")) or alias.get(r.get("video_path"))
            if seg is None:
                continue
            if r.get("video_path"):
                alias[r["video_path"]] = seg
            c = clips[seg]
            for k in KEEP:
                if k in r and value(k, r[k]) is not None:
                    c[k] = value(k, r[k])
            if c["status"] == "rejected":
                continue
            c["stage"] = step
            reason = r.get(reason_col) or ""
            if reason:
                c.update(status="rejected", reject_step=step, reject_reason=reason)
            elif step == "i2v":
                c["status"] = "ready"
    for c in clips.values():
        if c["status"] == "ready" and c["source"] in sources:
            sources[c["source"]]["ready"] += 1
    return clips, sources


def sync(fo, name, root):
    clips, sources = collect(root)
    for ds_name, records, media in ((name, clips, "video"), (f"{name}_sources", sources, "video")):
        ds = fo.load_dataset(ds_name) if fo.dataset_exists(ds_name) else fo.Dataset(ds_name, persistent=True)
        samples = []
        for rec in records.values():
            if not Path(rec["filepath"]).exists():
                continue
            s = fo.Sample(filepath=rec["filepath"])
            for k, v in rec.items():
                if k != "filepath" and v is not None:
                    s[k] = v
            tags = [rec.get("status", "")] if ds_name == name else (["rejected"] if rec.get("step1_reject") else ["kept"])
            s.tags = [t for t in tags if t]
            samples.append(s)
        if samples:
            ds.merge_samples(samples, key_field="filepath", insert_new=True, overwrite=True)
        print(f"[{time.strftime('%H:%M:%S')}] {ds_name}: {len(ds)} 个样本")
    return fo.load_dataset(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--root", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--port", type=int, default=5151)
    ap.add_argument("--interval", type=int, default=300, help="每隔多少秒重新同步一次；0=只同步一次不开网页")
    a = ap.parse_args()
    import fiftyone as fo
    ds = sync(fo, a.name, a.root)
    if a.interval <= 0:
        return
    session = fo.launch_app(ds, remote=True, address="0.0.0.0", port=a.port, auto=False)
    print(f"FiftyOne 已启动：http://<这台机器>:{a.port}", flush=True)
    while True:
        time.sleep(a.interval)
        try:
            sync(fo, a.name, a.root)
            session.refresh()
        except Exception as e:  # 同步失败不退出，下次再试
            print(f"同步失败: {e}", flush=True)


if __name__ == "__main__":
    main()
