"""
Step 6（标注版）：用原片自带的标注给片段写 MiniMax-H3 训练描述，不看视频。

适用于像 CineDance 这样带结构化标注的数据：原片旁边有同名 .json，里面有每个镜头的起止时间、画面描述、
音频描述，以及按人物、带时间戳的逐句对白。做法：
  1. 按片段在原片里的起止秒（Step 2 的 clip_start_sec / clip_end_sec），取出落在这段时间里的镜头描述、
     音频描述和对白；对白直接拼成 <d>[语言] 原文</d>，按人物分配 (S1)、(S2)；
  2. 把这些素材交给一个文本模型（默认 Qwen3-VL-8B-Instruct，只用文字），按 H3 格式改写成三段英文描述；
     切点时间用 Step 2 检测到的 cut_frames；
  3. 按 h3_caption_format 的规则检查，不合格重写一次；对白仍缺就在描述末尾补上（记 speech_appended 警告）。
比 Qwen3-Omni 看视频快得多，对白和说话人也更准（来自人工质量的标注）。
"""
import argparse
import csv
import fcntl
import json
import re
import sys
from pathlib import Path

_FILTER_ROOT = Path(__file__).resolve().parents[1]
if str(_FILTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FILTER_ROOT))

from filter.filter_utils import env_or, worker_syspath_setup, init_csv, load_processed_inputs, INPUT_PATH_FIELD  # noqa: E402
from filter.annotation_utils import load_annotation, character_labels, clip_dialogue  # noqa: E402
from filter.h3_caption_format import (  # noqa: E402
    H3_RULES, CAPTION_PREFIX, SOFT_ISSUES, format_cut_time, parse_cut_frames,
    check_caption, normalize_caption, describe_issue,
)

CAPTION_FIELDS = ["caption", "caption_asr", "caption_words", "caption_speakers", "caption_has_cutoff",
                  "caption_warnings", "caption_reject_reason"]

SYSTEM_PROMPT = """You are a meticulous annotator writing training captions for MiniMax-H3, a model that generates video and audio together. You cannot see the clip. Instead you get structured notes written by careful human-quality annotators: what each shot shows, what can be heard, and the exact dialogue. Turn the notes into the caption H3 should be trained with. Use only facts from the notes; never add events, objects or sounds they do not mention. The notes describe whole shots, which may extend beyond this clip, so keep to what fits a clip of the given length.

""" + H3_RULES

def build_notes(ann, start, end, cut_times):
    """把片段 [start, end) 秒内的标注整理成给文本模型的素材；返回 (素材文字, 对白 <d> 行, 说话人数, 是否截断)。"""
    chars = character_labels(ann)

    def name(text):
        return re.sub(r"<char\d+>", lambda m: f"[{chars.get(m.group(0), 'a person')}]", text or "")

    lines = [f"Clip length: {end - start:.2f} seconds."]
    style = (ann.get("visual") or {}).get("style", "")
    if style:
        lines.append("Visual style of the source film: " + style.split(". ")[0].rstrip(".") + ".")
    if cut_times:
        lines.append(f"The clip has {len(cut_times) + 1} shots; hard cuts at " + ", ".join(cut_times) + ".")
    else:
        lines.append("The clip is one continuous shot.")

    shots = [s for s in ann.get("shots", []) if s["end_sec"] > start + 0.05 and s["start_sec"] < end - 0.05]
    for i, s in enumerate(shots, 1):
        st = s.get("shot_type") or {}
        lines.append(
            f"Source shot {i} ({max(s['start_sec'], start) - start:.2f}s-{min(s['end_sec'], end) - start:.2f}s of this clip; "
            f"{st.get('scale', '')}, {st.get('angle', '')}, camera {st.get('movement', '')}): "
            f"{name(s.get('detailed_caption'))} Audio: {s.get('audio_prompt', '')}")

    dialogue = clip_dialogue(ann, start, end)
    if dialogue:
        lines.append("Dialogue, in order (speaker ID, who, start time in this clip, exact words to copy):")
        lines += [f"- ({x['sid']}) {x['who']}, at {x['t']:.2f}s: {x['d']}" for x in dialogue]
    else:
        lines.append("Nobody speaks or sings in this clip.")
    return ("\n".join(lines), [x["d"] for x in dialogue], len({x["sid"] for x in dialogue}),
            any("<cutoff>" in x["d"] for x in dialogue))


class TextRewriter:
    def __init__(self, model_path):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map="cuda", attn_implementation=args.attn_implementation).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

    def generate(self, user_text):
        conv = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [{"type": "text", "text": user_text}]}]
        text = self.processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False) + CAPTION_PREFIX
        inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                      repetition_penalty=args.repetition_penalty)
        return CAPTION_PREFIX + self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                                            skip_special_tokens=True)[0]


def caption_clip(rewriter, item):
    row = dict(item)
    row.update({k: "" for k in CAPTION_FIELDS})
    ann = load_annotation(item.get("source_path", ""), args.annotation_suffix)
    if ann is None:
        row["caption_reject_reason"] = "no_annotation"
        return row
    try:
        start, end = float(item["clip_start_sec"]), float(item["clip_end_sec"])
    except (KeyError, ValueError, TypeError):
        row["caption_reject_reason"] = "no_clip_time"
        return row
    fps = args.clip_fps
    cut_times = [format_cut_time(c / fps) for c in parse_cut_frames(item.get("cut_frames"))]
    notes, d_blocks, n_speakers, _ = build_notes(ann, start, end, cut_times)
    task = (notes + "\n\nWrite the caption. Begin each later shot with its cut time, like \"[Shot 2] At "
            + (cut_times[0] if cut_times else "00:02.000") + ", the camera cuts to\". "
            + ("Include every dialogue block above exactly as given, with its speaker ID outside the block."
               if d_blocks else "Write no <d> block."))
    caption, appended = "", False
    for attempt in range(2):
        caption = normalize_caption(rewriter.generate(task))
        issues, _ = check_caption(caption, has_speech=bool(d_blocks), cut_times=cut_times)
        if not issues or attempt == 1:
            break
        task += "\nYour previous attempt broke these rules, fix them: " + "; ".join(map(describe_issue, issues)) + "."
    if d_blocks and "<d>" not in caption:
        said = " ".join(d_blocks)
        caption = re.sub(r"(integrated_multimodal_description:.*?)(\n\noverall_soundscape:)",
                         lambda m: f"{m.group(1)} A voice (S1) says: {said}{m.group(2)}", caption, count=1, flags=re.S)
        appended = True
    issues, stats = check_caption(caption, has_speech=bool(d_blocks), cut_times=cut_times)
    warnings = [i for i in issues if i in SOFT_ISSUES] + (["speech_appended"] if appended else [])
    row.update(stats)
    row["caption"] = caption
    row["caption_asr"] = " | ".join(re.sub(r"</?d>", "", d) for d in d_blocks) or "NONE"
    row["caption_warnings"] = ",".join(warnings)
    row["caption_reject_reason"] = ",".join(i for i in issues if i not in SOFT_ISSUES)
    print(f"  {'不通过' if row['caption_reject_reason'] else '通过'}: {Path(item['video_path']).name} | "
          f"{stats.get('caption_words', 0)} words {row['caption_reject_reason']}")
    return row


class Worker:
    def __init__(self):
        worker_syspath_setup()
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.all_csv, self.qual_csv = out / "6_caption_all.csv", out / "6_caption_qualified.csv"
        with open(args.input_csv, encoding="utf-8") as f:
            fields = csv.DictReader(f).fieldnames or []
        fields = list(fields) + [c for c in CAPTION_FIELDS if c not in fields]
        self.csv_fields = {self.all_csv: init_csv(self.all_csv, fields + [INPUT_PATH_FIELD]),
                           self.qual_csv: init_csv(self.qual_csv, fields)}
        self.processed = load_processed_inputs(self.all_csv)
        print(f"加载文本模型: {args.model_path}")
        self.rewriter = TextRewriter(args.model_path)

    def _append(self, path, row):
        with open(path, "a", newline="", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            csv.DictWriter(f, fieldnames=self.csv_fields[path], extrasaction="ignore").writerow(row)
            fcntl.flock(f, fcntl.LOCK_UN)

    def __call__(self, item):
        clip = item.get(args.video_path_column, "")
        if not clip or clip in self.processed:
            return item
        row = caption_clip(self.rewriter, item)
        self._append(self.all_csv, {**row, INPUT_PATH_FIELD: clip})
        if not row["caption_reject_reason"]:
            self._append(self.qual_csv, row)
        self.processed.add(clip)
        return row


parser = argparse.ArgumentParser(description="用原片标注写 MiniMax-H3 格式的片段描述（文本模型改写，不看视频）")
parser.add_argument("--input_csv", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--video_path_column", default="video_path")
parser.add_argument("--annotation_suffix", default=".json", help="原片标注 = 原片路径换成这个后缀（CineDance 为 .json）")
parser.add_argument("--clip_fps", type=float, default=24.0, help="片段帧率，用于把 cut_frames 换成秒")
parser.add_argument("--model_path", default=str(env_or("CAPTION_TEXT_MODEL", "/mnt/shared/lihaoran/ssd/ckps/Qwen3-VL-8B-Instruct")))
parser.add_argument("--max_new_tokens", type=int, default=800)
parser.add_argument("--repetition_penalty", type=float, default=1.05)
parser.add_argument("--attn_implementation", default="flash_attention_2", choices=["flash_attention_2", "sdpa"])
args = parser.parse_args()

if __name__ == "__main__":
    rows = list(csv.DictReader(open(args.input_csv, encoding="utf-8")))
    print(f"读取 {len(rows)} 条记录 from {args.input_csv}")
    worker = Worker()
    for i, item in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {item.get(args.video_path_column, '')}")
        worker(item)
    print("完成")
