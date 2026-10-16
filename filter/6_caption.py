"""
Step 6: 用 Qwen3-Omni（同时看画面、听声音）给片段写 MiniMax-H3 训练用的描述。

格式按 MiniMax-H3 官方写 prompt 指南（skills/h3-prompt-writing/references/base-en.txt）的 T2VA：

    integrated_multimodal_description: [Shot 1] ...   画面、动作、运镜、对白（<d>[Chinese] 原文</d>）、和画面同步的声音
    overall_soundscape: ...                            环境声、动作声、非语言人声，1–4 句
    non_diegetic_music: ...                            画外配乐，1–3 句；没有就写 N/A

片段来自 Step 2。单镜头片段只写 [Shot 1]；多镜头片段（Step 2 multi_shot 模式，cut_frames 列）把切点时间告诉模型，
按 [Shot 2] At 00:02.375, the camera cuts to… 的格式写，并检查镜头数和时间戳。每个片段跑两遍：
  1. 只听音频，逐字转写人声（没有就输出 NONE）——单独转写比边看边写更不容易编造对白；
  2. 看画面 + 听音频写描述，把第 1 遍的转写告诉模型（对白照抄进 <d>；没有人声则不许写 <d>）。
生成后按官方校验器（verifyH3Prompt.ts）里的格式规则检查：不合格的只写进 all CSV（caption_reject_reason 写原因），
不进 qualified CSV。
"""
import argparse
import csv
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

_FILTER_ROOT = Path(__file__).resolve().parents[1]
if str(_FILTER_ROOT) not in sys.path:
    sys.path.insert(0, str(_FILTER_ROOT))

from filter.filter_utils import (  # noqa: E402
    ckps_root, env_or, worker_syspath_setup, init_csv, load_processed_inputs, INPUT_PATH_FIELD,
    unique_output_name,
)
from filter.obs_helper import add_obs_args, ObsIO  # noqa: E402
from filter.annotation_utils import load_annotation, clip_dialogue  # noqa: E402

CAPTION_FIELDS = ["caption", "caption_asr", "caption_words", "caption_speakers", "caption_has_cutoff",
                  "caption_warnings", "caption_reject_reason"]

# 指南原文改写成"描述看到和听到的"，规则和措辞尽量照抄指南，让训练描述和推理时的 prompt 同分布
from filter.h3_caption_format import (  # noqa: E402
    H3_RULES, CAPTION_PREFIX, FIELDS, SOFT_ISSUES, format_cut_time, parse_cut_frames,
    check_caption, normalize_caption, describe_issue,
)

SYSTEM_PROMPT = """You are a meticulous annotator of video clips for training MiniMax-H3, a model that generates video and audio together. Watch the clip, listen to its soundtrack, and write the caption H3 should be trained with: a faithful description of exactly what is seen and heard.

""" + H3_RULES

# 第 1 遍：只听音频转写人声。和描述分开做，模型不会因为画面里有人就编台词
ASR_PROMPT = """Transcribe all human speech and singing in this audio verbatim, in the original language and script, without translating. Put each utterance on its own line, starting with its language in square brackets, for example:
[Chinese] 你先回去吧。
[English] Wait for me!
Write [unclear] for words you cannot make out. Use basic punctuation and end every sentence with a full stop, question mark or exclamation mark (。？！ for Chinese and Japanese). Do not describe music, noise, ambience or animal sounds.
If there is no human speech or singing at all, output exactly: NONE"""

# ─── 读片段 ───────────────────────────────────────────────────────────────────
def probe_clip(path):
    """返回 (时长秒, 是否有音轨, 帧率)。"""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    info = json.loads(r.stdout)
    streams = info.get("streams", [])
    video = next(s for s in streams if s.get("codec_type") == "video")
    num, den = (video.get("r_frame_rate") or "0/1").split("/")
    fps = int(num) / max(int(den), 1)
    nb = video.get("nb_frames")
    if nb and str(nb).isdigit() and fps > 0:
        duration = int(nb) / fps
    else:
        duration = float(info.get("format", {}).get("duration") or 0)
    return duration, any(s.get("codec_type") == "audio" for s in streams), fps


def load_frames(path, fps, long_side):
    """按 fps 抽帧并缩放到长边 long_side，返回 (T, H, W, 3) uint8；T 取偶数（视觉编码器按 2 帧一组）。"""
    vf = (f"fps={fps},scale='if(gte(iw,ih),{long_side},-2)':'if(gte(iw,ih),-2,{long_side})'")
    probe = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf, "-frames:v", "1", "-f", "image2pipe",
         "-vcodec", "ppm", "-"], capture_output=True, timeout=120, check=True,
    ).stdout
    w, h = map(int, probe.split(b"\n")[1].split())  # ppm 头：P6\nW H\n255\n
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, timeout=300, check=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    if len(frames) % 2:
        frames = frames[:-1]
    return frames


def load_audio(path, sr=16000):
    """单声道 16 kHz float32（Qwen3-Omni 音频编码器的输入格式）。"""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True, timeout=300, check=True,
    ).stdout
    return np.frombuffer(raw, np.float32).copy()


# ─── 模型 ────────────────────────────────────────────────────────────────────
class OmniCaptioner:
    """transformers 后端：进程内加载 Qwen3-Omni（单卡单路，慢；大批量用 VLLMCaptioner）。"""

    def __init__(self, model_path, device="cuda"):
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self.torch = torch
        # enable_audio_output=False：不加载说话用的 talker / code2wav，省约 5 GB 显存
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device,
            attn_implementation=args.attn_implementation, enable_audio_output=False,
        ).eval()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
        self._media = {}

    def prepare(self, path, has_audio):
        """解码片段：抽帧 + 16 kHz 音频（重写时复用）。返回 (帧, 音频或 None)。"""
        if path not in self._media:
            frames = load_frames(path, args.fps, args.max_long_side)
            audio = load_audio(path) if has_audio else None
            if audio is not None and len(audio) < 1600:  # 不到 0.1 秒，当作无音轨
                audio = None
            self._media = {path: (frames, audio)}
        return self._media[path]

    def release(self, path):
        self._media.pop(path, None)

    def generate(self, system, user_text, path, with_video, with_audio, prefix="", max_new_tokens=1024,
                 repetition_penalty=1.0, temperature=0.0):
        frames, audio = self.prepare(path, with_audio)
        content = ([{"type": "video", "video": str(path)}] if with_video else []) + \
                  ([{"type": "audio", "audio": "clip"}] if with_audio and not with_video else []) + \
                  [{"type": "text", "text": user_text}]
        conversation = ([{"role": "system", "content": [{"type": "text", "text": system}]}] if system else []) + \
                       [{"role": "user", "content": content}]
        use_audio_in_video = with_video and with_audio
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False) + prefix
        inputs = self.processor(
            text=text, videos=[frames] if with_video else None, audio=[audio] if with_audio else None,
            fps=float(args.fps), use_audio_in_video=use_audio_in_video, return_tensors="pt", padding=True,
        ).to(self.model.device)
        for k, v in inputs.items():
            if self.torch.is_floating_point(v):
                inputs[k] = v.to(self.model.dtype)
        with self.torch.inference_mode():
            out = self.model.generate(
                **inputs, use_audio_in_video=use_audio_in_video, return_audio=False,
                thinker_max_new_tokens=max_new_tokens, thinker_do_sample=temperature > 0,
                **({"thinker_temperature": temperature} if temperature > 0 else {}),
                thinker_repetition_penalty=repetition_penalty,
            )
        if isinstance(out, tuple):  # 不出语音时返回 (thinker 结果, None)
            out = out[0]
        seq = out.sequences if hasattr(out, "sequences") else out
        new_tokens = seq[:, inputs["input_ids"].shape[1]:]
        return prefix + self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


class VLLMCaptioner:
    """vLLM 后端：向 `vllm serve` 起的 Qwen3-Omni 发 OpenAI 格式请求，可多线程并发（服务端自动拼批）。

    视频和音频分开传（缩小后的无声 mp4 + 16 kHz wav），同 LTX-2 打标器：vLLM 的 use_audio_in_video 走不通。
    回答开头用 continue_final_message 预填，和 transformers 后端一致。
    """

    def __init__(self, server_url, served_model, media_dir):
        import requests
        self.session = requests.Session()
        self.url = server_url.rstrip("/") + "/v1/chat/completions"
        self.model = served_model
        self.media_dir = Path(media_dir)
        self.media_dir.mkdir(parents=True, exist_ok=True)

    def prepare(self, path, has_audio):
        """写出缩小后的无声 mp4 和 16 kHz 单声道 wav，返回 (视频路径, 音频路径或 None)。"""
        stem = self.media_dir / unique_output_name(path, suffix="")
        video, wav = stem.with_suffix(".mp4"), stem.with_suffix(".wav")
        scale = (f"scale='if(gte(iw,ih),{args.max_long_side},-2)':'if(gte(iw,ih),-2,{args.max_long_side})'")
        if not video.exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vf", scale, "-an", "-c:v", "libx264",
                            "-crf", "20", "-pix_fmt", "yuv420p", str(video)], check=True, timeout=300)
        if has_audio and not wav.exists():
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                            str(wav)], check=True, timeout=300)
        has_audio = has_audio and wav.exists() and wav.stat().st_size > 44 + 3200  # 不到 0.1 秒当作无音轨
        return video, (wav if has_audio else None)

    def release(self, path):
        stem = self.media_dir / unique_output_name(path, suffix="")
        for p in (stem.with_suffix(".mp4"), stem.with_suffix(".wav")):
            p.unlink(missing_ok=True)

    def generate(self, system, user_text, path, with_video, with_audio, prefix="", max_new_tokens=1024,
                 repetition_penalty=1.0, temperature=0.0):
        video, wav = self.prepare(path, with_audio)
        content = []
        if with_video:
            content.append({"type": "video_url", "video_url": {"url": f"file://{video}"}})
        if with_audio and wav is not None:
            content.append({"type": "audio_url", "audio_url": {"url": f"file://{wav}"}})
        content.append({"type": "text", "text": user_text})
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
        body = {"model": self.model, "messages": messages, "max_tokens": max_new_tokens, "temperature": temperature,
                "repetition_penalty": repetition_penalty, "mm_processor_kwargs": {"fps": args.fps}}
        if prefix:
            body.update(messages=messages + [{"role": "assistant", "content": prefix}],
                        continue_final_message=True, add_generation_prompt=False)
        r = self.session.post(self.url, json=body, timeout=600)
        r.raise_for_status()
        return prefix + (r.json()["choices"][0]["message"]["content"] or "")


def transcribe(captioner, path):
    """第 1 遍：只听音频，返回人声转写的行列表；没有人声返回 []。"""
    text = captioner.generate(None, ASR_PROMPT, path, with_video=False, with_audio=True, max_new_tokens=512).strip()
    if not text or text.upper().startswith("NONE"):
        return []
    lines, tag = [], ""
    for line in (l.strip() for l in text.splitlines()):
        if not line or line.upper() == "NONE":
            continue
        if re.fullmatch(r"\[[A-Za-z-]+\]", line):  # 语言标签单独一行，属于下一行
            tag = line
            continue
        m = re.match(r"\[[A-Za-z-]+\]", line)
        if m:
            tag = m.group(0)  # 记住最近的语言标签：模型常常只在第一句标语言
        elif tag:
            line = f"{tag} {line}"
        lines.append(line)
    return lines


def caption_clip(captioner, path, cut_frames=(), dialogue=None):
    """返回 (描述, 对白行列表；无音轨时为 None, 切点时间列表, 是否补了对白)。

    dialogue：原片标注里这段的对白（annotation_utils.clip_dialogue）；给了就不跑转写那一遍，直接用标注的
    原文和说话人。
    """
    duration, has_audio, fps = probe_clip(path)
    cut_times = [format_cut_time(c / fps) for c in cut_frames] if fps > 0 else []
    if cut_times:
        shot_hint = (f"The clip has {len(cut_times) + 1} shots separated by hard cuts at "
                     + ", ".join(cut_times) + ". Write [Shot 1] to [Shot " + str(len(cut_times) + 1)
                     + "], beginning each later shot with its cut time, for example \"[Shot 2] At "
                     + cut_times[0] + ", the camera cuts to\".")
    else:
        shot_hint = "The clip is one continuous shot: write only [Shot 1]."
    try:
        _, audio = captioner.prepare(path, has_audio)
        has_audio = audio is not None
        if not has_audio:
            transcript = None
            hint = ("The clip has no audio track: write N/A for overall_soundscape and non_diegetic_music, "
                    "and write no <d> block.")
        elif dialogue is not None:
            transcript = [x["d"][3:-4] for x in dialogue]
            legend = {}
            for x in dialogue:
                legend.setdefault(x["sid"], x["who"])
            hint = ("Speakers (from a human transcript): " + "; ".join(f"{k} = {v}" for k, v in legend.items())
                    + ". Their lines, in order:\n" + "\n".join(f"{x['sid']}: {x['d']}" for x in dialogue)
                    + "\nWork every line into integrated_multimodal_description at the moment it is spoken, copying "
                      "the <d> block exactly, in a natural sentence such as: The woman in the grey shirt (S2) "
                      "says: <d>[English] I'm fine.</d>") if dialogue else \
                   "Nobody speaks or sings in this clip, so write no <d> block."
        else:
            transcript = transcribe(captioner, path)
            hint = ("This clip contains speech. integrated_multimodal_description MUST include each of these "
                    "dialogue blocks, copied exactly and in this order, each attributed to the person who says it "
                    "with a speaker ID:\n" + "\n".join(f"<d>{line}</d>" for line in transcript)) if transcript else \
                   "No speech or singing was detected in this clip's audio, so write no <d> block."
        user_text = f"Write the caption for this {duration:.2f}-second clip. {shot_hint} {hint}"
        caption = ""
        for attempt in range(2):  # 不合格时把问题告诉模型重写一次
            caption = normalize_caption(captioner.generate(
                SYSTEM_PROMPT, user_text, path, with_video=True, with_audio=has_audio, prefix=CAPTION_PREFIX,
                max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty,
                temperature=0.3 * attempt))  # 重写时加一点随机性，避免和第一次一样陷进重复
            issues, _ = check_caption(caption, has_speech=bool(transcript), cut_times=cut_times)
            if not issues or attempt == 1:
                break
            user_text += "\nYour previous attempt broke these rules, fix them: " + "; ".join(map(describe_issue, issues)) + "."
    finally:
        captioner.release(path)
    appended = False
    if transcript and "<d>" not in caption:
        # 重写后仍漏对白：在画面描述末尾补一句不指认说话人的对白（记 speech_appended 警告）
        said = " ".join(f"<d>{line}</d>" for line in transcript)
        caption = re.sub(r"(integrated_multimodal_description:.*?)(\n\noverall_soundscape:)",
                         lambda m: f"{m.group(1)} A voice (S1) says: {said}{m.group(2)}", caption, count=1, flags=re.S)
        appended = True
    return caption, transcript, cut_times, appended


def clip_annotation_dialogue(item):
    """片段对应的原片标注对白；没有标注（或没开 --annotation_suffix）返回 None。"""
    if not args.annotation_suffix:
        return None
    ann = load_annotation(item.get("source_path"), args.annotation_suffix)
    try:
        start, end = float(item["clip_start_sec"]), float(item["clip_end_sec"])
    except (KeyError, TypeError, ValueError):
        return None
    return None if ann is None else clip_dialogue(ann, start, end)


def process_clip(captioner, item):
    row = dict(item)
    row.update({k: "" for k in CAPTION_FIELDS})
    path = item.get(args.video_path_column, "")
    if not path or not Path(path).is_file():
        row["caption_reject_reason"] = "file_not_found"
        return row
    try:
        caption, transcript, cut_times, appended = caption_clip(
            captioner, path, parse_cut_frames(item.get("cut_frames")), clip_annotation_dialogue(item))
    except Exception as e:
        row["caption_reject_reason"] = f"caption_failed:{type(e).__name__}:{str(e)[:200]}"
        print(f"  [失败] {path}: {e}")
        return row
    issues, stats = check_caption(caption, has_speech=bool(transcript), cut_times=cut_times)
    warnings = [i for i in issues if i in SOFT_ISSUES] + (["speech_appended"] if appended else [])
    issues = [i for i in issues if i not in SOFT_ISSUES]
    row["caption_warnings"] = ",".join(warnings)
    row["caption"] = caption
    row["caption_asr"] = "NO_AUDIO" if transcript is None else (" | ".join(transcript) or "NONE")
    row.update(stats)
    row["caption_reject_reason"] = ",".join(issues)
    print(f"  {'不通过' if issues else '通过'}: {Path(path).name} | {stats.get('caption_words', 0)} words"
          + (f" | {','.join(issues)}" if issues else ""), flush=True)
    return row


class Worker:
    def __init__(self):
        worker_syspath_setup()  # Ray worker 端把 data_process/ 加进 sys.path
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_csv = self.output_dir / "6_caption_all.csv"
        self.qual_csv = self.output_dir / "6_caption_qualified.csv"
        with open(args.input_csv, encoding="utf-8") as f:
            input_fields = csv.DictReader(f).fieldnames or []
        self.fieldnames = list(input_fields) + [c for c in CAPTION_FIELDS if c not in input_fields]
        # 以文件实际表头追加（旧 CSV 缺列时原地补列）；all CSV 额外记录输入路径作断点续跑 key
        self.csv_fields = {
            self.all_csv: init_csv(self.all_csv, self.fieldnames + [INPUT_PATH_FIELD]),
            self.qual_csv: init_csv(self.qual_csv, self.fieldnames),
        }
        self.processed = load_processed_inputs(self.all_csv)
        if args.backend == "vllm":
            print(f"使用 vLLM 服务: {args.server_url}（{args.served_model}）")
            self.captioner = VLLMCaptioner(args.server_url, args.served_model, self.output_dir / "_media")
        else:
            print(f"加载 Qwen3-Omni: {args.model_path}")
            self.captioner = OmniCaptioner(args.model_path)
        self._lock = threading.Lock()
        self.obs_io = ObsIO(args)
        print(f"模型加载完成. OBS={self.obs_io.enabled}. 已处理 {len(self.processed)} 条。")

    def _append_csv(self, filepath, row):
        with self._lock, open(filepath, "a", newline="", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            csv.DictWriter(f, fieldnames=self.csv_fields[filepath], extrasaction="ignore").writerow(row)
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)

    def __call__(self, item):
        clip_path = item.get(args.video_path_column, "")
        if not clip_path or clip_path in self.processed:
            row = dict(item)
            row.update({k: "" for k in CAPTION_FIELDS})
            return row
        local_video = None
        try:
            # ── OBS: 下载片段（描述写进 CSV，片段本身不再上传）──
            if self.obs_io.enabled:
                local_video = self.obs_io.download(clip_path)
                item = {**item, args.video_path_column: local_video}
            else:
                local_video = clip_path
            row = process_clip(self.captioner, item)
            row[args.video_path_column] = clip_path
            if row["caption_reject_reason"].startswith("caption_failed"):
                return row  # 推理出错（OOM 等）不记入 CSV，续跑时重试
            self._append_csv(self.all_csv, {**row, INPUT_PATH_FIELD: clip_path})
            if not row.get("caption_reject_reason"):
                self._append_csv(self.qual_csv, row)
            self.processed.add(clip_path)
            return row
        finally:
            if self.obs_io.enabled and local_video and local_video != clip_path:
                self.obs_io.cleanup(local_video)


# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Qwen3-Omni 生成 MiniMax-H3 格式的片段描述")
parser.add_argument("--input_csv", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--video_path_column", default="video_path")
parser.add_argument("--model_path", default=str(env_or("QWEN3_OMNI_PATH", ckps_root() / "Qwen3-Omni-30B-A3B-Instruct")))
parser.add_argument("--fps", type=float, default=2.0, help="送给模型的抽帧率")
parser.add_argument("--max_long_side", type=int, default=768, help="送给模型的帧长边像素")
parser.add_argument("--max_new_tokens", type=int, default=1024)
parser.add_argument("--repetition_penalty", type=float, default=1.05)
parser.add_argument("--attn_implementation", default="flash_attention_2", choices=["flash_attention_2", "sdpa"])
parser.add_argument("--backend", default="transformers", choices=["transformers", "vllm"],
                    help="vllm：连 script/serve_omni.sh 起的服务，--workers 路并发；transformers：进程内单路")
parser.add_argument("--server_url", default="http://127.0.0.1:8901")
parser.add_argument("--served_model", default="qwen3-omni")
parser.add_argument("--workers", type=int, default=16, help="vllm 后端的并发请求数")
parser.add_argument("--annotation_suffix", default="",
                    help="原片带标注时（如 CineDance 填 .json），对白直接用标注的原文和说话人，不跑转写那一遍；"
                         "需要输入 CSV 有 source_path / clip_start_sec / clip_end_sec（Step 2 输出）")
add_obs_args(parser)
parser.add_argument("--ray", action="store_true", help="启用 Ray 分布式模式")
parser.add_argument("--ray_num_gpus", type=float, default=1.0, help="30B 模型 bf16 约 62 GB，一卡一个 actor")
parser.add_argument("--ray_num_cpus", type=float, default=4)
parser.add_argument("--concurrency", type=int, default=1)
args = parser.parse_args()


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv(args.input_csv, encoding="utf-8").drop_duplicates(subset=[args.video_path_column])

    if not args.ray:
        samples = df.fillna("").to_dict("records")
        total = len(samples)
        print(f"读取 {total} 条记录 from {args.input_csv}")
        worker = Worker()
        if args.backend == "vllm" and args.workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(args.workers) as pool:
                list(pool.map(worker, samples))
        else:
            for i, item in enumerate(samples, 1):
                print(f"[{i}/{total}] {item.get(args.video_path_column, '')}")
                worker(item)
        print("完成")
    else:
        import ray
        ray.init(address="auto", runtime_env={"env_vars": {
            "PYTHONPATH": str(_FILTER_ROOT), "DATA_PROCESS_ROOT": str(_FILTER_ROOT)}})
        from ray.data import ActorPoolStrategy
        samples = ray.data.from_pandas(df.fillna("")).materialize()
        target_parallelism = min(args.concurrency, samples.count())
        if samples.num_blocks() < target_parallelism:
            samples = samples.repartition(target_parallelism)
        samples.map(
            Worker, num_gpus=args.ray_num_gpus, num_cpus=max(args.ray_num_cpus, 1),
            compute=ActorPoolStrategy(size=target_parallelism),
        ).take_all()
        print("Ray 完成")
