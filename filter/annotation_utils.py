"""读原片自带的结构化标注（CineDance 格式：原片旁边同名 .json）。Step 6 两个打标脚本共用。"""
import json
import re
from pathlib import Path

_KANA = re.compile(r"[぀-ヿ]")
_HANGUL = re.compile(r"[가-힯]")
_HAN = re.compile(r"[一-鿿]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def guess_language(text):
    """对白语言标签：按文字判断（够 CineDance 的中英日韩俄；拉丁字母一律当英文）。"""
    if _KANA.search(text):
        return "Japanese"
    if _HANGUL.search(text):
        return "Korean"
    if _HAN.search(text):
        return "Chinese"
    if _CYRILLIC.search(text):
        return "Russian"
    return "English"


def load_annotation(source_path, suffix=".json"):
    """原片标注 = 原片路径换后缀；没有返回 None。"""
    if not source_path or not suffix:
        return None
    p = Path(str(source_path)).with_suffix(suffix)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def character_labels(ann):
    """<charN> → 简短身份描述。"""
    return {c["id"]: c.get("identity_label") or c.get("description", "")[:80] for c in ann.get("characters", [])}


def clip_dialogue(ann, start, end):
    """片段 [start, end) 秒内的对白：至少一半落在片段里的句子；越过片段结尾超过 0.3 秒的末尾加 <cutoff>。

    返回 [{"sid": "S1", "who": 身份描述, "d": "<d>[语言] 原文</d>", "t": 在片段里的开始秒}]，按时间顺序；
    同一人物（char，没有就用 speaker）在片段内用同一个 ID。
    """
    chars = character_labels(ann)
    speakers, out = {}, []
    for t in ann.get("transcript", []):
        a, b = t["start_sec"], t["end_sec"]
        inside = min(b, end) - max(a, start)
        if b <= a or inside < 0.5 * (b - a):
            continue
        text = t["text"].strip()
        if not text:
            continue
        key = t.get("char") or t.get("speaker") or "?"
        sid = speakers.setdefault(key, f"S{len(speakers) + 1}")
        cut = "<cutoff>" if b > end + 0.3 else ""
        out.append({"sid": sid, "who": chars.get(t.get("char"), "an unseen speaker"),
                    "d": f"<d>[{guess_language(text)}] {text}{cut}</d>", "t": max(a - start, 0.0)})
    return out
