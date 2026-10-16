"""MiniMax-H3 训练描述的格式：写作规则、格式检查和整理。Step 6 的两种打标方式（看视频 / 读原片标注）共用。

规则按官方写 prompt 指南（skills/h3-prompt-writing/references/base-en.txt）改写成"描述看到和听到的"；
检查按团队校验器 verifyH3Prompt.ts 的 error 级规则，加上镜头数 / 切点时间、全英文、对白与人声转写一致等几条。
"""
import re

# 三个字段的写法、对白、配乐等规则（不含开头那句"你在看什么"，由调用方各自加）
H3_RULES = """Output exactly these three fields, in this order, separated by one blank line, and nothing else (no Markdown, no headings, no notes):
integrated_multimodal_description: [Shot 1] ...
overall_soundscape: ...
non_diegetic_music: ...

integrated_multimodal_description
- You will be told whether the clip is one continuous shot or where its hard cuts are. Write one [Shot k] section per shot, numbered from 1. [Shot 1] has no timestamp; every later shot begins with exactly the cut time you are given, for example "[Shot 2] At 00:02.375, the camera cuts to a close-up of ...". Never mention the clip's length or count seconds.
- Begin with the visual style and the initial composition, for example "[Shot 1] Live-action, cinematic, a medium close-up frames ...". Judge the style from how the images are rendered: real camera footage is live-action; otherwise name the medium, such as hand-drawn 2D anime, 3D CG animation, painterly digital illustration, claymation, watercolor or vintage film. Characters drawn with outlines and flat or cel shading over painted backgrounds are hand-drawn 2D anime, not 3D CG.
- Then follow the clip in time order and describe only concrete, observable things: the subjects (apparent age, gender, build, hair, clothing, position in the frame), the setting, key objects, lighting, actions and reactions, camera behavior, speech, singing, visible on-screen text, and sounds synchronized with on-screen events.
- Describe camera behavior with these terms inside a natural sentence: static shot, push in / pull out, pan left / pan right, truck left / truck right, tilt up / tilt down, pedestal up / pedestal down, arc shot, tracking shot, zoom in / zoom out, shake slightly / shake strongly, POV, roll clockwise / roll counterclockwise. Add "with small amplitude" or "with large amplitude" and "at slow speed" or "at fast speed" only when clearly noticeable. Example: "The camera pushes in with small amplitude at slow speed toward her face."
- Refer to people by appearance, never by the name of an actor or a character.
- Be detailed and specific: usually 150-250 words.

Speech and singing
- Give every person who speaks or sings a stable ID in order of first vocalization: (S1), (S2), and so on. People who never vocalize get no ID.
- When a speaker first appears, identify them from what is visible and audible: on-screen or off-screen, apparent gender and age, voice pitch, timbre, pace and accent.
- Put the speaker phrase, ID, action and delivery outside <d>. Inside <d> put only a language tag and the exact words heard, in the original language and script, without translating or correcting them, for example: The young woman with a soft, breathy voice (S1) says: <d>[Chinese] 你先回去吧。</d>
- Write [unclear] for words you cannot make out. Use only basic punctuation, and end every complete sentence with . ? or !
- If speech is cut off by the end of the clip, put <cutoff> right after the last audible word, inside the block: <d>[Chinese] 我跟你说过<cutoff></d>
- If one line of speech continues across a cut, put <scenetrans> at the connecting points in both parts and state that the audio continues across the cut.
- A voice from someone outside the frame: An off-screen male voice (S2) calls out: <d>[English] Wait for me!</d>
- Narration or inner monologue by an on-screen person whose lips do not move: The man (S1) says in an off-screen voiceover: <d>[English] I still remember that road.</d> while his lips remain completely closed.
- If nobody speaks or sings, write no <d> block. Never invent words.

Visible text: put signs, labels or other text actually visible on screen in English double quotation marks, verbatim, for example a sign reading "营业中".

overall_soundscape
- 1-4 English sentences in one paragraph about the ambient sound, the sounds of physical actions, and non-verbal human sounds such as breathing, laughter or footsteps that are actually audible. Do not repeat speech or singing here, and do not mention music.
- If the clip has no sound at all, write N/A.

non_diegetic_music
- 1-3 English sentences about background music the characters cannot hear: instruments, tempo, rhythm and changes in loudness, for example "Sparse piano notes at a slow tempo, joined by sustained low strings that gradually increase in volume before fading out." Never use mood or atmosphere words such as melancholic, somber, mysterious, uplifting or tense.
- Music the characters can hear (a radio, someone playing or singing on screen) belongs in integrated_multimodal_description instead.
- If there is no background music, write N/A.

A complete example for a clip without speech:
integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a middle-aged baker in a flour-dusted white apron standing behind the wooden counter of a small street bakery before sunrise, warm tungsten light spilling from the shelves behind him. He lifts a fresh round loaf from a metal tray, turns it once in his hands to inspect the crust, and sets it in a wicker basket beside the till. A thin wisp of steam rises from the bread as he wipes his palms on the apron and glances toward the dark street outside the shop window. The camera pushes in with small amplitude at slow speed toward the counter while the baker straightens the basket and steps back.

overall_soundscape: Metal trays clink softly against the counter while a low refrigerator hum fills the room. The crust crackles faintly as he turns the loaf, followed by the dry rustle of the wicker basket and his quiet footsteps on the tiled floor.

non_diegetic_music: N/A"""

CAPTION_PREFIX = "integrated_multimodal_description: [Shot 1]"  # 预填回答开头，固定输出格式

FIELDS = ["integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"]
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_SECOND_COUNT = re.compile(
    r"\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen)[\s-]?seconds?\b", re.I)


def format_cut_time(seconds):
    """秒 → H3 的切镜时间格式 MM:SS.mmm。"""
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def parse_cut_frames(value):
    """Step 2 的 cut_frames 列（"52;97"）→ [52, 97]。空值、NaN 返回 []；整列只有单个切点时 pandas 会读成数字。"""
    if isinstance(value, (int, float)):
        return [] if value != value else [int(value)]
    if not isinstance(value, str) or not value.strip():
        return []
    return [int(float(x)) for x in value.split(";") if x.strip()]


def _count_sentences(text):
    text = text.strip()
    return len([s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]) if text else 0


_MOOD_WORDS = re.compile(
    r"\b(?:melanchol\w*|somber|sombre|mysterious|serene|contemplative|reflective|calm(?:ing)?|peaceful|tranquil|"
    r"uplifting|tense|suspenseful|emotional|joyful|sad|happy|hopeful|nostalgic|romantic|eerie|ominous|mood|"
    r"atmosphere|atmospheric|feeling)\b", re.I)
SOFT_ISSUES = ("music_mood_words",)  # 只警告不拒：重写一次仍有则保留描述，记进 caption_warnings

# 重写时告诉模型的问题说明（检查项代号模型看不懂）
ISSUE_HINTS = {
    "music_mood_words": "non_diegetic_music may only describe instruments, tempo, rhythm and loudness; "
                        "remove every mood or atmosphere word and any clause about the effect of the music",
    "speech_missing": "the dialogue blocks given above are missing from integrated_multimodal_description",
    "dialogue_without_speech": "nobody speaks in this clip, so remove every <d> block",
    "dialogue_outside_description": "dialogue belongs inside integrated_multimodal_description, "
                                    "not in overall_soundscape or non_diegetic_music",
    "multi_shot": "the clip is a single shot, so write only [Shot 1]",
    "shot_count": "write exactly one [Shot k] section per shot, as many as the cuts given above plus one",
    "cut_time_mismatch": "begin each later shot with exactly the cut times given above, like [Shot 2] At 00:02.375,",
    "second_count": "do not mention the clip length or count seconds",
    "non_english_description": "write everything in English except the words inside <d> blocks and quoted on-screen text",
    # 按代号前缀匹配（soundscape_sentences_5、field_missing:… 等）
    "soundscape": "overall_soundscape must be 1-4 sentences in one paragraph",
    "music": "non_diegetic_music must be 1-3 sentences, or N/A",
    "field": "output all three fields, each starting with its name and a colon",
    "too": "write a more detailed integrated_multimodal_description",
}


def describe_issue(code):
    return ISSUE_HINTS.get(code) or ISSUE_HINTS.get(re.split(r"[_:]", code)[0], code.replace("_", " "))


def check_caption(text, has_speech=True, cut_times=None):
    """按 verifyH3Prompt.ts 的格式规则（error 级）检查，另加单镜头、描述须为英文、没人声不许有对白三条。
    返回 (问题列表, 统计)。"""
    issues = []
    at = [m.start() if (m := re.search(rf"^\s*{f}\s*:", text, re.M)) else -1 for f in FIELDS]
    for f, i in zip(FIELDS, at):
        if i < 0:
            issues.append(f"field_missing:{f}")
    if issues:
        return issues, {}
    if not (at[0] < at[1] < at[2]):
        return ["field_order"], {}

    def section(a, b):
        return re.sub(r"^\s*\w+\s*:", "", text[a:b] if b >= 0 else text[a:]).strip()

    imd, soundscape, music = section(at[0], at[1]), section(at[1], at[2]), section(at[2], -1)

    if not imd.startswith("[Shot 1]"):
        issues.append("no_shot1")
    shots = sorted({int(x) for x in re.findall(r"\[Shot (\d+)\]", imd)})
    if cut_times:
        if shots != list(range(1, len(cut_times) + 2)):
            issues.append("shot_count")
        elif [t for _, t in re.findall(r"\[Shot (\d+)\] At (\d{2}:\d{2}\.\d{3})", imd)] != list(cut_times):
            issues.append("cut_time_mismatch")
    elif any(k > 1 for k in shots):
        issues.append("multi_shot")  # 单镜头片段写出了切镜，模型看错了
    if re.search(r"\[Shot 1\]\s*At\s", imd):
        issues.append("shot1_timestamp")
    if _SECOND_COUNT.search(imd):
        issues.append("second_count")

    ss_n = _count_sentences(soundscape)
    if soundscape != "N/A" and not 1 <= ss_n <= 4:
        issues.append(f"soundscape_sentences_{ss_n}")
    mu_n = _count_sentences(music)
    if music != "N/A" and not 1 <= mu_n <= 3:
        issues.append(f"music_sentences_{mu_n}")
    if music != "N/A" and _MOOD_WORDS.search(music):
        issues.append("music_mood_words")

    blocks = re.findall(r"<d>(.*?)</d>", text, re.S)
    if blocks and not has_speech:
        issues.append("dialogue_without_speech")  # 转写没听到人声，描述里的对白是编的
    if has_speech and not blocks:
        issues.append("speech_missing")  # 转写有人声，描述漏了对白
    if "<d>" in soundscape or "<d>" in music:
        issues.append("dialogue_outside_description")
    if text.count("<d>") != text.count("</d>"):
        issues.append("dialogue_unbalanced")
    for d in blocks:
        if not re.match(r"\s*\[[A-Za-z-]+\]", d):
            issues.append("dialogue_no_language_tag")
        if re.search(r"\(S\d+(?:,S\d+)*\)", d):
            issues.append("dialogue_speaker_inside")
        spoken = re.sub(r"[.!?,]", "", re.sub(r"^\s*\[[A-Za-z-]+\]\s*", "", d)).strip()
        if len(spoken) > 8 and spoken in re.sub(r"[.!?,]", "", soundscape):
            issues.append("dialogue_in_soundscape")
    for m in re.finditer(r"says in an off-screen voiceover[^<]*<d>.*?</d>([^.]*)", text, re.S):
        if not re.search(r"lips remain", m.group(1), re.I):
            issues.append("voiceover_no_lips")

    # 对白和画面文字之外都应是英文
    outside = re.sub(r'<d>.*?</d>|"[^"]*"|“[^”]*”', "", text, flags=re.S)
    if _CJK.search(outside):
        issues.append("non_english_description")

    words = len(re.sub(r"<d>.*?</d>", " ", imd, flags=re.S).split())
    if words < 40:
        issues.append(f"too_short_{words}words")
    stats = {
        "caption_words": words,
        "caption_speakers": len(set(re.findall(r"\(S(\d+)\)", imd))),
        "caption_has_cutoff": "<cutoff>" in text,
    }
    return sorted(set(issues)), stats


_PROMPT_HEADINGS = re.compile(r"^\s*(?:speech and singing|visible text)\s*:?\s*$", re.I | re.M)


def normalize_caption(text):
    """去掉代码块标记和模型抄来的系统提示小标题；每个字段合成一段，字段之间统一一个空行。"""
    text = re.sub(r"^```\w*\s*|\s*```$", "", text.strip())
    text = _PROMPT_HEADINGS.sub("", text)
    for i, f in enumerate(FIELDS):  # 字段名大小写不对、或后两个字段没换行（"… Overall_soundscape: …"）时改回来
        text = re.sub(rf"(?im)(^|\s){f}\s*:", lambda m: (m.group(1) if i == 0 else "\n\n") + f + ":", text)
    for f in FIELDS:
        text = re.sub(rf"^({f})\s*\n+", r"\1: ", text, flags=re.M)  # 字段名单独一行、没冒号
    parts = re.split(rf"^\s*({'|'.join(FIELDS)})\s*:", text, flags=re.M)
    if len(parts) < 3:
        return text.strip()
    fields = {name: " ".join(body.split()) for name, body in zip(parts[1::2], parts[2::2])}
    # 模型偶尔把对白句写到环境声 / 配乐字段后面（"N/A The woman (S1) says: <d>…</d>"）：挪回描述末尾
    for name in FIELDS[1:]:
        if FIELDS[0] in fields and "<d>" in fields.get(name, ""):
            sents = _split_sentences(fields[name])
            fields[FIELDS[0]] += " " + " ".join(x for x in sents if "<d>" in x)
            fields[name] = " ".join(x for x in sents if "<d>" not in x) or "N/A"
    return "\n\n".join(f"{name}: {body}" for name, body in fields.items())


def _split_sentences(text):
    """按句末标点分句；<d>…</d> 里的标点不算。开头的 N/A 单独算一句。"""
    blocks = []

    def hide(m):
        blocks.append(m.group(0))
        return f"\x00{len(blocks) - 1}\x00"

    masked = re.sub(r"<d>.*?</d>", hide, text, flags=re.S)
    head = []
    if masked.startswith("N/A "):
        head, masked = ["N/A"], masked[4:]
    sents = head + [x for x in re.split(r"(?<=[.!?])\s+", masked) if x.strip()]
    return [re.sub(r"\x00(\d+)\x00", lambda m: blocks[int(m.group(1))], x) for x in sents]


