#!/usr/bin/env bash
#
# 视频数据处理全流程 Pipeline (支持 Ray 分布式)
#
# 流程: 1_video_filter → 2_cutShot → 3_motion_filter → 4_quality_filter → 5_lipsync → 6_caption
#
# 用法:
#   bash filter/run_pipeline.sh --input_dir /path/to/videos --output_root /path/to/output
#   bash filter/run_pipeline.sh --input_dir /path/to/videos --output_root /path/to/output --use_ray
#   bash filter/run_pipeline.sh --input_csv /path/to/existing.csv --output_root /path/to/output
#
# 可选参数:
#   --input_dir DIR        视频文件目录（自动生成 CSV）
#   --input_csv CSV        已有输入 CSV（跳过目录扫描）
#   --output_root DIR      输出根目录
#   --use_ray              启用 Ray 分布式模式
#   --steps "1,2,3,4,5"   指定运行哪些步骤（默认全部）
#   --skip_to N            从第 N 步开始（跳过之前的步骤，使用已有输出）
#
set -euo pipefail

# ─── 基础参数 ───────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_DIR=""
INPUT_CSV=""
OUTPUT_ROOT=""
USE_RAY=false
# 默认针对 MiniMax-H3 通用镜头数据：Step 5（唇形同步，面向说话人数据）默认不跑，需要时 --steps 1,2,3,4,5
STEPS="1,2,3,4"
SKIP_TO=0

# ─── GPU / 并发 ──────────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-8}"
RAY_CONCURRENCY_STEP1="${RAY_CONCURRENCY_STEP1:-4}"
RAY_CONCURRENCY_STEP2="${RAY_CONCURRENCY_STEP2:-$((2 * NUM_GPUS))}"
RAY_CONCURRENCY_STEP3="${RAY_CONCURRENCY_STEP3:-$((2 * NUM_GPUS))}"
RAY_CONCURRENCY_STEP4="${RAY_CONCURRENCY_STEP4:-$((2 * NUM_GPUS))}"
RAY_CONCURRENCY_STEP5="${RAY_CONCURRENCY_STEP5:-$((1 * NUM_GPUS))}"
RAY_CONCURRENCY_STEP6="${RAY_CONCURRENCY_STEP6:-$((1 * NUM_GPUS))}"

# ─── Step 1: 视频质量过滤 + 转码 超参数 ──────────────────────────────────────
# 默认值按 MiniMax-H3 训练数据：24 fps、768p、片段 124 帧（17n+5）、音频 32 kHz 立体声
S1_MIN_RESOLUTION="${S1_MIN_RESOLUTION:-none}"         # 总像素门槛: none/360p/480p/720p/1080p 或像素数
S1_MIN_SHORT_SIDE="${S1_MIN_SHORT_SIDE:-768}"          # 短边门槛(像素)，0=关闭；不误杀 1920x800 宽银幕
S1_MIN_DURATION="${S1_MIN_DURATION:-124}"              # 最短帧数（至少能切出一个 124 帧片段）
S1_MAX_DURATION="${S1_MAX_DURATION:-0}"                # 最长帧数 (0=不限)
S1_TARGET_FPS="${S1_TARGET_FPS:-24}"                   # 目标帧率；23.5~24.5 fps 的源不转码
S1_TARGET_AUDIO_SR="${S1_TARGET_AUDIO_SR:-0}"          # 0=保留源采样率（Step 2 切片时统一到 32 kHz）
S1_COPY_MODE="${S1_COPY_MODE:-symlink}"                # 无需转码时复制模式: symlink/copy/none
S1_LETTERBOX_SAMPLES="${S1_LETTERBOX_SAMPLES:-12}"     # 黑边检测采样点数，0=不检测；Step 2 切片时按检测结果裁掉黑边
S1_RAY_NUM_CPUS="${S1_RAY_NUM_CPUS:-12}"               # 每个 worker 的 CPU 数

# ─── Step 2: 场景切分 超参数 ─────────────────────────────────────────────────
S2_MODE="${S2_MODE:-multi_shot}"                       # 切分模式: multi_shot（默认，片段可跨硬切）/ clean_shot（片段只取单个镜头内部）/ default
S2_NUM_CONTEXT_FRAMES="${S2_NUM_CONTEXT_FRAMES:-5}"    # 上下文帧数
S2_CLIP_FRAMES="${S2_CLIP_FRAMES:-124}"                # 片段帧数，H3 要求 17n+5（73/124/141/192…）
S2_TRIM_FRAMES="${S2_TRIM_FRAMES:-3}"                  # 每个镜头两端各裁掉的帧数
S2_AUDIO_SR="${S2_AUDIO_SR:-32000}"                    # 片段音频采样率（H3 音频 VAE 为 32 kHz），0=保留源
S2_AUDIO_CHANNELS="${S2_AUDIO_CHANNELS:-2}"            # 片段音频声道（H3 为立体声），0=保留源
S2_RAY_NUM_GPUS="${S2_RAY_NUM_GPUS:-0.5}"              # 每个 worker 的 GPU 份额

# ─── Step 3: RAFT 运动过滤 超参数 ────────────────────────────────────────────
# 运动分 = 长边缩到 512 后、相隔约 0.5 s（2 fps 采样）两帧间的平均光流位移（像素）
# 参考值: <0.5 静止, 0.5~2 talking head, 2~6 正常, 6~15 运动, >15 剧烈
# CineDance 实测（10-15，35 个片段）：只有嘴在动的对话镜头 0.04–0.4，手持动作戏 65–84；
# 平均光流分不开"对话特写"和"完全静止"，所以只拦明显异常，其余保留分数以后按需筛
S3_MIN_MOTION="${S3_MIN_MOTION:-0.02}"                 # 最低运动分：只去掉完全不动的画面
S3_MAX_MOTION="${S3_MAX_MOTION:-100}"                  # 最高运动分：只去掉极端乱晃、闪烁
S3_SAMPLING_FPS="${S3_SAMPLING_FPS:-2}"                # 采样帧率
S3_MAX_SIZE="${S3_MAX_SIZE:-512}"                      # 输入帧最大尺寸
S3_COPY_MODE="${S3_COPY_MODE:-symlink}"                # 通过时复制模式
S3_RAY_NUM_GPUS="${S3_RAY_NUM_GPUS:-0.5}"

# ─── Step 4: 质量评估过滤 超参数 ─────────────────────────────────────────────
# CineDance 实测（10-15，35 个片段）中位数 DOVER 0.35 / 美学 4.33 / KVQ 0.0065；原阈值 0.4/4.5/0.008 只剩 14%。
# 现在约取 p10：只拦每项最差的约 10%，分数都留在 CSV，训练时再按需筛
S4_MIN_DOVER="${S4_MIN_DOVER:-0.23}"                   # DOVER 总分阈值（原 0.4）
S4_MIN_AES="${S4_MIN_AES:-3.5}"                        # Aesthetic V2.5 均分阈值（原 4.5）
S4_MIN_KVQ="${S4_MIN_KVQ:--0.004}"                     # KVQ 分数阈值（原 0.008）
S4_AES_FRAMES="${S4_AES_FRAMES:-16}"                   # 美学评估采样帧数
S4_DISABLE_DOVER="${S4_DISABLE_DOVER:-false}"          # 禁用 DOVER
S4_DISABLE_AES="${S4_DISABLE_AES:-false}"              # 禁用 Aesthetic
S4_DISABLE_KVQ="${S4_DISABLE_KVQ:-false}"              # 禁用 KVQ
S4_DISABLE_OCR="${S4_DISABLE_OCR:-false}"              # 禁用 OCR 字幕检测（PaddleOCR 未装时可先关）
S4_RAY_NUM_GPUS="${S4_RAY_NUM_GPUS:-0.5}"

# ─── Step 5: 唇形同步过滤 超参数 ─────────────────────────────────────────────
# Face tracking
S5_INSIGHTFACE_DET_SIZE="${S5_INSIGHTFACE_DET_SIZE:-640}"
S5_FACE_IOU_THRESHOLD="${S5_FACE_IOU_THRESHOLD:-0.5}"
S5_FACE_MAX_MISSING="${S5_FACE_MAX_MISSING:-5}"
S5_MIN_FACE_TRACK_FRAMES="${S5_MIN_FACE_TRACK_FRAMES:-20}"
# TalkNCE
S5_ACTIVE_SPEAKER_THRESHOLD="${S5_ACTIVE_SPEAKER_THRESHOLD:-0.6}"
S5_ACTIVE_SPEAKER_MIN_RATIO="${S5_ACTIVE_SPEAKER_MIN_RATIO:-0.2}"
S5_MIN_TALKNCE_FRAMES="${S5_MIN_TALKNCE_FRAMES:-35}"
S5_MAX_TALKNCE_FRAMES="${S5_MAX_TALKNCE_FRAMES:-200}"
S5_DISABLE_TALKNCE="${S5_DISABLE_TALKNCE:-false}"
# SyncNet
S5_SYNCNET_MIN_CONFIDENCE="${S5_SYNCNET_MIN_CONFIDENCE:-1.0}"
S5_SYNCNET_MIN_OFFSET="${S5_SYNCNET_MIN_OFFSET:--6}"
S5_SYNCNET_MAX_OFFSET="${S5_SYNCNET_MAX_OFFSET:-6}"
S5_SYNCNET_CROP_SCALE="${S5_SYNCNET_CROP_SCALE:-0.40}"
S5_DISABLE_SYNCNET="${S5_DISABLE_SYNCNET:-false}"
# PyAnnote speaker diarization
S5_DISABLE_PYANNOTE="${S5_DISABLE_PYANNOTE:-false}"
S5_MIN_SPEAKERS="${S5_MIN_SPEAKERS:-0}"
# Audio enhance
S5_DISABLE_AUDIO_ENHANCE="${S5_DISABLE_AUDIO_ENHANCE:-false}"
# Other
S5_ALLOW_NON_TARGET_SPEAKER="${S5_ALLOW_NON_TARGET_SPEAKER:-false}"
S5_QUALIFIED_COPY_MODE="${S5_QUALIFIED_COPY_MODE:-symlink}"
S5_RAY_NUM_GPUS="${S5_RAY_NUM_GPUS:-1.0}"

# ─── Step 6: 打标（Qwen3-Omni 写 MiniMax-H3 格式描述）超参数 ──────────────────
S6_FPS="${S6_FPS:-2}"                                  # 送给模型的抽帧率
S6_MAX_LONG_SIDE="${S6_MAX_LONG_SIDE:-768}"            # 送给模型的帧长边像素
S6_RAY_NUM_GPUS="${S6_RAY_NUM_GPUS:-1.0}"              # 30B 模型 bf16 约 62 GB，一卡一个

# ─── 解析命令行参数 ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_dir)   INPUT_DIR="$2"; shift 2 ;;
        --input_csv)   INPUT_CSV="$2"; shift 2 ;;
        --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
        --use_ray)     USE_RAY=true; shift ;;
        --steps)       STEPS="$2"; shift 2 ;;
        --skip_to)     SKIP_TO="$2"; shift 2 ;;
        --num_gpus)    NUM_GPUS="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# ─── 参数校验 ───────────────────────────────────────────────────────────────
if [[ -z "$INPUT_DIR" && -z "$INPUT_CSV" ]]; then
    echo "错误: 必须指定 --input_dir 或 --input_csv"
    exit 1
fi
if [[ -z "$OUTPUT_ROOT" ]]; then
    echo "错误: 必须指定 --output_root"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"

# ─── Step 0: 从视频目录生成初始 CSV ────────────────────────────────────────
if [[ -n "$INPUT_DIR" && -z "$INPUT_CSV" ]]; then
    INPUT_CSV="$OUTPUT_ROOT/input_videos.csv"
    echo "═══ Step 0: 扫描视频目录 → $INPUT_CSV ═══"

    python3 -c "
import os, csv, sys
input_dir = sys.argv[1]
output_csv = sys.argv[2]
exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
videos = []
for root, dirs, files in os.walk(input_dir):
    for f in sorted(files):
        if os.path.splitext(f)[1].lower() in exts:
            videos.append(os.path.join(root, f))
# 去重
videos = list(dict.fromkeys(videos))
with open(output_csv, 'w', newline='', encoding='utf-8') as fout:
    w = csv.DictWriter(fout, fieldnames=['video_path'])
    w.writeheader()
    for v in videos:
        w.writerow({'video_path': v})
print(f'扫描到 {len(videos)} 个视频文件 → {output_csv}')
" "$INPUT_DIR" "$INPUT_CSV"
fi

echo "输入 CSV: $INPUT_CSV"
echo "输出根目录: $OUTPUT_ROOT"
echo "运行步骤: $STEPS"
echo "Ray 模式: $USE_RAY"
echo ""

# ─── 辅助函数 ───────────────────────────────────────────────────────────────
should_run_step() {
    local step=$1
    if [[ $SKIP_TO -gt $step ]]; then
        return 1
    fi
    if echo "$STEPS" | grep -q "$step"; then
        return 0
    fi
    return 1
}

ray_flag() {
    if $USE_RAY; then echo "--ray"; else echo ""; fi
}

# ─── 各步骤输出目录 ─────────────────────────────────────────────────────────
STEP1_DIR="$OUTPUT_ROOT/1_video_filter"
STEP2_DIR="$OUTPUT_ROOT/2_cutshot"
STEP3_DIR="$OUTPUT_ROOT/3_motion_filter"
STEP4_DIR="$OUTPUT_ROOT/4_quality_filter"
STEP5_DIR="$OUTPUT_ROOT/5_lipsync"
STEP6_DIR="$OUTPUT_ROOT/6_caption"

# 输出 CSV 路径（用于步骤间传递）
STEP1_OUTPUT="$STEP1_DIR/video_filter_qualified.csv"
STEP2_OUTPUT="$STEP2_DIR/2_cutshot_qualified.csv"
STEP3_OUTPUT="$STEP3_DIR/3_motion_filter_qualified.csv"
STEP4_OUTPUT="$STEP4_DIR/4_quality_qualified.csv"
STEP5_OUTPUT="$STEP5_DIR/5_lipsync_qualified.csv"
STEP6_OUTPUT="$STEP6_DIR/6_caption_qualified.csv"

# 追踪最新可用的 CSV（用于跳步时正确接续）
LAST_CSV="$INPUT_CSV"

# ─── Step 1: 视频质量过滤 + 转码 ───────────────────────────────────────────
if should_run_step 1; then
    echo "═══ Step 1: 视频质量过滤 + FPS/音频标准化转码 ═══"
    STEP1_CMD="python3 $SCRIPT_DIR/1_video_filter.py \
        --input_csv \"$LAST_CSV\" \
        --output_dir \"$STEP1_DIR\" \
        --video_path_column video_path \
        --min_resolution $S1_MIN_RESOLUTION \
        --min_short_side $S1_MIN_SHORT_SIDE \
        --min_duration $S1_MIN_DURATION \
        --max_duration $S1_MAX_DURATION \
        --target_fps $S1_TARGET_FPS \
        --target_audio_sr $S1_TARGET_AUDIO_SR \
        --copy_mode $S1_COPY_MODE \
        --letterbox_samples $S1_LETTERBOX_SAMPLES \
        --ray_num_cpus $S1_RAY_NUM_CPUS \
        $(ray_flag) \
        --concurrency $RAY_CONCURRENCY_STEP1"
    echo "$STEP1_CMD"
    eval "$STEP1_CMD"
    LAST_CSV="$STEP1_OUTPUT"
    echo ""
elif [[ -f "$STEP1_OUTPUT" ]]; then
    LAST_CSV="$STEP1_OUTPUT"
fi

# ─── Step 2: 场景切分 ──────────────────────────────────────────────────────
if should_run_step 2; then
    echo "═══ Step 2: OmniShotCut 场景切分 ═══"
    STEP2_CMD="python3 $SCRIPT_DIR/2_filter_cutShot.py \
        --input_csv \"$LAST_CSV\" \
        --output_dir \"$STEP2_DIR\" \
        --video_path_column video_path \
        --mode $S2_MODE \
        --num_context_frames $S2_NUM_CONTEXT_FRAMES \
        --clip_frames $S2_CLIP_FRAMES \
        --trim_frames $S2_TRIM_FRAMES \
        --audio_sr $S2_AUDIO_SR \
        --audio_channels $S2_AUDIO_CHANNELS \
        $(ray_flag) \
        --ray_num_gpus $S2_RAY_NUM_GPUS \
        --concurrency $RAY_CONCURRENCY_STEP2"
    echo "$STEP2_CMD"
    eval "$STEP2_CMD"
    LAST_CSV="$STEP2_OUTPUT"
    echo ""
elif [[ -f "$STEP2_OUTPUT" ]]; then
    LAST_CSV="$STEP2_OUTPUT"
fi

# ─── Step 3: RAFT 运动过滤 ─────────────────────────────────────────────────
if should_run_step 3; then
    echo "═══ Step 3: RAFT 光流运动评分过滤 ═══"
    STEP3_CMD="python3 $SCRIPT_DIR/3_motion_filter_raft.py \
        --input_csv \"$LAST_CSV\" \
        --output_dir \"$STEP3_DIR\" \
        --video_path_column video_path \
        --min_motion $S3_MIN_MOTION \
        --max_motion $S3_MAX_MOTION \
        --sampling_fps $S3_SAMPLING_FPS \
        --max_size $S3_MAX_SIZE \
        --copy_mode $S3_COPY_MODE \
        $(ray_flag) \
        --ray_num_gpus $S3_RAY_NUM_GPUS \
        --concurrency $RAY_CONCURRENCY_STEP3"
    echo "$STEP3_CMD"
    eval "$STEP3_CMD"
    LAST_CSV="$STEP3_OUTPUT"
    echo ""
elif [[ -f "$STEP3_OUTPUT" ]]; then
    LAST_CSV="$STEP3_OUTPUT"
fi

# ─── Step 4: 质量评估过滤 ──────────────────────────────────────────────────
if should_run_step 4; then
    echo "═══ Step 4: DOVER + AesV2.5 + KVQ 质量评估 ═══"
    S4_EXTRA_FLAGS=""
    [[ "$S4_DISABLE_DOVER" == "true" ]] && S4_EXTRA_FLAGS="$S4_EXTRA_FLAGS --no-dover"
    [[ "$S4_DISABLE_AES" == "true" ]]   && S4_EXTRA_FLAGS="$S4_EXTRA_FLAGS --no-aes"
    [[ "$S4_DISABLE_KVQ" == "true" ]]   && S4_EXTRA_FLAGS="$S4_EXTRA_FLAGS --no-kvq"
    [[ "$S4_DISABLE_OCR" == "true" ]]   && S4_EXTRA_FLAGS="$S4_EXTRA_FLAGS --no-ocr"
    STEP4_CMD="python3 $SCRIPT_DIR/4_quality_filter.py \
        --input_csv \"$LAST_CSV\" \
        --output_dir \"$STEP4_DIR\" \
        --video_path_column video_path \
        --min_dover_overall $S4_MIN_DOVER \
        --min_aes_mean $S4_MIN_AES \
        --min_kvq_score $S4_MIN_KVQ \
        --aes-frames $S4_AES_FRAMES \
        $S4_EXTRA_FLAGS \
        $(ray_flag) \
        --ray_num_gpus $S4_RAY_NUM_GPUS \
        --concurrency $RAY_CONCURRENCY_STEP4"
    echo "$STEP4_CMD"
    eval "$STEP4_CMD"
    LAST_CSV="$STEP4_OUTPUT"
    echo ""
elif [[ -f "$STEP4_OUTPUT" ]]; then
    LAST_CSV="$STEP4_OUTPUT"
fi

# ─── Step 5: 唇形同步过滤 ──────────────────────────────────────────────────
if should_run_step 5; then
    echo "═══ Step 5: 唇形同步检测 (SyncNet + PyAnnote + SemanticSync) ═══"
    S5_EXTRA_FLAGS=""
    [[ "$S5_DISABLE_TALKNCE" == "true" ]]        && S5_EXTRA_FLAGS="$S5_EXTRA_FLAGS --disable_talknce"
    [[ "$S5_DISABLE_SYNCNET" == "true" ]]        && S5_EXTRA_FLAGS="$S5_EXTRA_FLAGS --disable_syncnet"
    [[ "$S5_DISABLE_PYANNOTE" == "true" ]]       && S5_EXTRA_FLAGS="$S5_EXTRA_FLAGS --disable_pyannote"
    [[ "$S5_DISABLE_AUDIO_ENHANCE" == "true" ]]  && S5_EXTRA_FLAGS="$S5_EXTRA_FLAGS --disable_audio_enhance"
    [[ "$S5_ALLOW_NON_TARGET_SPEAKER" == "true" ]] && S5_EXTRA_FLAGS="$S5_EXTRA_FLAGS --allow_non_target_speaker"
    STEP5_CMD="python3 $SCRIPT_DIR/5_filter_lipsync.py \
        --input_csv \"$LAST_CSV\" \
        --output_dir \"$STEP5_DIR\" \
        --video_path_column video_path \
        --insightface_det_size $S5_INSIGHTFACE_DET_SIZE \
        --face_iou_threshold $S5_FACE_IOU_THRESHOLD \
        --face_max_missing $S5_FACE_MAX_MISSING \
        --min_face_track_frames $S5_MIN_FACE_TRACK_FRAMES \
        --active_speaker_threshold $S5_ACTIVE_SPEAKER_THRESHOLD \
        --active_speaker_min_ratio $S5_ACTIVE_SPEAKER_MIN_RATIO \
        --min_talknce_frames $S5_MIN_TALKNCE_FRAMES \
        --max_talknce_frames $S5_MAX_TALKNCE_FRAMES \
        --syncnet_min_confidence $S5_SYNCNET_MIN_CONFIDENCE \
        --syncnet_min_offset $S5_SYNCNET_MIN_OFFSET \
        --syncnet_max_offset $S5_SYNCNET_MAX_OFFSET \
        --syncnet_crop_scale $S5_SYNCNET_CROP_SCALE \
        --min_speakers $S5_MIN_SPEAKERS \
        --qualified_copy_mode $S5_QUALIFIED_COPY_MODE \
        $S5_EXTRA_FLAGS \
        $(ray_flag) \
        --ray_num_gpus $S5_RAY_NUM_GPUS \
        --concurrency $RAY_CONCURRENCY_STEP5"
    echo "$STEP5_CMD"
    eval "$STEP5_CMD"
    LAST_CSV="$STEP5_OUTPUT"
    echo ""
elif [[ -f "$STEP5_OUTPUT" ]]; then
    LAST_CSV="$STEP5_OUTPUT"
fi

# ─── Step 6: 打标 ──────────────────────────────────────────────────────────
if should_run_step 6; then
    echo "═══ Step 6: Qwen3-Omni 打标（MiniMax-H3 格式描述） ═══"
    STEP6_CMD="python3 $SCRIPT_DIR/6_caption.py \
        --input_csv \"$LAST_CSV\" \
        --output_dir \"$STEP6_DIR\" \
        --video_path_column video_path \
        --fps $S6_FPS \
        --max_long_side $S6_MAX_LONG_SIDE \
        $(ray_flag) \
        --ray_num_gpus $S6_RAY_NUM_GPUS \
        --concurrency $RAY_CONCURRENCY_STEP6"
    echo "$STEP6_CMD"
    eval "$STEP6_CMD"
    LAST_CSV="$STEP6_OUTPUT"
    echo ""
elif [[ -f "$STEP6_OUTPUT" ]]; then
    LAST_CSV="$STEP6_OUTPUT"
fi

# ─── 汇总 ──────────────────────────────────────────────────────────────────
echo "═══ Pipeline 完成 ═══"
echo "最终结果: $LAST_CSV"   # 最后一个跑过（或已有输出）的步骤
echo ""
echo "各步骤输出:"
for f in "$STEP1_OUTPUT" "$STEP2_OUTPUT" "$STEP3_OUTPUT" "$STEP4_OUTPUT" "$STEP5_OUTPUT" "$STEP6_OUTPUT"; do
    if [[ -f "$f" ]]; then
        cnt=$(wc -l < "$f")
        echo "  $f → $((cnt - 1)) 条"
    fi
done
