#!/usr/bin/env bash
#
# 处理 0615_movie_2 视频数据
#
# 用法:
#   bash script/run_0615_movie_2.sh           # 本地模式
#   bash script/run_0615_movie_2.sh --use_ray # Ray 分布式模式
#   bash script/run_0615_movie_2.sh --use_ray --steps "1,2,3"
#   bash script/run_0615_movie_2.sh --use_ray --skip_to 3
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT_DIR="$PROJECT_ROOT/test_data/0615_movie_1"
OUTPUT_ROOT="$PROJECT_ROOT/test/result/0615/pipeline_movie_3"

# ─── 基础参数 ─────────────────────────────────────────────────────────────────
USE_RAY=""
STEPS="1,2,3,4,5"
SKIP_TO=0

export NUM_GPUS=16
export RAY_CONCURRENCY_STEP1=$((4 * NUM_GPUS))    
export RAY_CONCURRENCY_STEP2=$((2 * NUM_GPUS))
export RAY_CONCURRENCY_STEP3=$((2 * NUM_GPUS))
export RAY_CONCURRENCY_STEP4=$((2 * NUM_GPUS))
export RAY_CONCURRENCY_STEP5=$((1 * NUM_GPUS))


# ─── Step 1: 视频质量过滤 + 转码 ─────────────────────────────────────────────
export S1_MIN_RESOLUTION="1080p"       # 最低分辨率: 360p / 480p / 720p / 1080p 或像素数
export S1_MIN_DURATION=121             # 最短帧数 (25fps 下 81帧 ≈ 3.2秒)
export S1_MAX_DURATION=0               # 最长帧数 (0=不限)
export S1_TARGET_FPS=25                # 目标帧率
export S1_TARGET_AUDIO_SR=48000        # 目标音频采样率
export S1_COPY_MODE="symlink"          # 无需转码时: symlink / copy / none
export S1_RAY_NUM_CPUS=2               # 每个 worker 占用 CPU 数 (转码吃 CPU)

# ─── Step 2: 场景切分 (OmniShotCut) ──────────────────────────────────────────
export S2_MODE="clean_shot"            # 切分模式: default(所有转场) / clean_shot(仅硬切)
export S2_NUM_CONTEXT_FRAMES=10         # 转场检测上下文帧数
export S2_RAY_NUM_GPUS=0.5             # 每个 worker GPU 份额

# ─── Step 3: RAFT 运动过滤 ───────────────────────────────────────────────────
# min_motion 参考: <0.5 静止, 0.5~2 talking head, 2~6 正常, 6~15 运动, >15 剧烈
export S3_MIN_MOTION=10                # 运动分下限 (低于此视为静止丢弃)
export S3_MAX_MOTION=200               # 运动分上限 (高于此视为抖动丢弃)
export S3_SAMPLING_FPS=2               # 光流采样帧率
export S3_MAX_SIZE=512                 # 输入帧缩放最大尺寸
export S3_COPY_MODE="symlink"          # 通过时: symlink / copy / none
export S3_RAY_NUM_GPUS=0.5

# ─── Step 4: 质量评估过滤 (DOVER + AesV2.5 + KVQ) ───────────────────────────
export S4_MIN_DOVER=0.4                # DOVER 总分阈值 (0~1, 越高越严)
export S4_MIN_AES=4.2                  # Aesthetic V2.5 均分阈值 (1~10)
export S4_MIN_KVQ=0.01                 # KVQ 分数阈值
export S4_AES_FRAMES=16                # 美学评估采样帧数
export S4_RAY_NUM_GPUS=0.5

# ─── Step 5: 唇形同步过滤 ────────────────────────────────────────────────────
# Face tracking
export S5_INSIGHTFACE_DET_SIZE=640     # 人脸检测分辨率
export S5_FACE_IOU_THRESHOLD=0.5       # 人脸跟踪 IoU 阈值
export S5_FACE_MAX_MISSING=5           # 允许丢失最大帧数
export S5_MIN_FACE_TRACK_FRAMES=20     # 最短人脸轨迹帧数
# TalkNCE
export S5_ACTIVE_SPEAKER_THRESHOLD=0.6 # 说话者检测概率阈值
export S5_ACTIVE_SPEAKER_MIN_RATIO=0.2 # 说话帧最低占比
export S5_MIN_TALKNCE_FRAMES=35        # TalkNCE 最少帧数
export S5_MAX_TALKNCE_FRAMES=200       # TalkNCE 最多帧数
# SyncNet
export S5_SYNCNET_MIN_CONFIDENCE=1.0   # SyncNet 最低置信度
export S5_SYNCNET_MIN_OFFSET=-6        # SyncNet 允许偏移下界
export S5_SYNCNET_MAX_OFFSET=6         # SyncNet 允许偏移上界
export S5_SYNCNET_CROP_SCALE=0.40      # SyncNet 人脸裁剪缩放
# PyAnnote 说话人分割
export S5_MIN_SPEAKERS=0               # 最少说话人数 (0=不限)
# Audio enhance
export S5_DISABLE_AUDIO_ENHANCE=false  # 禁用音频增强 (人声分离+降噪)
# Semantic sync
export S5_SEMANTIC_SYNC_THRESHOLD=0.15 # 语义同步阈值
export S5_SEMANTIC_SYNC_MAX_FRAMES=15  # 语义同步最大帧数
# Other
export S5_QUALIFIED_COPY_MODE="symlink"  # 通过视频: symlink / copy / none
export S5_RAY_NUM_GPUS=1.0

# ─── 解析命令行参数 ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --use_ray) USE_RAY="--use_ray"; shift ;;
        --steps)   STEPS="$2"; shift 2 ;;
        --skip_to) SKIP_TO="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "══════════════════════════════════════════════════════════"
echo "  数据处理 Pipeline: 0615_movie_2"
echo "  输入: $INPUT_DIR"
echo "  输出: $OUTPUT_ROOT"
echo "  步骤: $STEPS"
echo "  GPU:  $NUM_GPUS"
echo "══════════════════════════════════════════════════════════"
echo ""

exec bash "$PROJECT_ROOT/filter/run_pipeline.sh" \
    --input_dir "$INPUT_DIR" \
    --output_root "$OUTPUT_ROOT" \
    --steps "$STEPS" \
    --skip_to "$SKIP_TO" \
    --num_gpus "$NUM_GPUS" \
    $USE_RAY
