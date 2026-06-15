#!/usr/bin/env bash
#
# 处理 0615_movie_1 目录下的视频数据
#
# 用法:
#   bash run_0615_movie_1.sh           # 本地模式（顺序处理）
#   bash run_0615_movie_1.sh --use_ray # Ray 分布式模式
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_DIR="$PROJECT_ROOT/test_data/0615_movie_1"
OUTPUT_ROOT="$PROJECT_ROOT/test/result/0615/pipeline_movie_1"

# 默认参数
USE_RAY=""
STEPS="1,2,3,4,5"
SKIP_TO=0

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --use_ray) USE_RAY="--use_ray"; shift ;;
        --steps)   STEPS="$2"; shift 2 ;;
        --skip_to) SKIP_TO="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "══════════════════════════════════════════════════════════"
echo "  数据处理 Pipeline: 0615_movie_1"
echo "  输入: $INPUT_DIR"
echo "  输出: $OUTPUT_ROOT"
echo "  步骤: $STEPS"
echo "══════════════════════════════════════════════════════════"
echo ""

exec bash "$PROJECT_ROOT/filter/run_pipeline.sh" \
    --input_dir "$INPUT_DIR" \
    --output_root "$OUTPUT_ROOT" \
    --steps "$STEPS" \
    --skip_to "$SKIP_TO" \
    $USE_RAY
