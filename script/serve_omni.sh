#!/usr/bin/env bash
# 起 Qwen3-Omni 的 vLLM 服务，给 Step 6（filter/6_caption.py --backend vllm）用。占一整张卡，跑完记得停掉。
#
#   bash script/serve_omni.sh            # 前台运行，Ctrl-C 停
#   PORT=8901 bash script/serve_omni.sh
#
# 做法同团队 LTX-2 打标器（packages/ltx-trainer/scripts/serve_captioner.py）：vllm[audio]==0.11.2、
# 允许读本地 file:// 媒体、每个请求 1 段视频 + 1 段音频。
set -euo pipefail
DP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$DP_ROOT/script/env_ckps.sh"

VLLM_ENV="${VLLM_ENV:-/mnt/shared/lihaoran/ssd/framework/conda/envs/vllm}"
PORT="${PORT:-8901}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"         # 同时处理的请求数；显存不够就调小
GPU_MEM="${GPU_MEM:-0.92}"                # bf16 权重约 62 GB，剩下给 KV cache
QUANT="${QUANT:-}"                        # 设成 fp8 可以省一半权重显存，多放并发

exec "$VLLM_ENV/bin/vllm" serve "$QWEN3_OMNI_PATH" \
    --served-model-name qwen3-omni \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --allowed-local-media-path / \
    --limit-mm-per-prompt '{"video": 1, "audio": 1}' \
    ${QUANT:+--quantization "$QUANT"}
