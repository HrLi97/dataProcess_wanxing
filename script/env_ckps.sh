#!/usr/bin/env bash
# Model-weight locations for the dataProcess_wanxing pipeline (filter/run_pipeline.sh, steps 1-5).
# Every weight lives under /mnt/shared/lihaoran/ckps/dataprocess (provenance: *.download.log there).
#
#   source script/env_ckps.sh        # then: bash filter/run_pipeline.sh ...  (launch from the repo root)
#
# Ray mode: workers inherit the environment of `ray start`, not of the driver, so source this file
# before `ray start --head` / `ray start --address=...` on every node.
(return 0 2>/dev/null) || echo "warning: run 'source $0' (executing it does not export anything)" >&2

DP_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CKPS_ROOT=/mnt/shared/lihaoran/ckps/dataprocess

# torch.hub / torchvision cache ($TORCH_HOME/hub/checkpoints):
#   resnet18-f37072fd.pth          step 2, OmniShotCut backbone init (resnet18(pretrained=True))
#   convnext_tiny_1k_224_ema.pth   step 4, DOVER aesthetic backbone init (convnext_3d_tiny(pretrained=True))
export TORCH_HOME=$CKPS_ROOT/torch_home

# Step 2  OmniShotCut: no env var. third_part/OmniShotCut/checkpoints/OmniShotCut_ckpt.pth -> $CKPS_ROOT/OmniShotCut/
# Step 3  RAFT: third_part/raft-things.pth (tracked in the repo).

# Step 4  quality filter
export SIGLIP_ENCODER_PATH=$CKPS_ROOT/siglip-so400m-patch14-384     # AesV2.5 encoder, google/siglip-so400m-patch14-384 @ 9fdffc58
export PADDLE_PDX_CACHE_HOME=$CKPS_ROOT/paddlex                     # PaddleOCR/PaddleX cache: official_models/PP-OCRv4_mobile_{det,rec}
export PADDLEOCR_MODEL_DIR=$CKPS_ROOT/paddlex/official_models       # read by filter_ocr.py / 4_quality_filter.py (paths only)
export PADDLEOCR_ROOT=$DP_ROOT/third_part/PaddleOCR                 # code root (same as the default)
# DOVER.pth, KVQ.pth, aesthetic_predictor_v2_5.pth: no env vars; the fixed paths inside
# third_part/{DOVER-master,KVQ,aesthetic-predictor-v2-5} are symlinks into $CKPS_ROOT.

# Step 5  lipsync
export INSIGHTFACE_ROOT=$CKPS_ROOT/insightface                      # models/buffalo_l/*.onnx
export TALKNCE_CKPT=$CKPS_ROOT/lipsync/talknce_ava_pretrained.model
export SYNCNET_CKPT=$CKPS_ROOT/syncnet/syncnet_v2.model
export PYANNOTE_CFG=$CKPS_ROOT/pyannote/config.yaml                 # pyannote/speaker-diarization-community-1, loads offline
export AUDIO_SEP_ROOT=$DP_ROOT/third_part/Music-Source-Separation-Training   # code root (same as the default)
export AUDIO_SEP_CONFIG=$CKPS_ROOT/MelBandRoformer/config_vocals_mel_band_roformer_kj.yaml
export AUDIO_SEP_CKPT=$CKPS_ROOT/MelBandRoformer/MelBandRoformer.ckpt
# ClearVoice MossFormer2_SE_48K resolves checkpoints/MossFormer2_SE_48K relative to the CWD; audio_enhance_runner.py
# builds it with CWD=$CLEARVOICE_DIR (default ckps_root()/clearvoice), so step 5 no longer depends on the launch dir.
export CLEARVOICE_DIR=$CKPS_ROOT/clearvoice                         # checkpoints/MossFormer2_SE_48K/last_best_checkpoint{,.pt}
export QWEN3_OMNI_PATH=$CKPS_ROOT/Qwen3-Omni-30B-A3B-Instruct   # Step 6 打标

# Optional: stop pyannote.audio 4.x from sending usage telemetry.
# export PYANNOTE_METRICS_ENABLED=false
