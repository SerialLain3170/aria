#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-3}"
ROOT="${T2A_ROOT:-/data/shasegawa/t2a}"
ANISORA_CODE="${ANISORA_CODE:-${ROOT}/code/index-anisora/anisoraV3.2}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/models/Index-anisora/V3.2}"
PROMPT_LIST="${PROMPT_LIST:-${ROOT}/manifests/anisora_v32_i2v_anita4.txt}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/outputs/anisora_v32_raw_i2v_anita4_short_bf16}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTORCH_ALLOC_CONF=expandable_segments:True \
PYTHONPATH="${ANISORA_CODE}" \
python "${ANISORA_CODE}/generate_txt_new.py" \
  --task i2v-A14B \
  --size '832*480' \
  --ckpt_dir "${CKPT_DIR}" \
  --prompt_list "${PROMPT_LIST}" \
  --save_dir "${SAVE_DIR}" \
  --sample_steps 8 \
  --sample_shift 5 \
  --sample_guide_scale 1 \
  --ckpt_dir_lowname low_noise_model \
  --ckpt_dir_highname high_noise_model \
  --base_seed 4096 \
  --frame_num 17 \
  --offload_model True \
  --t5_cpu \
  --convert_model_dtype
