#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONPATH="${PYTHONPATH:-src}"
export HF_HOME="${HF_HOME:-/data/shasegawa/t2a/hf-cache}"

accelerate launch \
  --num_processes 2 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  -m text_to_anime.train_wan_i2v_lora \
  --config configs/wan22_i2v_anita_360p_lora.yaml \
  --manifest /data/shasegawa/t2a/manifests/anita_i2v_train.jsonl \
  --validation-manifest /data/shasegawa/t2a/manifests/anita_i2v_val50.jsonl
