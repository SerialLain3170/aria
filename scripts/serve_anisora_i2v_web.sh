#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
GPU_ID="${GPU_ID:-3}"
ROOT="${T2A_ROOT:-/data/shasegawa/t2a}"

PYTHONPATH=src python -m text_to_anime.anisora_i2v_web \
  --host "${HOST}" \
  --port "${PORT}" \
  --gpu-id "${GPU_ID}" \
  --root "${ROOT}"
