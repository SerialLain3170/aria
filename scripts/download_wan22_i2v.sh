#!/usr/bin/env bash
set -euo pipefail

ROOT="${T2A_ROOT:-/data/shasegawa/t2a}"
export HF_HOME="${HF_HOME:-${ROOT}/hf-cache}"
DEST="${ROOT}/models/Wan2.2-I2V-A14B-Diffusers"
mkdir -p "${DEST}" "${HF_HOME}"

hf download Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --type model \
  --local-dir "${DEST}"

echo "Wan2.2 I2V Diffusers checkpoint is under ${DEST}"
