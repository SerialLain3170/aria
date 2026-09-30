#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/download_wan21_vace.sh [1.3B|14B]
SIZE="${1:-1.3B}"
ROOT="${T2A_ROOT:-/data/shasegawa/t2a}"
export HF_HOME="${HF_HOME:-${ROOT}/hf-cache}"
DEST="${ROOT}/models/Wan2.1-VACE-${SIZE}-diffusers"
mkdir -p "${DEST}" "${HF_HOME}"

hf download "Wan-AI/Wan2.1-VACE-${SIZE}-diffusers" \
  --type model \
  --local-dir "${DEST}"

echo "Wan2.1 VACE ${SIZE} Diffusers checkpoint is under ${DEST}"
