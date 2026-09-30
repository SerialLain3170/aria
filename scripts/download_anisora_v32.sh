#!/usr/bin/env bash
set -euo pipefail

ROOT="${T2A_ROOT:-/data/shasegawa/t2a}"
export HF_HOME="${HF_HOME:-${ROOT}/hf-cache}"
DEST="${ROOT}/models/Index-anisora"
mkdir -p "${DEST}" "${HF_HOME}"

hf download IndexTeam/Index-anisora \
  --type model \
  --include 'V3.2/*' \
  --local-dir "${DEST}"

echo "AniSora V3.2 native checkpoint is under ${DEST}/V3.2"
echo "For this trainer, convert or expose it as a WanPipeline-compatible Diffusers folder at:"
echo "  ${ROOT}/models/Index-anisora-diffusers/V3.2"
