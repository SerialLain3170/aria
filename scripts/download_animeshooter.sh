#!/usr/bin/env bash
set -euo pipefail

ROOT="${T2A_ROOT:-/data/shasegawa/t2a}"
export HF_HOME="${HF_HOME:-${ROOT}/hf-cache}"
DEST="${ROOT}/datasets/animeshooter/raw"
mkdir -p "${DEST}" "${HF_HOME}" "${ROOT}/datasets/animeshooter/source_videos" "${ROOT}/datasets/animeshooter/clips"

hf download qiulu66/AnimeShooter \
  --type dataset \
  --local-dir "${DEST}"

echo "AnimeShooter annotations/files are under ${DEST}"
echo "If video_ids.txt is present, source videos can be fetched with yt-dlp into:"
echo "  ${ROOT}/datasets/animeshooter/source_videos"
