#!/usr/bin/env bash
set -euo pipefail

ROOT="${T2A_ROOT:-/data/shasegawa/t2a}"
IDS="${ROOT}/datasets/animeshooter/raw/video_ids.txt"
OUT="${ROOT}/datasets/animeshooter/source_videos"
mkdir -p "${OUT}"

if [[ ! -f "${IDS}" ]]; then
  echo "Missing ${IDS}; run scripts/download_animeshooter.sh first." >&2
  exit 1
fi

yt-dlp --batch-file "${IDS}" \
  -o "${OUT}/%(id)s.%(ext)s" \
  -f "bv*[height<=720][height>=360][ext=mp4]/bv*[height<=720][ext=mp4]/bv*[height<=720]" \
  --merge-output-format mp4 \
  --ignore-errors \
  --no-overwrites
