#!/usr/bin/env bash
set -euo pipefail

ROOT="${T2A_ROOT:-/data/shasegawa/t2a}"
IDS="${1:-${ROOT}/manifests/animeshooter_subset_video_urls.txt}"
OUT="${ROOT}/datasets/animeshooter/source_videos"
mkdir -p "${OUT}"

if [[ ! -f "${IDS}" ]]; then
  echo "Missing ${IDS}. Build it with t2a-export-source-ids --as-urls." >&2
  exit 1
fi

yt-dlp --batch-file "${IDS}" \
  -o "${OUT}/%(id)s.%(ext)s" \
  -f "bv*[height<=720][height>=360][ext=mp4]/bv*[height<=720][ext=mp4]/bv*[height<=720]" \
  --merge-output-format mp4 \
  --ignore-errors \
  --no-overwrites
