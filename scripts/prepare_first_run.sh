#!/usr/bin/env bash
set -euo pipefail

ROOT="${T2A_ROOT:-/data/shasegawa/t2a}"
mkdir -p \
  "${ROOT}/models" \
  "${ROOT}/hf-cache" \
  "${ROOT}/datasets/animeshooter/raw" \
  "${ROOT}/datasets/animeshooter/source_videos" \
  "${ROOT}/datasets/animeshooter/clips" \
  "${ROOT}/datasets/sakuga-42m/aesthetic/parquet" \
  "${ROOT}/datasets/sakuga-42m/aesthetic/clips" \
  "${ROOT}/manifests" \
  "${ROOT}/outputs/samples"

cat <<EOF
Prepared directory layout under ${ROOT}

Next required external inputs:
1. Download AniSora V3.2 with scripts/download_anisora_v32.sh
2. Download AnimeShooter with scripts/download_animeshooter.sh
3. Convert AnimeShooter annotations to JSONL at:
   ${ROOT}/manifests/animeshooter_raw.jsonl
4. Download source videos with scripts/download_animeshooter_sources.sh when video_ids.txt is present
5. Extract shot clips to:
   ${ROOT}/datasets/animeshooter/clips
EOF
