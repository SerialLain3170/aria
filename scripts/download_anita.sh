#!/usr/bin/env bash
set -euo pipefail

ROOT="${T2A_ROOT:-/data/shasegawa/t2a}"
DEST="${ROOT}/datasets/anita/raw"
mkdir -p "${DEST}" "${ROOT}/datasets/anita/clips"

FILE_ID="${ANITA_GDRIVE_ID:-1ctfD0sMpT2pVutJUOlyEYKhAxufMYmZ_}"
OUT="${DEST}/Anita_Dataset.zip"

python -m gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${OUT}"

echo "Downloaded AnitaDataset archive to ${OUT}"
echo "Extract it with: unzip -q ${OUT} -d ${ROOT}/datasets/anita"
