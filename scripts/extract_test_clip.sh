#!/usr/bin/env bash
# Extract a short test clip from the first video in data/
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
OUT="$ROOT/tests/fixtures"
mkdir -p "$OUT"

VIDEO="$(find "$DATA" -name '*.mp4' | head -1)"
if [ -z "$VIDEO" ]; then
  echo "No MP4 found in data/"
  exit 1
fi

ffmpeg -y -i "$VIDEO" -t 30 -vn -acodec pcm_s16le -ar 16000 -ac 1 "$OUT/sample_30s.wav"
echo "Created $OUT/sample_30s.wav from $VIDEO"
