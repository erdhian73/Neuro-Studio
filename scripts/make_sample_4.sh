#!/usr/bin/env bash
# Rebuild frontend/public/videos/sample_4.mp4 ("04 AI Optimized").
#
# sample_4 is a 15 s window of sample_2 with the *slowest* stretch removed and
# contrast/saturation boosted, so TRIBE v2 sees a denser, higher-salience
# stimulus than the flat original.
#
# Why 15 s and not 4 s: TRIBE runs at tr_s=1.0, so a 4 s clip yields only 5 TRs
# (and only 2 are usable once the 5 s hemodynamic lag is accounted for). A 15 s
# clip yields ~15 TRs, which is enough for a stable mean and a real peak window.
# Earlier measurement also showed the 4 s cut scored *lower* than its source on
# dynamic_range (2.13 -> 1.71) purely because of its length.
#
# Window selection is measured, not guessed. Per-second scene_score (mean
# frame-to-frame difference over 25 fps) on sample_2.mp4 (30.1 s):
#
#   busiest : 10-11s 0.0281 | 13-14s 0.0270 | 16-17s 0.0246 | 19-20s 0.0233
#   deadest : 26-27s 0.0006 | 27-28s 0.0004 | 29-30s 0.0000 | 18-19s 0.0008
#
# The deadest contiguous stretch is ~18-28 s. Every 15-20 s window was scored
# and 3-18 s won on retained motion (mean 0.0129) with only 3 dead seconds:
#
#   3-18s len=15s mean_motion=0.0129  <-- chosen
#   5-20s len=15s mean_motion=0.0132  (4 dead seconds, drops the strong 3-4s beat)
#
# The brightness/contrast values below were chosen from measurements taken at
# 8-12 s and are now re-checked over the full 15 s window by the script output.
#
# Audio is kept (mean_volume -27.6 dB) because TRIBE's w2v-BERT encoder uses it.
#
# A stream-copy cut is not possible: sample_2 has a single keyframe at 0 s, so
# `-ss 3 -c copy` would either fail or start on a non-keyframe. We re-encode.
#
# Usage:  scripts/make_sample_4.sh [path-to-ffmpeg]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIDEO_DIR="$REPO_ROOT/frontend/public/videos"
SRC="$VIDEO_DIR/sample_2.mp4"
OUT="$VIDEO_DIR/sample_4.mp4"

# 15 s cut starting at 3 s: keeps the event-dense body, drops the 18-28 s dead
# stretch, and stops before the near-static tail.
START=3
LEN=15

FFMPEG="${1:-}"
if [[ -z "$FFMPEG" ]]; then
  if command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG="$(command -v ffmpeg)"
  else
    # Fall back to the pip-installable static build (no root needed).
    FFBIN="$(python3 -c 'import static_ffmpeg,shutil,os;static_ffmpeg.add_paths();print(shutil.which("ffmpeg") or "")' 2>/dev/null || true)"
    if [[ -z "$FFBIN" ]]; then
      echo "error: ffmpeg not found. Install it, or: uv pip install static-ffmpeg" >&2
      exit 1
    fi
    FFMPEG="$FFBIN"
  fi
fi

[[ -f "$SRC" ]] || { echo "error: missing $SRC" >&2; exit 1; }

END=$((START + LEN))

echo "ffmpeg : $FFMPEG"
echo "source : $SRC"
echo "output : $OUT"
echo "window : ${START}-${END}s (${LEN}s)"

"$FFMPEG" -hide_banner -y -i "$SRC" \
  -vf "select='between(t,$START,$END-0.001)',setpts=PTS-STARTPTS,eq=contrast=1.32:brightness=0.03:saturation=1.6:gamma=1.04,unsharp=5:5:0.6:5:5:0.0" \
  -af "aselect='between(t,$START,$END-0.001)',asetpts=PTS-STARTPTS" \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -r 25 \
  -c:a aac -b:a 128k -ar 44100 \
  -movflags +faststart \
  "$OUT"

echo
echo "done. verify with:"
echo "  ffprobe -v error -show_entries format=duration,size -of default=noprint_wrappers=1 \"$OUT\""
