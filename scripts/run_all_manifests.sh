#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOG_LEVEL="${LOG_LEVEL:-INFO}"
SINGLE_POSE_INDEX="${SINGLE_POSE_INDEX:-1}"
TARGET_SIZE="${TARGET_SIZE:-224}"

echo "[1/6] Building raw manifest"
"$SCRIPT_DIR/run_build_manifest.sh" --log-level "$LOG_LEVEL"

echo "[2/6] Creating train/val/test splits"
"$SCRIPT_DIR/run_split_data.sh" --log-level "$LOG_LEVEL"

echo "[3/6] Building images-only manifest"
"$SCRIPT_DIR/run_images_only_manifest.sh" --log-level "$LOG_LEVEL"

echo "[4/6] Building single-image manifest"
"$SCRIPT_DIR/run_single_image_manifest.sh" \
  --log-level "$LOG_LEVEL" \
  --single-pose-index "$SINGLE_POSE_INDEX"

echo "[5/6] Isolating faces for images-only manifest"
"$SCRIPT_DIR/run_face_isolation_images_only.sh" \
  --log-level "$LOG_LEVEL" \
  --target-size "$TARGET_SIZE"

echo "[6/6] Isolating faces for single-image manifest"
"$SCRIPT_DIR/run_face_isolation_single_image.sh" \
  --log-level "$LOG_LEVEL" \
  --target-size "$TARGET_SIZE"

echo "Manifest pipeline complete."
