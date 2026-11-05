#!/usr/bin/env bash
# Versioned real Minecraft practice kit. Never use the heldout seed in training.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 2 ]];then
  echo 'Usage: tools/run-numeric-pickup-v2.sh train|resume|eval|gate TAG [extra options]' >&2
  echo 'Example: tools/run-numeric-pickup-v2.sh train egolearn1105b --episodes 300' >&2
  exit 2
fi
mode="$1"; shift
case "$mode" in
 train|resume)
    tag="$1";shift
    args=(--kit egoreplay_v2 --run-tag "$tag" --mode train --episodes 300 --max-actions 24 --case-seed 441151 --policy-seed 881151 --device cpu --visible --report-every 25)
    if [[ "$mode" == resume ]];then args+=(--resume);fi
    args+=("$@")
    GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u experiments/train_numeric_item_pickup.py "${args[@]}"
    ;;
 eval)
    tag="$1";shift
    # Explicit distinct frozen model, case seed, control and run tag required.
    GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh .venv/bin/python -u experiments/train_numeric_item_pickup.py --kit egoreplay_v2 --mode eval --run-tag "$tag" --visible "$@"
    ;;
 gate)
    .venv/bin/python tools/gate-numeric-pickup-v2.py "$@"
    ;;
 *)echo 'Invalid mode: train|resume|eval|gate' >&2;exit 2;;
esac
