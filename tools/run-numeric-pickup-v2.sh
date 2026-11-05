#!/usr/bin/env bash
# Actual Minecraft item pickup training with authoritative checkpoint validation.
# Do not trust headless Gamescope's occasional post-child exit code 139.
set -euo pipefail
cd "$(dirname "$0")/.."
usage(){
  echo 'Usage: tools/run-numeric-pickup-v2.sh train|resume TAG [extra options]' >&2
  echo '       tools/run-numeric-pickup-v2.sh start TAG [extra options]  # detached new train' >&2
  echo '       tools/run-numeric-pickup-v2.sh eval TAG --model-tag SOURCE --episodes N --case-seed S [options]' >&2
  echo '       tools/run-numeric-pickup-v2.sh status|stop TAG' >&2
  echo '       tools/run-numeric-pickup-v2.sh gate LEARNED RANDOM FORWARD [repeat for more seeds]' >&2
  exit 2
}
[[ $# -ge 2 ]] || usage
mode="$1"; shift
if [[ "$mode" == gate ]];then
  .venv/bin/python tools/gate-numeric-pickup-v2.py "$@"
  exit $?
fi
tag="$1";shift
[[ "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,22}$ ]] || usage
mkdir -p logs checkpoints
journal="logs/numeric_pickup_${tag}.jsonl"
checkpoint="checkpoints/numeric_pickup_${tag}.pt"
lock="checkpoints/numeric_pickup_${tag}.lock"
marker="logs/numeric_pickup_${tag}.stop"
case "$mode" in
 status)
    [[ $# == 0 ]] || usage
    .venv/bin/python tools/verify-numeric-pickup-v2.py "$tag"
    exit $?
    ;;
 stop)
    [[ $# == 0 ]] || usage
    .venv/bin/python - "$lock" "$marker" <<'PY'
from pathlib import Path
import fcntl,sys
lock=Path(sys.argv[1]);marker=Path(sys.argv[2])
if not lock.is_file():sys.exit('No run lock yet; nothing to stop')
with lock.open('a+') as handle:
 try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:
  marker.touch(exist_ok=False)
  print('Graceful stop requested. Check status before starting another Minecraft instance.')
 else:
  fcntl.flock(handle,fcntl.LOCK_UN)
  sys.exit('No trainer currently holds the run lock')
PY
    exit $?
    ;;
 start)
    [[ ! -e "$journal" && ! -e "$checkpoint" ]] || {
      echo "Existing experiment $tag; use status or resume, never overwrite it." >&2;exit 1;
    }
    [[ ! -e "logs/numeric_pickup_${tag}_launcher.console" ]] || {
      echo "Launcher log already exists for $tag" >&2;exit 1;
    }
    # setsid/nohup survives termination of the requesting Gentoo Shell call.
    nohup setsid bash "$0" train "$tag" "$@" > "logs/numeric_pickup_${tag}_launcher.console" 2>&1 < /dev/null &
    pid=$!
    echo "LAUNCHED $tag wrapper_pid=$pid"
    echo "WATCH: tail -f logs/numeric_pickup_${tag}_launcher.console"
    echo "STATUS: tools/run-numeric-pickup-v2.sh status $tag"
    echo "STOP: tools/run-numeric-pickup-v2.sh stop $tag"
    exit 0
    ;;
 train|resume|eval)
    ;;
 *)usage;;
esac
if [[ "$mode" == eval ]];then
  args=(--kit egoreplay_v2 --mode eval --run-tag "$tag" --visible "$@")
  # Re-read the explicit eval source and episode budget through parse_args.
  read -r expected source < <(.venv/bin/python - "${args[@]}" <<'PY'
import sys
sys.path.insert(0,'experiments')
from train_numeric_item_pickup import parse_args
args=parse_args(sys.argv[1:])
print(args.episodes,args.model_tag)
PY
)
  check=(--mode eval --source "$source")
else
  args=(--kit egoreplay_v2 --run-tag "$tag" --mode train
        --episodes 300 --max-actions 24 --case-seed 441151
        --policy-seed 881151 --device cpu --visible --report-every 25)
  [[ "$mode" == resume ]] && args+=(--resume)
  args+=("$@")
  read -r expected source < <(.venv/bin/python - "${args[@]}" <<'PY'
import sys
sys.path.insert(0,'experiments')
from train_numeric_item_pickup import parse_args
args=parse_args(sys.argv[1:])
print(args.episodes,'-')
PY
)
  check=(--mode train)
fi
console="logs/numeric_pickup_${tag}_gamescope.console"
if [[ -e "$console" ]];then
  echo "Existing console $console; use a new tag, or inspect before resuming." >&2
  exit 1
fi
echo "NUMERIC PICKUP LAUNCH $mode tag=$tag target=$expected; raw log $console"
set +e
.venv/bin/python -u tools/stream-numeric-pickup.py "$console" -- \
  tools/run-mcio-viewer.sh .venv/bin/python -u experiments/train_numeric_item_pickup.py "${args[@]}"
outer_code=$?
set -e
# The independent full journal/checkpoint audit is THE success criterion.
if .venv/bin/python tools/verify-numeric-pickup-v2.py "$tag" --target "$expected" \
    --require-complete "${check[@]}"; then
  if [[ "$outer_code" == 139 ]];then
    echo "NOTE: Gamescope exited 139 AFTER completed verified Minecraft training."
    echo "SUCCESS: verified real inventory and checkpoint; compositor teardown is nonfatal."
  elif [[ "$outer_code" != 0 ]];then
    echo "FAILED: abnormal launcher exit $outer_code is not the verified Gamescope teardown code 139." >&2
    echo "See raw log: $console" >&2
    exit 1
  else
    echo 'SUCCESS: verified Minecraft experiment and launcher exit 0.'
  fi
  exit 0
fi
echo "FAILED: $mode $tag did not produce a verified completed journal + checkpoint (outer code $outer_code)." >&2
echo "Inspect: tail -n 60 $console" >&2
tail -n 22 "$console" >&2
exit 1
