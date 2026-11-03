#!/usr/bin/env bash
# Forever-fresh vanilla Minecraft seeds, no cohort to prepare or repeat.
# A FOREGROUND job in the isolated /workspace, not a system service/daemon.
# Use EPISODE_DECISIONS=2400 by default to amortize JVM/vanilla worldgen startup.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
usage() {
  echo 'Usage: tools/run-fresh-treadmill.sh start|resume RUN_TAG 100000|forever [EPISODE_DECISIONS] [REWARD_PROFILE]' >&2
  echo '       tools/run-fresh-treadmill.sh stop|status RUN_TAG' >&2
  echo 'New Minecraft noise seeds generated automatically, never reused for completed episodes.' >&2
  exit 2
}
(( $# >= 2 && $# <= 5 )) || usage
operation="$1"; tag="$2"
[[ "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,22}$ ]] || usage
mkdir -p -- logs checkpoints
journal="logs/fresh_stream_${tag}.jsonl"
marker="logs/fresh_stream_${tag}.stop"
case "$operation" in
  stop)
    (( $# == 2 )) || usage
    "$root/.venv/bin/python" - "checkpoints/fresh_stream_${tag}.lock" "$marker" <<'PY_STOP'
import fcntl,sys
from pathlib import Path
lock=Path(sys.argv[1]);marker=Path(sys.argv[2])
if not lock.exists():
    sys.exit('No active treadmill has created this run lock yet')
with lock.open('a+') as handle:
    try:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        marker.touch()
        print('Graceful stop requested; Minecraft will close and commit after its current action.')
    else:
        fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
        sys.exit('No active treadmill holds this run lock; nothing to stop')
PY_STOP
    exit 0
    ;;
  status)
    (( $# == 2 )) || usage
    "$root/.venv/bin/python" - "$journal" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists():
    print('No training journal for this run tag yet');sys.exit(0)
rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
commits=[x for x in rows if x.get('stage')=='episode_committed']
print('Completed seed episodes:',len(commits))
if commits:
    last=commits[-1]
    print('Decisions:',last['committed_decisions'],'last seed:',last['seed'])
    print('Latest episode:',last['counts'],'worldgen seconds:',last['worldgen_s'],
          'Minecraft startup seconds:',last['client_startup_s'])
print('Last journal stage:',rows[-1].get('stage'))
PY
    exit 0
    ;;
  start|resume)
    (( $# >= 3 && $# <= 5 )) || usage
    ;;
  *) usage;;
esac
limit="$3"
episode_decisions="${4:-2400}"
reward_profile="${5:-wood_discovery_v2}"
if [[ -e "$marker" ]];then
  echo "Stale stop marker exists: $marker. Inspect the previous run before starting or resuming." >&2
  exit 1
fi
if [[ "$reward_profile" != wood_discovery_v2 && "$reward_profile" != wood_progress_v3 ]]; then usage; fi
if [[ "$limit" != 'forever' && ! "$limit" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$episode_decisions" =~ ^[1-9][0-9]*$ ]] ||
   (( episode_decisions < 12 || episode_decisions > 100000 )); then usage;fi
# Check the advisory lock before launching a new nested Gamescope/Java
# session. A crash in a different bubblewrap PID namespace can leave the
# original Python blocked even though ps here shows no visible process.
# NEVER delete/recreate the lock file to evade an existing flock.
"$root/.venv/bin/python" - "checkpoints/fresh_stream_${tag}.lock" <<'PY_LOCK'
import fcntl,sys
from pathlib import Path
lock=Path(sys.argv[1])
with lock.open('a+') as handle:
    try:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit('TRAINING LOCK HELD: previous trainer or orphaned sandbox '
                 'still owns this run. Stop/close the ORIGINAL Gentoo Shell '
                 'connection and its game process before resuming. '
                 'Do NOT delete the lock file or launch a second trainer.')
    else:
        fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
PY_LOCK
console="logs/fresh_treadmill_${tag}_$(date +%Y%m%d_%H%M%S)_$$.console"
if [[ -e "$console" ]]; then echo 'Console log already exists' >&2;exit 1;fi
ln -sfn -- "$(basename "$console")" "logs/fresh_treadmill_${tag}_latest.log"
opts=(--run-tag "$tag" --seed-root 917211 --policy-seed 731117
      --episode-decisions "$episode_decisions" --update-every 96 --report-every 100
      --reward-profile "$reward_profile" --retain-worlds 2 --device cuda --visible)
if [[ "$limit" == 'forever' ]]; then opts+=(--until-stopped)
else opts+=(--decisions "$limit");fi
if [[ "$operation" == resume ]];then opts+=(--resume);fi
old_bytes=0
if [[ -f "$journal" ]]; then old_bytes="$(wc -c < "$journal")";fi
printf 'Fresh-seed natural survival training: %s | run %s | target %s\n' "$operation" "$tag" "$limit"
printf 'Every episode generates a NEW vanilla Minecraft seed; seed RNG is not model input.\n'
printf 'Watch: tail -f logs/fresh_treadmill_%s_latest.log\n' "$tag"
printf 'Status: tools/run-fresh-treadmill.sh status %s\n' "$tag"
printf 'Stop: tools/run-fresh-treadmill.sh stop %s\n' "$tag"
set +e
GMAINCRAFT_GAMESCOPE_BACKEND="${GMAINCRAFT_GAMESCOPE_BACKEND:-headless}" \
  "$root/tools/run-mcio-viewer.sh" "$root/.venv/bin/python" -u \
  "$root/experiments/fresh_seed_treadmill.py" "${opts[@]}" 2>&1 \
    | tee "$console" \
    | grep --line-buffered -E '^(FRESH|INVALID MINING STAGE|Traceback|ValueError:|RuntimeError:|AssertionError:)'
status=("${PIPESTATUS[@]}")
set -e
# Validate actual NEW journal rows and committed checkpoint, not headless
# Gamescope's known post-child shutdown signal 139.
"$root/.venv/bin/python" - "$journal" "$old_bytes" "$limit" "checkpoints/fresh_stream_${tag}.pt" <<'PY'
import json,sys
from pathlib import Path
journal=Path(sys.argv[1]);old_bytes=int(sys.argv[2]);limit=sys.argv[3]
checkpoint=Path(sys.argv[4])
try:
    data=journal.read_bytes()
    if len(data)<=old_bytes:raise ValueError('Journal did not grow')
    rows=[json.loads(s) for s in data.decode('utf8').splitlines()]
    last=rows[-1]
except (OSError,ValueError,IndexError) as exc:
    sys.exit('FAILED: no new complete journal: '+str(exc))
if last.get('stage') not in ('completed','paused'):
    sys.exit('FAILED: run did not end cleanly: '+str(last))
if limit!='forever' and last['stage']=='completed' and last['decisions']!=int(limit):
    sys.exit('FAILED: expected target decisions were not actually completed')
if not checkpoint.is_file():sys.exit('FAILED: missing persisted training checkpoint')
import torch
saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
if (saved['next_episode']!=last['episodes'] or
    saved['decisions']!=last['decisions']):
    sys.exit('FAILED: model checkpoint and journal cursor disagree')
print('VERIFIED checkpoint + journal:',last['stage'],'episodes',last['episodes'],
      'decisions',last['decisions'],'updates',last['updates'])
PY
