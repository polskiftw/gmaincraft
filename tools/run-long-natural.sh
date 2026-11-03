#!/usr/bin/env bash
# Simple, bounded, resumable real-world Minecraft RL command.
# LEGACY fixed-cohort runner retained for reproducing old checkpoints.
# Primary fresh-random-seed runner: tools/run-fresh-treadmill.sh
# This is a FOREGROUND job you can let run in a terminal; it does NOT install
# a daemon/system service or edit personal Minecraft/Gentoo settings.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
printf 'NOTE: legacy fixed-cohort launcher. For a NEW random Minecraft seed every episode, use tools/run-fresh-treadmill.sh.\n' >&2
if (( $# != 4 )); then
  echo 'Usage: tools/run-long-natural.sh start|resume RUN_TAG SOURCE_COHORT TOTAL_DECISIONS' >&2
  echo 'Generate SOURCE_COHORT first: .venv/bin/python tools/prepare-natural-training-worlds.py --tag myworlds --count 8' >&2
  exit 2
fi
operation="$1"; tag="$2"; cohort="$3"; decisions="$4"
if [[ "$operation" != 'start' && "$operation" != 'resume' ]] ||
   [[ ! "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,28}$ ]] ||
   [[ ! "$cohort" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,28}$ ]] ||
   [[ ! "$decisions" =~ ^[1-9][0-9]*$ ]]; then
  echo 'Invalid start/resume, tags or total decisions' >&2
  exit 2
fi
[[ -f "$root/logs/natural_sources_${cohort}.json" ]] || {
  echo 'Natural-world source cohort not prepared or verified' >&2;exit 2;
}
mkdir -p -- "$root/logs"
console="$root/logs/natural_console_${tag}_$(date +%Y%m%d_%H%M%S).log"
if [[ -e "$console" ]]; then
  echo 'An existing console log has this timestamp; refuse overwriting it' >&2
  exit 2
fi
# Convenience symlink for a second terminal: tail -f logs/natural_console_TAG_latest.log
ln -sfn -- "$(basename "$console")" "$root/logs/natural_console_${tag}_latest.log"
options=(--run-tag "$tag" --source-cohort "$cohort" --decisions "$decisions"
         --switch-every 1200 --update-every 96 --report-every 200
         --reward-profile wood_discovery_v2 --seed 731117 --device cuda --visible)
if [[ "$operation" == 'resume' ]]; then options+=(--resume);fi
printf 'Natural survival training (%s): %s\n' "$operation" "$tag"
printf 'Full Game/Minecraft log: %s\n' "$console"
printf 'Graceful stop from another terminal: touch logs/natural_online_%s.stop\n' "$tag"
printf 'Checkpoint: checkpoints/natural_online_%s.pt\n' "$tag"
journal="$root/logs/natural_online_${tag}.jsonl"
old_bytes=0
if [[ -f "$journal" ]]; then old_bytes="$(wc -c < "$journal")";fi
set +e
GMAINCRAFT_GAMESCOPE_BACKEND="${GMAINCRAFT_GAMESCOPE_BACKEND:-headless}" \
  "$root/tools/run-mcio-viewer.sh" "$root/.venv/bin/python" -u \
  "$root/experiments/natural_online_train.py" "${options[@]}" 2>&1 \
  | tee "$console" \
  | grep --line-buffered -E '^(NATURAL|INVALID MINING STAGE|REQUESTED GRACEFUL STOP|Traceback|ValueError:|RuntimeError:|AssertionError:)'
statuses=("${PIPESTATUS[@]}")
set -e
# Nested headless Gamescope currently sometimes exits 139 AFTER its Python
# child has cleanly written final results. Verify durable journal rather
# than blindly trusting process wrapper status or hiding actual failures.
"$root/.venv/bin/python" - "$journal" "$decisions" "$old_bytes" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1]);target=int(sys.argv[2]);prior_bytes=int(sys.argv[3])
try:
    content=path.read_bytes()
    if len(content)<=prior_bytes:
        raise ValueError('Journal did not grow during this execution')
    rows=[json.loads(line) for line in content.decode('utf8').splitlines()]
    last=rows[-1]
except (OSError,IndexError,ValueError) as e:
    print('FAILED: No complete natural survival training report:',e,file=sys.stderr)
    sys.exit(1)
if ((last.get('stage')=='completed' and int(last.get('decisions',-1))==target) or
    (last.get('stage')=='paused' and 0<=int(last.get('decisions',-1))<=target)):
    print('VERIFIED SAVED RUN:',last['stage'],'decisions=',last['decisions'],
          'updates=',last.get('updates'),'summary=',last.get('cumulative'))
else:
    print('FAILED/INCOMPLETE RUN: last record',last,file=sys.stderr)
    sys.exit(1)
PY
