#!/usr/bin/env bash
# Frozen, GOAL-CONDITIONED skill-only evaluation on unseen natural seeds.
# This is NOT a high-level planner or scored full-game Any% run.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
if (( $# < 4 || $# > 8 )); then
  echo 'Usage: tools/run-hindsight-evaluation.sh GOAL_MODEL EVAL_TAG minecraft:ITEM TOTAL_DECISIONS [EPISODE_DECISIONS] [EVAL_SEED_START] [ACTION_RNG_SEED] [goal_sampled|goal_greedy]' >&2
  exit 2
fi
model="$1";tag="$2";item="$3";decisions="$4"
horizon="${5:-1200}"; seed_start="${6:-0}";rng="${7:-731117}";mode="${8:-goal_sampled}"
if [[ ! "$model" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,22}$ ]] ||
   [[ ! "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,22}$ ]] ||
   [[ ! "$item" =~ ^minecraft:[a-z0-9_./-]+$ ]] ||
   [[ ! "$decisions" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$horizon" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$seed_start" =~ ^[0-9]{1,12}$ ]] ||
   [[ ! "$rng" =~ ^[0-9]{1,15}$ ]] ||
   [[ "$mode" != goal_sampled && "$mode" != goal_greedy ]] ||
   (( horizon < 12 || horizon > 100000 )); then
  echo 'Invalid frozen goal-evaluation arguments' >&2;exit 2
fi
eval_device="${GMAINCRAFT_EVAL_DEVICE:-cuda}"
if [[ "$eval_device" != cuda && "$eval_device" != cpu && "$eval_device" != auto ]];then
  echo 'GMAINCRAFT_EVAL_DEVICE must be cuda, cpu or auto' >&2;exit 2
fi
reward_profile="$("$root/.venv/bin/python" - "$model" <<'PY'
import sys
from pathlib import Path
import torch
root=Path.cwd()
goal=torch.load(root/'checkpoints'/('hindsight_'+sys.argv[1]+'.pt'),map_location='cpu',weights_only=True)
source=goal['metadata']['source_run']
prior=torch.load(root/'checkpoints'/('fresh_stream_'+source+'.pt'),map_location='cpu',weights_only=True)
print(prior['metadata']['reward_profile'])
PY
)"
mkdir -p logs
journal="logs/fresh_eval_${tag}.jsonl"
if [[ -e "$journal" ]];then echo 'Refusing to overwrite an existing held-out evaluation report' >&2;exit 1;fi
console="logs/fresh_eval_${tag}_$(date +%Y%m%d_%H%M%S)_$$.console"
printf 'Goal-conditioned frozen evaluation of %s for observed item %s\n' "$model" "$item"
printf 'Unseen natural seed stream, %s game-time units, %s neural policy\n' "$decisions" "$mode"
set +e
GMAINCRAFT_GAMESCOPE_BACKEND="${GMAINCRAFT_GAMESCOPE_BACKEND:-headless}" \
  "$root/tools/run-mcio-viewer.sh" "$root/.venv/bin/python" -u \
  "$root/experiments/fresh_seed_treadmill.py" \
  --mode eval --control "$mode" --run-tag "$tag" --goal-model "$model" --goal-item "$item" \
  --decisions "$decisions" --episode-decisions "$horizon" --eval-seed-start "$seed_start" \
  --seed-root 917211 --policy-seed "$rng" --reward-profile "$reward_profile" \
  --retain-worlds 2 --device "$eval_device" --visible 2>&1 | tee "$console" \
   | grep --line-buffered -E '^(FRESH|Traceback|ValueError:|RuntimeError:|AssertionError:)'
statuses=("${PIPESTATUS[@]}")
set -e
"$root/.venv/bin/python" - "$journal" "$decisions" "$item" <<'PY'
import json,sys
from pathlib import Path
journal=Path(sys.argv[1])
try:
    rows=[json.loads(row) for row in journal.read_text().splitlines()]
    last=rows[-1]
    episodes=[row for row in rows if row.get('stage')=='episode_committed']
    config=next(row for row in rows if row.get('stage')=='config')
except (OSError,ValueError,StopIteration,IndexError) as exc:
    sys.exit('FAILED: no verified frozen goal evaluation: '+str(exc))
if (last.get('stage')!='completed' or last.get('decisions')!=int(sys.argv[2])
    or config.get('goal_item')!=sys.argv[3] or
    not config.get('restricted_goal_conditioned_skill_evaluation') or
    len(episodes)!=last.get('evaluated_episodes')):
    sys.exit('FAILED: goal evaluation journal incomplete or wrong controller')
wins=sum(bool(row.get('goal_achieved')) for row in episodes)
print('VERIFIED GOAL FROZEN EVAL:',wins,'/',len(episodes),'fresh worlds gained',sys.argv[3],
      '| THIS IS NOT AN ANY% SCORE')
PY
