#!/usr/bin/env bash
# Evaluate frozen policy or control on disjoint never-trained natural seeds.
# Pass TRAIN_RUN='-' for no checkpoint (uniform_random/untrained controls).
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
if (( $# < 3 || $# > 5 )); then
  echo 'Usage: tools/run-fresh-evaluation.sh TRAIN_RUN|- EVAL_TAG TOTAL_DECISIONS [policy_sampled|policy_greedy|uniform_random|untrained] [EPISODE_DECISIONS]' >&2
  exit 2
fi
training="$1"; tag="$2"; decisions="$3"; mode="${4:-policy_sampled}"
horizon="${5:-1200}"
if [[ ! "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,22}$ ]] ||
   [[ "$training" != '-' && ! "$training" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,22}$ ]] ||
   [[ ! "$decisions" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$horizon" =~ ^[1-9][0-9]*$ ]] ||
   (( horizon < 12 || horizon > 100000 )) ||
   [[ "$mode" != policy_sampled && "$mode" != policy_greedy &&
      "$mode" != uniform_random && "$mode" != untrained ]];then
  echo 'Invalid frozen evaluation arguments' >&2;exit 2
fi
if [[ "$training" == '-' && ( "$mode" == policy_sampled || "$mode" == policy_greedy ) ]] ||
   [[ "$training" != '-' && ( "$mode" == uniform_random || "$mode" == untrained ) ]];then
  echo 'Policy needs a trained checkpoint; random/untrained controls MUST use TRAIN_RUN=-' >&2
  exit 2
fi
mkdir -p -- logs
journal="logs/fresh_eval_${tag}.jsonl"
if [[ -e "$journal" ]];then echo 'Refusing to overwrite existing evaluation report' >&2;exit 1;fi
console="logs/fresh_eval_${tag}_$(date +%Y%m%d_%H%M%S)_$$.console"
args=(--mode eval --control "$mode" --run-tag "$tag" --decisions "$decisions"
      --episode-decisions "$horizon" --seed-root 917211 --policy-seed 731117
      --reward-profile wood_discovery_v2 --retain-worlds 2 --device cuda --visible)
if [[ "$training" != '-' ]];then args+=(--model-run "$training");fi
printf 'Frozen evaluation: %s, %s, %s decisions; fresh held-out Minecraft seeds\n' "$training" "$mode" "$decisions"
set +e
GMAINCRAFT_GAMESCOPE_BACKEND="${GMAINCRAFT_GAMESCOPE_BACKEND:-headless}" \
  "$root/tools/run-mcio-viewer.sh" "$root/.venv/bin/python" -u \
  "$root/experiments/fresh_seed_treadmill.py" "${args[@]}" 2>&1 \
    | tee "$console" \
    | grep --line-buffered -E '^(FRESH|INVALID MINING STAGE|Traceback|ValueError:|RuntimeError:|AssertionError:)'
status=("${PIPESTATUS[@]}")
set -e
"$root/.venv/bin/python" - "$journal" "$decisions" <<'PY'
import json,sys
from pathlib import Path
try:
    rows=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines()]
    last=rows[-1]
except (OSError,ValueError,IndexError) as e:
    sys.exit('FAILED: Missing or invalid frozen-eval journal: '+str(e))
if last.get('stage')!='completed' or last['decisions']!=int(sys.argv[2]) or last['mode']!='eval':
    sys.exit('FAILED: Frozen evaluation did not complete')
print('VERIFIED frozen holdout:',last['episodes'],'fresh seeds,',last['decisions'],
      'decisions, wood inventory events:',last['totals'].get('wood_inventory_gain',0))
PY
