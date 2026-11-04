#!/usr/bin/env bash
# Evaluate frozen policy or control on disjoint never-trained natural seeds.
# Pass TRAIN_RUN='-' for no checkpoint (uniform_random/untrained controls).
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
if (( $# < 3 || $# > 8 )); then
  echo 'Usage: tools/run-fresh-evaluation.sh TRAIN_RUN|- EVAL_TAG TOTAL_DECISIONS [policy_sampled|policy_greedy|uniform_random|untrained] [EPISODE_DECISIONS] [REWARD_PROFILE] [EVAL_SEED_START] [EVAL_ACTION_RNG_SEED]' >&2
  exit 2
fi
training="$1"; tag="$2"; decisions="$3"; mode="${4:-policy_sampled}"
horizon="${5:-1200}"
reward_profile="${6:-wood_discovery_v2}"
eval_seed_start="${7:-0}"
# Optional independent rollout RNG for repeated frozen comparisons. This
# changes sampled actions, NOT model weights or the Minecraft seed stream.
eval_action_rng_seed="${8:-731117}"
if [[ ! "$eval_action_rng_seed" =~ ^[0-9]{1,15}$ ]]; then
  echo "Invalid evaluation action RNG seed" >&2; exit 2
fi
# Preserve the CUDA default, but permit safe CPU-only frozen inference
# when an isolated shell cannot initialize the NVIDIA CUDA device.
eval_device="${GMAINCRAFT_EVAL_DEVICE:-cuda}"
if [[ "$eval_device" != cuda && "$eval_device" != cpu && "$eval_device" != auto ]]; then
  echo "Invalid GMAINCRAFT_EVAL_DEVICE: use cuda, cpu, or auto" >&2; exit 2
fi
if [[ "$reward_profile" != wood_discovery_v2 && "$reward_profile" != wood_progress_v3 && "$reward_profile" != wood_persistence_v4 && "$reward_profile" != wood_target_v5 && "$reward_profile" != wood_engagement_v6 && "$reward_profile" != wood_motor_v7 && "$reward_profile" != wood_sustain_v8 && "$reward_profile" != wood_temporal_v9 ]]; then echo "Invalid reward profile" >&2; exit 2; fi
if [[ ! "$eval_seed_start" =~ ^[0-9]{1,9}$ ]]; then
  echo "Invalid held-out seed start index" >&2; exit 2
fi
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
      --eval-seed-start "$eval_seed_start"
      --episode-decisions "$horizon" --seed-root 917211 --policy-seed "$eval_action_rng_seed"
      --reward-profile "$reward_profile" --retain-worlds 2 --device "$eval_device" --visible)
if [[ "$training" != '-' ]];then args+=(--model-run "$training");fi
printf 'Frozen evaluation: %s, %s, %s decisions; fresh held-out Minecraft seeds\n' "$training" "$mode" "$decisions"
printf 'Frozen policy compute device: %s; rollout RNG: %s\n' "$eval_device" "$eval_action_rng_seed"
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
print('VERIFIED frozen holdout:',last.get('evaluated_episodes',last['episodes']),'fresh seeds,',last['decisions'],
      'decisions, wood inventory events:',last['totals'].get('wood_inventory_gain',0))
PY
