#!/usr/bin/env bash
# Frozen, experimental off-policy skill test on unseen real noise worlds.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
if (( $# < 4 || $# > 8 )); then
  echo 'Usage: tools/run-replay-q-evaluation.sh Q_MODEL EVAL_TAG GOAL TOTAL_UNITS [UNITS_PER_SEED] [EVAL_SEED_START] [ACTION_RNG_SEED] [q_sampled|q_greedy]' >&2
  exit 2
fi
model="$1";tag="$2";goal="$3";units="$4"
horizon="${5:-1200}";seed_start="${6:-0}";rng="${7:-731117}";mode="${8:-q_sampled}"
if [[ ! "$model" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,22}$ ]] ||
   [[ ! "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,22}$ ]] ||
   [[ "$goal" != effect:visible_crack_progress && "$goal" != effect:visible_wood_crack_progress_v2 && ! "$goal" =~ ^minecraft:[a-z0-9_./-]+$ ]] ||
   [[ ! "$units" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$horizon" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$seed_start" =~ ^[0-9]{1,12}$ ]] ||
   [[ ! "$rng" =~ ^[0-9]{1,15}$ ]] ||
   [[ "$mode" != q_sampled && "$mode" != q_greedy ]] ||
   (( horizon < 12 || horizon > 100000 )); then
  echo 'Invalid frozen replay-Q evaluation arguments' >&2;exit 2
fi
if [[ "${GMAINCRAFT_EVAL_DEVICE:-cuda}" != cuda && "${GMAINCRAFT_EVAL_DEVICE:-cuda}" != cpu && "${GMAINCRAFT_EVAL_DEVICE:-cuda}" != auto ]];then
  echo 'Invalid evaluation device (use cuda, cpu or auto)' >&2;exit 2
fi
journal="logs/fresh_eval_${tag}.jsonl"
if [[ -e "$journal" ]];then echo 'Refusing to overwrite prior frozen evaluation journal' >&2;exit 1;fi
console="logs/fresh_eval_${tag}_$(date +%Y%m%d_%H%M%S)_$$.console"
args=(--mode eval --control "$mode" --run-tag "$tag" --q-model "$model"
      --q-goal "$goal" --decisions "$units" --episode-decisions "$horizon"
      --eval-seed-start "$seed_start" --seed-root 917211 --policy-seed "$rng"
      --reward-profile wood_temporal_v9 --retain-worlds 2 --device "${GMAINCRAFT_EVAL_DEVICE:-cuda}" --visible)
if [[ "$goal" == minecraft:* ]];then args+=(--goal-item "$goal");fi
if [[ "${GMAINCRAFT_MEMORY_ONLY_NO_RGB:-0}" == 1 ]];then args+=(--memory-only-no-rgb);fi
printf 'Frozen replay-Q practice goal %s, RNG %s, unseen survival seeds\n' "$goal" "$rng"
set +e
GMAINCRAFT_GAMESCOPE_BACKEND="${GMAINCRAFT_GAMESCOPE_BACKEND:-headless}" \
  "$root/tools/run-mcio-viewer.sh" "$root/.venv/bin/python" -u \
  "$root/experiments/fresh_seed_treadmill.py" "${args[@]}" 2>&1 \
  | tee "$console" \
  | grep --line-buffered -E '^(FRESH|Traceback|ValueError:|RuntimeError:|AssertionError:)'
status=("${PIPESTATUS[@]}")
set -e
"$root/.venv/bin/python" - "$journal" "$units" "$goal" <<'PY'
import json,sys
from pathlib import Path
try:
    rows=[json.loads(x) for x in Path(sys.argv[1]).read_text().splitlines()]
    config=next(x for x in rows if x.get('stage')=='config')
    episodes=[x for x in rows if x.get('stage')=='episode_committed']
    last=rows[-1]
except (ValueError,OSError,StopIteration,IndexError) as exc:
    sys.exit('FAILED: missing verified replay-Q evaluation: '+str(exc))
if (last.get('stage')!='completed' or last['decisions']!=int(sys.argv[2]) or
    not config.get('restricted_offpolicy_skill_evaluation') or
    config.get('q_goal')!=sys.argv[3] or len(episodes)!=last['evaluated_episodes'] or
    last['new_optimizer_updates_in_eval']!=0):
    sys.exit('FAILED: replay-Q frozen evaluation journal is incomplete or mismatched')
if sys.argv[3].startswith('minecraft:'):
    print('VERIFIED Q PRACTICE:',sum(bool(x.get('goal_achieved')) for x in episodes),
          '/',len(episodes),'unseen natural worlds gained',sys.argv[3])
elif sys.argv[3]=='effect:visible_wood_crack_progress_v2':
    print('VERIFIED Q V2 PRACTICE:',sum(x.get('counts',{}).get('v2_visible_wood_crack_rises',0) for x in episodes),
          'own visible same-wood crack-rise samples in',len(episodes),'unseen worlds')
else:
    print('VERIFIED Q PRACTICE:',sum(x.get('visible_target_history_audit',{}).get('visible_crack_rise_samples',0) for x in episodes),
          'player-visible crack-rise samples in',len(episodes),'unseen worlds')
print('This is NOT a Minecraft Any% or reliable skill claim')
PY
