#!/usr/bin/env bash
# Frozen independent item pickup comparison on identical heldout stage cases.
# The randomized, trained-greedy and trained-sampled actions NEVER train.
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$root"
for control in uniform_random greedy sampled; do
  case "$control" in
    uniform_random) tag=numitem_h2random1104;;
    greedy) tag=numitem_h2greedy1104;;
    sampled) tag=numitem_h2sample1104;;
  esac
  echo "HELDOUT START $tag ($control)"
  if [[ -e "logs/numeric_pickup_$tag.jsonl" ]];then
    echo "Existing journal, refusing overwrite: $tag" >&2
    exit 1
  fi
  set +e
  GMAINCRAFT_GAMESCOPE_BACKEND=headless tools/run-mcio-viewer.sh \
    .venv/bin/python -u experiments/train_numeric_item_pickup.py \
    --run-tag "$tag" --mode eval --model-tag numitem_train200a \
    --episodes 100 --max-actions 24 --case-seed 991771 \
    --policy-seed 774499 --device cpu --control "$control" \
    --report-every 20 --visible > "logs/numeric_pickup_$tag.console" 2>&1
  outer=$?
  set -e
  .venv/bin/python - "$tag" "$control" "$outer" <<'PY'
import json,sys
tag,control,outer=sys.argv[1:4]
rows=[json.loads(s) for s in open('logs/numeric_pickup_'+tag+'.jsonl') if s.strip()]
ep=[e for e in rows if e.get('stage')=='episode_committed']
last=rows[-1]
if last.get('stage')!='completed' or len(ep)!=100 or last.get('optimizer_updates')!=0:
    raise RuntimeError('Frozen pickup evaluation did not complete: '+tag)
if any(e['actor_updates']!=200 for e in ep):
    raise RuntimeError('Model weights were updated during frozen eval: '+tag)
print('FROZEN VALID',tag,'outer_exit',outer,'successes',sum(e['actual_acquired'] for e in ep),
      'out_of_bounds',sum(e.get('practice_out_of_bounds',False) for e in ep),
      'item_gains',sum(e['own_inventory_gain'] for e in ep),flush=True)
PY
  echo "HELDOUT FINISH $tag"
done
echo NUMERIC_PICKUP_HELDOUT_COMPLETED
